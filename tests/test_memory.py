import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from llm_negotiation.communication import MessageBus, MessageType, MessageVisibility
from llm_negotiation.domain import (
    AcceptAction,
    ActionType,
    CounterAction,
    IssueId,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ProposeAction,
    ReservationPolicy,
)
from llm_negotiation.memory import (
    ActionMemoryEvent,
    AgentMemory,
    AgentObservation,
    AgentProfile,
    DeterministicMemorySummarizer,
    MemoryIsolationError,
    MemoryLimits,
    MemoryPersistenceError,
    MemorySnapshot,
    ObservationMemoryEvent,
    OutcomeMemoryEvent,
    PublicAgentIdentity,
    ReceivedMessageMemoryEvent,
    SummaryMemoryEvent,
    load_memory_snapshot,
    save_memory_snapshot,
)
from llm_negotiation.protocol import NegotiationProtocol, ProtocolPhase


ALICE = ParticipantId("alice")
BOB = ParticipantId("bob")
MEDIATOR = ParticipantId("mediator")
AUDITOR = ParticipantId("auditor")
PRICE = IssueId("price")
NOW = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)


def scenario(scenario_id="memory-test"):
    return NegotiationScenario(
        scenario_id=scenario_id,
        title="Memory Test",
        participants=(
            Participant(
                participant_id=ALICE,
                display_name="Alice",
                role=ParticipantRole.BUYER,
            ),
            Participant(
                participant_id=BOB,
                display_name="Bob",
                role=ParticipantRole.SELLER,
            ),
        ),
        issues=(
            NumericIssue(
                issue_id=PRICE,
                name="Price",
                minimum=0.0,
                maximum=100.0,
                unit="USD",
            ),
        ),
    )


def profile(participant_id, direction, secret):
    participant = scenario().participant(participant_id)
    return AgentProfile(
        identity=PublicAgentIdentity(
            participant=participant,
            persona=f"Public persona for {participant.display_name}",
        ),
        preferences=ParticipantPreferences(
            participant_id=participant_id,
            issue_preferences=(
                NumericPreference(
                    issue_id=PRICE,
                    weight=1.0,
                    direction=direction,
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=0.3,
                batna_utility=0.2,
                batna_description=secret,
            ),
        ),
    )


def alice_profile():
    return profile(ALICE, PreferenceDirection.MINIMIZE, "ALICE PRIVATE BATNA")


def bob_profile():
    return profile(BOB, PreferenceDirection.MAXIMIZE, "BOB PRIVATE BATNA")


def offer(offer_id, price):
    return Offer(
        offer_id=OfferId(offer_id),
        values=(NumericIssueValue(issue_id=PRICE, value=price),),
    )


def observation(state, owner_profile, active_scenario=None):
    return AgentObservation(
        scenario=active_scenario or scenario(),
        session=state,
        own_profile=owner_profile,
    )


def bus():
    return MessageBus(
        negotiation_id="memory-test",
        participants=(ALICE, BOB, MEDIATOR),
        mediator_ids=(MEDIATOR,),
        audit_reader_ids=(AUDITOR,),
    )


def test_long_term_and_working_memory_contain_required_owner_specific_facts():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=6, initial_turn=ALICE)
    created = engine.create_session()
    state = engine.transition(
        created,
        ProposeAction(actor_id=ALICE, round_number=1, offer=offer("offer-1", 40.0)),
    ).state
    messages = bus()
    messages.send(
        sender=BOB,
        recipients=(ALICE,),
        timestamp=NOW,
        message_type=MessageType.QUESTION,
        visibility=MessageVisibility.DIRECT_PRIVATE,
        content="Can you explain the price?",
    )
    memory = AgentMemory(
        alice_profile(),
        active_scenario,
        goals=("Reach a valid agreement without crossing reservation utility.",),
        strategy_guidance=("Concede according to the configured baseline.",),
    )

    snapshot = memory.update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
        visible_messages=messages.inbox(ALICE),
        active_plan=("Evaluate the outstanding offer.", "Counter if it is below aspiration."),
    )

    assert snapshot.owner_id == ALICE
    assert snapshot.long_term.role is ParticipantRole.BUYER
    assert snapshot.long_term.goals
    assert snapshot.long_term.protocol_rules
    assert snapshot.long_term.issue_descriptions == active_scenario.issues
    assert snapshot.long_term.strategy_guidance
    assert snapshot.long_term.public_persona == "Public persona for Alice"
    assert snapshot.working.legal_actions == engine.legal_action_types(state, ALICE)
    assert snapshot.working.outstanding_offer == offer("offer-1", 40.0)
    assert snapshot.working.round_number == 1
    assert snapshot.working.deadline_round == 6
    assert snapshot.working.rounds_remaining == 5
    assert snapshot.working.visible_messages[0].content == "Can you explain the price?"
    assert snapshot.working.current_utility_estimates[0].utility == pytest.approx(0.6)
    assert snapshot.working.active_plan == (
        "Evaluate the outstanding offer.",
        "Counter if it is below aspiration.",
    )


def test_snapshot_replaces_full_session_at_policy_boundary():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    state = engine.create_session()
    memory = AgentMemory(alice_profile(), active_scenario)

    snapshot = memory.update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )

    assert isinstance(snapshot, MemorySnapshot)
    assert "session" not in MemorySnapshot.model_fields
    assert not hasattr(snapshot, "event_sequence")
    assert snapshot.working.legal_actions == (ActionType.PROPOSE, ActionType.WITHDRAW)

    with pytest.raises(MemoryIsolationError, match="authoritative protocol state"):
        AgentMemory(alice_profile(), active_scenario).update(
            observation(state, alice_profile()),
            legal_actions=(ActionType.ACCEPT,),
        )


def test_episodic_memory_records_owned_actions_observations_messages_and_outcomes():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    created = engine.create_session()
    proposed = engine.transition(
        created,
        ProposeAction(actor_id=ALICE, round_number=1, offer=offer("offer-1", 40.0)),
    ).state
    agreed = engine.transition(
        proposed,
        AcceptAction(actor_id=BOB, round_number=1, offer_id=OfferId("offer-1")),
    ).state
    messages = bus()
    messages.send(
        sender=BOB,
        recipients=(ALICE,),
        timestamp=NOW,
        message_type=MessageType.ANSWER,
        visibility=MessageVisibility.DIRECT_PRIVATE,
        content="I can accept that price.",
    )
    memory = AgentMemory(alice_profile(), active_scenario)

    snapshot = memory.update(
        observation(agreed, alice_profile()),
        legal_actions=engine.legal_action_types(agreed, ALICE),
        visible_messages=messages.inbox(ALICE),
    )

    assert any(isinstance(event, ActionMemoryEvent) for event in snapshot.episodic.events)
    assert any(
        isinstance(event, ObservationMemoryEvent) and event.observed_action is not None
        for event in snapshot.episodic.events
    )
    assert any(isinstance(event, ReceivedMessageMemoryEvent) for event in snapshot.episodic.events)
    assert any(isinstance(event, OutcomeMemoryEvent) for event in snapshot.episodic.events)
    assert snapshot.working.phase is ProtocolPhase.AGREED
    assert snapshot.working.outcome == agreed.outcome


class FixedSummary:
    def __init__(self):
        self.calls = 0

    def summarize(self, events, max_characters):
        self.calls += 1
        return "Deterministic factual compacted history."[:max_characters]


def long_state(action_count=12):
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=20)
    state = engine.create_session()
    for index in range(action_count):
        actor = state.current_turn
        next_offer = offer(f"offer-{index + 1}", 20.0 + index * 3.0)
        if state.latest_valid_offer is None:
            action = ProposeAction(
                actor_id=actor,
                round_number=state.round_number,
                offer=next_offer,
            )
        else:
            action = CounterAction(
                actor_id=actor,
                round_number=state.round_number,
                offer=next_offer,
                responds_to=state.latest_valid_offer.offer_id,
            )
        state = engine.transition(state, action).state
    return active_scenario, engine, state


def test_long_negotiation_compacts_deterministically_within_event_and_token_limits():
    active_scenario, engine, state = long_state()
    limits = MemoryLimits(
        episodic_event_limit=4,
        episodic_character_limit=900,
        episodic_token_limit=225,
        summary_character_limit=120,
        working_message_limit=2,
        working_character_limit=512,
        working_token_limit=128,
    )
    first_summarizer = FixedSummary()
    second_summarizer = FixedSummary()
    first = AgentMemory(alice_profile(), active_scenario, limits=limits, summarizer=first_summarizer)
    second = AgentMemory(alice_profile(), active_scenario, limits=limits, summarizer=second_summarizer)

    first_snapshot = first.update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )
    second_snapshot = second.update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )

    assert first_snapshot == second_snapshot
    assert first_summarizer.calls > 0
    assert first_snapshot.episodic.compaction_count > 0
    assert len(first_snapshot.episodic.events) <= limits.episodic_event_limit
    assert first_snapshot.episodic.approximate_characters <= 900
    assert first_snapshot.episodic.approximate_tokens <= 225
    assert any(isinstance(event, SummaryMemoryEvent) for event in first_snapshot.episodic.events)
    assert first_snapshot.working.outstanding_offer == state.latest_valid_offer
    assert first_snapshot.working.round_number == state.round_number
    assert first_snapshot.working.public_event_count == len(state.event_sequence)
    assert first_snapshot.working.recent_offers[-1].offer == state.latest_valid_offer
    assert len(state.event_sequence) == 12


def test_default_non_llm_summarizer_is_deterministic_and_fact_only():
    active_scenario, engine, state = long_state(action_count=6)
    limits = MemoryLimits(episodic_event_limit=2, episodic_character_limit=700)
    first = AgentMemory(
        alice_profile(),
        active_scenario,
        limits=limits,
        summarizer=DeterministicMemorySummarizer(),
    ).update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )
    second = AgentMemory(
        alice_profile(),
        active_scenario,
        limits=limits,
        summarizer=DeterministicMemorySummarizer(),
    ).update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )

    assert first.episodic == second.episodic
    summary = next(event for event in first.episodic.events if isinstance(event, SummaryMemoryEvent))
    assert "reasoning" not in summary.content.lower()
    assert "observed" in summary.content or "round" in summary.content


def test_working_messages_are_bounded_by_count_and_approximate_tokens():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    state = engine.create_session()
    messages = bus()
    for index in range(5):
        messages.send(
            sender=BOB,
            recipients=(ALICE,),
            timestamp=NOW + timedelta(seconds=index),
            message_type=MessageType.QUESTION,
            visibility=MessageVisibility.DIRECT_PRIVATE,
            content=(f"message-{index} " + "x" * 900),
        )
    limits = MemoryLimits(
        working_message_limit=2,
        working_character_limit=512,
        working_token_limit=128,
    )
    memory = AgentMemory(alice_profile(), active_scenario, limits=limits)

    snapshot = memory.update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
        visible_messages=messages.inbox(ALICE),
    )

    assert len(snapshot.working.visible_messages) <= 2
    assert snapshot.working.approximate_characters <= 512
    assert snapshot.working.approximate_tokens <= 128
    assert snapshot.working.visible_messages[-1].message_id.value == "message-5"


def test_memory_isolation_rejects_other_agent_observations_and_private_messages():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    state = engine.create_session()
    alice_memory = AgentMemory(alice_profile(), active_scenario)
    messages = bus()
    private_for_mediator = messages.send(
        sender=BOB,
        recipients=(MEDIATOR,),
        timestamp=NOW,
        message_type=MessageType.CRITIQUE,
        visibility=MessageVisibility.MEDIATOR_ONLY,
        content="Private mediator critique.",
    )

    with pytest.raises(MemoryIsolationError, match="another participant"):
        alice_memory.update(
            observation(state, bob_profile()),
            legal_actions=engine.legal_action_types(state, BOB),
        )
    with pytest.raises(MemoryIsolationError, match="outside participant"):
        alice_memory.update(
            observation(state, alice_profile()),
            legal_actions=engine.legal_action_types(state, ALICE),
            visible_messages=(private_for_mediator,),
        )


def test_agent_snapshots_do_not_leak_opponent_private_preferences():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    state = engine.create_session()
    alice_snapshot = AgentMemory(alice_profile(), active_scenario).update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )
    bob_snapshot = AgentMemory(bob_profile(), active_scenario).update(
        observation(state, bob_profile()),
        legal_actions=engine.legal_action_types(state, BOB),
    )

    alice_json = alice_snapshot.model_dump_json()
    bob_json = bob_snapshot.model_dump_json()
    assert "ALICE PRIVATE BATNA" in alice_json
    assert "BOB PRIVATE BATNA" not in alice_json
    assert "BOB PRIVATE BATNA" in bob_json
    assert "ALICE PRIVATE BATNA" not in bob_json
    assert "opponent_preferences" not in alice_json


def test_reset_clears_transient_memory_and_optionally_retains_learning():
    first_scenario = scenario()
    engine = NegotiationProtocol(first_scenario, maximum_rounds=3)
    state = engine.create_session()
    memory = AgentMemory(
        alice_profile(),
        first_scenario,
        strategy_guidance=("Use measured concessions.",),
        learned_strategy_data=("Past counterparts responded to clear explanations.",),
    )
    memory.update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )
    memory.remember_strategy_fact("A newly learned bounded strategy fact.")

    second_scenario = scenario("memory-test-2")
    memory.reset_for_negotiation(second_scenario, retain_learned_strategy=True)

    assert memory.working is None
    assert memory.episodic.events == ()
    assert memory.long_term.negotiation_id == "memory-test-2"
    assert memory.long_term.strategy_guidance == ("Use measured concessions.",)
    assert memory.long_term.learned_strategy_data == (
        "Past counterparts responded to clear explanations.",
        "A newly learned bounded strategy fact.",
    )

    memory.reset_for_negotiation(scenario("memory-test-3"), retain_learned_strategy=False)
    assert memory.long_term.learned_strategy_data == ()


def test_learned_strategy_facts_are_concise_deduplicated_and_bounded():
    memory = AgentMemory(
        alice_profile(),
        scenario(),
        limits=MemoryLimits(learned_strategy_limit=2),
    )

    memory.remember_strategy_fact("fact one")
    memory.remember_strategy_fact("fact two")
    memory.remember_strategy_fact("fact two")
    memory.remember_strategy_fact("fact three")

    assert memory.long_term.learned_strategy_data == ("fact two", "fact three")
    with pytest.raises(ValueError, match="at most 500"):
        memory.remember_strategy_fact("x" * 501)


def test_versioned_snapshot_persistence_and_reload(tmp_path):
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    state = engine.create_session()
    configured_limits = MemoryLimits(
        episodic_event_limit=7,
        episodic_character_limit=2_000,
        working_message_limit=3,
    )
    snapshot = AgentMemory(alice_profile(), active_scenario, limits=configured_limits).update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
    )
    path = tmp_path / "alice-memory.json"

    save_memory_snapshot(snapshot, path)
    restored = load_memory_snapshot(path)
    resumed = AgentMemory.from_snapshot(restored)

    assert restored == snapshot
    assert restored.schema_version == "1.0"
    assert restored.limits == configured_limits
    assert resumed.limits == configured_limits
    assert resumed.snapshot() == snapshot

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "99.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MemoryPersistenceError, match="unsupported memory snapshot"):
        load_memory_snapshot(path)


def test_raw_protocol_and_message_audits_remain_outside_compacted_memory():
    active_scenario, engine, state = long_state(action_count=10)
    messages = bus()
    for index in range(4):
        messages.send(
            sender=BOB,
            recipients=(ALICE,),
            timestamp=NOW + timedelta(seconds=index),
            message_type=MessageType.INTENT_SIGNAL,
            visibility=MessageVisibility.DIRECT_PRIVATE,
            content=f"Publicly deliverable fact {index}.",
        )
    limits = MemoryLimits(episodic_event_limit=3, episodic_character_limit=700)
    snapshot = AgentMemory(alice_profile(), active_scenario, limits=limits).update(
        observation(state, alice_profile()),
        legal_actions=engine.legal_action_types(state, ALICE),
        visible_messages=messages.inbox(ALICE),
    )

    assert len(snapshot.episodic.events) <= 3
    assert len(state.event_sequence) == 10
    assert len(messages.audit_view(AUDITOR)) == 4
    assert "event_sequence" not in snapshot.model_dump_json()


def test_reload_preserves_message_cursor_when_working_view_was_truncated():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    state = engine.create_session()
    messages = bus()
    for index in range(5):
        messages.send(
            sender=BOB,
            recipients=(ALICE,),
            timestamp=NOW + timedelta(seconds=index),
            message_type=MessageType.QUESTION,
            visibility=MessageVisibility.DIRECT_PRIVATE,
            content=f"Question {index}.",
        )
    limits = MemoryLimits(working_message_limit=1, working_character_limit=512)
    memory = AgentMemory(alice_profile(), active_scenario, limits=limits)
    snapshot = memory.update(
        observation(state, alice_profile()),
        visible_messages=messages.inbox(ALICE),
    )
    received_before = sum(
        isinstance(event, ReceivedMessageMemoryEvent) for event in snapshot.episodic.events
    )

    resumed = AgentMemory.from_snapshot(snapshot)
    after = resumed.update(
        observation(state, alice_profile()),
        visible_messages=messages.inbox(ALICE),
    )
    received_after = sum(
        isinstance(event, ReceivedMessageMemoryEvent) for event in after.episodic.events
    )

    assert len(snapshot.working.visible_messages) == 1
    assert snapshot.working.last_visible_message_sequence == 5
    assert received_before == 5
    assert received_after == received_before


def test_memory_rejects_a_forked_previously_observed_protocol_history():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    created = engine.create_session()
    proposed = engine.transition(
        created,
        ProposeAction(actor_id=ALICE, round_number=1, offer=offer("offer-1", 40.0)),
    ).state
    countered = engine.transition(
        proposed,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=offer("offer-2", 60.0),
            responds_to=OfferId("offer-1"),
        ),
    ).state
    memory = AgentMemory(alice_profile(), active_scenario)
    memory.update(observation(countered, alice_profile()))

    first_event = countered.event_sequence[0]
    changed_action = first_event.action.model_copy(
        update={"offer": offer("offer-1", 41.0)}
    )
    changed_event = first_event.model_copy(update={"action": changed_action})
    forked = countered.model_copy(
        update={"event_sequence": (changed_event, countered.event_sequence[1])}
    )

    with pytest.raises(MemoryIsolationError, match="does not continue"):
        memory.update(observation(forked, alice_profile()))


def test_failed_memory_update_is_atomic():
    active_scenario = scenario()
    engine = NegotiationProtocol(active_scenario, maximum_rounds=3)
    created = engine.create_session()
    proposed = engine.transition(
        created,
        ProposeAction(actor_id=ALICE, round_number=1, offer=offer("offer-1", 40.0)),
    ).state
    memory = AgentMemory(alice_profile(), active_scenario)

    with pytest.raises(ValidationError, match="active plan"):
        memory.update(
            observation(proposed, alice_profile()),
            active_plan=("x" * 501,),
        )

    assert memory.working is None
    assert memory.episodic.events == ()
