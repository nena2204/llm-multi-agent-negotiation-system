import json

import pytest
from pydantic import ValidationError

from llm_negotiation.beliefs import (
    GoalKind,
    OpponentBeliefState,
    OpponentModelConfiguration,
    OpponentModelMode,
    StrategyKind,
)
from llm_negotiation.communication import MessageType, MessageVisibility
from llm_negotiation.domain import (
    CategoricalIssue,
    CategoricalIssueValue,
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
    WithdrawAction,
)
from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    LLMProvider,
    ModelConfiguration,
    RetryConfiguration,
)
from llm_negotiation.llm_policy import LLMNegotiationPolicy
from llm_negotiation.memory import (
    AgentMemory,
    AgentObservation,
    AgentProfile,
    BeliefSnapshotMemoryEvent,
    MemorySnapshot,
    PublicAgentIdentity,
)
from llm_negotiation.opponent import (
    HeuristicOpponentModeller,
    LLMOpponentModeller,
    ObservableOpponentEvidence,
    ObservedAction,
    ObservedMessage,
    calibrate_belief,
    evidence_from_memory,
)
from llm_negotiation.protocol import NegotiationProtocol


ALICE = ParticipantId("alice")
BOB = ParticipantId("bob")
CAROL = ParticipantId("carol")
PRICE = IssueId("price")
QUALITY = IssueId("quality")
DELIVERY = IssueId("delivery")


def scenario(multi_issue=False):
    issues = [NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0)]
    if multi_issue:
        issues.extend(
            (
                NumericIssue(issue_id=QUALITY, name="Quality", minimum=0.0, maximum=10.0),
                CategoricalIssue(
                    issue_id=DELIVERY,
                    name="Delivery",
                    choices=("slow", "fast"),
                ),
            )
        )
    return NegotiationScenario(
        scenario_id="belief-test",
        title="Belief Test",
        participants=(
            Participant(participant_id=ALICE, display_name="Alice", role=ParticipantRole.BUYER),
            Participant(participant_id=BOB, display_name="Bob", role=ParticipantRole.SELLER),
        ),
        issues=tuple(issues),
    )


def offer(identifier, price, quality=None, delivery=None):
    values = [NumericIssueValue(issue_id=PRICE, value=price)]
    if quality is not None:
        values.append(NumericIssueValue(issue_id=QUALITY, value=quality))
    if delivery is not None:
        values.append(CategoricalIssueValue(issue_id=DELIVERY, value=delivery))
    return Offer(offer_id=OfferId(identifier), values=tuple(values))


def observed(identifier, round_number, proposed_offer):
    return ObservedAction(
        evidence_event_id=identifier,
        round_number=round_number,
        action=ProposeAction(
            actor_id=BOB,
            recipients=(ALICE,),
            round_number=round_number,
            offer=proposed_offer,
        ),
    )


def evidence(actions=(), multi_issue=False, round_number=1):
    return ObservableOpponentEvidence(
        negotiation_id="belief-test",
        observer_id=ALICE,
        opponent_id=BOB,
        scenario=scenario(multi_issue=multi_issue),
        current_round=round_number,
        actions=actions,
    )


def fake_configuration():
    return ModelConfiguration(
        provider=LLMProvider.FAKE,
        model="fake-opponent-model",
        retry=RetryConfiguration(max_retries=0),
    )


def alice_profile():
    return AgentProfile(
        identity=PublicAgentIdentity(
            participant=scenario().participant(ALICE),
            persona="Public Alice",
        ),
        preferences=ParticipantPreferences(
            participant_id=ALICE,
            issue_preferences=(
                NumericPreference(
                    issue_id=PRICE,
                    weight=1.0,
                    direction=PreferenceDirection.MINIMIZE,
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=0.4,
                batna_utility=0.3,
                batna_description="ALICE PRIVATE BATNA",
            ),
        ),
    )


def memory_with_bob_offer():
    public = scenario()
    protocol = NegotiationProtocol(public, maximum_rounds=4, initial_turn=BOB)
    state = protocol.create_session()
    state = protocol.transition(
        state,
        ProposeAction(
            actor_id=BOB,
            recipients=(ALICE,),
            round_number=1,
            offer=offer("bob-offer", 80.0),
        ),
    ).state
    profile = alice_profile()
    memory = AgentMemory(profile, public)
    snapshot = memory.update(
        AgentObservation(scenario=public, session=state, own_profile=profile),
        legal_actions=protocol.legal_action_types(state, ALICE),
    )
    return memory, snapshot


def test_no_information_remains_explicitly_unknown_and_stable():
    modeller = HeuristicOpponentModeller()
    first = modeller.update(None, evidence())
    repeated = modeller.update(first, evidence(round_number=2))

    assert first.confidence == 0.0
    assert first.evidence_event_ids == ()
    assert first.unknowns
    assert first.strategy_hypotheses[0].strategy is StrategyKind.UNKNOWN
    assert repeated == first


def test_price_concessions_update_goal_reservation_strategy_and_confidence():
    modeller = HeuristicOpponentModeller()
    first_evidence = evidence(
        actions=(observed("action-1", 1, offer("offer-1", 90.0)),),
    )
    first = modeller.update(None, first_evidence)
    second = modeller.update(
        first,
        evidence(
            actions=(
                observed("action-1", 1, offer("offer-1", 90.0)),
                observed("action-2", 2, offer("offer-2", 70.0)),
            ),
            round_number=2,
        ),
    )

    maximize = next(item for item in second.goal_hypotheses if item.kind is GoalKind.MAXIMIZE)
    assert maximize.probability > 0.5
    assert second.numeric_reservation_ranges[0].upper == 70.0
    assert second.reservation_utility_upper == pytest.approx(0.7)
    assert second.confidence > first.confidence
    assert second.evidence_event_ids == ("action-1", "action-2")
    assert all(fact.status.value == "fact" for fact in second.facts)
    inferences = (
        second.goal_hypotheses
        + second.numeric_reservation_ranges
        + second.strategy_hypotheses
        + second.next_action_hypotheses
    )
    assert all(item.status.value == "inference" for item in inferences)
    assert all(
        set(item.evidence_event_ids).issubset(second.evidence_event_ids)
        for item in inferences
    )


def test_contradictory_concession_evidence_reduces_confidence():
    modeller = HeuristicOpponentModeller()
    consistent = modeller.update(
        None,
        evidence(
            actions=(
                observed("action-1", 1, offer("offer-1", 90.0)),
                observed("action-2", 2, offer("offer-2", 70.0)),
            ),
            round_number=2,
        ),
    )
    contradictory = modeller.update(
        consistent,
        evidence(
            actions=(
                observed("action-1", 1, offer("offer-1", 90.0)),
                observed("action-2", 2, offer("offer-2", 70.0)),
                observed("action-3", 3, offer("offer-3", 85.0)),
            ),
            round_number=3,
        ),
    )

    assert contradictory.confidence < consistent.confidence
    maximize = next(
        item for item in contradictory.goal_hypotheses if item.kind is GoalKind.MAXIMIZE
    )
    assert maximize.probability == pytest.approx(0.5)

    fresh_batch = modeller.update(
        None,
        evidence(
            actions=(
                observed("action-1", 1, offer("offer-1", 90.0)),
                observed("action-2", 2, offer("offer-2", 70.0)),
                observed("action-3", 3, offer("offer-3", 85.0)),
            ),
            round_number=3,
        ),
    )
    fresh_maximize = next(
        item for item in fresh_batch.goal_hypotheses if item.kind is GoalKind.MAXIMIZE
    )
    assert fresh_maximize.probability == pytest.approx(maximize.probability)


def test_message_only_update_does_not_double_count_an_old_concession():
    modeller = HeuristicOpponentModeller()
    actions = (
        observed("action-1", 1, offer("offer-1", 90.0)),
        observed("action-2", 2, offer("offer-2", 70.0)),
    )
    before_message = modeller.update(
        None,
        evidence(actions=actions, round_number=2),
    )
    after_message = modeller.update(
        before_message,
        ObservableOpponentEvidence(
            negotiation_id="belief-test",
            observer_id=ALICE,
            opponent_id=BOB,
            scenario=scenario(),
            current_round=3,
            actions=actions,
            messages=(
                ObservedMessage(
                    evidence_event_id="message-1",
                    round_number=3,
                    negotiation_id="belief-test",
                    sender=BOB,
                    recipients=(ALICE,),
                    visibility=MessageVisibility.DIRECT_PRIVATE,
                    message_type=MessageType.QUESTION,
                    content="Can delivery be changed?",
                ),
            ),
        ),
    )

    probability_before = next(
        item.probability
        for item in before_message.goal_hypotheses
        if item.kind is GoalKind.MAXIMIZE
    )
    probability_after = next(
        item.probability
        for item in after_message.goal_hypotheses
        if item.kind is GoalKind.MAXIMIZE
    )
    assert probability_after == pytest.approx(probability_before)
    assert after_message.information_needs[0].evidence_event_ids == ("message-1",)


def test_multi_issue_concessions_produce_per_issue_and_categorical_hypotheses():
    state = HeuristicOpponentModeller().update(
        None,
        evidence(
            actions=(
                observed("action-1", 1, offer("offer-1", 90.0, 9.0, "fast")),
                observed("action-2", 2, offer("offer-2", 70.0, 7.0, "slow")),
            ),
            multi_issue=True,
            round_number=2,
        ),
    )

    assert {item.issue_id for item in state.numeric_reservation_ranges} == {PRICE, QUALITY}
    categorical = [item for item in state.goal_hypotheses if item.issue_id == DELIVERY]
    assert any(item.kind is GoalKind.PREFER_CATEGORY and item.category == "fast" for item in categorical)


def test_belief_serialization_and_typed_episodic_storage_round_trip():
    memory, snapshot = memory_with_bob_offer()
    belief = HeuristicOpponentModeller().update(None, evidence_from_memory(snapshot, BOB))

    stored = memory.record_belief_snapshot(belief)
    restored = MemorySnapshot.model_validate_json(stored.model_dump_json())

    belief_events = [
        event for event in restored.episodic.events
        if isinstance(event, BeliefSnapshotMemoryEvent)
    ]
    assert belief_events[-1].belief == belief

    leaked = stored.model_dump(mode="json")
    leaked_belief = next(
        event["belief"]
        for event in leaked["episodic"]["events"]
        if event["event_type"] == "opponent_belief"
    )
    leaked_belief["observer_id"] = str(BOB)
    leaked_belief["opponent_id"] = str(ALICE)
    with pytest.raises(ValidationError, match="observer must match memory owner"):
        MemorySnapshot.model_validate_json(json.dumps(leaked))


def test_llm_modeller_is_strict_and_prompt_contains_no_private_ground_truth():
    observable = evidence(
        actions=(
            observed("action-1", 1, offer("offer-1", 90.0)),
            observed("action-2", 2, offer("offer-2", 80.0)),
        ),
        round_number=2,
    )
    baseline = HeuristicOpponentModeller().update(None, observable)
    client = FakeLLMClient(
        queued=(FakeResponse(text=baseline.model_dump_json()),),
        record_requests=True,
    )

    result = LLMOpponentModeller(client, fake_configuration()).update(None, observable)

    assert result == baseline
    prompt = "\n".join(
        message.content for request in client.recorded_requests for message in request.messages
    )
    assert "PRIVATE BATNA" not in prompt
    assert "chain-of-thought" in prompt
    assert '"prompt_version":"opponent-model-v1"' in prompt

    malformed = FakeLLMClient(queued=(FakeResponse(text='{"unexpected": true}'),))
    fallback = LLMOpponentModeller(malformed, fake_configuration()).update(None, observable)
    assert fallback == baseline

    invalid_range = baseline.model_dump(mode="json")
    invalid_range["numeric_reservation_ranges"][0]["upper"] = 1000.0
    out_of_bounds = FakeLLMClient(
        queued=(FakeResponse(text=json.dumps(invalid_range)),)
    )
    bounded_fallback = LLMOpponentModeller(
        out_of_bounds, fake_configuration()
    ).update(None, observable)
    assert bounded_fallback == baseline


def test_evidence_schema_rejects_private_preference_leakage():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservableOpponentEvidence.model_validate(
            {
                **evidence().model_dump(),
                "opponent_private_preferences": alice_profile().preferences,
            }
        )

    with pytest.raises(ValidationError, match="not visible"):
        ObservableOpponentEvidence(
            negotiation_id="belief-test",
            observer_id=ALICE,
            opponent_id=BOB,
            scenario=scenario(),
            current_round=1,
            messages=(
                ObservedMessage(
                    evidence_event_id="private-to-someone-else",
                    round_number=1,
                    negotiation_id="belief-test",
                    sender=BOB,
                    recipients=(ParticipantId("someone-else"),),
                    visibility=MessageVisibility.DIRECT_PRIVATE,
                    message_type=MessageType.INTENT_SIGNAL,
                    content="Not authorized for Alice",
                ),
            ),
        )


def test_tom_can_be_disabled_or_enabled_without_ground_truth_access():
    _, snapshot = memory_with_bob_offer()
    disabled = LLMNegotiationPolicy(
        FakeLLMClient(),
        fake_configuration(),
        opponent_configuration=OpponentModelConfiguration(mode=OpponentModelMode.DISABLED),
    ).update_opponent_beliefs(snapshot)
    enabled = LLMNegotiationPolicy(
        FakeLLMClient(),
        fake_configuration(),
        opponent_configuration=OpponentModelConfiguration(mode=OpponentModelMode.HEURISTIC),
    ).update_opponent_beliefs(snapshot)[0]

    assert disabled == ()
    assert enabled.evidence_event_ids == ("action-1",)
    assert not hasattr(enabled, "ground_truth")


def test_llm_tom_mode_uses_the_gateway_modeller_and_multiparty_is_per_opponent():
    _, snapshot = memory_with_bob_offer()
    observable = evidence_from_memory(snapshot, BOB)
    expected = HeuristicOpponentModeller().update(None, observable)
    client = FakeLLMClient(queued=(FakeResponse(text=expected.model_dump_json()),))
    policy = LLMNegotiationPolicy(
        FakeLLMClient(),
        fake_configuration(),
        opponent_configuration=OpponentModelConfiguration(mode=OpponentModelMode.LLM),
        opponent_modeller=LLMOpponentModeller(client, fake_configuration()),
    )

    assert policy.update_opponent_beliefs(snapshot) == (expected,)
    assert client.call_count == 1

    public = NegotiationScenario(
        scenario_id="multiparty-belief-test",
        title="Multiparty belief test",
        participants=(
            scenario().participant(ALICE),
            scenario().participant(BOB),
            Participant(
                participant_id=CAROL,
                display_name="Carol",
                role=ParticipantRole.MEDIATOR,
            ),
        ),
        issues=scenario().issues,
    )
    base_profile = alice_profile()
    profile = AgentProfile(
        identity=PublicAgentIdentity(
            participant=public.participant(ALICE),
            persona=base_profile.identity.persona,
        ),
        preferences=base_profile.preferences,
    )
    protocol = NegotiationProtocol(public, maximum_rounds=3, initial_turn=ALICE)
    session = protocol.create_session()
    multiparty_memory = AgentMemory(profile, public).update(
        AgentObservation(scenario=public, session=session, own_profile=profile),
        legal_actions=protocol.legal_action_types(session, ALICE),
    )
    states = LLMNegotiationPolicy(
        FakeLLMClient(), fake_configuration()
    ).update_opponent_beliefs(multiparty_memory)

    assert {state.opponent_id for state in states} == {BOB, CAROL}


def test_post_episode_calibration_uses_ground_truth_only_when_called():
    belief = HeuristicOpponentModeller().update(
        None,
        evidence(
            actions=(
                observed("action-1", 1, offer("offer-1", 90.0)),
                observed("action-2", 2, offer("offer-2", 70.0)),
            ),
            round_number=2,
        ),
    )
    truth = ParticipantPreferences(
        participant_id=BOB,
        issue_preferences=(
            NumericPreference(
                issue_id=PRICE,
                weight=1.0,
                direction=PreferenceDirection.MAXIMIZE,
            ),
        ),
        reservation=ReservationPolicy(reservation_utility=0.6, batna_utility=0.5),
    )

    protocol = NegotiationProtocol(scenario(), maximum_rounds=4, initial_turn=ALICE)
    active_session = protocol.create_session()
    with pytest.raises(ValueError, match="completed terminal"):
        calibrate_belief(belief, truth, StrategyKind.LINEAR, active_session)
    completed_session = protocol.transition(
        active_session,
        WithdrawAction(actor_id=ALICE, round_number=1, reason="End calibration episode"),
    ).state

    metrics = calibrate_belief(
        belief,
        truth,
        StrategyKind.LINEAR,
        completed_session,
    )

    assert metrics.reservation_utility_covered is True
    assert metrics.reservation_interval_width == pytest.approx(0.7)
    assert 0.0 <= metrics.strategy_probability <= 1.0
    assert metrics.goal_brier_score < 0.25
    assert metrics.evidence_count == 2
