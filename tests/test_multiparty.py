from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from llm_negotiation.communication import MessageBus, MessageType, MessageVisibility, summarize_transcript
from llm_negotiation.domain import (
    AcceptAction,
    CounterAction,
    IllegalActionError,
    MessageAction,
    NumericIssueValue,
    Offer,
    OfferId,
    ParticipantId,
    ProposeAction,
    RejectAction,
)
from llm_negotiation.memory import AgentMemory, MemorySnapshot
from llm_negotiation.llm import LLMProvider, ModelConfiguration
from llm_negotiation.multiparty import (
    AcceptanceRule,
    Coalition,
    CoalitionRestrictions,
    DeliberativePolicy,
    MultipartyNegotiationProtocol,
    MultipartyProtocolConfiguration,
    MultipartyProtocolFactory,
    heterogeneous_three_principal_team,
    three_principal_scenario,
)
from llm_negotiation.orchestration import (
    AUDIT_READER_ID,
    InMemoryMemoryStore,
    NegotiationOrchestrator,
    OrchestratorConfiguration,
)
from llm_negotiation.policies import ConcederPolicy, LinearConcessionPolicy
from llm_negotiation.protocol import (
    FeasibleRegionStatus,
    NegotiationSession,
    ProtocolPhase,
    ProtocolStage,
)
from llm_negotiation.team_experiment import TeamComposition, run_team_comparison
from llm_negotiation.verification import (
    DeterministicActionVerifier,
    VerificationContext,
    VerificationCoordinator,
    VerificationReasonCode,
    VerificationVerdict,
)


class FixedClock:
    def now(self) -> datetime:
        return datetime(2025, 1, 1, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return 0.0


def configuration(
    participants: tuple[ParticipantId, ...],
    *,
    rule: AcceptanceRule = AcceptanceRule.UNANIMITY,
    required: tuple[ParticipantId, ...] = (),
    coalition: CoalitionRestrictions | None = None,
) -> MultipartyProtocolConfiguration:
    return MultipartyProtocolConfiguration(
        speaking_order=participants,
        eligible_proposers=participants,
        eligible_voters=participants,
        required_participants=required,
        acceptance_rule=rule,
        coalition_restrictions=coalition or CoalitionRestrictions(),
    )


def offer(number: int, budget: float, launch: str = "balanced") -> Offer:
    from llm_negotiation.domain import CategoricalIssueValue

    return Offer(
        offer_id=OfferId(f"offer-{number}"),
        values=(
            NumericIssueValue(issue_id="budget_share", value=budget),
            CategoricalIssueValue(issue_id="launch", value=launch),
        ),
    )


def advance_to_deliberation(protocol: MultipartyNegotiationProtocol):
    state = protocol.create_session()
    for index, actor in enumerate(protocol.configuration.eligible_proposers, start=1):
        state = protocol.transition(
            state,
            ProposeAction(
                actor_id=actor,
                recipients=tuple(item for item in state.participants if item != actor),
                round_number=state.round_number,
                offer=offer(index, 20.0 * index),
            ),
        ).state
    return state


def advance_to_voting(
    protocol: MultipartyNegotiationProtocol,
    *,
    final_budget: float = 50.0,
    final_launch: str = "balanced",
):
    state = advance_to_deliberation(protocol)
    for actor in protocol.configuration.speaking_order:
        state = protocol.transition(
            state,
            MessageAction(
                actor_id=actor,
                recipients=tuple(item for item in state.participants if item != actor),
                round_number=state.round_number,
                content=f"Critique from {actor}",
            ),
        ).state
    for index, actor in enumerate(protocol.configuration.eligible_proposers, start=10):
        state = protocol.transition(
            state,
            CounterAction(
                actor_id=actor,
                recipients=tuple(item for item in state.participants if item != actor),
                round_number=state.round_number,
                offer=offer(
                    index,
                    final_budget if actor == protocol.configuration.eligible_proposers[-1] else 50.0,
                    final_launch if actor == protocol.configuration.eligible_proposers[-1] else "balanced",
                ),
                responds_to=state.latest_valid_offer.offer_id,
            ),
        ).state
    return state


def test_acceptance_rules_are_explicit_and_validated():
    ids = tuple(ParticipantId(item) for item in ("a", "b", "c", "d"))
    assert configuration(ids, rule=AcceptanceRule.UNANIMITY).required_approval_count == 4
    assert configuration(ids, rule=AcceptanceRule.MAJORITY).required_approval_count == 3
    threshold = MultipartyProtocolConfiguration(
        speaking_order=ids,
        eligible_proposers=ids,
        eligible_voters=ids,
        acceptance_rule=AcceptanceRule.THRESHOLD,
        acceptance_threshold=0.75,
    )
    assert threshold.required_approval_count == 3
    with pytest.raises(ValueError, match="acceptance_threshold"):
        MultipartyProtocolConfiguration(
            speaking_order=ids,
            eligible_proposers=ids,
            eligible_voters=ids,
            acceptance_rule=AcceptanceRule.THRESHOLD,
        )
    with pytest.raises(ValueError, match="every principal"):
        MultipartyProtocolConfiguration(
            speaking_order=ids,
            eligible_proposers=ids[:3],
            eligible_voters=ids,
        )


@pytest.mark.parametrize(
    ("rule", "threshold", "approvals"),
    (
        (AcceptanceRule.MAJORITY, None, 2),
        (AcceptanceRule.THRESHOLD, 0.66, 2),
        (AcceptanceRule.UNANIMITY, None, 3),
    ),
)
def test_voting_rules_terminate_at_the_configured_quorum(rule, threshold, approvals):
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    config = MultipartyProtocolConfiguration(
        speaking_order=ids,
        eligible_proposers=ids,
        eligible_voters=ids,
        acceptance_rule=rule,
        acceptance_threshold=threshold,
    )
    protocol = MultipartyNegotiationProtocol(scenario, 6, config)
    state = advance_to_voting(protocol)
    for index, actor in enumerate(ids[:approvals]):
        state = protocol.transition(
            state,
            AcceptAction(
                actor_id=actor,
                recipients=tuple(item for item in ids if item != actor),
                round_number=state.round_number,
                offer_id=state.latest_valid_offer.offer_id,
            ),
        ).state
        assert (state.phase is ProtocolPhase.AGREED) == (index + 1 == approvals)
    assert set(state.outcome.agreement.accepted_by) == set(ids[:approvals])


def test_speaking_order_proposal_ownership_and_bounded_deliberation():
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    order = (ids[1], ids[2], ids[0])
    protocol = MultipartyNegotiationProtocol(scenario, 6, configuration(order))
    state = protocol.create_session()
    assert state.current_turn == order[0]
    state = advance_to_deliberation(protocol)
    assert tuple(item.owner_id for item in state.proposal_ownership) == order
    assert state.protocol_stage is ProtocolStage.DELIBERATION
    for index, actor in enumerate(order):
        state = protocol.transition(
            state,
            MessageAction(
                actor_id=actor,
                recipients=tuple(item for item in ids if item != actor),
                round_number=state.round_number,
                content="Bounded public critique",
            ),
        ).state
        if index < 2:
            assert state.protocol_stage is ProtocolStage.DELIBERATION
    assert state.protocol_stage is ProtocolStage.REVISION
    assert state.deliberation_actions == 3


def test_proposals_and_votes_cannot_be_routed_to_a_private_subset():
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    protocol = MultipartyNegotiationProtocol(scenario, 6, configuration(ids))
    state = protocol.create_session()
    before = state.model_dump_json()
    with pytest.raises(IllegalActionError, match="every other principal"):
        protocol.transition(
            state,
            ProposeAction(
                actor_id=ids[0],
                recipients=(ids[1],),
                round_number=state.round_number,
                offer=offer(99, 50.0),
            ),
        )
    assert state.model_dump_json() == before


def test_private_coalition_attempt_is_rejected_without_mutation():
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    protocol = MultipartyNegotiationProtocol(scenario, 6, configuration(ids))
    state = advance_to_deliberation(protocol)
    before = state.model_dump_json()
    with pytest.raises(IllegalActionError, match="coalition"):
        protocol.transition(
            state,
            MessageAction(
                actor_id=ids[0],
                recipients=(ids[1],),
                round_number=state.round_number,
                content="Exclude the holdout.",
            ),
        )
    assert state.model_dump_json() == before

    allowed = CoalitionRestrictions(
        private_coalitions_allowed=True,
        allowed_coalitions=(Coalition(members=(ids[0], ids[1])),),
        maximum_coalition_size=2,
    )
    allowed_protocol = MultipartyNegotiationProtocol(scenario, 6, configuration(ids, coalition=allowed))
    allowed_state = advance_to_deliberation(allowed_protocol)
    result = allowed_protocol.transition(
        allowed_state,
        MessageAction(
            actor_id=ids[0], recipients=(ids[1],), round_number=2, content="Permitted caucus."
        ),
    )
    assert result.state.current_turn == ids[1]


def test_required_holdout_can_block_a_majority():
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    protocol = MultipartyNegotiationProtocol(
        scenario,
        6,
        configuration(ids, rule=AcceptanceRule.MAJORITY, required=(ids[2],)),
    )
    state = advance_to_voting(protocol)
    for actor in ids[:2]:
        state = protocol.transition(
            state,
            AcceptAction(
                actor_id=actor,
                recipients=tuple(item for item in ids if item != actor),
                round_number=state.round_number,
                offer_id=state.latest_valid_offer.offer_id,
            ),
        ).state
    assert state.phase is ProtocolPhase.ACTIVE
    state = protocol.transition(
        state,
        RejectAction(
            actor_id=ids[2],
            recipients=ids[:2],
            round_number=state.round_number,
            offer_id=state.latest_valid_offer.offer_id,
        ),
    ).state
    assert state.phase is ProtocolPhase.FAILED
    assert "required participant" in state.outcome.reason


def test_serialized_session_rejects_an_agreement_below_its_quorum():
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    protocol = MultipartyNegotiationProtocol(
        scenario, 6, configuration(ids, rule=AcceptanceRule.MAJORITY)
    )
    state = advance_to_voting(protocol)
    for actor in ids[:2]:
        state = protocol.transition(
            state,
            AcceptAction(
                actor_id=actor,
                recipients=tuple(item for item in ids if item != actor),
                round_number=state.round_number,
                offer_id=state.latest_valid_offer.offer_id,
            ),
        ).state
    assert state.phase is ProtocolPhase.AGREED
    payload = state.model_dump(mode="python")
    payload["acceptance_semantics"]["approval_count"] = 3
    with pytest.raises(ValueError, match="approval quorum"):
        NegotiationSession.model_validate(payload)


def test_impossible_multiparty_episode_replays_its_system_termination():
    scenario, _profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    protocol = MultipartyNegotiationProtocol(
        scenario,
        6,
        configuration(ids),
        feasible_region=FeasibleRegionStatus.IMPOSSIBLE,
    )
    state = protocol.create_session()
    assert state.phase is ProtocolPhase.FAILED
    assert protocol.replay(state.event_sequence) == state


def test_required_holdout_cannot_accept_below_its_reservation_utility():
    scenario, profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    required = ids[1]
    protocol = MultipartyNegotiationProtocol(
        scenario,
        6,
        configuration(ids, rule=AcceptanceRule.MAJORITY, required=(required,)),
    )
    state = advance_to_voting(protocol, final_budget=0.0, final_launch="early")
    state = protocol.transition(
        state,
        AcceptAction(
            actor_id=ids[0],
            recipients=ids[1:],
            round_number=state.round_number,
            offer_id=state.latest_valid_offer.offer_id,
        ),
    ).state
    before = state.model_dump_json()
    result = DeterministicActionVerifier().verify(
        VerificationContext(
            protocol=protocol,
            state=state,
            actor_profile=profiles[required],
        ),
        AcceptAction(
            actor_id=required,
            recipients=(ids[0], ids[2]),
            round_number=state.round_number,
            offer_id=state.latest_valid_offer.offer_id,
        ),
        "holdout-check",
    )
    assert result.verdict is VerificationVerdict.REJECT
    assert VerificationReasonCode.RESERVATION_UTILITY_VIOLATION in {
        item.code for item in result.reasons
    }
    assert state.model_dump_json() == before


@dataclass
class RecordingPolicy:
    delegate: DeliberativePolicy
    initial_views: list[tuple[int, int, int]] = field(default_factory=list)
    name: str = "recording-deliberative"

    def choose_action(self, memory: MemorySnapshot):
        if memory.working.protocol_stage is ProtocolStage.INDEPENDENT_PROPOSALS:
            self.initial_views.append(
                (
                    memory.working.public_event_count,
                    len(memory.working.recent_offers),
                    len(memory.working.visible_messages),
                )
            )
        return self.delegate.choose_action(memory)


def test_three_party_episode_has_independent_initialization_and_replays():
    scenario, profiles = three_principal_scenario()
    ids = tuple(item.participant_id for item in scenario.participants)
    policies = {
        participant_id: RecordingPolicy(DeliberativePolicy(ConcederPolicy()))
        for participant_id in ids
    }
    memories = {participant_id: AgentMemory(profiles[participant_id], scenario) for participant_id in ids}
    factory = MultipartyProtocolFactory(
        configuration(ids, rule=AcceptanceRule.UNANIMITY)
    )
    orchestrator = NegotiationOrchestrator(
        scenario=scenario,
        profiles=profiles,
        policies=policies,
        message_bus=MessageBus(
            scenario.scenario_id,
            participants=ids,
            mediator_ids=(),
            audit_reader_ids=(AUDIT_READER_ID,),
        ),
        verifier=VerificationCoordinator(),
        mediator=None,
        model_gateway=None,
        memory_store=InMemoryMemoryStore(memories),
        clock=FixedClock(),
        random_seed=7,
        configuration=OrchestratorConfiguration(maximum_rounds=6),
        protocol_factory=factory,
    )
    result = orchestrator.run()
    assert result.outcome is not None
    assert result.outcome.status.value == "agreement"
    assert result.final_session.proposal_ownership[-1].owner_id == ids[-1]
    assert (
        result.final_session.proposal_ownership[-1].offer_id
        == result.outcome.agreement.offer.offer_id
    )
    assert all(policy.initial_views == [(0, 0, 0)] for policy in policies.values())
    assert len(result.public_transcript) == 3
    assert all(item.message_type is MessageType.CRITIQUE for item in result.public_transcript)
    for participant_id, memory in memories.items():
        snapshot = memory.snapshot()
        assert snapshot.working.acceptance_semantics is not None
        assert snapshot.long_term.own_preferences.participant_id == participant_id
        payload = snapshot.model_dump(mode="json")
        assert payload["long_term"]["own_preferences"]["participant_id"] == str(participant_id)
        assert "preferences" not in payload["working"]
    assert NegotiationSession.model_validate_json(
        result.final_session.model_dump_json()
    ) == result.final_session
    replayed, metrics = orchestrator.replay(result)
    assert replayed == result.final_session
    assert metrics == result.metrics


def test_heterogeneous_team_configuration_keeps_model_settings_separate():
    model_configuration = ModelConfiguration(provider=LLMProvider.FAKE, model="research-model")
    assignments = heterogeneous_three_principal_team(
        {ParticipantId("research"): model_configuration}
    )
    assert {item.strategy for item in assignments.values()} == {"boulware", "linear", "conceder"}
    assert len({item.profile.identity.persona for item in assignments.values()}) == 3
    assert len({item.profile.preferences for item in assignments.values()}) == 3
    assert assignments[ParticipantId("research")].model_configuration == model_configuration
    assert assignments[ParticipantId("research")].profile.model_dump().get("model_configuration") is None


def test_transcript_summary_preserves_every_source_event_id():
    ids = tuple(ParticipantId(item) for item in ("a", "b", "c"))
    bus = MessageBus("summary", participants=ids, mediator_ids=(), audit_reader_ids=(AUDIT_READER_ID,))
    for index in range(6):
        sender = ids[index % 3]
        bus.send(
            sender=sender,
            recipients=tuple(item for item in ids if item != sender),
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            message_type=MessageType.CRITIQUE,
            visibility=MessageVisibility.PUBLIC,
            content=f"Critique {index}",
        )
    summary = summarize_transcript(bus.public_transcript(), maximum_items=3)
    assert len(summary.items) == 3
    assert summary.items[0].summarized
    assert sum(len(item.source_event_ids) for item in summary.items) == 6
    assert sum(len(item.content) for item in summary.items) <= 2000


def test_comparison_experiment_covers_composition_and_team_size_deterministically():
    first = run_team_comparison()
    second = run_team_comparison()
    assert first == second
    assert {(run.team_size, run.composition) for run in first.runs} == {
        (size, composition)
        for size in (2, 3, 4)
        for composition in (TeamComposition.HOMOGENEOUS, TeamComposition.HETEROGENEOUS)
    }
    assert any(run.agreement for run in first.runs)
    assert any(not run.agreement for run in first.runs)
    by_cell = {(run.team_size, run.composition): run for run in first.runs}
    assert by_cell[(3, TeamComposition.HETEROGENEOUS)].agreement
    assert not by_cell[(4, TeamComposition.HETEROGENEOUS)].agreement
    assert all(0.0 <= run.mean_outcome_utility <= 1.0 for run in first.runs)
    for size in (3, 4):
        assert (
            by_cell[(size, TeamComposition.HETEROGENEOUS)].initial_proposal_diversity
            > by_cell[(size, TeamComposition.HOMOGENEOUS)].initial_proposal_diversity
        )
