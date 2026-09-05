import json
from datetime import datetime, timezone

import pytest

from llm_negotiation.domain import (
    AcceptAction,
    MessageAction,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    NegotiationScenario,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ProposeAction,
    ReservationPolicy,
    calculate_utility,
)
from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    FakeRule,
    LLMProvider,
    ModelConfiguration,
    RetryConfiguration,
)
from llm_negotiation.llm_policy import LLMNegotiationPolicy
from llm_negotiation.mediation import DeadlockConfiguration, MediatorConfiguration
from llm_negotiation.memory import (
    AgentProfile,
    BeliefSnapshotMemoryEvent,
    PublicAgentIdentity,
)
from llm_negotiation.opponent import HeuristicOpponentModeller
from llm_negotiation.orchestration import (
    EpisodeResult,
    FailurePolicy,
    NegotiationOrchestrator,
    OrchestrationError,
    OrchestratorConfiguration,
)
from llm_negotiation.policies import FixedPolicy, LinearConcessionPolicy
from llm_negotiation.protocol import MediatorInterventionEvent, ProtocolPhase


BUYER = ParticipantId("buyer")
SELLER = ParticipantId("seller")


def scenario():
    return NegotiationScenario(
        scenario_id="orchestration-test",
        title="Orchestration test",
        participants=(
            Participant(
                participant_id=BUYER,
                display_name="Buyer",
                role=ParticipantRole.BUYER,
            ),
            Participant(
                participant_id=SELLER,
                display_name="Seller",
                role=ParticipantRole.SELLER,
            ),
        ),
        issues=(NumericIssue(issue_id="price", name="Price", minimum=0, maximum=100),),
    )


def profile(participant_id, direction, private_description):
    return AgentProfile(
        identity=PublicAgentIdentity(
            participant=scenario().participant(participant_id),
            persona=f"Public {participant_id}",
        ),
        preferences=ParticipantPreferences(
            participant_id=participant_id,
            issue_preferences=(
                NumericPreference(issue_id="price", weight=1, direction=direction),
            ),
            reservation=ReservationPolicy(
                reservation_utility=0.2,
                batna_utility=0.1,
                batna_description=private_description,
            ),
        ),
    )


def profiles():
    return {
        BUYER: profile(BUYER, PreferenceDirection.MINIMIZE, "BUYER PRIVATE BATNA"),
        SELLER: profile(SELLER, PreferenceDirection.MAXIMIZE, "SELLER PRIVATE BATNA"),
    }


class FixedClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return datetime(2026, 1, 1, tzinfo=timezone.utc)

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class ReservationAcceptor:
    name = "reservation-acceptor"

    def choose_action(self, memory):
        outstanding = memory.working.outstanding_offer
        if outstanding is not None and calculate_utility(
            memory.scenario, outstanding, memory.long_term.own_preferences
        ) >= memory.long_term.own_preferences.reservation.reservation_utility:
            return AcceptAction(
                actor_id=memory.owner_id,
                recipients=tuple(
                    item.participant_id
                    for item in memory.long_term.public_participants
                    if item.participant_id != memory.owner_id
                ),
                round_number=memory.working.round_number,
                offer_id=outstanding.offer_id,
            )
        return LinearConcessionPolicy().choose_action(memory)


class CommunicatingAcceptor(ReservationAcceptor):
    name = "communicating-acceptor"

    def choose_action(self, memory):
        if not any(item.sender == memory.owner_id for item in memory.working.visible_messages):
            return MessageAction(
                actor_id=memory.owner_id,
                recipients=tuple(
                    item.participant_id
                    for item in memory.long_term.public_participants
                    if item.participant_id != memory.owner_id
                ),
                round_number=memory.working.round_number,
                content="I can accept a reservation-safe midpoint.",
            )
        return super().choose_action(memory)


def orchestrator(policies, **kwargs):
    return NegotiationOrchestrator.with_defaults(
        scenario=scenario(),
        profiles=profiles(),
        policies=policies,
        random_seed=kwargs.pop("random_seed", 7),
        configuration=kwargs.pop(
            "configuration", OrchestratorConfiguration(maximum_rounds=4)
        ),
        clock=kwargs.pop("clock", FixedClock()),
        **kwargs,
    )


def test_deterministic_episode_is_complete_private_and_replayable():
    service = orchestrator(
        {BUYER: LinearConcessionPolicy(), SELLER: LinearConcessionPolicy()}
    )
    result = service.run()

    assert result.final_session.phase is ProtocolPhase.AGREED
    assert result.outcome == result.final_session.outcome
    assert result.metrics.agreement_valid is True
    assert {item.participant_id for item in result.memory_references} == {BUYER, SELLER}
    assert "PRIVATE BATNA" not in result.model_dump_json()
    replayed, metrics = service.replay(result)
    assert replayed == result.final_session
    assert metrics == result.metrics
    assert EpisodeResult.model_validate_json(result.model_dump_json()) == result

    repeated = orchestrator(
        {BUYER: LinearConcessionPolicy(), SELLER: LinearConcessionPolicy()}
    ).run()
    assert repeated.audit_events == result.audit_events
    assert repeated.metrics == result.metrics


def objective_json():
    return json.dumps(
        {
            "schema_version": "1.0",
            "objective": "Reach an acceptable agreement.",
            "constraints": ["Use only legal actions."],
            "decision_rationale": "Public rules determine the objective.",
            "evidence": ["legal actions"],
        }
    )


def llm_response(request):
    payload = json.loads(request.messages[-1].content)
    stage = payload["stage"]
    if stage == "objective_constraints":
        return FakeResponse(text=objective_json())
    if stage == "opponent_hypothesis":
        return FakeResponse(
            text=json.dumps(
                {
                    "schema_version": "1.0",
                    "hypothesis": "A midpoint may be acceptable.",
                    "confidence": 0.5,
                    "decision_rationale": "Only visible evidence is used.",
                    "evidence": ["public events"],
                }
            )
        )
    if stage == "planning":
        return FakeResponse(
            text=json.dumps(
                {
                    "schema_version": "1.0",
                    "plan": ["Offer or accept the midpoint."],
                    "target_utility": 0.5,
                    "decision_rationale": "The midpoint is within bounds.",
                    "evidence": ["public issue bounds"],
                }
            )
        )
    observation = payload["participant_visible_context"]["observation"]
    if observation["outstanding_offer"] is None:
        action = {
            "action": "propose",
            "actor_id": observation["participant_id"],
            "recipients": observation["expected_recipients"],
            "round_number": observation["round_number"],
            "offer": {
                "offer_id": observation["required_offer_id"],
                "values": [{"kind": "numeric", "issue_id": "price", "value": 50}],
            },
        }
    else:
        action = {
            "action": "accept",
            "actor_id": observation["participant_id"],
            "recipients": observation["expected_recipients"],
            "round_number": observation["round_number"],
            "offer_id": observation["outstanding_offer"]["offer_id"],
        }
    return FakeResponse(
        text=json.dumps(
            {
                "schema_version": "1.0",
                "action": action,
                "decision_rationale": "This typed action is legal.",
                "evidence": ["participant-visible observation"],
            }
        )
    )


def model_configuration():
    return ModelConfiguration(
        provider=LLMProvider.FAKE,
        model="fake-orchestrator",
        retry=RetryConfiguration(max_retries=0),
    )


def test_fake_llm_and_deterministic_policy_complete_offline():
    client = FakeLLMClient(
        rules=(FakeRule(predicate=lambda _: True, responder=llm_response),),
        record_requests=True,
    )
    result = orchestrator(
        {
            BUYER: LLMNegotiationPolicy(client, model_configuration()),
            SELLER: ReservationAcceptor(),
        },
        model_gateway=client,
        configuration=OrchestratorConfiguration(
            maximum_rounds=4,
            failure_policy=FailurePolicy.RAISE,
        ),
    ).run()

    assert result.final_session.phase is ProtocolPhase.AGREED
    assert result.usage.model_calls == client.call_count == 4
    assert all(item.provider is LLMProvider.FAKE for item in result.usage.records)
    assert all(item.model == "fake-orchestrator" for item in result.usage.records)
    buyer_prompts = "\n".join(
        message.content
        for request in client.recorded_requests
        if json.loads(request.messages[-1].content)["participant_visible_context"][
            "observation"
        ]["participant_id"] == "buyer"
        for message in request.messages
    )
    assert "SELLER PRIVATE BATNA" not in buyer_prompts


def test_typed_communication_is_delivered_without_changing_the_offer_itself():
    result = orchestrator(
        {BUYER: LinearConcessionPolicy(), SELLER: CommunicatingAcceptor()}
    ).run()

    assert result.final_session.phase is ProtocolPhase.AGREED
    assert len(result.public_transcript) == 1
    message = result.public_transcript[0]
    assert message.content == "I can accept a reservation-safe midpoint."
    message_event = next(
        event for event in result.audit_events
        if getattr(event, "action", None) is not None
        and isinstance(event.action, MessageAction)
    )
    assert message.correlation_id == message_event.correlation_id


def test_injected_opponent_modellers_write_only_owner_specific_belief_memory():
    service = orchestrator(
        {BUYER: LinearConcessionPolicy(), SELLER: LinearConcessionPolicy()},
        opponent_modellers={
            BUYER: HeuristicOpponentModeller(),
            SELLER: HeuristicOpponentModeller(),
        },
    )
    service.run()

    for participant_id in (BUYER, SELLER):
        snapshot = service.memory_store.get(participant_id).snapshot()
        beliefs = [
            item.belief for item in snapshot.episodic.events
            if isinstance(item, BeliefSnapshotMemoryEvent)
        ]
        assert beliefs
        assert all(item.observer_id == participant_id for item in beliefs)
        assert all(not hasattr(item, "preferences") for item in beliefs)


def test_mediation_runs_as_correlated_non_binding_protocol_intervention():
    result = orchestrator(
        {BUYER: FixedPolicy(), SELLER: FixedPolicy()},
        configuration=OrchestratorConfiguration(
            maximum_rounds=3,
            mediation_enabled=True,
        ),
        mediator_configuration=MediatorConfiguration(
            deadlock=DeadlockConfiguration(deadline_rounds_remaining=1)
        ),
    ).run()

    interventions = [
        item for item in result.audit_events if isinstance(item, MediatorInterventionEvent)
    ]
    assert len(interventions) == 1
    assert interventions[0].intervention.offer is not None
    assert result.public_transcript[0].correlation_id == interventions[0].correlation_id


class SlowPolicy:
    name = "slow"

    def __init__(self, clock):
        self.clock = clock

    def choose_action(self, memory):
        self.clock.advance(2)
        return LinearConcessionPolicy().choose_action(memory)


class TickingClock(FixedClock):
    def monotonic(self):
        current = self.value
        self.value += 1
        return current


def test_action_timeout_uses_explicit_withdraw_failure_policy():
    clock = FixedClock()
    result = orchestrator(
        {BUYER: SlowPolicy(clock), SELLER: LinearConcessionPolicy()},
        clock=clock,
        configuration=OrchestratorConfiguration(
            maximum_rounds=3,
            action_timeout_seconds=1,
            failure_policy=FailurePolicy.WITHDRAW,
        ),
    ).run()
    assert result.final_session.phase is ProtocolPhase.WITHDRAWN
    assert "timeout" in result.outcome.reason.lower()


def test_episode_timeout_terminates_before_requesting_an_agent_action():
    result = orchestrator(
        {BUYER: LinearConcessionPolicy(), SELLER: LinearConcessionPolicy()},
        clock=TickingClock(),
        configuration=OrchestratorConfiguration(
            maximum_rounds=3,
            episode_timeout_seconds=0.5,
            failure_policy=FailurePolicy.WITHDRAW,
        ),
    ).run()
    assert result.final_session.phase is ProtocolPhase.WITHDRAWN
    assert result.verification_log[0].selected_action_type.value == "withdraw"
    assert "episode timeout" in result.outcome.reason.lower()


class InvalidFirstPolicy:
    name = "invalid-first"

    def choose_action(self, memory):
        return AcceptAction(
            actor_id=memory.owner_id,
            recipients=(SELLER,),
            round_number=memory.working.round_number,
            offer_id=OfferId("fabricated"),
        )


def test_invalid_action_gets_one_correction_before_transition():
    corrected = ProposeAction(
        actor_id=BUYER,
        recipients=(SELLER,),
        round_number=1,
        offer=Offer(
            offer_id=OfferId("corrected-offer"),
            values=(NumericIssueValue(issue_id="price", value=50),),
        ),
    )
    result = orchestrator(
        {BUYER: InvalidFirstPolicy(), SELLER: LinearConcessionPolicy()},
        correction_providers={BUYER: lambda _feedback: corrected},
    ).run()
    assert result.verification_log[0].correction_count == 1
    assert result.verification_log[0].selected_action_type.value == "propose"
    assert result.audit_events[0].action == corrected


def test_provider_failure_follows_withdraw_or_raise_configuration():
    failed_client = FakeLLMClient()
    policies = {
        BUYER: LLMNegotiationPolicy(failed_client, model_configuration()),
        SELLER: LinearConcessionPolicy(),
    }
    result = orchestrator(
        policies,
        model_gateway=failed_client,
        configuration=OrchestratorConfiguration(
            maximum_rounds=3,
            failure_policy=FailurePolicy.WITHDRAW,
        ),
    ).run()
    assert result.final_session.phase is ProtocolPhase.WITHDRAWN

    raise_client = FakeLLMClient()
    with pytest.raises(OrchestrationError, match="used fallback"):
        orchestrator(
            {
                BUYER: LLMNegotiationPolicy(raise_client, model_configuration()),
                SELLER: LinearConcessionPolicy(),
            },
            model_gateway=raise_client,
            configuration=OrchestratorConfiguration(
                maximum_rounds=3,
                failure_policy=FailurePolicy.RAISE,
            ),
        ).run()

    fallback_client = FakeLLMClient()
    fallback_service = orchestrator(
        {
            BUYER: LLMNegotiationPolicy(fallback_client, model_configuration()),
            SELLER: ReservationAcceptor(),
        },
        model_gateway=fallback_client,
        configuration=OrchestratorConfiguration(
            maximum_rounds=4,
            failure_policy=FailurePolicy.FALLBACK,
        ),
    )
    fallback_result = fallback_service.run()
    assert fallback_client.call_count >= 1
    assert fallback_result.final_session.phase in {
        ProtocolPhase.AGREED,
        ProtocolPhase.EXPIRED,
    }
    assert fallback_service.policies[BUYER].last_trace.used_fallback is True


def test_zero_model_call_budget_prevents_provider_execution_and_falls_back():
    client = FakeLLMClient(
        rules=(FakeRule(predicate=lambda _: True, responder=llm_response),)
    )
    result = orchestrator(
        {
            BUYER: LLMNegotiationPolicy(client, model_configuration()),
            SELLER: ReservationAcceptor(),
        },
        model_gateway=client,
        configuration=OrchestratorConfiguration(
            maximum_rounds=4,
            maximum_model_calls=0,
            failure_policy=FailurePolicy.FALLBACK,
        ),
    ).run()

    assert client.call_count == 0
    assert result.usage.model_calls == 0
    assert result.final_session.phase in {
        ProtocolPhase.AGREED,
        ProtocolPhase.EXPIRED,
    }


def test_model_call_budget_is_hard_across_cognitive_stages():
    client = FakeLLMClient(
        rules=(FakeRule(predicate=lambda _: True, responder=llm_response),)
    )
    service = orchestrator(
        {
            BUYER: LLMNegotiationPolicy(client, model_configuration()),
            SELLER: ReservationAcceptor(),
        },
        model_gateway=client,
        configuration=OrchestratorConfiguration(
            maximum_rounds=4,
            maximum_model_calls=3,
            failure_policy=FailurePolicy.FALLBACK,
        ),
    )
    result = service.run()

    assert client.call_count == 3
    assert service.episode_model_gateway.used_calls == 3
    assert result.usage.model_calls == 3
    assert service.policies[BUYER].last_trace.used_fallback is True
    assert service.policies[BUYER].last_trace.fallback_reason == "llm_budget_exceeded"
