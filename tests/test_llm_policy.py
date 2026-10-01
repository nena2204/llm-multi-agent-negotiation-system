import json
from datetime import datetime, timezone

import pytest

from llm_negotiation.benchmark import run_policy_session
from llm_negotiation.communication import MessageBus, MessageType, MessageVisibility
from llm_negotiation.domain import (
    AcceptAction,
    ActionType,
    CorrelationId,
    IssueId,
    NegotiationScenario,
    NumericIssue,
    NumericPreference,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ProposeAction,
    ReservationPolicy,
)
from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    FakeRule,
    LLMClientError,
    LLMErrorCode,
    LLMErrorInfo,
    LLMProvider,
    LLMUsage,
    ModelConfiguration,
    RetryConfiguration,
)
from llm_negotiation.llm_policy import (
    StructuredActionDecision,
    normalize_stage_json,
    CognitiveStage,
    LLMNegotiationPolicy,
)
from llm_negotiation.memory import (
    AgentMemory,
    AgentObservation,
    AgentProfile,
    BeliefSnapshotMemoryEvent,
    PublicAgentIdentity,
)
from llm_negotiation.policies import NegotiationPolicy
from llm_negotiation.protocol import NegotiationProtocol, ProtocolPhase


SELLER = ParticipantId("seller")
BUYER = ParticipantId("buyer")
MEDIATOR = ParticipantId("mediator")
PRICE = IssueId("price")


def price_scenario(include_mediator=False):
    participants = [
        Participant(participant_id=SELLER, display_name="Seller", role=ParticipantRole.SELLER),
        Participant(participant_id=BUYER, display_name="Buyer", role=ParticipantRole.BUYER),
    ]
    if include_mediator:
        participants.append(
            Participant(
                participant_id=MEDIATOR,
                display_name="Mediator",
                role=ParticipantRole.MEDIATOR,
            )
        )
    return NegotiationScenario(
        scenario_id="llm-policy-test",
        title="LLM Policy Test",
        participants=tuple(participants),
        issues=(NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),),
    )


def profile(participant_id, direction, private_batna, include_mediator=False):
    scenario = price_scenario(include_mediator=include_mediator)
    return AgentProfile(
        identity=PublicAgentIdentity(
            participant=scenario.participant(participant_id),
            persona=f"Public {participant_id} persona",
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
                reservation_utility=0.4,
                batna_utility=0.3,
                batna_description=private_batna,
            ),
        ),
    )


def seller_profile(include_mediator=False):
    return profile(
        SELLER,
        PreferenceDirection.MAXIMIZE,
        "SELLER PRIVATE BATNA",
        include_mediator,
    )


def buyer_profile(include_mediator=False):
    return profile(
        BUYER,
        PreferenceDirection.MINIMIZE,
        "BUYER PRIVATE BATNA",
        include_mediator,
    )


def fake_model_configuration():
    return ModelConfiguration(
        provider=LLMProvider.FAKE,
        model="fake-negotiator",
        retry=RetryConfiguration(max_retries=0),
    )


def memory_snapshot(participant_id=SELLER, visible_messages=(), include_mediator=False):
    scenario = price_scenario(include_mediator=include_mediator)
    protocol = NegotiationProtocol(scenario, maximum_rounds=3, initial_turn=SELLER)
    session = protocol.create_session()
    own_profile = (
        seller_profile(include_mediator)
        if participant_id == SELLER
        else buyer_profile(include_mediator)
    )
    memory = AgentMemory(own_profile, scenario)
    snapshot = memory.update(
        AgentObservation(scenario=scenario, session=session, own_profile=own_profile),
        legal_actions=protocol.legal_action_types(session, participant_id),
        visible_messages=visible_messages,
    )
    return snapshot


def objective_json():
    return json.dumps(
        {
            "schema_version": "1.0",
            "objective": "Reach an acceptable agreement.",
            "constraints": ["Use only legal actions.", "Respect reservation utility."],
            "decision_rationale": "The stated goals and protocol define the objective.",
            "evidence": ["working legal actions", "owner preferences"],
        }
    )


def hypothesis_json():
    return json.dumps(
        {
            "schema_version": "1.0",
            "hypothesis": "The partner may accept a balanced price.",
            "confidence": 0.5,
            "decision_rationale": "Only public behavior is available.",
            "evidence": ["visible offers and messages only"],
        }
    )


def plan_json():
    return json.dumps(
        {
            "schema_version": "1.0",
            "plan": ["Make a valid offer within the reservation bound."],
            "target_utility": 0.5,
            "decision_rationale": "A balanced valid offer can enable agreement.",
            "evidence": ["round and deadline", "legal action list"],
        }
    )


def propose_json(snapshot, price=50.0, action_name="propose"):
    policy_observation = LLMNegotiationPolicy(
        FakeLLMClient(), fake_model_configuration()
    ).build_agent_visible_observation(snapshot)
    return json.dumps(
        {
            "schema_version": "1.0",
            "action": {
                "action": action_name,
                "actor_id": str(snapshot.owner_id),
                "recipients": [str(item) for item in policy_observation.expected_recipients],
                "round_number": snapshot.working.round_number,
                "offer": {
                    "offer_id": str(policy_observation.required_offer_id),
                    "values": [
                        {"kind": "numeric", "issue_id": "price", "value": price}
                    ],
                },
            },
            "decision_rationale": "This is a legal and acceptable proposal.",
            "evidence": ["required offer id", "public issue bounds"],
        }
    )


def valid_queue(action_json, usage=None):
    return tuple(
        FakeResponse(text=text, usage=usage or LLMUsage())
        for text in (objective_json(), hypothesis_json(), plan_json(), action_json)
    )


def stage_metrics(policy):
    return {item.stage: item for item in policy.last_trace.stages}


def test_valid_structured_action_runs_all_cognitive_stages_and_tracks_telemetry():
    snapshot = memory_snapshot()
    snapshot_before = snapshot.model_dump_json()
    usage = LLMUsage(input_tokens=2, output_tokens=1, total_tokens=3)
    ticks = iter(index / 1000.0 for index in range(8))
    client = FakeLLMClient(
        queued=valid_queue(propose_json(snapshot), usage=usage),
        clock=lambda: next(ticks),
        record_requests=True,
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    action = policy.choose_action(snapshot)

    assert isinstance(policy, NegotiationPolicy)
    assert snapshot.model_dump_json() == snapshot_before
    assert isinstance(action, ProposeAction)
    assert action.offer.values[0].value == 50.0
    assert policy.last_trace.used_fallback is False
    metrics = stage_metrics(policy)
    for stage in (
        CognitiveStage.OBJECTIVE_CONSTRAINTS,
        CognitiveStage.OPPONENT_HYPOTHESIS,
        CognitiveStage.PLANNING,
        CognitiveStage.ACTION_GENERATION,
    ):
        assert metrics[stage].model_calls == 1
        assert metrics[stage].latency_ms == pytest.approx(1.0)
        assert metrics[stage].usage == usage
        assert metrics[stage].succeeded is True
    assert metrics[CognitiveStage.OBSERVATION].model_calls == 0
    assert metrics[CognitiveStage.GROUNDING].model_calls == 0
    action_request = next(
        json.loads(request.messages[-1].content)
        for request in client.recorded_requests
        if json.loads(request.messages[-1].content)["stage"] == "action_generation"
    )
    schema_text = json.dumps(action_request["response_schema"], sort_keys=True)
    assert "ProposeAction" in schema_text
    assert "WithdrawAction" in schema_text
    assert "AcceptAction" not in schema_text
    assert "CounterAction" not in schema_text
    assert "RejectAction" not in schema_text
    belief_contexts = {
        payload["stage"]: payload["participant_visible_context"]
        for payload in (
            json.loads(request.messages[-1].content)
            for request in client.recorded_requests
        )
        if payload["stage"] in {"opponent_hypothesis", "planning"}
    }
    assert belief_contexts["opponent_hypothesis"]["opponent_beliefs"]
    assert "uncertain inferences" in belief_contexts["planning"]["belief_guidance"]
    assert policy.last_trace.opponent_beliefs == policy.last_belief_states


def test_opponent_hypothesis_is_updated_from_its_concise_prior_record():
    snapshot = memory_snapshot()
    action_json = propose_json(snapshot)
    client = FakeLLMClient(
        queued=valid_queue(action_json) + valid_queue(action_json),
        record_requests=True,
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    policy.choose_action(snapshot)
    policy.choose_action(snapshot)

    hypothesis_contexts = []
    for request in client.recorded_requests:
        payload = json.loads(request.messages[-1].content)
        if payload["stage"] == "opponent_hypothesis":
            hypothesis_contexts.append(payload["participant_visible_context"])
    assert hypothesis_contexts[0]["previous_hypothesis"] is None
    assert hypothesis_contexts[1]["previous_hypothesis"]["hypothesis"] == (
        "The partner may accept a balanced price."
    )
    assert "prompt" not in type(policy.last_trace).model_fields


def test_malformed_json_gets_exactly_one_repair_attempt():
    snapshot = memory_snapshot()
    client = FakeLLMClient(
        queued=(
            FakeResponse(text=objective_json()),
            FakeResponse(text=hypothesis_json()),
            FakeResponse(text=plan_json()),
            FakeResponse(text="not-json"),
            FakeResponse(text=propose_json(snapshot)),
        )
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    action = policy.choose_action(snapshot)

    assert isinstance(action, ProposeAction)
    metrics = stage_metrics(policy)[CognitiveStage.ACTION_GENERATION]
    assert metrics.model_calls == 2
    assert metrics.repair_attempted is True
    assert client.call_count == 5


def test_context_illegal_action_is_repaired_without_fuzzy_action_selection():
    snapshot = memory_snapshot()
    illegal_accept = json.dumps(
        {
            "schema_version": "1.0",
            "action": {
                "action": "accept",
                "actor_id": "seller",
                "recipients": ["buyer"],
                "round_number": 1,
                "offer_id": "offer-does-not-exist",
            },
            "decision_rationale": "Invalid test action.",
            "evidence": ["none"],
        }
    )
    client = FakeLLMClient(
        queued=valid_queue(illegal_accept) + (FakeResponse(text=propose_json(snapshot)),)
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    action = policy.choose_action(snapshot)

    assert isinstance(action, ProposeAction)
    assert client.call_count == 5
    assert stage_metrics(policy)[CognitiveStage.ACTION_GENERATION].repair_attempted is True


def test_offer_below_reservation_utility_cannot_bypass_grounding():
    snapshot = memory_snapshot()
    client = FakeLLMClient(
        queued=valid_queue(propose_json(snapshot, price=10.0))
        + (FakeResponse(text=propose_json(snapshot, price=50.0)),)
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    action = policy.choose_action(snapshot)

    assert isinstance(action, ProposeAction)
    assert action.offer.values[0].value == 50.0
    assert client.call_count == 5
    assert stage_metrics(policy)[CognitiveStage.GROUNDING].succeeded is True
    assert stage_metrics(policy)[CognitiveStage.ACTION_GENERATION].repair_attempted is True


def test_refusal_then_failed_repair_uses_deterministic_safe_fallback():
    snapshot = memory_snapshot()
    client = FakeLLMClient(
        queued=(
            FakeResponse(text=objective_json()),
            FakeResponse(text=hypothesis_json()),
            FakeResponse(text=plan_json()),
            FakeResponse(text="I refuse to choose an action."),
            FakeResponse(text="I still refuse."),
        )
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    first = policy.choose_action(snapshot)

    assert isinstance(first, ProposeAction)
    assert policy.last_trace.used_fallback is True
    assert policy.last_trace.fallback_reason == "repair_failed"
    assert stage_metrics(policy)[CognitiveStage.FALLBACK].succeeded is True
    assert client.call_count == 5

    repeated_client = FakeLLMClient(
        queued=(
            FakeResponse(text=objective_json()),
            FakeResponse(text=hypothesis_json()),
            FakeResponse(text=plan_json()),
            FakeResponse(text="I refuse to choose an action."),
            FakeResponse(text="I still refuse."),
        )
    )
    repeated = LLMNegotiationPolicy(repeated_client, fake_model_configuration()).choose_action(
        snapshot
    )
    assert repeated == first


def test_timeout_uses_fallback_without_unbounded_policy_repair():
    snapshot = memory_snapshot()
    timeout = LLMClientError(
        LLMErrorInfo(
            code=LLMErrorCode.TIMEOUT,
            message="timeout",
            retryable=True,
            provider=LLMProvider.FAKE,
        )
    )
    client = FakeLLMClient(queued=(timeout,))
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    action = policy.choose_action(snapshot)

    assert isinstance(action, ProposeAction)
    assert policy.last_trace.used_fallback is True
    assert policy.last_trace.fallback_reason == "llm_retries_exhausted"
    assert client.call_count == 1
    assert stage_metrics(policy)[CognitiveStage.OBJECTIVE_CONSTRAINTS].model_calls == 1


def test_requests_contain_visible_bus_messages_but_not_opponent_private_preferences():
    scenario = price_scenario(include_mediator=True)
    bus = MessageBus(
        negotiation_id=scenario.scenario_id,
        participants=(SELLER, BUYER, MEDIATOR),
        mediator_ids=(MEDIATOR,),
        audit_reader_ids=(),
    )
    bus.send(
        sender=BUYER,
        recipients=(SELLER, MEDIATOR),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message_type=MessageType.INTENT_SIGNAL,
        visibility=MessageVisibility.PUBLIC,
        content="PUBLIC BUYER SIGNAL",
        correlation_id=CorrelationId("public-signal"),
    )
    bus.send(
        sender=BUYER,
        recipients=(MEDIATOR,),
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        message_type=MessageType.QUESTION,
        visibility=MessageVisibility.DIRECT_PRIVATE,
        content="BUYER MEDIATOR SECRET",
        correlation_id=CorrelationId("private-signal"),
    )
    snapshot = memory_snapshot(
        visible_messages=bus.inbox(SELLER),
        include_mediator=True,
    )
    client = FakeLLMClient(
        queued=valid_queue(propose_json(snapshot)),
        record_requests=True,
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    policy.choose_action(snapshot)

    prompt_text = "\n".join(
        message.content
        for request in client.recorded_requests
        for message in request.messages
    )
    assert "PUBLIC BUYER SIGNAL" in prompt_text
    assert "BUYER MEDIATOR SECRET" not in prompt_text
    assert "BUYER PRIVATE BATNA" not in prompt_text
    assert "SELLER PRIVATE BATNA" in prompt_text
    assert "chain-of-thought" in prompt_text
    assert policy.last_trace.decision_rationale
    assert not hasattr(policy.last_trace, "prompt")


def stage_rule_response(request):
    payload = json.loads(request.messages[-1].content)
    stage = payload["stage"]
    if stage == "objective_constraints":
        return FakeResponse(text=objective_json())
    if stage == "opponent_hypothesis":
        return FakeResponse(text=hypothesis_json())
    if stage == "planning":
        return FakeResponse(text=plan_json())
    context = payload["participant_visible_context"]
    observation = context["observation"]
    if observation["outstanding_offer"] is None:
        action = {
            "action": "propose",
            "actor_id": observation["participant_id"],
            "recipients": observation["expected_recipients"],
            "round_number": observation["round_number"],
            "offer": {
                "offer_id": observation["required_offer_id"],
                "values": [{"kind": "numeric", "issue_id": "price", "value": 50.0}],
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
                "decision_rationale": "The action is legal and meets the reservation bound.",
                "evidence": ["participant-visible observation"],
            }
        )
    )


def test_fake_llm_policies_complete_a_full_protocol_negotiation():
    scenario = price_scenario()
    seller = seller_profile()
    buyer = buyer_profile()
    seller_client = FakeLLMClient(
        rules=(FakeRule(predicate=lambda _request: True, responder=stage_rule_response),)
    )
    buyer_client = FakeLLMClient(
        rules=(FakeRule(predicate=lambda _request: True, responder=stage_rule_response),)
    )
    policies = {
        SELLER: LLMNegotiationPolicy(seller_client, fake_model_configuration()),
        BUYER: LLMNegotiationPolicy(buyer_client, fake_model_configuration()),
    }

    result = run_policy_session(
        scenario,
        profiles={SELLER: seller, BUYER: buyer},
        policies=policies,
        maximum_rounds=3,
    )

    assert result.session.phase is ProtocolPhase.AGREED
    assert result.session.outcome.agreement.offer.values[0].value == 50.0
    assert len(result.offer_trajectory) == 1
    assert seller_client.call_count == 4
    assert buyer_client.call_count == 4
    assert isinstance(policies[BUYER].last_trace.selected_action, AcceptAction)
    assert policies[SELLER].last_trace.used_fallback is False
    assert policies[BUYER].last_trace.used_fallback is False
    assert all(
        any(
            isinstance(event, BeliefSnapshotMemoryEvent)
            for event in result.memory_snapshots[participant_id].episodic.events
        )
        for participant_id in (SELLER, BUYER)
    )
    assert len(result.verification_log) == 2
    assert all(
        entry.results[-1].verdict.value == "pass"
        and entry.correction_count == 0
        and entry.used_fallback is False
        for entry in result.verification_log
    )


def test_long_explanations_from_real_models_are_shortened_not_rejected():
    # Observed with gpt-4.1-mini: rationales longer than 500 characters, 12 constraints and
    # fenced JSON caused repair failures and needless fallbacks.
    snapshot = memory_snapshot()
    long_text = "The seller has not moved enough on price, so I keep my position. " * 15
    objective = json.loads(objective_json())
    objective["decision_rationale"] = long_text
    objective["constraints"] = [f"Constraint number {index}." for index in range(12)]
    plan = json.loads(plan_json())
    plan["decision_rationale"] = long_text
    action = "```json\n" + propose_json(snapshot) + "\n```"
    client = FakeLLMClient(
        queued=(
            FakeResponse(text=json.dumps(objective)),
            FakeResponse(text=hypothesis_json()),
            FakeResponse(text=json.dumps(plan)),
            FakeResponse(text=action),
        )
    )
    policy = LLMNegotiationPolicy(client, fake_model_configuration())

    chosen = policy.choose_action(snapshot)

    assert isinstance(chosen, ProposeAction)
    assert policy.last_trace.used_fallback is False
    assert client.call_count == 4  # no repair call was needed
    assert len(policy.last_trace.objective.constraints) == 10
    assert len(policy.last_trace.objective.decision_rationale) <= 500
    assert len(policy.last_trace.plan.decision_rationale) <= 500


def test_normalization_never_relaxes_the_typed_action():
    snapshot = memory_snapshot()
    action = json.loads(propose_json(snapshot))
    action["action"]["unexpected"] = "field"
    action["action"]["offer"]["values"][0]["value"] = "50"
    normalized = json.loads(normalize_stage_json(json.dumps(action)))
    assert normalized["action"] == action["action"]
    with pytest.raises(Exception):
        StructuredActionDecision.model_validate_json(json.dumps(normalized))
