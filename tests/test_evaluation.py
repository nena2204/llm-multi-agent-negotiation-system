import json
from datetime import datetime, timezone

import pytest

from llm_negotiation.communication import MessageBus, MessageType, MessageVisibility
from llm_negotiation.domain import (
    AcceptAction,
    ActionType,
    CategoryUtility,
    CategoricalIssue,
    CategoricalIssueValue,
    CategoricalPreference,
    CounterAction,
    IssueId,
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
)
from llm_negotiation.evaluation import (
    EvaluationBundle,
    EvaluationInput,
    JudgeAccessMode,
    JudgeConfiguration,
    JudgeOutputError,
    LLMJudge,
    ModelCallRecord,
    ParetoStatus,
    DeterministicEvaluation,
    evaluate_episode,
    evaluate_judge_panel,
    export_evaluation_json,
    legacy_midpoint_fairness,
)
from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    LLMProvider,
    LLMUsage,
    ModelConfiguration,
    RetryConfiguration,
)
from llm_negotiation.protocol import NegotiationProtocol, ProtocolPhase
from llm_negotiation.verification import (
    VerificationLogEntry,
    VerificationReason,
    VerificationReasonCode,
    VerificationResult,
    VerificationSeverity,
    VerificationVerdict,
    VerifierKind,
    VerifierMetadata,
)


SELLER = ParticipantId("seller")
BUYER = ParticipantId("buyer")
AUDITOR = ParticipantId("auditor")
PRICE = IssueId("price")
DELIVERY = IssueId("delivery")


def price_scenario():
    return NegotiationScenario(
        scenario_id="evaluation-price",
        title="Alice and Bob price negotiation",
        participants=(
            Participant(
                participant_id=SELLER,
                display_name="Alice",
                role=ParticipantRole.SELLER,
            ),
            Participant(
                participant_id=BUYER,
                display_name="Bob",
                role=ParticipantRole.BUYER,
            ),
        ),
        issues=(NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),),
    )


def price_preferences(
    seller_reservation=0.2,
    buyer_reservation=0.2,
    seller_batna=0.1,
    buyer_batna=0.1,
):
    return (
        ParticipantPreferences(
            participant_id=SELLER,
            issue_preferences=(
                NumericPreference(
                    issue_id=PRICE,
                    weight=1.0,
                    direction=PreferenceDirection.MAXIMIZE,
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=seller_reservation,
                batna_utility=seller_batna,
                batna_description="SELLER SECRET BATNA",
            ),
        ),
        ParticipantPreferences(
            participant_id=BUYER,
            issue_preferences=(
                NumericPreference(
                    issue_id=PRICE,
                    weight=1.0,
                    direction=PreferenceDirection.MINIMIZE,
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=buyer_reservation,
                batna_utility=buyer_batna,
                batna_description="BUYER SECRET BATNA",
            ),
        ),
    )


def price_offer(identifier, price):
    return Offer(
        offer_id=OfferId(identifier),
        values=(NumericIssueValue(issue_id=PRICE, value=price),),
    )


def agreement_session(price=50.0, maximum_rounds=3):
    scenario = price_scenario()
    protocol = NegotiationProtocol(scenario, maximum_rounds, initial_turn=SELLER)
    state = protocol.create_session()
    state = protocol.transition(
        state,
        ProposeAction(
            actor_id=SELLER,
            recipients=(BUYER,),
            round_number=1,
            offer=price_offer("agreement-offer", price),
        ),
    ).state
    state = protocol.transition(
        state,
        AcceptAction(
            actor_id=BUYER,
            recipients=(SELLER,),
            round_number=1,
            offer_id=OfferId("agreement-offer"),
        ),
    ).state
    return scenario, state


def test_hand_verifiable_asymmetric_price_metrics_and_legacy_formula():
    scenario, session = agreement_session(price=80.0)
    report = evaluate_episode(
        EvaluationInput(
            scenario=scenario,
            session=session,
            preferences=price_preferences(),
            elapsed_time_ms=1250.0,
        )
    )

    assert report.agreement_reached is True
    assert report.agreement_valid is True
    assert report.individually_rational is True
    assert report.rounds_to_agreement == 1
    assert report.elapsed_time_to_agreement_ms == 1250.0
    assert report.seller_utility == pytest.approx(0.8)
    assert report.buyer_utility == pytest.approx(0.2)
    assert report.social_welfare == pytest.approx(1.0)
    assert report.nash_product == pytest.approx(0.07)
    assert report.utility_balance == pytest.approx(0.4)
    assert report.pareto.status is ParetoStatus.COMPUTED
    assert report.pareto.is_pareto_efficient is True
    assert report.pareto.distance_to_frontier == pytest.approx(0.0)
    assert report.legacy_midpoint_fairness_score == 50.0
    assert legacy_midpoint_fairness(80.0, 20.0, 80.0) == 50.0


def test_no_deal_uses_batna_outcomes_and_has_no_agreement_fairness_or_pareto():
    scenario = price_scenario()
    protocol = NegotiationProtocol(scenario, 1, initial_turn=SELLER)
    state = protocol.create_session()
    state = protocol.transition(
        state,
        ProposeAction(actor_id=SELLER, round_number=1, offer=price_offer("no-deal-1", 80.0)),
    ).state
    state = protocol.transition(
        state,
        CounterAction(
            actor_id=BUYER,
            round_number=1,
            responds_to=OfferId("no-deal-1"),
            offer=price_offer("no-deal-2", 20.0),
        ),
    ).state
    assert state.phase is ProtocolPhase.EXPIRED
    preferences = price_preferences(seller_batna=0.1, buyer_batna=0.2)
    report = evaluate_episode(
        EvaluationInput(scenario=scenario, session=state, preferences=preferences)
    )

    assert report.agreement_reached is False
    assert report.agreement_valid is False
    assert report.rounds_to_agreement is None
    assert report.buyer_utility == 0.2
    assert report.seller_utility == 0.1
    assert [item.outcome_utility for item in report.participant_utilities] == [0.1, 0.2]
    assert report.social_welfare == pytest.approx(0.3)
    assert report.nash_product == 0.0
    assert report.utility_balance is None
    assert report.pareto.status is ParetoStatus.NOT_APPLICABLE


def test_zero_bargaining_range_is_finite_and_hand_verifiable():
    scenario, session = agreement_session(price=50.0)
    report = evaluate_episode(
        EvaluationInput(
            scenario=scenario,
            session=session,
            preferences=price_preferences(
                seller_reservation=0.5,
                buyer_reservation=0.5,
                seller_batna=0.5,
                buyer_batna=0.5,
            ),
        )
    )
    assert report.seller_utility == report.buyer_utility == 0.5
    assert report.social_welfare == 1.0
    assert report.nash_product == 0.0
    assert report.utility_balance == 1.0
    assert report.legacy_midpoint_fairness_score == 100.0
    assert legacy_midpoint_fairness(50.0, 50.0, 50.0) == 100.0
    assert legacy_midpoint_fairness(51.0, 50.0, 50.0) == 100.0


def multi_issue_input():
    scenario = NegotiationScenario(
        scenario_id="evaluation-multi",
        title="Multi issue",
        participants=(
            Participant(participant_id=SELLER, display_name="Alice"),
            Participant(participant_id=BUYER, display_name="Bob"),
        ),
        issues=(
            NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),
            CategoricalIssue(
                issue_id=DELIVERY,
                name="Delivery",
                choices=("slow", "fast"),
            ),
        ),
    )
    preferences = tuple(
        ParticipantPreferences(
            participant_id=participant_id,
            issue_preferences=(
                NumericPreference(
                    issue_id=PRICE,
                    weight=0.5,
                    direction=direction,
                ),
                CategoricalPreference(
                    issue_id=DELIVERY,
                    weight=0.5,
                    category_utilities=(
                        CategoryUtility(category="slow", utility=0.0),
                        CategoryUtility(category="fast", utility=1.0),
                    ),
                ),
            ),
            reservation=ReservationPolicy(reservation_utility=0.0, batna_utility=0.0),
        )
        for participant_id, direction in (
            (SELLER, PreferenceDirection.MAXIMIZE),
            (BUYER, PreferenceDirection.MINIMIZE),
        )
    )
    offer = Offer(
        offer_id=OfferId("multi-agreement"),
        values=(
            NumericIssueValue(issue_id=PRICE, value=50.0),
            CategoricalIssueValue(issue_id=DELIVERY, value="slow"),
        ),
    )
    protocol = NegotiationProtocol(scenario, 3, initial_turn=SELLER)
    state = protocol.transition(
        protocol.create_session(),
        ProposeAction(actor_id=SELLER, round_number=1, offer=offer),
    ).state
    state = protocol.transition(
        state,
        AcceptAction(actor_id=BUYER, round_number=1, offer_id=offer.offer_id),
    ).state
    return EvaluationInput(scenario=scenario, session=state, preferences=preferences)


def test_multi_issue_pareto_distance_detects_a_dominated_agreement():
    report = evaluate_episode(multi_issue_input())
    assert [item.agreement_utility for item in report.participant_utilities] == [0.25, 0.25]
    assert report.social_welfare == 0.5
    assert report.nash_product == pytest.approx(0.0625)
    assert report.utility_balance == 1.0
    assert report.pareto.status is ParetoStatus.COMPUTED
    assert report.pareto.is_pareto_efficient is False
    assert report.pareto.distance_to_frontier == pytest.approx(2**-0.5)


def verification_logs():
    metadata = VerifierMetadata(
        verifier_name="test-verifier",
        verifier_version="1.0",
        kind=VerifierKind.DETERMINISTIC,
    )
    rejected = VerificationResult(
        verdict=VerificationVerdict.REJECT,
        reasons=(
            VerificationReason(
                code=VerificationReasonCode.SCHEMA_INVALID,
                message="Invalid action.",
            ),
        ),
        severity=VerificationSeverity.ERROR,
        checked_action_id="checked-1",
        verifier_metadata=(metadata,),
    )
    passed = VerificationResult(
        verdict=VerificationVerdict.PASS,
        severity=VerificationSeverity.INFO,
        checked_action_id="checked-2",
        verifier_metadata=(metadata,),
    )
    return (
        VerificationLogEntry(
            negotiation_id="evaluation-price",
            sequence_number=1,
            results=(rejected, passed),
            correction_count=1,
            used_fallback=False,
            selected_action_type=ActionType.PROPOSE,
        ),
        VerificationLogEntry(
            negotiation_id="evaluation-price",
            sequence_number=2,
            results=(passed.model_copy(update={"checked_action_id": "checked-3"}),),
            correction_count=0,
            used_fallback=False,
            selected_action_type=ActionType.ACCEPT,
        ),
    )


def test_action_message_model_latency_and_usage_metrics_aggregate_exactly():
    scenario, session = agreement_session()
    bus = MessageBus(
        negotiation_id=scenario.scenario_id,
        participants=(SELLER, BUYER),
        mediator_ids=(),
        audit_reader_ids=(AUDITOR,),
    )
    bus.send(
        sender=SELLER,
        recipients=(BUYER,),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        message_type=MessageType.PROPOSAL_EXPLANATION,
        visibility=MessageVisibility.PUBLIC,
        content="Public explanation.",
    )
    records = (
        ModelCallRecord(
            source="policy",
            participant_id=SELLER,
            provider=LLMProvider.FAKE,
            model="fake-a",
            model_calls=2,
            latency_ms=12.5,
            usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        ),
        ModelCallRecord(
            source="verifier",
            provider=LLMProvider.FAKE,
            model="fake-b",
            model_calls=1,
            latency_ms=7.5,
            usage=LLMUsage(input_tokens=2, output_tokens=1, total_tokens=3),
        ),
    )
    report = evaluate_episode(
        EvaluationInput(
            scenario=scenario,
            session=session,
            preferences=price_preferences(),
            messages=bus.audit_view(AUDITOR),
            verification_log=verification_logs(),
            model_calls=records,
        )
    )
    assert report.action_quality.submitted_action_count == 2
    assert report.action_quality.invalid_action_rate == 0.5
    assert report.action_quality.correction_rate == 0.5
    assert report.telemetry.message_count == 1
    assert report.telemetry.public_message_count == 1
    assert report.telemetry.model_call_count == 3
    assert report.telemetry.model_latency_ms == 20.0
    assert report.telemetry.model_usage == LLMUsage(
        input_tokens=5, output_tokens=3, total_tokens=8
    )


def fake_model(model="fake-judge"):
    return ModelConfiguration(
        provider=LLMProvider.FAKE,
        model=model,
        retry=RetryConfiguration(max_retries=0),
    )


def judge_json(score=0.5, citation="action-1"):
    return json.dumps(
        {
            "schema_version": "1.0",
            "scores": [
                {
                    "rubric": rubric,
                    "score": score,
                    "rationale": "Supported by the cited public event.",
                    "cited_event_ids": [citation],
                }
                for rubric in (
                    "process_fairness",
                    "communication_quality",
                    "justification_quality",
                    "coercion_safety",
                )
            ],
            "cited_event_ids": [citation],
            "summary": "A bounded qualitative assessment.",
        }
    )


def judge_input():
    scenario, session = agreement_session()
    return EvaluationInput(
        scenario=scenario,
        session=session,
        preferences=price_preferences(),
    )


def test_llm_judge_is_anonymized_bounded_public_only_and_cannot_change_outcome():
    input = judge_input()
    state_before = input.session.model_dump_json()
    client = FakeLLMClient(
        queued=(FakeResponse(text=judge_json(citation="action-2")),),
        record_requests=True,
    )
    judge = LLMJudge(
        client,
        fake_model(),
        JudgeConfiguration(
            judge_id="judge-a",
            transcript_event_limit=1,
            transcript_character_limit=500,
        ),
    )
    sample = judge.evaluate(input)
    prompt = "\n".join(
        message.content
        for request in client.recorded_requests
        for message in request.messages
    )
    assert sample.cited_event_ids == ("action-2",)
    assert sample.model_calls == 1
    assert "participant-1" in prompt
    assert "participant-2" in prompt
    for secret in ("Alice", "Bob", "seller", "buyer", "SECRET BATNA"):
        assert secret.casefold() not in prompt.casefold()
    payload = json.loads(client.recorded_requests[0].messages[-1].content)
    transcript = payload["judge_context"]["bounded_transcript"]
    assert len(transcript) == 1
    assert sum(len(item["text"]) for item in transcript) <= 500
    assert input.session.model_dump_json() == state_before


def test_private_judge_mode_is_explicitly_research_labelled_and_still_anonymized():
    with pytest.raises(ValueError, match="explicit"):
        JudgeConfiguration(
            judge_id="unsafe",
            access_mode=JudgeAccessMode.RESEARCH_PRIVATE,
        )
    input = judge_input()
    client = FakeLLMClient(
        queued=(FakeResponse(text=judge_json()),),
        record_requests=True,
    )
    judge = LLMJudge(
        client,
        fake_model(),
        JudgeConfiguration(
            judge_id="research-judge",
            access_mode=JudgeAccessMode.RESEARCH_PRIVATE,
            allow_private_research_context=True,
        ),
    )
    judge.evaluate(input)
    prompt = client.recorded_requests[0].messages[-1].content
    assert "research_only_private_preferences" in prompt
    assert "research_private" in prompt
    assert "SELLER SECRET BATNA" not in prompt
    assert "participant-1 SECRET BATNA" in prompt


def test_malformed_or_fabricated_judge_output_is_rejected():
    input = judge_input()
    malformed = LLMJudge(
        FakeLLMClient(queued=(FakeResponse(text="not-json"),)),
        fake_model(),
        JudgeConfiguration(judge_id="malformed"),
    )
    with pytest.raises(JudgeOutputError, match="malformed"):
        malformed.evaluate(input)

    fabricated = LLMJudge(
        FakeLLMClient(queued=(FakeResponse(text=judge_json(citation="invented-event")),)),
        fake_model(),
        JudgeConfiguration(judge_id="fabricated"),
    )
    with pytest.raises(JudgeOutputError, match="outside"):
        fabricated.evaluate(input)


def test_multiple_models_and_samples_are_aggregated_without_majority_claim():
    input = judge_input()
    judge_a = LLMJudge(
        FakeLLMClient(
            queued=(FakeResponse(text=judge_json(0.2)), FakeResponse(text=judge_json(0.4)))
        ),
        fake_model("judge-model-a"),
        JudgeConfiguration(judge_id="judge-a"),
    )
    judge_b = LLMJudge(
        FakeLLMClient(
            queued=(FakeResponse(text=judge_json(0.8)), FakeResponse(text=judge_json(1.0)))
        ),
        fake_model("judge-model-b"),
        JudgeConfiguration(judge_id="judge-b"),
    )
    qualitative = evaluate_judge_panel(input, (judge_a, judge_b), samples_per_judge=2)
    assert len(qualitative.samples) == 4
    assert {sample.judge_id for sample in qualitative.samples} == {"judge-a", "judge-b"}
    assert {sample.model for sample in qualitative.samples} == {
        "judge-model-a",
        "judge-model-b",
    }
    assert all(item.mean == pytest.approx(0.6) for item in qualitative.rubric_aggregates)
    assert all(item.median == pytest.approx(0.6) for item in qualitative.rubric_aggregates)
    assert "not ground truth" in qualitative.interpretation

    deterministic = evaluate_episode(input)
    bundle = EvaluationBundle(deterministic=deterministic, qualitative=qualitative)
    restored = EvaluationBundle.model_validate_json(bundle.model_dump_json())
    assert restored == bundle
    assert EvaluationBundle.model_validate_json(export_evaluation_json(bundle)) == bundle
    assert DeterministicEvaluation.model_validate_json(
        deterministic.model_dump_json()
    ) == deterministic
