import json
from datetime import datetime, timezone

import pytest

from llm_negotiation.communication import MessageBus, MessageType, build_episode
from llm_negotiation.domain import (
    AcceptAction,
    CategoryUtility,
    CategoricalIssue,
    CategoricalIssueValue,
    CategoricalPreference,
    CounterAction,
    IssueId,
    MediationAccessMode,
    MediationTrigger,
    MediatorInterventionKind,
    MediatorInterventionId,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    PreferenceDirection,
    ProposeAction,
    RejectAction,
    RequestMediationAction,
    ReservationPolicy,
    calculate_utility,
)
from llm_negotiation.llm import (
    FakeLLMClient,
    FakeResponse,
    LLMProvider,
    ModelConfiguration,
    RetryConfiguration,
)
from llm_negotiation.mediation import (
    ConfidentialPreferenceSummary,
    DeadlockConfiguration,
    DeadlockDetector,
    DeadlockSignalType,
    DeterministicMediator,
    LLMMediator,
    MediationObjective,
    MediationService,
    MediatorConfiguration,
    MediatorContext,
)
from llm_negotiation.protocol import (
    MediatorInterventionEvent,
    NegotiationProtocol,
    NegotiationSession,
    ProtocolPhase,
)


ALICE = ParticipantId("alice")
BOB = ParticipantId("bob")
MEDIATOR = ParticipantId("mediator")
AUDITOR = ParticipantId("auditor")
PRICE = IssueId("price")
DELIVERY = IssueId("delivery")


def price_scenario():
    return NegotiationScenario(
        scenario_id="mediation-price",
        title="Price mediation",
        participants=(
            Participant(participant_id=ALICE, display_name="Alice"),
            Participant(participant_id=BOB, display_name="Bob"),
        ),
        issues=(NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0),),
    )


def price_offer(identifier, value):
    return Offer(
        offer_id=OfferId(identifier),
        values=(NumericIssueValue(issue_id=PRICE, value=value),),
    )


def price_preferences(participant_id, direction, reservation):
    return ParticipantPreferences(
        participant_id=participant_id,
        issue_preferences=(
            NumericPreference(issue_id=PRICE, weight=1.0, direction=direction),
        ),
        reservation=ReservationPolicy(
            reservation_utility=reservation,
            batna_utility=reservation,
            batna_description=f"{participant_id} confidential BATNA",
        ),
    )


def active_price_session(maximum_rounds=5):
    protocol = NegotiationProtocol(price_scenario(), maximum_rounds, initial_turn=ALICE)
    state = protocol.create_session()
    state = protocol.transition(
        state,
        ProposeAction(actor_id=ALICE, round_number=1, offer=price_offer("alice-1", 20.0)),
    ).state
    state = protocol.transition(
        state,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            offer=price_offer("bob-1", 80.0),
            responds_to=OfferId("alice-1"),
        ),
    ).state
    return protocol, state


def enter_requested_mediation(protocol, state):
    return protocol.transition(
        state,
        RequestMediationAction(
            actor_id=BOB,
            round_number=state.round_number,
            reason="Please mediate.",
        ),
    ).state


def test_access_modes_default_public_and_oracle_requires_explicit_research_opt_in():
    assert MediatorConfiguration().access_mode is MediationAccessMode.PUBLIC_ONLY
    assert MediatorConfiguration().production_safe is True
    with pytest.raises(ValueError, match="research-only"):
        MediatorConfiguration(access_mode=MediationAccessMode.SIMULATION_ORACLE)

    protocol, active = active_price_session()
    mediated = enter_requested_mediation(protocol, active)
    summary = ConfidentialPreferenceSummary(
        participant_id=ALICE,
        issue_preferences=price_preferences(
            ALICE, PreferenceDirection.MAXIMIZE, 0.4
        ).issue_preferences,
        reservation_utility=0.4,
    )
    with pytest.raises(ValueError, match="public-only"):
        MediatorContext(
            scenario=price_scenario(),
            session=mediated,
            confidential_summaries=(summary,),
        )

    oracle = MediatorConfiguration(
        access_mode=MediationAccessMode.SIMULATION_ORACLE,
        allow_simulation_oracle=True,
    )
    assert oracle.production_safe is False


def test_explicit_deadlock_invalid_action_and_deadline_triggers_are_deterministic():
    protocol, active = active_price_session(maximum_rounds=3)
    requested = enter_requested_mediation(protocol, active)
    detector = DeadlockDetector(price_scenario(), DeadlockConfiguration())
    explicit = detector.assess(requested)
    assert explicit.trigger is MediationTrigger.EXPLICIT_REQUEST

    deadline = detector.assess(active)
    assert deadline.should_enter is True
    assert deadline.trigger is MediationTrigger.DEADLINE

    repeated_protocol = NegotiationProtocol(price_scenario(), 10, initial_turn=ALICE)
    state = repeated_protocol.create_session()
    state = repeated_protocol.transition(
        state,
        ProposeAction(actor_id=ALICE, round_number=1, offer=price_offer("r1", 50.0)),
    ).state
    for identifier, actor in (("r2", BOB), ("r3", ALICE)):
        state = repeated_protocol.transition(
            state,
            CounterAction(
                actor_id=actor,
                round_number=state.round_number,
                offer=price_offer(identifier, 50.0),
                responds_to=state.latest_valid_offer.offer_id,
            ),
        ).state
    repeated = DeadlockDetector(
        price_scenario(),
        DeadlockConfiguration(
            low_concession_steps=2,
            deadline_rounds_remaining=None,
        ),
    ).assess(state)
    assert repeated.trigger is MediationTrigger.DEADLOCK
    assert {
        DeadlockSignalType.REPEATED_OFFERS,
        DeadlockSignalType.LOW_CONCESSION,
    }.issubset({signal.signal_type for signal in repeated.signals})

    invalid = detector.assess(active, invalid_action_actor_ids=(BOB, BOB, BOB))
    assert invalid.trigger is MediationTrigger.DEADLOCK
    assert invalid.signals[0].signal_type is DeadlockSignalType.REPEATED_INVALID_ACTIONS


def test_cycle_signal_and_automatic_mediation_entry_are_replayable():
    protocol = NegotiationProtocol(price_scenario(), 10, initial_turn=ALICE)
    state = protocol.create_session()
    state = protocol.transition(
        state,
        ProposeAction(actor_id=ALICE, round_number=1, offer=price_offer("c1", 40.0)),
    ).state
    for identifier, actor, value in (
        ("c2", BOB, 60.0),
        ("c3", ALICE, 40.0),
        ("c4", BOB, 60.0),
    ):
        state = protocol.transition(
            state,
            CounterAction(
                actor_id=actor,
                round_number=state.round_number,
                offer=price_offer(identifier, value),
                responds_to=state.latest_valid_offer.offer_id,
            ),
        ).state
    decision = DeadlockDetector(
        price_scenario(),
        DeadlockConfiguration(
            repeated_offer_count=5,
            low_concession_steps=5,
            deadline_rounds_remaining=None,
        ),
    ).assess(state)
    assert [signal.signal_type for signal in decision.signals] == [
        DeadlockSignalType.CYCLING
    ]

    entered = MediationService().enter_if_triggered(protocol, state, decision)
    assert entered.state.phase is ProtocolPhase.MEDIATION
    assert protocol.replay(entered.state.event_sequence) == entered.state


def test_price_midpoint_is_non_binding_and_participants_can_accept_or_reject_it():
    protocol, active = active_price_session()
    mediated = enter_requested_mediation(protocol, active)
    mediator = DeterministicMediator()
    intervention = mediator.intervene(
        MediatorContext(scenario=price_scenario(), session=mediated),
        MediationTrigger.EXPLICIT_REQUEST,
    )
    price = intervention.offer.value_for(PRICE)
    assert price.value == 50.0
    assert intervention.kind is MediatorInterventionKind.PROPOSAL

    proposed = protocol.apply_mediator_intervention(mediated, intervention).state
    assert proposed.phase is ProtocolPhase.MEDIATION
    assert proposed.outcome is None
    assert proposed.offer_acceptances == ()
    assert isinstance(proposed.event_sequence[-1], MediatorInterventionEvent)
    assert protocol.replay(proposed.event_sequence) == proposed
    assert NegotiationSession.model_validate_json(proposed.model_dump_json()) == proposed

    after_alice = protocol.transition(
        proposed,
        AcceptAction(
            actor_id=ALICE,
            round_number=proposed.round_number,
            offer_id=intervention.offer.offer_id,
        ),
    ).state
    agreed = protocol.transition(
        after_alice,
        AcceptAction(
            actor_id=BOB,
            round_number=after_alice.round_number,
            offer_id=intervention.offer.offer_id,
        ),
    ).state
    assert agreed.phase is ProtocolPhase.AGREED

    proposed_again = protocol.apply_mediator_intervention(
        mediated,
        intervention.model_copy(
            update={
                "intervention_id": MediatorInterventionId("mediator-intervention-reject"),
                "offer": price_offer("mediator-offer-reject", 50.0),
            }
        ),
    ).state
    rejected = protocol.transition(
        proposed_again,
        RejectAction(
            actor_id=ALICE,
            round_number=proposed_again.round_number,
            offer_id=OfferId("mediator-offer-reject"),
            reason="I prefer to continue negotiating.",
        ),
    ).state
    assert rejected.phase is ProtocolPhase.MEDIATION
    assert rejected.latest_valid_offer is None
    assert rejected.outcome is None

    counter_intervention = intervention.model_copy(
        update={
            "intervention_id": MediatorInterventionId("mediator-intervention-counter"),
            "offer": price_offer("mediator-offer-counter", 50.0),
        }
    )
    counter_state = protocol.apply_mediator_intervention(
        mediated, counter_intervention
    ).state
    countered = protocol.transition(
        counter_state,
        CounterAction(
            actor_id=ALICE,
            round_number=counter_state.round_number,
            responds_to=OfferId("mediator-offer-counter"),
            offer=price_offer("participant-counter", 55.0),
        ),
    ).state
    assert countered.phase is ProtocolPhase.MEDIATION
    assert countered.latest_valid_offer.offer_id == OfferId("participant-counter")


def test_confidential_and_oracle_infeasible_zones_refuse_without_public_leakage():
    protocol, active = active_price_session()
    mediated = enter_requested_mediation(protocol, active)
    seller = price_preferences(ALICE, PreferenceDirection.MAXIMIZE, 0.8)
    buyer = price_preferences(BOB, PreferenceDirection.MINIMIZE, 0.8)
    summaries = tuple(
        ConfidentialPreferenceSummary(
            participant_id=item.participant_id,
            issue_preferences=item.issue_preferences,
            reservation_utility=item.reservation.reservation_utility,
        )
        for item in (seller, buyer)
    )
    configuration = MediatorConfiguration(
        access_mode=MediationAccessMode.CONFIDENTIAL_SUMMARY
    )
    refusal = DeterministicMediator(configuration).intervene(
        MediatorContext(
            scenario=price_scenario(),
            session=mediated,
            access_mode=MediationAccessMode.CONFIDENTIAL_SUMMARY,
            confidential_summaries=summaries,
        ),
        MediationTrigger.EXPLICIT_REQUEST,
    )
    assert refusal.kind is MediatorInterventionKind.REFUSAL
    assert refusal.offer is None
    assert "0.8" not in refusal.public_explanation
    assert "confidential BATNA" not in refusal.public_explanation
    after_refusal = protocol.apply_mediator_intervention(mediated, refusal).state
    assert after_refusal.phase is ProtocolPhase.MEDIATION
    assert after_refusal.latest_valid_offer == mediated.latest_valid_offer


def multi_issue_scenario():
    return NegotiationScenario(
        scenario_id="mediation-multi",
        title="Multi-issue mediation",
        participants=(
            Participant(participant_id=ALICE, display_name="Alice"),
            Participant(participant_id=BOB, display_name="Bob"),
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


def multi_preferences(participant_id, price_direction, preferred_delivery):
    return ParticipantPreferences(
        participant_id=participant_id,
        issue_preferences=(
            NumericPreference(
                issue_id=PRICE,
                weight=0.5,
                direction=price_direction,
            ),
            CategoricalPreference(
                issue_id=DELIVERY,
                weight=0.5,
                category_utilities=tuple(
                    CategoryUtility(
                        category=choice,
                        utility=1.0 if choice == preferred_delivery else 0.0,
                    )
                    for choice in ("slow", "fast")
                ),
            ),
        ),
        reservation=ReservationPolicy(reservation_utility=0.2, batna_utility=0.2),
    )


def multi_offer(identifier, price, delivery):
    return Offer(
        offer_id=OfferId(identifier),
        values=(
            NumericIssueValue(issue_id=PRICE, value=price),
            CategoricalIssueValue(issue_id=DELIVERY, value=delivery),
        ),
    )


@pytest.mark.parametrize("objective", [MediationObjective.NASH_PRODUCT, MediationObjective.MAX_MIN])
def test_multi_issue_candidate_search_is_deterministic_complete_and_reservation_safe(objective):
    scenario = multi_issue_scenario()
    protocol = NegotiationProtocol(scenario, 5, initial_turn=ALICE)
    state = protocol.create_session()
    state = protocol.transition(
        state,
        ProposeAction(
            actor_id=ALICE,
            round_number=1,
            offer=multi_offer("multi-a", 80.0, "fast"),
        ),
    ).state
    state = protocol.transition(
        state,
        CounterAction(
            actor_id=BOB,
            round_number=1,
            responds_to=OfferId("multi-a"),
            offer=multi_offer("multi-b", 20.0, "slow"),
        ),
    ).state
    state = enter_requested_mediation(protocol, state)
    preferences = (
        multi_preferences(ALICE, PreferenceDirection.MAXIMIZE, "fast"),
        multi_preferences(BOB, PreferenceDirection.MINIMIZE, "slow"),
    )
    configuration = MediatorConfiguration(
        access_mode=MediationAccessMode.SIMULATION_ORACLE,
        allow_simulation_oracle=True,
        objective=objective,
    )
    context = MediatorContext(
        scenario=scenario,
        session=state,
        access_mode=MediationAccessMode.SIMULATION_ORACLE,
        simulation_ground_truth=preferences,
    )
    mediator = DeterministicMediator(configuration)
    first = mediator.intervene(context, MediationTrigger.EXPLICIT_REQUEST)
    second = mediator.intervene(context, MediationTrigger.EXPLICIT_REQUEST)
    assert first == second
    assert {value.issue_id for value in first.offer.values} == {PRICE, DELIVERY}
    assert all(
        calculate_utility(scenario, first.offer, item)
        >= item.reservation.reservation_utility
        for item in preferences
    )


def test_mediation_service_correlates_public_message_and_keeps_separate_intervention_log():
    protocol, active = active_price_session()
    mediated = enter_requested_mediation(protocol, active)
    intervention = DeterministicMediator().intervene(
        MediatorContext(scenario=price_scenario(), session=mediated),
        MediationTrigger.EXPLICIT_REQUEST,
    )
    bus = MessageBus(
        negotiation_id=mediated.scenario_id,
        participants=(ALICE, BOB, MEDIATOR),
        mediator_ids=(MEDIATOR,),
        audit_reader_ids=(AUDITOR,),
    )
    service = MediationService()
    result = service.apply(
        protocol,
        mediated,
        intervention,
        bus,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert service.interventions == (intervention,)
    assert result.message.message_type is MessageType.MEDIATOR_PROPOSAL
    assert result.message.correlation_id == result.transition.events[0].correlation_id
    assert bus.public_transcript() == (result.message,)
    episode = build_episode(result.transition.state, bus.audit_view(AUDITOR))
    assert episode.messages[0].referenced_offer_id == intervention.offer.offer_id


def test_llm_mediator_uses_strict_typed_output_and_only_public_context():
    protocol, active = active_price_session()
    mediated = enter_requested_mediation(protocol, active)
    private = (
        price_preferences(ALICE, PreferenceDirection.MAXIMIZE, 0.4),
        price_preferences(BOB, PreferenceDirection.MINIMIZE, 0.4),
    )
    context = MediatorContext(
        scenario=price_scenario(),
        session=mediated,
        access_mode=MediationAccessMode.SIMULATION_ORACLE,
        simulation_ground_truth=private,
    )
    response = json.dumps(
        {
            "schema_version": "1.0",
            "kind": "clarifying_question",
            "public_disagreement_summary": "The public price offers remain separated.",
            "public_explanation": "A public clarification may help.",
            "offer": None,
            "question": "Which public issue should we address first?",
        }
    )
    client = FakeLLMClient(
        queued=(FakeResponse(text=response),),
        record_requests=True,
    )
    mediator = LLMMediator(
        client,
        ModelConfiguration(
            provider=LLMProvider.FAKE,
            model="fake-mediator",
            retry=RetryConfiguration(max_retries=0),
        ),
        MediatorConfiguration(
            access_mode=MediationAccessMode.SIMULATION_ORACLE,
            allow_simulation_oracle=True,
        ),
    )
    intervention = mediator.intervene(context, MediationTrigger.EXPLICIT_REQUEST)
    assert intervention.kind is MediatorInterventionKind.CLARIFYING_QUESTION
    assert intervention.public_disagreement_summary == (
        "The public price offers remain separated."
    )
    assert intervention.offer is None
    prompt = "\n".join(
        message.content
        for request in client.recorded_requests
        for message in request.messages
    )
    assert "confidential BATNA" not in prompt
    assert "reservation_utility" not in prompt
    assert "simulation_oracle" in prompt


def test_llm_mediator_proposal_is_grounded_to_complete_domain_offer():
    protocol, active = active_price_session()
    mediated = enter_requested_mediation(protocol, active)
    output = json.dumps(
        {
            "schema_version": "1.0",
            "kind": "proposal",
            "public_disagreement_summary": "The public price offers remain separated.",
            "public_explanation": "This public midpoint suggestion is non-binding.",
            "offer": price_offer("llm-mediated-offer", 50.0).model_dump(mode="json"),
            "question": None,
        }
    )
    configuration = ModelConfiguration(
        provider=LLMProvider.FAKE,
        model="fake-mediator",
        retry=RetryConfiguration(max_retries=0),
    )
    intervention = LLMMediator(
        FakeLLMClient(queued=(FakeResponse(text=output),)), configuration
    ).intervene(
        MediatorContext(scenario=price_scenario(), session=mediated),
        MediationTrigger.EXPLICIT_REQUEST,
    )
    assert intervention.kind is MediatorInterventionKind.PROPOSAL
    applied = protocol.apply_mediator_intervention(mediated, intervention).state
    assert applied.latest_valid_offer == intervention.offer

    refusal_output = json.dumps(
        {
            "schema_version": "1.0",
            "kind": "refusal",
            "public_disagreement_summary": "The public record is currently insufficient.",
            "public_explanation": "I do not have enough public information to intervene.",
            "offer": None,
            "question": None,
        }
    )
    refusal = LLMMediator(
        FakeLLMClient(queued=(FakeResponse(text=refusal_output),)), configuration
    ).intervene(
        MediatorContext(scenario=price_scenario(), session=mediated),
        MediationTrigger.EXPLICIT_REQUEST,
    )
    refused_state = protocol.apply_mediator_intervention(mediated, refusal).state
    assert refusal.kind is MediatorInterventionKind.REFUSAL
    assert refused_state.phase is ProtocolPhase.MEDIATION
    assert refused_state.latest_valid_offer == mediated.latest_valid_offer

    malformed = LLMMediator(
        FakeLLMClient(queued=(FakeResponse(text='{"kind":"accept"}'),)),
        configuration,
    )
    with pytest.raises(ValueError):
        malformed.intervene(
            MediatorContext(scenario=price_scenario(), session=mediated),
            MediationTrigger.EXPLICIT_REQUEST,
        )


def test_deterministic_mediation_can_improve_outcome_without_forcing_it():
    unmediated_protocol, unmediated = active_price_session(maximum_rounds=2)
    unmediated = unmediated_protocol.transition(
        unmediated,
        CounterAction(
            actor_id=ALICE,
            round_number=2,
            responds_to=unmediated.latest_valid_offer.offer_id,
            offer=price_offer("unmediated-a", 30.0),
        ),
    ).state
    unmediated = unmediated_protocol.transition(
        unmediated,
        CounterAction(
            actor_id=BOB,
            round_number=2,
            responds_to=unmediated.latest_valid_offer.offer_id,
            offer=price_offer("unmediated-b", 70.0),
        ),
    ).state
    assert unmediated.phase is ProtocolPhase.EXPIRED

    protocol, active = active_price_session(maximum_rounds=2)
    mediated = enter_requested_mediation(protocol, active)
    intervention = DeterministicMediator().intervene(
        MediatorContext(scenario=price_scenario(), session=mediated),
        MediationTrigger.EXPLICIT_REQUEST,
    )
    mediated = protocol.apply_mediator_intervention(mediated, intervention).state
    for actor in (ALICE, BOB):
        mediated = protocol.transition(
            mediated,
            AcceptAction(
                actor_id=actor,
                round_number=mediated.round_number,
                offer_id=intervention.offer.offer_id,
            ),
        ).state
    assert mediated.phase is ProtocolPhase.AGREED
    assert mediated.outcome.agreement.offer == intervention.offer
