import math

import pytest
from pydantic import ValidationError

from llm_negotiation.agents import BuyerAgent, SellerAgent
from llm_negotiation.domain import (
    AcceptAction,
    Agreement,
    AgreementId,
    CategoryUtility,
    CategoricalIssue,
    CategoricalIssueValue,
    CategoricalPreference,
    CounterAction,
    IncompleteOfferError,
    InvalidWeightsError,
    IssueId,
    IssueValueError,
    MalformedActionError,
    MessageAction,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    OfferId,
    OutcomeStatus,
    OutcomeValidationError,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    PreferenceIssueMismatchError,
    ProposeAction,
    RejectAction,
    RequestMediationAction,
    ReservationPolicy,
    TerminalOutcome,
    UnknownIssueError,
    UnknownParticipantError,
    WithdrawAction,
    calculate_utility,
    parse_action,
    price_only_from_legacy,
    price_only_to_legacy,
)


BUYER = ParticipantId("buyer")
SELLER = ParticipantId("seller")
OBSERVER = ParticipantId("observer")
PRICE = IssueId("price")
DELIVERY = IssueId("delivery")


def multi_issue_scenario(include_observer=False):
    participants = [
        Participant(participant_id=BUYER, display_name="Buyer", role=ParticipantRole.BUYER),
        Participant(participant_id=SELLER, display_name="Seller", role=ParticipantRole.SELLER),
    ]
    if include_observer:
        participants.append(
            Participant(participant_id=OBSERVER, display_name="Observer", role=ParticipantRole.NEGOTIATOR)
        )
    return NegotiationScenario(
        scenario_id="procurement",
        title="Procurement",
        participants=tuple(participants),
        issues=(
            NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=100.0, unit="USD"),
            CategoricalIssue(
                issue_id=DELIVERY,
                name="Delivery",
                choices=("standard", "express"),
            ),
        ),
    )


def buyer_preferences(price_weight=0.7, delivery_weight=0.3):
    return ParticipantPreferences(
        participant_id=BUYER,
        issue_preferences=(
            NumericPreference(
                issue_id=PRICE,
                weight=price_weight,
                direction=PreferenceDirection.MINIMIZE,
            ),
            CategoricalPreference(
                issue_id=DELIVERY,
                weight=delivery_weight,
                category_utilities=(
                    CategoryUtility(category="standard", utility=0.2),
                    CategoryUtility(category="express", utility=1.0),
                ),
            ),
        ),
        reservation=ReservationPolicy(
            reservation_utility=0.4,
            batna_utility=0.35,
            batna_description="Buy from another supplier",
        ),
    )


def complete_offer(price=25.0, delivery="express", offer_id="offer-1"):
    return Offer(
        offer_id=OfferId(offer_id),
        values=(
            NumericIssueValue(issue_id=PRICE, value=price),
            CategoricalIssueValue(issue_id=DELIVERY, value=delivery),
        ),
    )


def test_multi_issue_utility_is_weighted_and_normalized():
    utility = calculate_utility(multi_issue_scenario(), complete_offer(), buyer_preferences())

    assert utility == pytest.approx(0.825)
    assert 0.0 <= utility <= 1.0


@pytest.mark.parametrize(
    ("price", "delivery", "expected"),
    [
        (0.0, "express", 1.0),
        (100.0, "standard", 0.06),
        (0.0, "standard", 0.76),
    ],
)
def test_utility_boundary_values(price, delivery, expected):
    utility = calculate_utility(
        multi_issue_scenario(),
        complete_offer(price=price, delivery=delivery),
        buyer_preferences(),
    )

    assert utility == pytest.approx(expected)


def test_weight_tolerance_is_accepted_but_invalid_sum_is_rejected():
    within_tolerance = buyer_preferences(price_weight=0.7000004, delivery_weight=0.3)
    assert sum(item.weight for item in within_tolerance.issue_preferences) == pytest.approx(1.0000004)

    with pytest.raises(ValidationError, match="issue weights must sum to 1") as exc_info:
        buyer_preferences(price_weight=0.6, delivery_weight=0.3)
    assert isinstance(exc_info.value.errors()[0]["ctx"]["error"], InvalidWeightsError)


def test_private_preferences_are_not_part_of_public_scenario():
    scenario = multi_issue_scenario()
    private = buyer_preferences()

    public_payload = scenario.model_dump(mode="json")
    assert "preferences" not in public_payload
    assert "reservation" not in str(public_payload)
    assert private.participant_id == scenario.participants[0].participant_id


def test_offer_must_be_complete_and_known_to_scenario():
    scenario = multi_issue_scenario()
    incomplete = Offer(
        offer_id=OfferId("incomplete"),
        values=(NumericIssueValue(issue_id=PRICE, value=50.0),),
    )
    unknown = Offer(
        offer_id=OfferId("unknown"),
        values=complete_offer().values
        + (NumericIssueValue(issue_id=IssueId("warranty"), value=1.0),),
    )

    with pytest.raises(IncompleteOfferError, match="delivery"):
        scenario.validate_offer(incomplete)
    with pytest.raises(UnknownIssueError, match="warranty"):
        scenario.validate_offer(unknown)


@pytest.mark.parametrize(
    "offer",
    [
        complete_offer(price=-0.01),
        complete_offer(price=100.01),
        complete_offer(delivery="overnight"),
        Offer(
            offer_id=OfferId("wrong-kind"),
            values=(
                CategoricalIssueValue(issue_id=PRICE, value="cheap"),
                CategoricalIssueValue(issue_id=DELIVERY, value="express"),
            ),
        ),
    ],
)
def test_out_of_range_or_wrong_issue_values_are_rejected(offer):
    with pytest.raises(IssueValueError):
        multi_issue_scenario().validate_offer(offer)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numeric_values_are_rejected_at_boundary(value):
    with pytest.raises(ValidationError, match="finite number"):
        NumericIssueValue(issue_id=PRICE, value=value)


def test_numeric_issue_rejects_a_non_finite_span():
    with pytest.raises(ValidationError, match="finite span"):
        NumericIssue(
            issue_id=PRICE,
            name="Extreme range",
            minimum=-1e308,
            maximum=1e308,
        )


def test_preference_issues_and_categories_must_match_scenario():
    scenario = multi_issue_scenario()
    price_only_preferences = ParticipantPreferences(
        participant_id=BUYER,
        issue_preferences=(
            NumericPreference(issue_id=PRICE, weight=1.0, direction=PreferenceDirection.MINIMIZE),
        ),
        reservation=ReservationPolicy(reservation_utility=0.5, batna_utility=0.4),
    )
    incomplete_categories = ParticipantPreferences(
        participant_id=BUYER,
        issue_preferences=(
            NumericPreference(issue_id=PRICE, weight=0.5, direction=PreferenceDirection.MINIMIZE),
            CategoricalPreference(
                issue_id=DELIVERY,
                weight=0.5,
                category_utilities=(CategoryUtility(category="express", utility=1.0),),
            ),
        ),
        reservation=ReservationPolicy(reservation_utility=0.5, batna_utility=0.4),
    )

    with pytest.raises(PreferenceIssueMismatchError, match="missing: delivery"):
        calculate_utility(scenario, complete_offer(), price_only_preferences)
    with pytest.raises(PreferenceIssueMismatchError, match="cover exactly"):
        calculate_utility(scenario, complete_offer(), incomplete_categories)


@pytest.mark.parametrize(
    ("payload", "action_class"),
    [
        (
            {
                "action": "propose",
                "actor_id": "buyer",
                "recipients": ["seller"],
                "round_number": 1,
                "offer": {
                    "offer_id": "offer-1",
                    "values": [
                        {"kind": "numeric", "issue_id": "price", "value": 25.0},
                        {"kind": "categorical", "issue_id": "delivery", "value": "express"},
                    ],
                },
            },
            ProposeAction,
        ),
        (
            {
                "action": "counter",
                "actor_id": "seller",
                "round_number": 2,
                "offer": {
                    "offer_id": "offer-2",
                    "values": [
                        {"kind": "numeric", "issue_id": "price", "value": 30.0},
                        {"kind": "categorical", "issue_id": "delivery", "value": "standard"},
                    ],
                },
                "responds_to": "offer-1",
            },
            CounterAction,
        ),
        ({"action": "accept", "actor_id": "buyer", "round_number": 3, "offer_id": "offer-2"}, AcceptAction),
        (
            {"action": "reject", "actor_id": "seller", "round_number": 3, "offer_id": "offer-2"},
            RejectAction,
        ),
        (
            {"action": "message", "actor_id": "buyer", "round_number": 1, "content": "Please clarify."},
            MessageAction,
        ),
        (
            {"action": "request_mediation", "actor_id": "buyer", "round_number": 4},
            RequestMediationAction,
        ),
        ({"action": "withdraw", "actor_id": "seller", "round_number": 4}, WithdrawAction),
    ],
)
def test_all_external_action_types_parse(payload, action_class):
    action = parse_action(payload)

    assert isinstance(action, action_class)
    multi_issue_scenario().validate_action(action)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "propose", "actor_id": "buyer", "round_number": 1},
        {"action": "dance", "actor_id": "buyer", "round_number": 1},
        {"action": "message", "actor_id": "buyer", "round_number": 1, "content": " "},
        {"action": "accept", "actor_id": "buyer", "round_number": -1, "offer_id": "offer-1"},
    ],
)
def test_malformed_external_actions_raise_explicit_domain_error(payload):
    with pytest.raises(MalformedActionError, match="malformed negotiation action"):
        parse_action(payload)


def test_actions_reject_unknown_participants():
    action = MessageAction(
        actor_id=ParticipantId("intruder"),
        round_number=1,
        content="Hello",
    )

    with pytest.raises(UnknownParticipantError, match="intruder"):
        multi_issue_scenario().validate_action(action)


def test_multiparty_agreement_requires_every_participant():
    scenario = multi_issue_scenario(include_observer=True)
    incomplete = Agreement(
        agreement_id=AgreementId("agreement-1"),
        offer=complete_offer(),
        accepted_by=(BUYER, SELLER),
        round_number=3,
    )
    complete = incomplete.model_copy(update={"accepted_by": (BUYER, SELLER, OBSERVER)})

    with pytest.raises(OutcomeValidationError, match="every scenario participant"):
        scenario.validate_agreement(incomplete)
    assert scenario.validate_agreement(complete) is complete


def test_terminal_outcome_requires_agreement_only_for_agreement_status():
    agreement = Agreement(
        agreement_id=AgreementId("agreement-1"),
        offer=complete_offer(),
        accepted_by=(BUYER, SELLER),
        round_number=3,
    )
    outcome = TerminalOutcome(
        status=OutcomeStatus.AGREEMENT,
        final_round=3,
        agreement=agreement,
    )
    assert outcome.agreement == agreement

    with pytest.raises(ValidationError, match="must contain an agreement"):
        TerminalOutcome(status=OutcomeStatus.AGREEMENT, final_round=3)
    with pytest.raises(ValidationError, match="only an agreement outcome"):
        TerminalOutcome(
            status=OutcomeStatus.WITHDRAWN,
            final_round=3,
            agreement=agreement,
        )


def test_domain_models_are_frozen():
    offer = complete_offer()
    with pytest.raises(ValidationError, match="frozen"):
        offer.offer_id = OfferId("changed")


def test_price_only_scenario_round_trips_legacy_configuration():
    buyer = BuyerAgent(
        name="Legacy Buyer",
        role="buyer",
        strategy="cooperative",
        max_price=120.0,
        current_offer=20.0,
    )
    seller = SellerAgent(
        name="Legacy Seller",
        role="seller",
        strategy="aggressive",
        initial_price=150.0,
        min_acceptable=80.0,
    )

    migration = price_only_from_legacy("Widget", buyer, seller, rounds=6)
    restored_buyer, restored_seller, restored_rounds = price_only_to_legacy(migration)

    assert len(migration.scenario.issues) == 1
    assert isinstance(migration.scenario.issues[0], NumericIssue)
    assert migration.scenario.issues[0].maximum == 1000.0
    assert migration.scenario.issues[0].maximum not in {buyer.max_price, seller.min_acceptable}
    assert "reservation" not in str(migration.scenario.model_dump(mode="json"))
    assert restored_buyer.name == buyer.name
    assert restored_buyer.strategy == buyer.strategy
    assert restored_buyer.initial_offer == buyer.initial_offer
    assert restored_buyer.max_price == buyer.max_price
    assert restored_seller.name == seller.name
    assert restored_seller.strategy == seller.strategy
    assert restored_seller.initial_price == seller.initial_price
    assert restored_seller.min_acceptable == seller.min_acceptable
    assert restored_rounds == 6


def test_price_only_reservation_utilities_match_legacy_bounds():
    buyer = BuyerAgent(name="Buyer", role="buyer", max_price=120.0, current_offer=20.0)
    seller = SellerAgent(
        name="Seller",
        role="seller",
        initial_price=150.0,
        min_acceptable=80.0,
    )
    migration = price_only_from_legacy("Widget", buyer, seller, rounds=6)
    buyer_bound = Offer(
        offer_id=OfferId("buyer-bound"),
        values=(NumericIssueValue(issue_id=IssueId("price"), value=120.0),),
    )
    seller_bound = Offer(
        offer_id=OfferId("seller-bound"),
        values=(NumericIssueValue(issue_id=IssueId("price"), value=80.0),),
    )

    assert calculate_utility(
        migration.scenario, buyer_bound, migration.buyer_preferences
    ) == pytest.approx(migration.buyer_preferences.reservation.reservation_utility)
    assert calculate_utility(
        migration.scenario, seller_bound, migration.seller_preferences
    ) == pytest.approx(migration.seller_preferences.reservation.reservation_utility)


def test_price_adapter_handles_extremely_large_finite_prices():
    buyer = BuyerAgent(name="Buyer", role="buyer", max_price=1e308, current_offer=0.0)
    seller = SellerAgent(
        name="Seller",
        role="seller",
        initial_price=1e308,
        min_acceptable=1e307,
    )

    migration = price_only_from_legacy("Extreme", buyer, seller, rounds=1)
    restored_buyer, restored_seller, _ = price_only_to_legacy(migration)

    assert restored_buyer.max_price == 1e308
    assert restored_seller.initial_price == 1e308
    assert restored_seller.min_acceptable == pytest.approx(1e307)
