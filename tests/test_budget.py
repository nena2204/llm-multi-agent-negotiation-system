"""Conditional budget concession: exceed the price limit by at most 30% only when the
offer satisfies the participant's required attributes."""

import pytest
from pydantic import ValidationError

from llm_negotiation.benchmark import run_policy_session
from llm_negotiation.domain import (
    AcceptAction,
    AttributeRequirement,
    BudgetPolicy,
    CategoricalIssue,
    CategoricalIssueValue,
    CategoricalPreference,
    CategoryUtility,
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
from llm_negotiation.memory import AgentProfile, PublicAgentIdentity
from llm_negotiation.policies import (
    LinearConcessionPolicy,
    generate_offer_for_utility,
)
from llm_negotiation.protocol import NegotiationProtocol, ProtocolPhase
from llm_negotiation.verification import (
    DeterministicActionVerifier,
    VerificationContext,
    VerificationReasonCode,
)


BUYER = ParticipantId("buyer")
SELLER = ParticipantId("seller")
PRICE = IssueId("price")
WARRANTY = IssueId("warranty")
DELIVERY_DAYS = IssueId("delivery_days")


def scenario():
    return NegotiationScenario(
        scenario_id="budget",
        title="Laptop",
        participants=(
            Participant(participant_id=BUYER, display_name="Buyer", role=ParticipantRole.BUYER),
            Participant(participant_id=SELLER, display_name="Seller", role=ParticipantRole.SELLER),
        ),
        issues=(
            NumericIssue(issue_id=PRICE, name="Price", minimum=0.0, maximum=1000.0),
            CategoricalIssue(issue_id=WARRANTY, name="Warranty", choices=("none", "1y", "3y")),
            NumericIssue(issue_id=DELIVERY_DAYS, name="Delivery days", minimum=1.0, maximum=30.0),
        ),
    )


def buyer_budget(**overrides):
    values = dict(
        price_issue_id=PRICE,
        limit_price=300.0,
        required_attributes=(
            AttributeRequirement(issue_id=WARRANTY, acceptable_values=("3y",)),
            AttributeRequirement(issue_id=DELIVERY_DAYS, maximum=7.0),
        ),
    )
    values.update(overrides)
    return BudgetPolicy(**values)


def preferences(participant_id, direction, budget=None, reservation=0.2):
    return ParticipantPreferences(
        participant_id=participant_id,
        issue_preferences=(
            NumericPreference(issue_id=PRICE, weight=0.5, direction=direction),
            CategoricalPreference(
                issue_id=WARRANTY,
                weight=0.3,
                category_utilities=(
                    CategoryUtility(category="none", utility=0.0 if direction.value == "minimize" else 1.0),
                    CategoryUtility(category="1y", utility=0.5),
                    CategoryUtility(category="3y", utility=1.0 if direction.value == "minimize" else 0.0),
                ),
            ),
            NumericPreference(
                issue_id=DELIVERY_DAYS,
                weight=0.2,
                direction=(
                    PreferenceDirection.MINIMIZE
                    if direction is PreferenceDirection.MINIMIZE
                    else PreferenceDirection.MAXIMIZE
                ),
            ),
        ),
        reservation=ReservationPolicy(reservation_utility=reservation, batna_utility=reservation),
        budget=budget,
    )


def profile(participant_id, prefs):
    return AgentProfile(
        identity=PublicAgentIdentity(participant=scenario().participant(participant_id), persona="test"),
        preferences=prefs,
    )


def offer(price, warranty="3y", days=5.0, offer_id="o"):
    return Offer(
        offer_id=OfferId(offer_id),
        values=(
            NumericIssueValue(issue_id=PRICE, value=price),
            CategoricalIssueValue(issue_id=WARRANTY, value=warranty),
            NumericIssueValue(issue_id=DELIVERY_DAYS, value=days),
        ),
    )


# --- domain rule -----------------------------------------------------------------------------


def test_limit_stretches_by_30_percent_only_when_all_attributes_are_met():
    buyer = preferences(BUYER, PreferenceDirection.MINIMIZE, buyer_budget())
    assert buyer.price_limit_for(offer(350, "3y", 5)) == pytest.approx(390.0)
    assert buyer.price_limit_for(offer(350, "1y", 5)) == 300.0
    assert buyer.price_limit_for(offer(350, "3y", 10)) == 300.0

    assert buyer.budget_violation(offer(280, "none", 20)) is None
    assert buyer.budget_violation(offer(350, "3y", 5)) is None
    assert buyer.budget_violation(offer(390, "3y", 7)) is None
    assert "exceeds" in buyer.budget_violation(offer(391, "3y", 5))
    assert "exceeds" in buyer.budget_violation(offer(350, "1y", 5))


def test_concession_fraction_is_configurable():
    buyer = preferences(
        BUYER, PreferenceDirection.MINIMIZE, buyer_budget(maximum_concession_fraction=0.1)
    )
    assert buyer.price_limit_for(offer(0, "3y", 5)) == pytest.approx(330.0)


def test_seller_budget_concedes_downwards():
    seller = preferences(
        SELLER,
        PreferenceDirection.MAXIMIZE,
        BudgetPolicy(
            price_issue_id=PRICE,
            limit_price=400.0,
            required_attributes=(AttributeRequirement(issue_id=WARRANTY, acceptable_values=("none",)),),
        ),
    )
    assert seller.price_limit_for(offer(0, "none")) == pytest.approx(280.0)
    assert seller.budget_violation(offer(300, "none")) is None
    assert seller.budget_violation(offer(300, "3y")) is not None


def test_invalid_budget_configurations_are_rejected():
    with pytest.raises(ValidationError):
        AttributeRequirement(issue_id=WARRANTY)
    with pytest.raises(ValidationError):
        AttributeRequirement(issue_id=WARRANTY, acceptable_values=("3y",), maximum=3.0)
    with pytest.raises(ValidationError):
        buyer_budget(maximum_concession_fraction=1.5)
    with pytest.raises(ValidationError):
        buyer_budget(required_attributes=())
    with pytest.raises(ValidationError):
        preferences(
            BUYER,
            PreferenceDirection.MINIMIZE,
            buyer_budget(
                required_attributes=(AttributeRequirement(issue_id=IssueId("color"), acceptable_values=("red",)),)
            ),
        )
    with pytest.raises(ValidationError):
        preferences(
            BUYER,
            PreferenceDirection.MINIMIZE,
            buyer_budget(required_attributes=(AttributeRequirement(issue_id=WARRANTY, minimum=1.0),)),
        )


def test_budget_round_trips_through_json():
    buyer = preferences(BUYER, PreferenceDirection.MINIMIZE, buyer_budget())
    assert ParticipantPreferences.model_validate_json(buyer.model_dump_json()) == buyer


# --- deterministic policies --------------------------------------------------------------------


def test_generated_offers_never_exceed_the_applicable_limit():
    buyer = preferences(BUYER, PreferenceDirection.MINIMIZE, buyer_budget(), reservation=0.0)
    for step in range(0, 101):
        generated = generate_offer_for_utility(scenario(), buyer, step / 100, OfferId(f"g{step}"))
        assert buyer.budget_violation(generated) is None


class _ScriptedSeller:
    """Seller that always puts the same offer on the table."""

    name = "scripted-seller"

    def __init__(self, price, warranty, days):
        self._values = (price, warranty, days)

    def choose_action(self, memory):
        recipients = (BUYER,)
        new_offer = offer(*self._values, offer_id=f"s-{memory.working.public_event_count}")
        outstanding = memory.working.outstanding_offer
        if outstanding is None:
            return ProposeAction(
                actor_id=SELLER, recipients=recipients,
                round_number=memory.working.round_number, offer=new_offer,
            )
        return CounterAction(
            actor_id=SELLER, recipients=recipients,
            round_number=memory.working.round_number,
            responds_to=outstanding.offer_id, offer=new_offer,
        )


def _run(price, warranty, days):
    buyer = preferences(BUYER, PreferenceDirection.MINIMIZE, buyer_budget(), reservation=0.2)
    seller = preferences(SELLER, PreferenceDirection.MAXIMIZE)
    return run_policy_session(
        scenario(),
        {BUYER: profile(BUYER, buyer), SELLER: profile(SELLER, seller)},
        {BUYER: LinearConcessionPolicy(), SELLER: _ScriptedSeller(price, warranty, days)},
        maximum_rounds=6,
    )


def _agreed_price(result):
    assert result.session.phase is ProtocolPhase.AGREED
    return result.session.outcome.agreement.offer.value_for(PRICE).value


def test_buyer_concedes_above_budget_when_attributes_are_met():
    assert _agreed_price(_run(350.0, "3y", 5.0)) == 350.0


@pytest.mark.parametrize(
    "price, warranty, days",
    [
        (350.0, "1y", 5.0),   # warranty requirement not met -> hard limit 300
        (350.0, "3y", 14.0),  # delivery requirement not met -> hard limit 300
        (400.0, "3y", 5.0),   # attributes met but above 300 * 1.3 = 390
    ],
)
def test_buyer_refuses_when_attributes_missing_or_above_30_percent(price, warranty, days):
    result = _run(price, warranty, days)
    assert result.session.phase is not ProtocolPhase.AGREED
    buyer = preferences(BUYER, PreferenceDirection.MINIMIZE, buyer_budget())
    for item in result.offer_trajectory:
        if not str(item.offer_id).startswith("s-"):
            assert buyer.budget_violation(item) is None


def test_buyer_accepts_within_plain_budget_without_attributes():
    assert _agreed_price(_run(290.0, "none", 20.0)) == 290.0


# --- verification ------------------------------------------------------------------------------


def test_verifier_flags_over_budget_acceptance_and_offers():
    public = scenario()
    protocol = NegotiationProtocol(public, maximum_rounds=4, initial_turn=SELLER)
    state = protocol.create_session()
    over_budget = offer(350.0, "1y", 5.0, offer_id="seller-1")
    state = protocol.transition(
        state,
        ProposeAction(actor_id=SELLER, recipients=(BUYER,), round_number=1, offer=over_budget),
    ).state
    buyer = profile(BUYER, preferences(BUYER, PreferenceDirection.MINIMIZE, buyer_budget(), reservation=0.0))
    context = VerificationContext(protocol, state, buyer)
    verifier = DeterministicActionVerifier()

    accept = verifier.verify(
        context,
        AcceptAction(actor_id=BUYER, recipients=(SELLER,), round_number=state.round_number, offer_id=OfferId("seller-1")),
        "accept",
    )
    assert VerificationReasonCode.BUDGET_LIMIT_VIOLATION in {r.code for r in accept.reasons}

    counter = verifier.verify(
        context,
        CounterAction(
            actor_id=BUYER, recipients=(SELLER,), round_number=state.round_number,
            responds_to=OfferId("seller-1"), offer=offer(395.0, "3y", 5.0, offer_id="buyer-1"),
        ),
        "counter",
    )
    assert VerificationReasonCode.BUDGET_LIMIT_VIOLATION in {r.code for r in counter.reasons}

    fine = verifier.verify(
        context,
        CounterAction(
            actor_id=BUYER, recipients=(SELLER,), round_number=state.round_number,
            responds_to=OfferId("seller-1"), offer=offer(380.0, "3y", 5.0, offer_id="buyer-2"),
        ),
        "counter-ok",
    )
    assert VerificationReasonCode.BUDGET_LIMIT_VIOLATION not in {r.code for r in fine.reasons}
