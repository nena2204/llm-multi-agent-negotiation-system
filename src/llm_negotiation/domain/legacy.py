from dataclasses import dataclass
import math
from typing import Tuple

from llm_negotiation.agents import BuyerAgent, SellerAgent

from .identifiers import IssueId, OfferId, ParticipantId
from .models import (
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    Participant,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ReservationPolicy,
    calculate_utility,
)


PRICE_ISSUE_ID = IssueId("price")
BUYER_ID = ParticipantId("buyer")
SELLER_ID = ParticipantId("seller")


@dataclass(frozen=True)
class LegacyAgentMetadata:
    buyer_name: str
    buyer_strategy: str
    seller_name: str
    seller_strategy: str


@dataclass(frozen=True)
class PriceOnlyMigration:
    """Migration boundary holding public scenario data and separately named private profiles."""

    scenario: NegotiationScenario
    buyer_preferences: ParticipantPreferences
    seller_preferences: ParticipantPreferences
    buyer_initial_offer: Offer
    seller_initial_offer: Offer
    metadata: LegacyAgentMetadata
    rounds: int

    def preferences_for(self, participant_id: ParticipantId) -> ParticipantPreferences:
        if participant_id == BUYER_ID:
            return self.buyer_preferences
        if participant_id == SELLER_ID:
            return self.seller_preferences
        raise KeyError(f"no private preferences for participant '{participant_id}'")


def _price_offer(offer_id: str, price: float) -> Offer:
    return Offer(
        offer_id=OfferId(offer_id),
        values=(NumericIssueValue(issue_id=PRICE_ISSUE_ID, value=price),),
    )


def _public_price_upper_bound(*prices: float) -> float:
    """Choose a coarse magnitude boundary without publishing an exact private reservation price."""

    reference = max(1.0, *prices)
    exponent = math.floor(math.log10(reference)) + 1
    try:
        upper_bound = 10.0**exponent
    except OverflowError:
        return reference
    return upper_bound if math.isfinite(upper_bound) else reference


def price_only_from_legacy(
    product_name: str,
    buyer: BuyerAgent,
    seller: SellerAgent,
    rounds: int,
) -> PriceOnlyMigration:
    """Convert legacy buyer/seller configuration into the typed price-only domain."""

    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")

    upper_bound = _public_price_upper_bound(
        buyer.initial_offer,
        buyer.max_price,
        seller.initial_price,
        seller.min_acceptable,
    )
    price_issue = NumericIssue(
        issue_id=PRICE_ISSUE_ID,
        name="Price",
        minimum=0.0,
        maximum=upper_bound,
        unit="currency",
    )
    scenario = NegotiationScenario(
        scenario_id="price-only",
        title=product_name,
        participants=(
            Participant(participant_id=BUYER_ID, display_name=buyer.name, role=ParticipantRole.BUYER),
            Participant(participant_id=SELLER_ID, display_name=seller.name, role=ParticipantRole.SELLER),
        ),
        issues=(price_issue,),
    )
    buyer_reservation = 1.0 - (buyer.max_price / upper_bound)
    seller_reservation = seller.min_acceptable / upper_bound
    buyer_preferences = ParticipantPreferences(
        participant_id=BUYER_ID,
        issue_preferences=(
            NumericPreference(
                issue_id=PRICE_ISSUE_ID,
                weight=1.0,
                direction=PreferenceDirection.MINIMIZE,
            ),
        ),
        reservation=ReservationPolicy(
            reservation_utility=buyer_reservation,
            batna_utility=buyer_reservation,
            batna_description="Legacy buyer maximum price",
        ),
    )
    seller_preferences = ParticipantPreferences(
        participant_id=SELLER_ID,
        issue_preferences=(
            NumericPreference(
                issue_id=PRICE_ISSUE_ID,
                weight=1.0,
                direction=PreferenceDirection.MAXIMIZE,
            ),
        ),
        reservation=ReservationPolicy(
            reservation_utility=seller_reservation,
            batna_utility=seller_reservation,
            batna_description="Legacy seller minimum acceptable price",
        ),
    )
    buyer_offer = _price_offer("buyer-initial", buyer.initial_offer)
    seller_offer = _price_offer("seller-initial", seller.initial_price)
    scenario.validate_offer(buyer_offer)
    scenario.validate_offer(seller_offer)
    return PriceOnlyMigration(
        scenario=scenario,
        buyer_preferences=buyer_preferences,
        seller_preferences=seller_preferences,
        buyer_initial_offer=buyer_offer,
        seller_initial_offer=seller_offer,
        metadata=LegacyAgentMetadata(
            buyer_name=buyer.name,
            buyer_strategy=buyer.strategy,
            seller_name=seller.name,
            seller_strategy=seller.strategy,
        ),
        rounds=rounds,
    )


def _single_price(offer: Offer) -> float:
    value = offer.value_for(PRICE_ISSUE_ID)
    if not isinstance(value, NumericIssueValue):
        raise TypeError("price-only offer must contain a numeric price")
    return value.value


def _reservation_price(
    migration: PriceOnlyMigration,
    preferences: ParticipantPreferences,
    direction: PreferenceDirection,
) -> float:
    issue = migration.scenario.issue(PRICE_ISSUE_ID)
    if not isinstance(issue, NumericIssue):
        raise TypeError("price-only scenario must define a numeric price issue")
    utility = preferences.reservation.reservation_utility
    position = utility if direction is PreferenceDirection.MAXIMIZE else 1.0 - utility
    return round(issue.minimum + position * (issue.maximum - issue.minimum), 12)


def price_only_to_legacy(
    migration: PriceOnlyMigration,
) -> Tuple[BuyerAgent, SellerAgent, int]:
    """Reconstruct legacy agents from the typed price-only domain configuration."""

    migration.scenario.validate_offer(migration.buyer_initial_offer)
    migration.scenario.validate_offer(migration.seller_initial_offer)
    buyer_max = _reservation_price(migration, migration.buyer_preferences, PreferenceDirection.MINIMIZE)
    seller_min = _reservation_price(migration, migration.seller_preferences, PreferenceDirection.MAXIMIZE)
    buyer = BuyerAgent(
        name=migration.metadata.buyer_name,
        role=ParticipantRole.BUYER.value,
        strategy=migration.metadata.buyer_strategy,
        max_price=buyer_max,
        current_offer=_single_price(migration.buyer_initial_offer),
    )
    seller = SellerAgent(
        name=migration.metadata.seller_name,
        role=ParticipantRole.SELLER.value,
        strategy=migration.metadata.seller_strategy,
        initial_price=_single_price(migration.seller_initial_offer),
        min_acceptable=seller_min,
    )
    return buyer, seller, migration.rounds


def reservation_utilities(migration: PriceOnlyMigration) -> Tuple[float, float]:
    """Return utilities at the reconstructed legacy reservation prices for verification."""

    buyer_max = _reservation_price(migration, migration.buyer_preferences, PreferenceDirection.MINIMIZE)
    seller_min = _reservation_price(migration, migration.seller_preferences, PreferenceDirection.MAXIMIZE)
    return (
        calculate_utility(migration.scenario, _price_offer("buyer-reservation", buyer_max), migration.buyer_preferences),
        calculate_utility(
            migration.scenario,
            _price_offer("seller-reservation", seller_min),
            migration.seller_preferences,
        ),
    )
