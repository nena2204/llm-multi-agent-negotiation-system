"""Versioned, offline scenario dataset for reproducible negotiation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from .agents import BuyerAgent, SellerAgent
from .domain import (
    CategoryUtility,
    CategoricalIssue,
    CategoricalPreference,
    NegotiationScenario,
    NumericIssue,
    NumericPreference,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ReservationPolicy,
    price_only_from_legacy,
)
from .memory import AgentProfile, PublicAgentIdentity
from .multiparty import three_principal_scenario
from .protocol import FeasibleRegionStatus


SCENARIO_DATASET_VERSION = "1.0"


@dataclass(frozen=True)
class ExperimentScenario:
    """One public scenario plus simulation-only private profiles and labels."""

    case_id: str
    tags: Tuple[str, ...]
    scenario: NegotiationScenario
    profiles: Mapping[ParticipantId, AgentProfile]
    feasible_region: FeasibleRegionStatus
    default_rounds: int

    def __post_init__(self) -> None:
        participants = {item.participant_id for item in self.scenario.participants}
        if set(self.profiles) != participants:
            raise ValueError("experiment profiles must cover scenario participants exactly")
        if not self.case_id or len(set(self.tags)) != len(self.tags):
            raise ValueError("experiment case id must be nonblank and tags unique")
        if self.default_rounds <= 0:
            raise ValueError("experiment default rounds must be positive")


def _price_case(
    case_id: str,
    *,
    buyer_maximum: float,
    seller_minimum: float,
    buyer_initial: float,
    seller_initial: float,
    tags: Tuple[str, ...],
) -> ExperimentScenario:
    migration = price_only_from_legacy(
        case_id,
        BuyerAgent(
            name="Buyer",
            role="buyer",
            strategy="neutral",
            max_price=buyer_maximum,
            current_offer=buyer_initial,
        ),
        SellerAgent(
            name="Seller",
            role="seller",
            strategy="neutral",
            initial_price=seller_initial,
            min_acceptable=seller_minimum,
        ),
        rounds=6,
    )
    scenario = migration.scenario.model_copy(
        update={"scenario_id": case_id, "title": case_id.replace("-", " ").title()}
    )
    profiles = {
        participant.participant_id: AgentProfile(
            identity=PublicAgentIdentity(participant=participant, persona=participant.display_name),
            preferences=migration.preferences_for(participant.participant_id),
        )
        for participant in scenario.participants
    }
    feasible = (
        FeasibleRegionStatus.FEASIBLE
        if seller_minimum <= buyer_maximum
        else FeasibleRegionStatus.IMPOSSIBLE
    )
    return ExperimentScenario(
        case_id=case_id,
        tags=tags,
        scenario=scenario,
        profiles=profiles,
        feasible_region=feasible,
        default_rounds=6,
    )


def _multi_issue_case() -> ExperimentScenario:
    participants = (
        Participant(
            participant_id=ParticipantId("client"),
            display_name="Client",
            role=ParticipantRole.NEGOTIATOR,
        ),
        Participant(
            participant_id=ParticipantId("supplier"),
            display_name="Supplier",
            role=ParticipantRole.NEGOTIATOR,
        ),
    )
    price = NumericIssue(
        issue_id="price", name="Contract price", minimum=0.0, maximum=100.0
    )
    delivery = CategoricalIssue(
        issue_id="delivery",
        name="Delivery window",
        choices=("fast", "standard", "flexible"),
    )
    scenario = NegotiationScenario(
        scenario_id="multi-issue-asymmetric",
        title="Asymmetric price and delivery negotiation",
        participants=participants,
        issues=(price, delivery),
    )
    specifications = (
        (
            participants[0],
            "Cost-sensitive client",
            0.70,
            PreferenceDirection.MINIMIZE,
            (1.0, 0.7, 0.2),
            0.35,
        ),
        (
            participants[1],
            "Schedule-sensitive supplier",
            0.40,
            PreferenceDirection.MAXIMIZE,
            (0.1, 0.8, 1.0),
            0.30,
        ),
    )
    profiles = {}
    for participant, persona, weight, direction, categories, reservation in specifications:
        preferences = ParticipantPreferences(
            participant_id=participant.participant_id,
            issue_preferences=(
                NumericPreference(
                    issue_id=price.issue_id, weight=weight, direction=direction
                ),
                CategoricalPreference(
                    issue_id=delivery.issue_id,
                    weight=1.0 - weight,
                    category_utilities=tuple(
                        CategoryUtility(category=category, utility=utility)
                        for category, utility in zip(delivery.choices, categories)
                    ),
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=reservation,
                batna_utility=reservation,
            ),
        )
        profiles[participant.participant_id] = AgentProfile(
            identity=PublicAgentIdentity(participant=participant, persona=persona),
            preferences=preferences,
        )
    return ExperimentScenario(
        case_id=scenario.scenario_id,
        tags=("feasible", "asymmetric", "multi_issue", "bilateral"),
        scenario=scenario,
        profiles=profiles,
        feasible_region=FeasibleRegionStatus.FEASIBLE,
        default_rounds=7,
    )


def _multiparty_case() -> ExperimentScenario:
    scenario, profiles = three_principal_scenario()
    return ExperimentScenario(
        case_id=scenario.scenario_id,
        tags=("feasible", "asymmetric", "multi_issue", "multiparty"),
        scenario=scenario,
        profiles=profiles,
        feasible_region=FeasibleRegionStatus.FEASIBLE,
        default_rounds=8,
    )


def team_size_case(size: int, *, heterogeneous: bool) -> ExperimentScenario:
    """Build the existing deterministic 2/3/4-agent comparison cell as a dataset case."""

    if size not in (2, 3, 4):
        raise ValueError("team-size ablations support exactly 2, 3, or 4 participants")
    # Local import avoids making the core dataset and comparison helper depend cyclically.
    from .team_experiment import TeamComposition, team_scenario_and_profiles

    composition = (
        TeamComposition.HETEROGENEOUS if heterogeneous else TeamComposition.HOMOGENEOUS
    )
    scenario, profiles = team_scenario_and_profiles(size, composition)
    return ExperimentScenario(
        case_id=scenario.scenario_id,
        tags=(
            "feasible",
            "multi_issue",
            "multiparty" if size >= 3 else "bilateral",
            composition.value,
            f"team_size_{size}",
        ),
        scenario=scenario,
        profiles=profiles,
        feasible_region=FeasibleRegionStatus.FEASIBLE,
        default_rounds=8,
    )


def scenario_dataset() -> Tuple[ExperimentScenario, ...]:
    """Return the immutable built-in dataset in stable case-id order."""

    return (
        _price_case(
            "price-feasible-symmetric",
            buyer_maximum=60.0,
            seller_minimum=40.0,
            buyer_initial=20.0,
            seller_initial=80.0,
            tags=("feasible", "symmetric", "price_only", "bilateral"),
        ),
        _price_case(
            "price-infeasible",
            buyer_maximum=40.0,
            seller_minimum=60.0,
            buyer_initial=20.0,
            seller_initial=80.0,
            tags=("infeasible", "symmetric", "price_only", "bilateral"),
        ),
        _price_case(
            "price-feasible-asymmetric",
            buyer_maximum=90.0,
            seller_minimum=30.0,
            buyer_initial=10.0,
            seller_initial=95.0,
            tags=("feasible", "asymmetric", "price_only", "bilateral"),
        ),
        _multi_issue_case(),
        _multiparty_case(),
    )


def scenario_index() -> Mapping[str, ExperimentScenario]:
    cases = scenario_dataset()
    return {case.case_id: case for case in cases}
