"""Small deterministic comparison of team size and composition."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Tuple

from pydantic import Field, field_validator

from .communication import MessageBus
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
    ProposeAction,
    ReservationPolicy,
)
from .domain.models import DomainModel
from .memory import AgentMemory, AgentProfile, PublicAgentIdentity
from .multiparty import (
    AcceptanceRule,
    DeliberativePolicy,
    MultipartyProtocolConfiguration,
    MultipartyProtocolFactory,
)
from .orchestration import (
    AUDIT_READER_ID,
    InMemoryMemoryStore,
    NegotiationOrchestrator,
    OrchestratorConfiguration,
)
from .policies import (
    BoulwarePolicy,
    ConcederPolicy,
    FixedPolicy,
    LinearConcessionPolicy,
    NegotiationPolicy,
)
from .verification import VerificationCoordinator


class TeamComposition(str, Enum):
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


class TeamExperimentRun(DomainModel):
    team_size: int = Field(ge=2, le=4)
    composition: TeamComposition
    protocol: str = Field(min_length=1, max_length=100)
    agreement: bool
    social_welfare: float = Field(ge=0.0)
    mean_outcome_utility: float = Field(ge=0.0, le=1.0)
    unique_initial_proposals: int = Field(ge=1)
    initial_proposal_diversity: float = Field(ge=0.0, le=1.0)
    rounds: int = Field(ge=1)
    reason: str


class TeamComparisonResult(DomainModel):
    runs: Tuple[TeamExperimentRun, ...] = Field(min_length=6, max_length=6)

    @field_validator("runs", mode="before")
    @classmethod
    def runs_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class _ExperimentClock:
    def now(self) -> datetime:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return 0.0


def team_scenario_and_profiles(
    size: int, composition: TeamComposition
) -> tuple[NegotiationScenario, Mapping[ParticipantId, AgentProfile]]:
    participants = tuple(
        Participant(
            participant_id=ParticipantId(f"member-{index + 1}"),
            display_name=f"Member {index + 1}",
            role=ParticipantRole.NEGOTIATOR,
        )
        for index in range(size)
    )
    allocation = NumericIssue(
        issue_id="allocation", name="Allocation to project A", minimum=0.0, maximum=100.0
    )
    timing = CategoricalIssue(
        issue_id="timing", name="Delivery timing", choices=("early", "balanced", "late")
    )
    scenario = NegotiationScenario(
        scenario_id=f"team-{size}-{composition.value}",
        title=f"{size}-member {composition.value} comparison",
        participants=participants,
        issues=(allocation, timing),
    )
    profiles = {}
    for index, participant in enumerate(participants):
        if composition is TeamComposition.HOMOGENEOUS:
            weight, direction, category_values, reservation = (
                0.5,
                PreferenceDirection.MAXIMIZE,
                (0.4, 1.0, 0.6),
                0.25,
            )
            persona = "Shared analytical consensus profile"
        else:
            weight = (0.7, 0.4, 0.6, 0.3)[index]
            direction = (
                PreferenceDirection.MAXIMIZE
                if index % 2 == 0
                else PreferenceDirection.MINIMIZE
            )
            category_values = (
                ((1.0, 0.6, 0.2), (0.2, 0.7, 1.0), (0.5, 1.0, 0.3), (0.8, 0.4, 1.0))[index]
            )
            reservation = (0.30, 0.35, 0.25, 0.40)[index]
            persona = ("Analyst", "Risk steward", "Innovator", "Operations lead")[index]
        preferences = ParticipantPreferences(
            participant_id=participant.participant_id,
            issue_preferences=(
                NumericPreference(
                    issue_id=allocation.issue_id,
                    weight=weight,
                    direction=direction,
                ),
                CategoricalPreference(
                    issue_id=timing.issue_id,
                    weight=1.0 - weight,
                    category_utilities=tuple(
                        CategoryUtility(category=choice, utility=value)
                        for choice, value in zip(timing.choices, category_values)
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
    return scenario, profiles


def _policies(
    participants: Tuple[Participant, ...], composition: TeamComposition, multiparty: bool
) -> Mapping[ParticipantId, NegotiationPolicy]:
    strategies = (
        (LinearConcessionPolicy(),) * len(participants)
        if composition is TeamComposition.HOMOGENEOUS
        else (FixedPolicy(), BoulwarePolicy(), LinearConcessionPolicy(), ConcederPolicy())[: len(participants)]
    )
    if multiparty:
        return {
            participant.participant_id: DeliberativePolicy(
                strategy,
                critique=f"Public critique from speaking position {index + 1}.",
                name=f"deliberative-{strategy.name}",
            )
            for index, (participant, strategy) in enumerate(zip(participants, strategies))
        }
    return {
        participant.participant_id: strategy
        for participant, strategy in zip(participants, strategies)
    }


def _run(size: int, composition: TeamComposition) -> TeamExperimentRun:
    scenario, profiles = team_scenario_and_profiles(size, composition)
    participant_ids = tuple(item.participant_id for item in scenario.participants)
    policies = _policies(scenario.participants, composition, size >= 3)
    memories = {
        participant_id: AgentMemory(profiles[participant_id], scenario)
        for participant_id in participant_ids
    }
    factory = (
        MultipartyProtocolFactory(
            MultipartyProtocolConfiguration(
                speaking_order=participant_ids,
                eligible_proposers=participant_ids,
                eligible_voters=participant_ids,
                acceptance_rule=AcceptanceRule.UNANIMITY,
            )
        )
        if size >= 3
        else None
    )
    result = NegotiationOrchestrator(
        scenario=scenario,
        profiles=profiles,
        policies=policies,
        message_bus=MessageBus(
            scenario.scenario_id,
            participants=participant_ids,
            mediator_ids=(),
            audit_reader_ids=(AUDIT_READER_ID,),
        ),
        verifier=VerificationCoordinator(),
        mediator=None,
        model_gateway=None,
        memory_store=InMemoryMemoryStore(memories),
        clock=_ExperimentClock(),
        random_seed=0,
        configuration=OrchestratorConfiguration(maximum_rounds=6),
        protocol_factory=factory,
    ).run()
    initial_offers = tuple(
        event.action.offer
        for event in result.audit_events
        if getattr(event, "action", None) is not None
        and isinstance(event.action, ProposeAction)
    )
    unique_initial = len(
        {
            tuple(value.model_dump_json() for value in proposal.values)
            for proposal in initial_offers
        }
    )
    expected_initial_count = size if size >= 3 else len(initial_offers)
    return TeamExperimentRun(
        team_size=size,
        composition=composition,
        protocol=("multiparty_staged" if size >= 3 else "bilateral_alternating_offer"),
        agreement=result.metrics.agreement_reached,
        social_welfare=result.metrics.social_welfare,
        mean_outcome_utility=result.metrics.social_welfare / size,
        unique_initial_proposals=unique_initial,
        initial_proposal_diversity=unique_initial / expected_initial_count,
        rounds=result.final_session.round_number,
        reason=(result.outcome.reason or "agreement") if result.outcome is not None else "paused",
    )


def run_team_comparison() -> TeamComparisonResult:
    """Run all cells; results are observations, never a claim that larger teams are better."""

    return TeamComparisonResult(
        runs=tuple(
            _run(size, composition)
            for size in (2, 3, 4)
            for composition in (TeamComposition.HOMOGENEOUS, TeamComposition.HETEROGENEOUS)
        )
    )
