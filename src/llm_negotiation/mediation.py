from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from .communication import MessageBus, MessageEnvelope, MessageType, MessageVisibility
from .domain import (
    CategoricalIssue,
    CategoricalIssueValue,
    CorrelationId,
    MediationAccessMode,
    MediationTrigger,
    MediatorIntervention,
    MediatorInterventionId,
    MediatorInterventionKind,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    Offer,
    OfferId,
    ParticipantId,
    ParticipantPreferences,
    ReservationPolicy,
    calculate_utility,
)
from .domain.models import DomainModel, IssuePreference
from .llm import LLMClient, LLMRequest, LLMUsage, ModelConfiguration
from .llm_prompts.mediator_v1 import PROMPT_VERSION, mediator_messages
from .protocol import (
    ActionAppliedEvent,
    MediatorInterventionEvent,
    NegotiationProtocol,
    NegotiationSession,
    ProtocolPhase,
    TransitionResult,
)
from .domain import CounterAction, ProposeAction, RequestMediationAction


class MediationObjective(str, Enum):
    NASH_PRODUCT = "nash_product"
    MAX_MIN = "max_min"


class DeadlockSignalType(str, Enum):
    REPEATED_OFFERS = "repeated_offers"
    LOW_CONCESSION = "low_concession"
    CYCLING = "cycling"
    REPEATED_INVALID_ACTIONS = "repeated_invalid_actions"


class DeadlockConfiguration(DomainModel):
    enabled: bool = True
    repeated_offer_count: int = Field(default=3, ge=2, le=20)
    low_concession_steps: int = Field(default=3, ge=2, le=20)
    low_concession_epsilon: float = Field(default=0.01, ge=0.0, le=1.0)
    cycle_length: int = Field(default=4, ge=4, le=20)
    invalid_action_threshold: int = Field(default=3, ge=1, le=100)
    deadline_rounds_remaining: Optional[int] = Field(default=1, ge=0)


class DeadlockSignal(DomainModel):
    signal_type: DeadlockSignalType
    evidence_event_ids: Tuple[str, ...] = ()
    detail: str = Field(min_length=1, max_length=300)

    @field_validator("evidence_event_ids", mode="before")
    @classmethod
    def evidence_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class MediationTriggerDecision(DomainModel):
    should_enter: bool
    trigger: Optional[MediationTrigger] = None
    signals: Tuple[DeadlockSignal, ...] = ()
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("signals", mode="before")
    @classmethod
    def signals_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def trigger_matches_decision(self) -> MediationTriggerDecision:
        if self.should_enter != (self.trigger is not None):
            raise ValueError("mediation trigger is required exactly when entry is requested")
        return self


class ConfidentialPreferenceSummary(DomainModel):
    """Mediator-only utility summary; deliberately excludes BATNA prose and persona data."""

    participant_id: ParticipantId
    issue_preferences: Tuple[IssuePreference, ...] = Field(min_length=1)
    reservation_utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("issue_preferences", mode="before")
    @classmethod
    def preferences_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    def as_preferences(self) -> ParticipantPreferences:
        return ParticipantPreferences(
            participant_id=self.participant_id,
            issue_preferences=self.issue_preferences,
            reservation=ReservationPolicy(
                reservation_utility=self.reservation_utility,
                batna_utility=self.reservation_utility,
            ),
        )


class MediatorCapabilities(DomainModel):
    access_modes: Tuple[MediationAccessMode, ...] = (
        MediationAccessMode.PUBLIC_ONLY,
        MediationAccessMode.CONFIDENTIAL_SUMMARY,
        MediationAccessMode.SIMULATION_ORACLE,
    )
    can_ask_clarifying_questions: bool = True
    can_propose_non_binding_offers: bool = True
    can_accept_for_participants: bool = False


class MediatorConfiguration(DomainModel):
    mediator_id: ParticipantId = ParticipantId("mediator")
    access_mode: MediationAccessMode = MediationAccessMode.PUBLIC_ONLY
    objective: MediationObjective = MediationObjective.NASH_PRODUCT
    deadlock: DeadlockConfiguration = Field(default_factory=DeadlockConfiguration)
    numeric_grid_points: int = Field(default=11, ge=3, le=101)
    maximum_candidates: int = Field(default=20_000, ge=1, le=1_000_000)
    allow_simulation_oracle: bool = False

    @model_validator(mode="after")
    def oracle_is_explicit(self) -> MediatorConfiguration:
        if (
            self.access_mode is MediationAccessMode.SIMULATION_ORACLE
            and not self.allow_simulation_oracle
        ):
            raise ValueError(
                "simulation_oracle is research-only; set allow_simulation_oracle=True explicitly"
            )
        return self

    @property
    def production_safe(self) -> bool:
        return self.access_mode is not MediationAccessMode.SIMULATION_ORACLE


class MediatorContext(DomainModel):
    scenario: NegotiationScenario
    session: NegotiationSession
    access_mode: MediationAccessMode = MediationAccessMode.PUBLIC_ONLY
    public_messages: Tuple[MessageEnvelope, ...] = ()
    confidential_summaries: Tuple[ConfidentialPreferenceSummary, ...] = ()
    simulation_ground_truth: Tuple[ParticipantPreferences, ...] = ()

    @field_validator(
        "public_messages", "confidential_summaries", "simulation_ground_truth", mode="before"
    )
    @classmethod
    def tuples_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def enforce_access_boundary(self) -> MediatorContext:
        if self.session.scenario_id != self.scenario.scenario_id:
            raise ValueError("mediator context session and scenario must match")
        if self.session.participants != tuple(
            participant.participant_id for participant in self.scenario.participants
        ):
            raise ValueError("mediator context participants must match the public scenario")
        if any(message.visibility is not MessageVisibility.PUBLIC for message in self.public_messages):
            raise ValueError("public mediator context may contain only public messages")
        participants = {item.participant_id for item in self.scenario.participants}
        if self.access_mode is MediationAccessMode.PUBLIC_ONLY:
            if self.confidential_summaries or self.simulation_ground_truth:
                raise ValueError("public-only mediation cannot receive private preference data")
        elif self.access_mode is MediationAccessMode.CONFIDENTIAL_SUMMARY:
            if self.simulation_ground_truth:
                raise ValueError("confidential-summary mode cannot receive simulation ground truth")
            if (
                len(self.confidential_summaries) != len(participants)
                or {item.participant_id for item in self.confidential_summaries} != participants
            ):
                raise ValueError("confidential summaries must cover every participant exactly once")
        else:
            if self.confidential_summaries:
                raise ValueError("simulation-oracle mode uses ground truth, not confidential summaries")
            if (
                len(self.simulation_ground_truth) != len(participants)
                or {item.participant_id for item in self.simulation_ground_truth} != participants
            ):
                raise ValueError("simulation ground truth must cover every participant exactly once")
        private_preferences = _preferences_for_context(self)
        if private_preferences:
            probe = Offer(
                offer_id=OfferId("mediation-context-probe"),
                values=tuple(
                    NumericIssueValue(issue_id=issue.issue_id, value=issue.minimum)
                    if isinstance(issue, NumericIssue)
                    else CategoricalIssueValue(issue_id=issue.issue_id, value=issue.choices[0])
                    for issue in self.scenario.issues
                ),
            )
            for preference in private_preferences.values():
                calculate_utility(self.scenario, probe, preference)
        return self


def _offer_signature(offer: Offer) -> Tuple[Tuple[str, object], ...]:
    return tuple(
        sorted((str(item.issue_id), item.value) for item in offer.values)
    )


def _offer_distance(scenario: NegotiationScenario, left: Offer, right: Offer) -> float:
    distances = []
    for issue in scenario.issues:
        left_value = left.value_for(issue.issue_id)
        right_value = right.value_for(issue.issue_id)
        if isinstance(issue, NumericIssue):
            distances.append(
                abs(left_value.value - right_value.value) / (issue.maximum - issue.minimum)
            )
        else:
            distances.append(0.0 if left_value.value == right_value.value else 1.0)
    return math.fsum(distances) / len(distances)


class DeadlockDetector:
    def __init__(self, scenario: NegotiationScenario, configuration: DeadlockConfiguration) -> None:
        self.scenario = scenario
        self.configuration = configuration

    def assess(
        self,
        state: NegotiationSession,
        *,
        explicit_requested: bool = False,
        invalid_action_actor_ids: Tuple[ParticipantId, ...] = (),
    ) -> MediationTriggerDecision:
        if state.phase is ProtocolPhase.MEDIATION or explicit_requested:
            requested = explicit_requested or (
                bool(state.event_sequence)
                and isinstance(state.event_sequence[-1], ActionAppliedEvent)
                and isinstance(state.event_sequence[-1].action, RequestMediationAction)
            )
            if requested:
                return MediationTriggerDecision(
                    should_enter=True,
                    trigger=MediationTrigger.EXPLICIT_REQUEST,
                    reason="A participant explicitly requested mediation.",
                )
        if state.phase is not ProtocolPhase.ACTIVE:
            return MediationTriggerDecision(
                should_enter=False,
                reason="Mediation triggers are evaluated only during active negotiation.",
            )

        offers = [
            event.action.offer
            for event in state.event_sequence
            if isinstance(event, ActionAppliedEvent)
            and isinstance(event.action, (ProposeAction, CounterAction))
        ]
        offer_events = [
            event
            for event in state.event_sequence
            if isinstance(event, ActionAppliedEvent)
            and isinstance(event.action, (ProposeAction, CounterAction))
        ]
        signals: list[DeadlockSignal] = []
        config = self.configuration
        if config.enabled and len(offers) >= config.repeated_offer_count:
            recent = offers[-config.repeated_offer_count :]
            if len({_offer_signature(item) for item in recent}) == 1:
                signals.append(
                    DeadlockSignal(
                        signal_type=DeadlockSignalType.REPEATED_OFFERS,
                        evidence_event_ids=tuple(str(item.action_id) for item in offer_events[-len(recent) :]),
                        detail="The same complete offer was repeated without movement.",
                    )
                )
        if config.enabled and len(offers) >= config.low_concession_steps + 1:
            recent = offers[-(config.low_concession_steps + 1) :]
            distances = tuple(
                _offer_distance(self.scenario, left, right)
                for left, right in zip(recent, recent[1:])
            )
            if all(distance <= config.low_concession_epsilon for distance in distances):
                signals.append(
                    DeadlockSignal(
                        signal_type=DeadlockSignalType.LOW_CONCESSION,
                        evidence_event_ids=tuple(
                            str(item.action_id)
                            for item in offer_events[-(config.low_concession_steps + 1) :]
                        ),
                        detail="Recent normalized concessions stayed below the configured threshold.",
                    )
                )
        if config.enabled and len(offers) >= config.cycle_length:
            recent = tuple(_offer_signature(item) for item in offers[-config.cycle_length :])
            half = config.cycle_length // 2
            if config.cycle_length % 2 == 0 and recent[:half] == recent[half:] and len(set(recent)) > 1:
                signals.append(
                    DeadlockSignal(
                        signal_type=DeadlockSignalType.CYCLING,
                        evidence_event_ids=tuple(
                            str(item.action_id) for item in offer_events[-config.cycle_length :]
                        ),
                        detail="The public offer trajectory repeated a prior cycle.",
                    )
                )
        if config.enabled and invalid_action_actor_ids:
            counts = {
                actor_id: invalid_action_actor_ids.count(actor_id)
                for actor_id in set(invalid_action_actor_ids)
            }
            if max(counts.values()) >= config.invalid_action_threshold:
                signals.append(
                    DeadlockSignal(
                        signal_type=DeadlockSignalType.REPEATED_INVALID_ACTIONS,
                        detail="A participant repeatedly produced actions rejected by verification.",
                    )
                )
        if signals:
            return MediationTriggerDecision(
                should_enter=True,
                trigger=MediationTrigger.DEADLOCK,
                signals=tuple(signals),
                reason="Configured public deadlock signals were detected.",
            )
        remaining = state.maximum_rounds - state.round_number
        if (
            config.deadline_rounds_remaining is not None
            and remaining <= config.deadline_rounds_remaining
            and state.phase is ProtocolPhase.ACTIVE
        ):
            return MediationTriggerDecision(
                should_enter=True,
                trigger=MediationTrigger.DEADLINE,
                reason="The configured negotiation deadline threshold was reached.",
            )
        return MediationTriggerDecision(
            should_enter=False,
            reason="No mediation trigger is currently active.",
        )


def _preferences_for_context(
    context: MediatorContext,
) -> Mapping[ParticipantId, ParticipantPreferences]:
    if context.access_mode is MediationAccessMode.CONFIDENTIAL_SUMMARY:
        return {item.participant_id: item.as_preferences() for item in context.confidential_summaries}
    if context.access_mode is MediationAccessMode.SIMULATION_ORACLE:
        return {item.participant_id: item for item in context.simulation_ground_truth}
    return {}


class DeterministicMediator:
    capabilities = MediatorCapabilities()

    def __init__(self, configuration: Optional[MediatorConfiguration] = None) -> None:
        self.configuration = configuration or MediatorConfiguration()

    def intervene(
        self, context: MediatorContext, trigger: MediationTrigger
    ) -> MediatorIntervention:
        self._validate_context(context)
        offer = (
            self._price_midpoint(context)
            if len(context.scenario.issues) == 1
            and isinstance(context.scenario.issues[0], NumericIssue)
            else self._multi_issue_candidate(context)
        )
        intervention_number = 1 + sum(
            isinstance(event, MediatorInterventionEvent)
            for event in context.session.event_sequence
        )
        intervention_id = MediatorInterventionId(f"mediator-intervention-{intervention_number}")
        if offer is None:
            return MediatorIntervention(
                intervention_id=intervention_id,
                mediator_id=self.configuration.mediator_id,
                negotiation_id=context.session.scenario_id,
                round_number=context.session.round_number,
                trigger=trigger,
                access_mode=context.access_mode,
                kind=MediatorInterventionKind.REFUSAL,
                public_disagreement_summary=(
                    "The permitted information does not establish a mutually admissible region."
                ),
                public_explanation=(
                    "I cannot identify a mutually admissible compromise from the permitted "
                    "information. No agreement has been imposed."
                ),
            )
        algorithm = (
            "midpoint"
            if len(context.scenario.issues) == 1
            else self.configuration.objective.value.replace("_", "-")
        )
        return MediatorIntervention(
            intervention_id=intervention_id,
            mediator_id=self.configuration.mediator_id,
            negotiation_id=context.session.scenario_id,
            round_number=context.session.round_number,
            trigger=trigger,
            access_mode=context.access_mode,
            kind=MediatorInterventionKind.PROPOSAL,
            public_disagreement_summary=(
                "The public offer history shows unresolved differences across the negotiated issues."
            ),
            public_explanation=(
                f"I propose a non-binding compromise selected by the {algorithm} method using "
                "only information permitted by the configured access mode."
            ),
            offer=offer,
        )

    def _validate_context(self, context: MediatorContext) -> None:
        if context.session.phase is not ProtocolPhase.MEDIATION:
            raise ValueError("mediator intervention requires a session in mediation")
        if context.access_mode is not self.configuration.access_mode:
            raise ValueError("mediator context access mode does not match its configuration")
        if (
            context.access_mode is MediationAccessMode.SIMULATION_ORACLE
            and not self.configuration.allow_simulation_oracle
        ):
            raise ValueError("simulation-oracle access was not explicitly enabled")

    def _next_offer_id(self, context: MediatorContext) -> OfferId:
        count = 1 + sum(
            isinstance(event, MediatorInterventionEvent)
            for event in context.session.event_sequence
        )
        return OfferId(f"mediator-offer-{count}")

    def _price_midpoint(self, context: MediatorContext) -> Optional[Offer]:
        issue = context.scenario.issues[0]
        preferences = _preferences_for_context(context)
        lower, upper = issue.minimum, issue.maximum
        if preferences:
            for item in preferences.values():
                preference = item.issue_preferences[0]
                reservation = item.reservation.reservation_utility
                if not isinstance(preference, NumericPreference):
                    raise ValueError("price-only mediation requires numeric preferences")
                if preference.direction.value == "maximize":
                    lower = max(lower, issue.minimum + reservation * (issue.maximum - issue.minimum))
                else:
                    upper = min(upper, issue.minimum + (1.0 - reservation) * (issue.maximum - issue.minimum))
            if lower > upper + 1e-9:
                return None
        else:
            latest_by_actor: dict[ParticipantId, float] = {}
            for event in context.session.event_sequence:
                if isinstance(event, ActionAppliedEvent) and isinstance(
                    event.action, (ProposeAction, CounterAction)
                ):
                    value = event.action.offer.value_for(issue.issue_id)
                    latest_by_actor[event.action.actor_id] = value.value
            if len(latest_by_actor) >= 2:
                values = tuple(latest_by_actor.values())
                lower, upper = min(values), max(values)
        midpoint = (lower + upper) / 2.0
        return Offer(
            offer_id=self._next_offer_id(context),
            values=(NumericIssueValue(issue_id=issue.issue_id, value=midpoint),),
        )

    def _multi_issue_candidate(self, context: MediatorContext) -> Optional[Offer]:
        issue_values = []
        observed = [
            event.action.offer
            for event in context.session.event_sequence
            if isinstance(event, ActionAppliedEvent)
            and isinstance(event.action, (ProposeAction, CounterAction))
        ]
        for issue in context.scenario.issues:
            if isinstance(issue, NumericIssue):
                values = {
                    issue.minimum
                    + index * (issue.maximum - issue.minimum)
                    / (self.configuration.numeric_grid_points - 1)
                    for index in range(self.configuration.numeric_grid_points)
                }
                for offer in observed:
                    value = offer.value_for(issue.issue_id)
                    values.add(value.value)
                issue_values.append(
                    tuple(NumericIssueValue(issue_id=issue.issue_id, value=value) for value in sorted(values))
                )
            else:
                issue_values.append(
                    tuple(
                        CategoricalIssueValue(issue_id=issue.issue_id, value=choice)
                        for choice in issue.choices
                    )
                )
        candidate_count = math.prod(len(values) for values in issue_values)
        if candidate_count > self.configuration.maximum_candidates:
            raise ValueError("multi-issue mediator candidate grid exceeds maximum_candidates")

        preferences = _preferences_for_context(context)
        anchors: dict[ParticipantId, Offer] = {}
        for event in context.session.event_sequence:
            if isinstance(event, ActionAppliedEvent) and isinstance(
                event.action, (ProposeAction, CounterAction)
            ):
                anchors[event.action.actor_id] = event.action.offer

        best: Optional[tuple[float, float, Tuple[Tuple[str, object], ...], Offer]] = None
        for index, values in enumerate(itertools.product(*issue_values), start=1):
            offer = Offer(offer_id=OfferId(f"candidate-{index}"), values=tuple(values))
            if preferences:
                utilities = tuple(
                    calculate_utility(context.scenario, offer, preferences[participant.participant_id])
                    for participant in context.scenario.participants
                )
                reservations = tuple(
                    preferences[participant.participant_id].reservation.reservation_utility
                    for participant in context.scenario.participants
                )
                if any(utility + 1e-9 < reservation for utility, reservation in zip(utilities, reservations)):
                    continue
                gains = tuple(
                    (utility - reservation) / max(1e-12, 1.0 - reservation)
                    for utility, reservation in zip(utilities, reservations)
                )
            else:
                gains = tuple(
                    1.0 - _offer_distance(context.scenario, offer, anchors[participant.participant_id])
                    if participant.participant_id in anchors
                    else 0.5
                    for participant in context.scenario.participants
                )
            nash = math.prod(max(0.0, gain) for gain in gains)
            minimum = min(gains)
            primary, secondary = (
                (nash, minimum)
                if self.configuration.objective is MediationObjective.NASH_PRODUCT
                else (minimum, nash)
            )
            rank = (primary, secondary, _offer_signature(offer), offer)
            if best is None or rank[:2] > best[:2] or (
                rank[:2] == best[:2] and rank[2] < best[2]
            ):
                best = rank
        if best is None:
            return None
        return best[3].model_copy(update={"offer_id": self._next_offer_id(context)})


class LLMMediatorOutput(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: MediatorInterventionKind
    public_disagreement_summary: str = Field(min_length=1, max_length=1000)
    public_explanation: str = Field(min_length=1, max_length=1000)
    offer: Optional[Offer] = None
    question: Optional[str] = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def payload_matches_kind(self) -> LLMMediatorOutput:
        MediatorIntervention(
            intervention_id=MediatorInterventionId("validation"),
            mediator_id=ParticipantId("mediator"),
            negotiation_id="validation",
            round_number=1,
            trigger=MediationTrigger.DEADLOCK,
            kind=self.kind,
            public_disagreement_summary=self.public_disagreement_summary,
            public_explanation=self.public_explanation,
            offer=self.offer,
            question=self.question,
        )
        return self


class LLMMediator:
    """Optional structured mediator using public context only, even in research modes."""

    capabilities = MediatorCapabilities()

    def __init__(
        self,
        client: LLMClient,
        model_configuration: ModelConfiguration,
        configuration: Optional[MediatorConfiguration] = None,
    ) -> None:
        self.client = client
        self.model_configuration = model_configuration
        self.configuration = configuration or MediatorConfiguration()
        self.last_usage = LLMUsage()
        self.last_latency_ms = 0.0

    def intervene(
        self, context: MediatorContext, trigger: MediationTrigger
    ) -> MediatorIntervention:
        if context.session.phase is not ProtocolPhase.MEDIATION:
            raise ValueError("LLM mediator requires a session in mediation")
        if context.access_mode is not self.configuration.access_mode:
            raise ValueError("mediator context access mode does not match its configuration")
        public_events = tuple(
            event.model_dump(mode="json") for event in context.session.event_sequence
            if isinstance(event, ActionAppliedEvent)
        )
        request = LLMRequest(
            request_id=f"mediator-{len(context.session.event_sequence) + 1}",
            messages=mediator_messages(
                LLMMediatorOutput.model_json_schema(),
                {
                    "access_mode_label": context.access_mode.value,
                    "scenario": context.scenario.model_dump(mode="json"),
                    "phase": context.session.phase.value,
                    "round_number": context.session.round_number,
                    "deadline_round": context.session.maximum_rounds,
                    "latest_offer": (
                        context.session.latest_valid_offer.model_dump(mode="json")
                        if context.session.latest_valid_offer is not None
                        else None
                    ),
                    "public_action_events": public_events,
                    "public_messages": tuple(
                        message.model_dump(mode="json") for message in context.public_messages
                    ),
                },
            ),
            model_configuration=self.model_configuration,
        )
        response = self.client.generate(request)
        output = LLMMediatorOutput.model_validate_json(response.text)
        if output.offer is not None:
            context.scenario.validate_offer(output.offer)
        self.last_usage = response.usage
        self.last_latency_ms = response.latency.total_ms
        count = 1 + sum(
            isinstance(event, MediatorInterventionEvent)
            for event in context.session.event_sequence
        )
        return MediatorIntervention(
            intervention_id=MediatorInterventionId(f"mediator-intervention-{count}"),
            mediator_id=self.configuration.mediator_id,
            negotiation_id=context.session.scenario_id,
            round_number=context.session.round_number,
            trigger=trigger,
            access_mode=context.access_mode,
            kind=output.kind,
            public_disagreement_summary=output.public_disagreement_summary,
            public_explanation=output.public_explanation,
            offer=output.offer,
            question=output.question,
        )


@dataclass(frozen=True)
class MediatedTransition:
    transition: TransitionResult
    intervention: MediatorIntervention
    message: MessageEnvelope


class MediationService:
    """Coordinates replayable protocol entry/interventions with correlated public messages."""

    def __init__(self) -> None:
        self._interventions: Tuple[MediatorIntervention, ...] = ()

    @property
    def interventions(self) -> Tuple[MediatorIntervention, ...]:
        return self._interventions

    def enter_if_triggered(
        self,
        protocol: NegotiationProtocol,
        state: NegotiationSession,
        decision: MediationTriggerDecision,
    ) -> TransitionResult:
        if not decision.should_enter or decision.trigger is None:
            return TransitionResult(state=state, events=())
        if decision.trigger is MediationTrigger.EXPLICIT_REQUEST:
            if state.phase is not ProtocolPhase.MEDIATION:
                raise ValueError("explicit mediation entry must first be applied as a protocol action")
            return TransitionResult(state=state, events=())
        return protocol.enter_mediation(state, decision.trigger, decision.reason)

    def apply(
        self,
        protocol: NegotiationProtocol,
        state: NegotiationSession,
        intervention: MediatorIntervention,
        message_bus: MessageBus,
        timestamp: datetime,
    ) -> MediatedTransition:
        message_type = {
            MediatorInterventionKind.PROPOSAL: MessageType.MEDIATOR_PROPOSAL,
            MediatorInterventionKind.CLARIFYING_QUESTION: MessageType.MEDIATOR_QUESTION,
            MediatorInterventionKind.REFUSAL: MessageType.MEDIATOR_NOTICE,
        }[intervention.kind]
        content = intervention.question or intervention.public_explanation
        recipients = tuple(
            participant_id
            for participant_id in message_bus.participants
            if participant_id != intervention.mediator_id
        )
        # Untrusted model-authored mediator text must pass the message boundary
        # before the intervention can mutate protocol state.
        message_bus.preflight(
            sender=intervention.mediator_id,
            recipients=recipients,
            timestamp=timestamp,
            message_type=message_type,
            visibility=MessageVisibility.PUBLIC,
            content=content,
            referenced_offer_id=(
                intervention.offer.offer_id if intervention.offer is not None else None
            ),
        )
        transition = protocol.apply_mediator_intervention(state, intervention)
        event = transition.events[0]
        message = message_bus.send(
            sender=intervention.mediator_id,
            recipients=recipients,
            timestamp=timestamp,
            message_type=message_type,
            visibility=MessageVisibility.PUBLIC,
            content=content,
            correlation_id=event.correlation_id,
            referenced_offer_id=(
                intervention.offer.offer_id if intervention.offer is not None else None
            ),
        )
        self._interventions = self._interventions + (intervention,)
        return MediatedTransition(transition, intervention, message)
