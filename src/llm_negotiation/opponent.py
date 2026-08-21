from __future__ import annotations

import math
from typing import Any, Dict, Optional, Protocol, Tuple

from pydantic import Field, ValidationError, field_validator, model_validator

from .beliefs import (
    BeliefUnknown,
    CalibrationMetrics,
    GoalHypothesis,
    GoalKind,
    InformationNeed,
    NextActionHypothesis,
    NextActionKind,
    NumericReservationRange,
    ObservableFact,
    OpponentBeliefState,
    StrategyHypothesis,
    StrategyKind,
)
from .communication import MessageType, MessageVisibility
from .domain import (
    CategoricalIssue,
    CounterAction,
    NegotiationAction,
    NegotiationScenario,
    NumericIssue,
    NumericIssueValue,
    NumericPreference,
    ParticipantId,
    ParticipantPreferences,
    PreferenceDirection,
    ProposeAction,
)
from .domain.models import DomainModel
from .llm import LLMClient, LLMClientError, LLMRequest, ModelConfiguration
from .llm_prompts.opponent_model_v1 import opponent_model_messages
from .memory import (
    MemorySnapshot,
    ObservationMemoryEvent,
    ReceivedMessageMemoryEvent,
)
from .protocol import NegotiationSession, TERMINAL_PHASES


class ObservedAction(DomainModel):
    evidence_event_id: str = Field(min_length=1, max_length=100)
    round_number: int = Field(ge=1)
    action: NegotiationAction


class ObservedMessage(DomainModel):
    evidence_event_id: str = Field(min_length=1, max_length=100)
    round_number: int = Field(ge=1)
    negotiation_id: str = Field(min_length=1, max_length=100)
    sender: ParticipantId
    recipients: Tuple[ParticipantId, ...]
    visibility: MessageVisibility
    message_type: MessageType
    content: str = Field(min_length=1, max_length=1000)

    @field_validator("recipients", mode="before")
    @classmethod
    def recipients_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ObservableOpponentEvidence(DomainModel):
    negotiation_id: str
    observer_id: ParticipantId
    opponent_id: ParticipantId
    scenario: NegotiationScenario
    current_round: int = Field(ge=1)
    actions: Tuple[ObservedAction, ...] = ()
    messages: Tuple[ObservedMessage, ...] = ()

    @field_validator("actions", "messages", mode="before")
    @classmethod
    def tuples_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def visible_only(self) -> ObservableOpponentEvidence:
        if self.observer_id == self.opponent_id:
            raise ValueError("opponent evidence requires distinct participants")
        if self.negotiation_id != self.scenario.scenario_id:
            raise ValueError("opponent evidence negotiation must match its public scenario")
        self.scenario.participant(self.observer_id)
        self.scenario.participant(self.opponent_id)
        if any(item.action.actor_id != self.opponent_id for item in self.actions):
            raise ValueError("opponent action evidence must belong to the modelled opponent")
        if any(item.action.round_number != item.round_number for item in self.actions):
            raise ValueError("observed action rounds must match their evidence records")
        if any(item.sender != self.opponent_id for item in self.messages):
            raise ValueError("opponent message evidence must belong to the modelled opponent")
        if any(item.negotiation_id != self.negotiation_id for item in self.messages):
            raise ValueError("opponent messages must belong to the evidence negotiation")
        if any(
            item.visibility is MessageVisibility.SYSTEM_AUDIT
            or (
                item.visibility is not MessageVisibility.PUBLIC
                and self.observer_id not in item.recipients
            )
            for item in self.messages
        ):
            raise ValueError("opponent message is not visible to the belief observer")
        if any(
            item.round_number > self.current_round
            for item in self.actions + self.messages
        ):
            raise ValueError("opponent evidence cannot come from a future round")
        if any(
            later.round_number < earlier.round_number
            for collection in (self.actions, self.messages)
            for earlier, later in zip(collection, collection[1:])
        ):
            raise ValueError("opponent evidence must use deterministic round ordering")
        ids = tuple(item.evidence_event_id for item in self.actions + self.messages)
        if len(set(ids)) != len(ids):
            raise ValueError("observable evidence ids must be unique")
        return self


def evidence_from_memory(
    memory: MemorySnapshot, opponent_id: ParticipantId
) -> ObservableOpponentEvidence:
    actions = []
    messages = []
    for event in memory.episodic.events:
        if (
            isinstance(event, ObservationMemoryEvent)
            and event.observed_action is not None
            and event.observed_action.actor_id == opponent_id
        ):
            actions.append(
                ObservedAction(
                    evidence_event_id=event.source_event_id or f"memory-{event.sequence_number}",
                    round_number=event.round_number,
                    action=event.observed_action,
                )
            )
        elif (
            isinstance(event, ReceivedMessageMemoryEvent)
            and event.message.sender == opponent_id
        ):
            messages.append(
                ObservedMessage(
                    evidence_event_id=str(event.message.message_id),
                    round_number=memory.working.round_number,
                    negotiation_id=event.message.negotiation_id,
                    sender=event.message.sender,
                    recipients=event.message.recipients,
                    visibility=event.message.visibility,
                    message_type=event.message.message_type,
                    content=event.message.content,
                )
            )
    return ObservableOpponentEvidence(
        negotiation_id=memory.working.negotiation_id,
        observer_id=memory.owner_id,
        opponent_id=opponent_id,
        scenario=memory.scenario,
        current_round=memory.working.round_number,
        actions=tuple(actions),
        messages=tuple(messages),
    )


class OpponentModeller(Protocol):
    def update(
        self,
        previous: Optional[OpponentBeliefState],
        evidence: ObservableOpponentEvidence,
    ) -> OpponentBeliefState:
        """Update beliefs using observable evidence only."""


def unknown_belief(evidence: ObservableOpponentEvidence) -> OpponentBeliefState:
    return OpponentBeliefState(
        negotiation_id=evidence.negotiation_id,
        observer_id=evidence.observer_id,
        opponent_id=evidence.opponent_id,
        reservation_utility_lower=0.0,
        reservation_utility_upper=1.0,
        reservation_utility_probability=1.0,
        strategy_hypotheses=(
            StrategyHypothesis(strategy=StrategyKind.UNKNOWN, probability=1.0),
        ),
        next_action_hypotheses=(
            NextActionHypothesis(action=NextActionKind.UNKNOWN, probability=1.0),
        ),
        confidence=0.0,
        unknowns=(
            BeliefUnknown(topic="opponent goals"),
            BeliefUnknown(topic="opponent reservation utility"),
            BeliefUnknown(topic="opponent strategy"),
            BeliefUnknown(topic="opponent next action"),
            BeliefUnknown(topic="opponent information needs"),
        ),
        last_updated_round=evidence.current_round,
    )


def _normalize(values: Dict[Any, float]) -> Dict[Any, float]:
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


class HeuristicOpponentModeller:
    """Deterministic Bayesian-style concession updater for ablation baselines."""

    def update(
        self,
        previous: Optional[OpponentBeliefState],
        evidence: ObservableOpponentEvidence,
    ) -> OpponentBeliefState:
        if previous is not None and (
            previous.negotiation_id != evidence.negotiation_id
            or previous.observer_id != evidence.observer_id
            or previous.opponent_id != evidence.opponent_id
        ):
            raise ValueError("prior belief identity does not match observable evidence")
        evidence_ids = tuple(
            item.evidence_event_id for item in evidence.actions + evidence.messages
        )
        if previous is not None and set(evidence_ids) == set(previous.evidence_event_ids):
            return previous
        if not evidence_ids:
            return unknown_belief(evidence)

        facts = tuple(
            ObservableFact(
                evidence_event_id=item.evidence_event_id,
                round_number=item.round_number,
                statement=f"Opponent performed observable action {item.action.action.value}.",
            )
            for item in evidence.actions
        ) + tuple(
            ObservableFact(
                evidence_event_id=item.evidence_event_id,
                round_number=item.round_number,
                statement=f"Opponent sent observable {item.message_type.value} message.",
            )
            for item in evidence.messages
        )
        offers = [
            item for item in evidence.actions
            if isinstance(item.action, (ProposeAction, CounterAction))
        ]
        offer_evidence_ids = tuple(item.evidence_event_id for item in offers)
        previous_evidence_ids = set(previous.evidence_event_ids) if previous is not None else set()
        new_offer_indexes = tuple(
            index
            for index, item in enumerate(offers)
            if item.evidence_event_id not in previous_evidence_ids
        )
        prior_goals: Dict[object, Dict[GoalKind, float]] = {}
        if previous is not None:
            for item in previous.goal_hypotheses:
                if item.kind in {GoalKind.MAXIMIZE, GoalKind.MINIMIZE}:
                    prior_goals.setdefault(item.issue_id, {})[item.kind] = item.probability

        goals = []
        ranges = []
        reservation_upper = 1.0
        directional_certainties = []
        for issue in evidence.scenario.issues:
            if isinstance(issue, NumericIssue):
                prior = prior_goals.get(
                    issue.issue_id,
                    {GoalKind.MAXIMIZE: 0.5, GoalKind.MINIMIZE: 0.5},
                )
                values = [
                    item.action.offer.value_for(issue.issue_id).value
                    for item in offers
                    if isinstance(item.action.offer.value_for(issue.issue_id), NumericIssueValue)
                ]
                posterior = dict(prior)
                for new_index in new_offer_indexes:
                    if new_index == 0:
                        continue
                    delta = values[new_index] - values[new_index - 1]
                    likelihood = {GoalKind.MAXIMIZE: 1.0, GoalKind.MINIMIZE: 1.0}
                    if delta < 0:
                        likelihood = {GoalKind.MAXIMIZE: 0.8, GoalKind.MINIMIZE: 0.2}
                    elif delta > 0:
                        likelihood = {GoalKind.MAXIMIZE: 0.2, GoalKind.MINIMIZE: 0.8}
                    posterior = _normalize(
                        {
                            kind: posterior.get(kind, 0.5) * likelihood[kind]
                            for kind in likelihood
                        }
                    )
                for kind, probability in posterior.items():
                    goals.append(
                        GoalHypothesis(
                            issue_id=issue.issue_id,
                            kind=kind,
                            probability=probability,
                            evidence_event_ids=offer_evidence_ids,
                        )
                    )
                directional_certainties.append(abs(posterior[GoalKind.MAXIMIZE] - 0.5) * 2)
                if values and not math.isclose(
                    posterior[GoalKind.MAXIMIZE],
                    posterior[GoalKind.MINIMIZE],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    latest = values[-1]
                    maximizing = posterior[GoalKind.MAXIMIZE] >= posterior[GoalKind.MINIMIZE]
                    lower, upper = (issue.minimum, latest) if maximizing else (latest, issue.maximum)
                    ranges.append(
                        NumericReservationRange(
                            issue_id=issue.issue_id,
                            lower=lower,
                            upper=upper,
                            probability=max(posterior.values()),
                            evidence_event_ids=offer_evidence_ids,
                        )
                    )
                    if len(evidence.scenario.issues) == 1:
                        normalized = (latest - issue.minimum) / (issue.maximum - issue.minimum)
                        reservation_upper = normalized if maximizing else 1.0 - normalized
            elif isinstance(issue, CategoricalIssue) and offers:
                category = offers[0].action.offer.value_for(issue.issue_id).value
                goals.extend(
                    (
                        GoalHypothesis(
                            issue_id=issue.issue_id,
                            kind=GoalKind.PREFER_CATEGORY,
                            category=category,
                            probability=0.7,
                            evidence_event_ids=offer_evidence_ids,
                        ),
                        GoalHypothesis(
                            issue_id=issue.issue_id,
                            kind=GoalKind.UNKNOWN,
                            probability=0.3,
                            evidence_event_ids=offer_evidence_ids,
                        ),
                    )
                )

        strategy = self._strategy_distribution(offers, offer_evidence_ids)
        next_actions = self._next_action_distribution(bool(offers), offer_evidence_ids)
        info_needs = []
        questions = [item for item in evidence.messages if item.message_type is MessageType.QUESTION]
        if questions:
            info_needs.append(
                InformationNeed(
                    topic="answer to the opponent's observable question",
                    probability=0.9,
                    evidence_event_ids=tuple(item.evidence_event_id for item in questions),
                )
            )
        if not offers:
            info_needs.append(InformationNeed(topic="counterparty offer information", probability=0.8))
        confidence = min(
            0.95,
            (0.15 * len(evidence_ids))
            * (sum(directional_certainties) / len(directional_certainties) if directional_certainties else 0.5),
        )
        unknowns = []
        if not offers:
            unknowns.extend(
                (
                    BeliefUnknown(topic="opponent issue goals"),
                    BeliefUnknown(topic="opponent reservation range"),
                )
            )
        if confidence < 0.5:
            unknowns.append(BeliefUnknown(topic="opponent strategy remains low confidence"))
        return OpponentBeliefState(
            negotiation_id=evidence.negotiation_id,
            observer_id=evidence.observer_id,
            opponent_id=evidence.opponent_id,
            facts=facts,
            goal_hypotheses=tuple(goals),
            reservation_utility_lower=0.0,
            reservation_utility_upper=max(0.0, min(1.0, reservation_upper)),
            reservation_utility_probability=(
                ranges[0].probability if len(evidence.scenario.issues) == 1 and ranges else 1.0
            ),
            numeric_reservation_ranges=tuple(ranges),
            strategy_hypotheses=strategy,
            next_action_hypotheses=next_actions,
            information_needs=tuple(info_needs),
            confidence=confidence,
            evidence_event_ids=evidence_ids,
            unknowns=tuple(unknowns),
            last_updated_round=evidence.current_round,
        )

    @staticmethod
    def _strategy_distribution(
        offers: list, evidence_event_ids: Tuple[str, ...]
    ) -> Tuple[StrategyHypothesis, ...]:
        scores = {kind: 0.05 for kind in (
            StrategyKind.FIXED, StrategyKind.LINEAR, StrategyKind.BOULWARE,
            StrategyKind.CONCEDER, StrategyKind.TIME_DEPENDENT,
            StrategyKind.TIT_FOR_TAT, StrategyKind.RANDOM,
        )}
        if len(offers) < 2:
            scores[StrategyKind.UNKNOWN] = 0.7
        else:
            numeric_vectors = []
            for item in offers:
                numeric_vectors.append(tuple(
                    value.value for value in item.action.offer.values if isinstance(value, NumericIssueValue)
                ))
            deltas = [
                sum(abs(a - b) for a, b in zip(left, right))
                for left, right in zip(numeric_vectors, numeric_vectors[1:])
            ]
            if not any(deltas):
                scores[StrategyKind.FIXED] += 0.75
            elif len(deltas) >= 2 and deltas[-1] > deltas[0] * 1.5:
                scores[StrategyKind.BOULWARE] += 0.65
            elif len(deltas) >= 2 and deltas[0] > deltas[-1] * 1.5:
                scores[StrategyKind.CONCEDER] += 0.65
            else:
                scores[StrategyKind.LINEAR] += 0.6
        normalized = _normalize(scores)
        return tuple(
            StrategyHypothesis(
                strategy=key,
                probability=value,
                evidence_event_ids=evidence_event_ids,
            )
            for key, value in normalized.items()
        )

    @staticmethod
    def _next_action_distribution(
        has_offer: bool, evidence_event_ids: Tuple[str, ...]
    ) -> Tuple[NextActionHypothesis, ...]:
        values = (
            {NextActionKind.COUNTER: 0.45, NextActionKind.ACCEPT: 0.25,
             NextActionKind.REJECT: 0.1, NextActionKind.MESSAGE: 0.1,
             NextActionKind.REQUEST_MEDIATION: 0.05, NextActionKind.WITHDRAW: 0.05}
            if has_offer else {NextActionKind.UNKNOWN: 1.0}
        )
        return tuple(
            NextActionHypothesis(
                action=key,
                probability=value,
                evidence_event_ids=evidence_event_ids,
            )
            for key, value in values.items()
        )


class LLMOpponentModeller:
    def __init__(
        self,
        client: LLMClient,
        model_configuration: ModelConfiguration,
        fallback: Optional[OpponentModeller] = None,
    ) -> None:
        self.client = client
        self.model_configuration = model_configuration
        self.fallback = fallback or HeuristicOpponentModeller()

    def update(
        self,
        previous: Optional[OpponentBeliefState],
        evidence: ObservableOpponentEvidence,
    ) -> OpponentBeliefState:
        messages = opponent_model_messages(
            OpponentBeliefState.model_json_schema(),
            {
                "previous_belief": previous.model_dump(mode="json") if previous else None,
                "observable_evidence": evidence.model_dump(mode="json"),
            },
        )
        try:
            response = self.client.generate(
                LLMRequest(
                    request_id=f"opponent-belief-{evidence.opponent_id}-{evidence.current_round}",
                    messages=messages,
                    model_configuration=self.model_configuration,
                )
            )
            belief = OpponentBeliefState.model_validate_json(response.text)
            self._validate_output(belief, evidence)
            return belief
        except (LLMClientError, ValidationError, ValueError):
            return self.fallback.update(previous, evidence)

    @staticmethod
    def _validate_output(
        belief: OpponentBeliefState, evidence: ObservableOpponentEvidence
    ) -> None:
        if (
            belief.negotiation_id != evidence.negotiation_id
            or belief.observer_id != evidence.observer_id
            or belief.opponent_id != evidence.opponent_id
            or belief.last_updated_round != evidence.current_round
        ):
            raise ValueError("LLM belief identity or round does not match observable evidence")
        available = {item.evidence_event_id for item in evidence.actions + evidence.messages}
        if not set(belief.evidence_event_ids).issubset(available):
            raise ValueError("LLM belief references non-observable evidence")
        expected_facts = {
            item.evidence_event_id: (
                item.round_number,
                f"Opponent performed observable action {item.action.action.value}.",
            )
            for item in evidence.actions
        }
        expected_facts.update(
            {
                item.evidence_event_id: (
                    item.round_number,
                    f"Opponent sent observable {item.message_type.value} message.",
                )
                for item in evidence.messages
            }
        )
        if any(
            expected_facts.get(fact.evidence_event_id)
            != (fact.round_number, fact.statement)
            for fact in belief.facts
        ):
            raise ValueError("LLM facts must exactly reflect supplied observable evidence")
        issues = {issue.issue_id: issue for issue in evidence.scenario.issues}
        for hypothesis in belief.goal_hypotheses:
            issue = issues.get(hypothesis.issue_id)
            if issue is None:
                raise ValueError("LLM belief references an unknown issue")
            if hypothesis.kind is GoalKind.PREFER_CATEGORY:
                if not isinstance(issue, CategoricalIssue) or hypothesis.category not in issue.choices:
                    raise ValueError("LLM belief contains an invalid categorical goal")
            elif hypothesis.kind in {GoalKind.MAXIMIZE, GoalKind.MINIMIZE} and not isinstance(
                issue, NumericIssue
            ):
                raise ValueError("LLM belief applies a numeric goal to a non-numeric issue")
        for inferred_range in belief.numeric_reservation_ranges:
            issue = issues.get(inferred_range.issue_id)
            if not isinstance(issue, NumericIssue):
                raise ValueError("LLM belief range must reference a numeric scenario issue")
            if inferred_range.lower < issue.minimum or inferred_range.upper > issue.maximum:
                raise ValueError("LLM belief range exceeds the public issue bounds")


def calibrate_belief(
    belief: OpponentBeliefState,
    ground_truth: ParticipantPreferences,
    actual_strategy: StrategyKind,
    completed_session: NegotiationSession,
) -> CalibrationMetrics:
    """Compare a frozen belief with private simulation truth after terminal completion."""
    if completed_session.phase not in TERMINAL_PHASES or completed_session.outcome is None:
        raise ValueError("calibration requires a completed terminal negotiation session")
    if completed_session.scenario_id != belief.negotiation_id:
        raise ValueError("calibration session does not match the belief negotiation")
    if belief.opponent_id != ground_truth.participant_id:
        raise ValueError("calibration ground truth belongs to a different opponent")
    reservation = ground_truth.reservation.reservation_utility
    covered = belief.reservation_utility_lower <= reservation <= belief.reservation_utility_upper
    strategy_probability = next(
        (item.probability for item in belief.strategy_hypotheses if item.strategy is actual_strategy),
        0.0,
    )
    squared = []
    for preference in ground_truth.issue_preferences:
        if isinstance(preference, NumericPreference):
            expected = (
                GoalKind.MAXIMIZE
                if preference.direction is PreferenceDirection.MAXIMIZE
                else GoalKind.MINIMIZE
            )
            probability = next(
                (item.probability for item in belief.goal_hypotheses
                 if item.issue_id == preference.issue_id and item.kind is expected),
                0.0,
            )
            squared.append((1.0 - probability) ** 2)
    goal_brier = sum(squared) / len(squared) if squared else 1.0
    correctness = (float(covered) + strategy_probability + (1.0 - goal_brier)) / 3.0
    return CalibrationMetrics(
        reservation_utility_covered=covered,
        reservation_interval_width=(
            belief.reservation_utility_upper - belief.reservation_utility_lower
        ),
        strategy_probability=strategy_probability,
        goal_brier_score=goal_brier,
        confidence_error=abs(belief.confidence - correctness),
        evidence_count=len(belief.evidence_event_ids),
    )
