from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from .domain import IssueId, ParticipantId
from .domain.models import DomainModel


PROBABILITY_TOLERANCE = 1e-6


class BeliefStatus(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


class GoalKind(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    PREFER_CATEGORY = "prefer_category"
    UNKNOWN = "unknown"


class StrategyKind(str, Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    BOULWARE = "boulware"
    CONCEDER = "conceder"
    TIME_DEPENDENT = "time_dependent"
    TIT_FOR_TAT = "tit_for_tat"
    RANDOM = "random"
    UNKNOWN = "unknown"


class NextActionKind(str, Enum):
    PROPOSE = "propose"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"
    MESSAGE = "message"
    REQUEST_MEDIATION = "request_mediation"
    WITHDRAW = "withdraw"
    UNKNOWN = "unknown"


class ObservableFact(DomainModel):
    status: Literal[BeliefStatus.FACT] = BeliefStatus.FACT
    evidence_event_id: str = Field(min_length=1, max_length=100)
    round_number: int = Field(ge=1)
    statement: str = Field(min_length=1, max_length=500)


class InferenceModel(DomainModel):
    status: Literal[BeliefStatus.INFERENCE] = BeliefStatus.INFERENCE
    evidence_event_ids: Tuple[str, ...] = ()

    @field_validator("evidence_event_ids", mode="before")
    @classmethod
    def ids_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class GoalHypothesis(InferenceModel):
    issue_id: IssueId
    kind: GoalKind
    category: Optional[str] = Field(default=None, min_length=1, max_length=200)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def category_matches_kind(self) -> GoalHypothesis:
        if (self.kind is GoalKind.PREFER_CATEGORY) != (self.category is not None):
            raise ValueError("only categorical goal hypotheses may contain a category")
        return self


class NumericReservationRange(InferenceModel):
    issue_id: IssueId
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> NumericReservationRange:
        if self.lower > self.upper:
            raise ValueError("reservation range lower bound must not exceed upper bound")
        return self


class StrategyHypothesis(InferenceModel):
    strategy: StrategyKind
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class NextActionHypothesis(InferenceModel):
    action: NextActionKind
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class InformationNeed(InferenceModel):
    topic: str = Field(min_length=1, max_length=300)
    probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class BeliefUnknown(DomainModel):
    status: Literal[BeliefStatus.UNKNOWN] = BeliefStatus.UNKNOWN
    topic: str = Field(min_length=1, max_length=300)


def _distribution_sum(items: Tuple[Any, ...], label: str) -> None:
    if items and not math.isclose(
        math.fsum(item.probability for item in items),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{label} probabilities must sum to 1")


class OpponentBeliefState(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    negotiation_id: str = Field(min_length=1, max_length=100)
    observer_id: ParticipantId
    opponent_id: ParticipantId
    facts: Tuple[ObservableFact, ...] = ()
    goal_hypotheses: Tuple[GoalHypothesis, ...] = ()
    reservation_utility_lower: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reservation_utility_upper: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reservation_utility_probability: float = Field(
        default=1.0, ge=0.0, le=1.0, allow_inf_nan=False
    )
    numeric_reservation_ranges: Tuple[NumericReservationRange, ...] = ()
    strategy_hypotheses: Tuple[StrategyHypothesis, ...] = ()
    next_action_hypotheses: Tuple[NextActionHypothesis, ...] = ()
    information_needs: Tuple[InformationNeed, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_event_ids: Tuple[str, ...] = ()
    unknowns: Tuple[BeliefUnknown, ...] = ()
    last_updated_round: int = Field(ge=1)

    @field_validator(
        "facts", "goal_hypotheses", "numeric_reservation_ranges", "strategy_hypotheses",
        "next_action_hypotheses", "information_needs", "evidence_event_ids", "unknowns",
        mode="before",
    )
    @classmethod
    def tuples_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def consistent(self) -> OpponentBeliefState:
        if self.observer_id == self.opponent_id:
            raise ValueError("belief observer and opponent must differ")
        if self.reservation_utility_lower > self.reservation_utility_upper:
            raise ValueError("reservation utility range must be ordered")
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("belief evidence event ids must be unique")
        if any(fact.evidence_event_id not in self.evidence_event_ids for fact in self.facts):
            raise ValueError("every fact must reference retained observable evidence")
        inferences = (
            self.goal_hypotheses
            + self.numeric_reservation_ranges
            + self.strategy_hypotheses
            + self.next_action_hypotheses
            + self.information_needs
        )
        if any(
            evidence_id not in self.evidence_event_ids
            for inference in inferences
            for evidence_id in inference.evidence_event_ids
        ):
            raise ValueError("inferences may reference only retained observable evidence")
        for issue_id in {item.issue_id for item in self.goal_hypotheses}:
            _distribution_sum(
                tuple(item for item in self.goal_hypotheses if item.issue_id == issue_id),
                f"goal hypotheses for {issue_id}",
            )
        _distribution_sum(self.strategy_hypotheses, "strategy hypotheses")
        _distribution_sum(self.next_action_hypotheses, "next-action hypotheses")
        return self


class OpponentModelMode(str, Enum):
    DISABLED = "disabled"
    HEURISTIC = "heuristic"
    LLM = "llm"


class OpponentModelConfiguration(DomainModel):
    mode: OpponentModelMode = OpponentModelMode.HEURISTIC
    minimum_planning_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


class CalibrationMetrics(DomainModel):
    reservation_utility_covered: bool
    reservation_interval_width: float = Field(ge=0.0, le=1.0)
    strategy_probability: float = Field(ge=0.0, le=1.0)
    goal_brier_score: float = Field(ge=0.0, le=1.0)
    confidence_error: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
