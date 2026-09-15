"""Transparent, reproducible strategy learning between completed episodes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

from pydantic import Field, ValidationError, field_validator, model_validator

from .domain import (
    CategoricalPreference,
    IssueId,
    IssueKind,
    NegotiationScenario,
    NumericPreference,
    ParticipantId,
    ParticipantRole,
)
from .domain.models import DomainModel
from .evaluation import DeterministicEvaluation, ParetoStatus
from .memory import AgentProfile
from .protocol import ProtocolPhase, TERMINAL_PHASES


LEARNER_SCHEMA_VERSION = "1.0"
CONTEXT_BUCKETS = 10


class LearnerError(ValueError):
    """Base error for deterministic learning failures."""


class LearnerFrozenError(LearnerError):
    """Raised when an evaluation-only learner is asked to update."""


class LearnerPersistenceError(LearnerError):
    """Raised when learner state cannot be safely persisted or restored."""


class LearnerMode(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class BanditAlgorithm(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"


class ContextIssueFeature(DomainModel):
    """A bounded feature derived from one public issue and the owner's preference."""

    issue_id: IssueId
    issue_kind: IssueKind
    weight_bucket: int = Field(ge=0, le=CONTEXT_BUCKETS)
    numeric_direction: Optional[str] = Field(default=None, pattern="^(minimize|maximize)$")


class StrategyContext(DomainModel):
    """Context containing public metadata and derived owner-only profile features."""

    owner_id: ParticipantId
    owner_role: ParticipantRole
    participant_count: int = Field(ge=2)
    numeric_issue_count: int = Field(ge=0)
    categorical_issue_count: int = Field(ge=0)
    reservation_bucket: int = Field(ge=0, le=CONTEXT_BUCKETS)
    batna_bucket: int = Field(ge=0, le=CONTEXT_BUCKETS)
    persona_present: bool
    issue_features: Tuple[ContextIssueFeature, ...] = Field(min_length=1)

    @field_validator("issue_features", mode="before")
    @classmethod
    def features_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def valid_feature_counts(self) -> "StrategyContext":
        if len({item.issue_id for item in self.issue_features}) != len(self.issue_features):
            raise ValueError("context issue features must be unique")
        numeric = sum(item.issue_kind is IssueKind.NUMERIC for item in self.issue_features)
        categorical = sum(item.issue_kind is IssueKind.CATEGORICAL for item in self.issue_features)
        if numeric != self.numeric_issue_count or categorical != self.categorical_issue_count:
            raise ValueError("context issue counts must match issue features")
        return self

    @property
    def key(self) -> str:
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bucket(value: float) -> int:
    return min(CONTEXT_BUCKETS, max(0, int(math.floor(value * CONTEXT_BUCKETS + 0.5))))


def build_strategy_context(
    scenario: NegotiationScenario,
    own_profile: AgentProfile,
) -> StrategyContext:
    """Build context without accepting or inspecting any opponent preference object."""

    owner = own_profile.identity.participant
    if scenario.participant(owner.participant_id) != owner:
        raise LearnerError("own profile identity must match the public scenario participant")
    preferences = own_profile.preferences
    public_issues = {issue.issue_id: issue for issue in scenario.issues}
    if {item.issue_id for item in preferences.issue_preferences} != set(public_issues):
        raise LearnerError("own preferences must cover exactly the public scenario issues")
    feature_by_issue = {}
    for preference in preferences.issue_preferences:
        public_issue = public_issues[preference.issue_id]
        if isinstance(preference, NumericPreference):
            direction = preference.direction.value
        elif isinstance(preference, CategoricalPreference):
            direction = None
        else:  # pragma: no cover - closed Pydantic union
            raise LearnerError("unsupported owner preference type")
        feature_by_issue[preference.issue_id] = ContextIssueFeature(
            issue_id=preference.issue_id,
            issue_kind=public_issue.kind,
            weight_bucket=_bucket(preference.weight),
            numeric_direction=direction,
        )
    features = tuple(feature_by_issue[issue.issue_id] for issue in scenario.issues)
    return StrategyContext(
        owner_id=owner.participant_id,
        owner_role=owner.role,
        participant_count=len(scenario.participants),
        numeric_issue_count=sum(issue.kind is IssueKind.NUMERIC for issue in scenario.issues),
        categorical_issue_count=sum(
            issue.kind is IssueKind.CATEGORICAL for issue in scenario.issues
        ),
        reservation_bucket=_bucket(preferences.reservation.reservation_utility),
        batna_bucket=_bucket(preferences.reservation.batna_utility),
        persona_present=bool(own_profile.identity.persona.strip()),
        issue_features=features,
    )


class RewardConfiguration(DomainModel):
    """Reward weights and cost scales.

    ``reward = own_utility_weight*U + agreement_weight*A + efficiency_weight*E
    + fairness_weight*F - cost_weight*C``. All component signals are in ``[0, 1]``.
    Weights are deliberately not normalized, so their absolute scale remains explicit.
    """

    own_utility_weight: float = Field(default=0.50, ge=0.0, allow_inf_nan=False)
    agreement_weight: float = Field(default=0.20, ge=0.0, allow_inf_nan=False)
    efficiency_weight: float = Field(default=0.15, ge=0.0, allow_inf_nan=False)
    fairness_weight: float = Field(default=0.10, ge=0.0, allow_inf_nan=False)
    cost_weight: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)
    message_cost_scale: int = Field(default=100, ge=1)
    model_call_cost_scale: int = Field(default=20, ge=1)
    latency_cost_scale_ms: float = Field(default=60_000.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def has_reward_signal(self) -> "RewardConfiguration":
        if math.isclose(
            self.own_utility_weight
            + self.agreement_weight
            + self.efficiency_weight
            + self.fairness_weight
            + self.cost_weight,
            0.0,
        ):
            raise ValueError("at least one reward weight must be positive")
        return self


class RewardBreakdown(DomainModel):
    participant_id: ParticipantId
    own_utility: float = Field(ge=0.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    fairness: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    reward: float = Field(allow_inf_nan=False)


class CompletedEpisodeOutcome(DomainModel):
    """Terminal episode evidence; rewards are always derived from its objective metrics."""

    terminal_phase: ProtocolPhase
    evaluation: DeterministicEvaluation

    @model_validator(mode="after")
    def is_completed(self) -> "CompletedEpisodeOutcome":
        if self.terminal_phase not in TERMINAL_PHASES:
            raise ValueError("learning outcomes must come from a terminal episode")
        if (
            self.terminal_phase is ProtocolPhase.AGREED
        ) != self.evaluation.agreement_reached:
            raise ValueError("terminal phase and agreement metric must agree")
        return self


def compose_reward(
    evaluation: DeterministicEvaluation,
    participant_id: ParticipantId,
    configuration: Optional[RewardConfiguration] = None,
) -> RewardBreakdown:
    config = configuration or RewardConfiguration()
    metric = next(
        (
            item
            for item in evaluation.participant_utilities
            if item.participant_id == participant_id
        ),
        None,
    )
    if metric is None:
        raise LearnerError("evaluation has no utility metric for the learner participant")
    agreement = 1.0 if evaluation.agreement_valid else 0.0
    if evaluation.agreement_valid and evaluation.pareto.status is ParetoStatus.COMPUTED:
        maximum_distance = math.sqrt(len(evaluation.participant_utilities))
        efficiency = max(
            0.0,
            1.0 - (evaluation.pareto.distance_to_frontier or 0.0) / maximum_distance,
        )
    else:
        efficiency = 0.0
    fairness = evaluation.utility_balance if evaluation.utility_balance is not None else 0.0
    telemetry = evaluation.telemetry
    cost = min(
        1.0,
        (
            min(1.0, telemetry.message_count / config.message_cost_scale)
            + min(1.0, telemetry.model_call_count / config.model_call_cost_scale)
            + min(1.0, telemetry.model_latency_ms / config.latency_cost_scale_ms)
        )
        / 3.0,
    )
    reward = (
        config.own_utility_weight * metric.outcome_utility
        + config.agreement_weight * agreement
        + config.efficiency_weight * efficiency
        + config.fairness_weight * fairness
        - config.cost_weight * cost
    )
    return RewardBreakdown(
        participant_id=participant_id,
        own_utility=metric.outcome_utility,
        agreement=agreement,
        efficiency=efficiency,
        fairness=fairness,
        cost=cost,
        reward=reward,
    )


class BanditConfiguration(DomainModel):
    algorithm: BanditAlgorithm
    epsilon: float = Field(default=0.10, ge=0.0, le=1.0, allow_inf_nan=False)
    ucb_exploration: float = Field(default=math.sqrt(2.0), ge=0.0, allow_inf_nan=False)


class ArmStatistics(DomainModel):
    strategy: str = Field(min_length=1, max_length=100)
    observations: int = Field(default=0, ge=0)
    total_reward: float = Field(default=0.0, allow_inf_nan=False)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.observations if self.observations else 0.0


class ContextStatistics(DomainModel):
    context: StrategyContext
    selections: int = Field(default=0, ge=0)
    arms: Tuple[ArmStatistics, ...]
    pending_selections: Tuple[str, ...] = ()

    @field_validator("arms", "pending_selections", mode="before")
    @classmethod
    def arms_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class LearnerState(DomainModel):
    schema_version: str = LEARNER_SCHEMA_VERSION
    algorithm: BanditAlgorithm
    strategies: Tuple[str, ...] = Field(min_length=2)
    seed: int
    mode: LearnerMode
    bandit_configuration: BanditConfiguration
    reward_configuration: RewardConfiguration
    contexts: Tuple[ContextStatistics, ...] = ()

    @field_validator("strategies", "contexts", mode="before")
    @classmethod
    def tuples_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def consistent_state(self) -> "LearnerState":
        if self.schema_version != LEARNER_SCHEMA_VERSION:
            raise ValueError(f"unsupported learner schema version '{self.schema_version}'")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("registered strategies must be unique")
        if any(not item.strip() for item in self.strategies):
            raise ValueError("registered strategies must not be blank")
        if self.algorithm is not self.bandit_configuration.algorithm:
            raise ValueError("learner algorithm and configuration must match")
        keys = tuple(item.context.key for item in self.contexts)
        if len(set(keys)) != len(keys):
            raise ValueError("learner contexts must be unique")
        for context in self.contexts:
            if tuple(item.strategy for item in context.arms) != self.strategies:
                raise ValueError("every context must contain every registered strategy in order")
            if not set(context.pending_selections).issubset(set(self.strategies)):
                raise ValueError("pending selections must use registered strategies")
            if len(context.pending_selections) > 1:
                raise ValueError("a context can have at most one unresolved training episode")
            observations = sum(item.observations for item in context.arms)
            if observations + len(context.pending_selections) != context.selections:
                raise ValueError("every selection must be observed or pending")
        return self


@runtime_checkable
class StrategyLearner(Protocol):
    def select_strategy(self, context: StrategyContext) -> str:
        """Select one registered strategy without changing protocol state."""

    def observe_outcome(
        self,
        context: StrategyContext,
        strategy: str,
        outcome: CompletedEpisodeOutcome,
    ) -> RewardBreakdown:
        """Update learner state from one completed episode in training mode."""

    def save(self, path: Union[str, Path]) -> None:
        """Atomically persist versioned learner state."""

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StrategyLearner":
        """Load a validated learner with identical future behavior."""


LearnerT = TypeVar("LearnerT", bound="ContextualBanditLearner")


class ContextualBanditLearner:
    """Context-specific epsilon-greedy or UCB learner with deterministic tie-breaking."""

    expected_algorithm: ClassVar[Optional[BanditAlgorithm]] = None

    def __init__(
        self,
        strategies: Tuple[str, ...],
        *,
        bandit_configuration: BanditConfiguration,
        reward_configuration: Optional[RewardConfiguration] = None,
        seed: int = 0,
        mode: LearnerMode = LearnerMode.TRAINING,
    ) -> None:
        self._state = LearnerState(
            algorithm=bandit_configuration.algorithm,
            strategies=strategies,
            seed=seed,
            mode=mode,
            bandit_configuration=bandit_configuration,
            reward_configuration=reward_configuration or RewardConfiguration(),
        )
        self._validate_expected_algorithm()

    @property
    def state(self) -> LearnerState:
        return self._state

    @property
    def mode(self) -> LearnerMode:
        return self._state.mode

    def set_mode(self, mode: LearnerMode) -> None:
        if not isinstance(mode, LearnerMode):
            raise LearnerError("mode must be LearnerMode.TRAINING or LearnerMode.EVALUATION")
        if mode is LearnerMode.EVALUATION and any(
            context.pending_selections for context in self._state.contexts
        ):
            raise LearnerError(
                "cannot enter evaluation mode with an unresolved training episode"
            )
        self._state = self._state.model_copy(update={"mode": mode})

    def select_strategy(self, context: StrategyContext) -> str:
        statistics = self._context_statistics(context, create=self.mode is LearnerMode.TRAINING)
        if self.mode is LearnerMode.TRAINING and statistics.pending_selections:
            raise LearnerError(
                "observe the completed outcome before selecting another strategy "
                "for this context"
            )
        strategy = (
            self._epsilon_greedy(statistics)
            if self._state.algorithm is BanditAlgorithm.EPSILON_GREEDY
            else self._ucb(statistics)
        )
        if self.mode is LearnerMode.TRAINING:
            updated = statistics.model_copy(
                update={
                    "selections": statistics.selections + 1,
                    "pending_selections": statistics.pending_selections + (strategy,),
                }
            )
            self._replace_context(updated)
        return strategy

    def observe_outcome(
        self,
        context: StrategyContext,
        strategy: str,
        outcome: CompletedEpisodeOutcome,
    ) -> RewardBreakdown:
        if self.mode is LearnerMode.EVALUATION:
            raise LearnerFrozenError("evaluation/frozen learners cannot observe outcomes")
        if strategy not in self._state.strategies:
            raise LearnerError(f"strategy '{strategy}' is not registered")
        statistics = self._context_statistics(context, create=False)
        if strategy not in statistics.pending_selections:
            raise LearnerError(
                "outcome strategy was not selected or was already observed for this context"
            )
        reward = compose_reward(
            outcome.evaluation,
            context.owner_id,
            self._state.reward_configuration,
        )
        pending = list(statistics.pending_selections)
        pending.remove(strategy)
        arms = tuple(
            item.model_copy(
                update={
                    "observations": item.observations + 1,
                    "total_reward": item.total_reward + reward.reward,
                }
            )
            if item.strategy == strategy
            else item
            for item in statistics.arms
        )
        self._replace_context(
            statistics.model_copy(
                update={"arms": arms, "pending_selections": tuple(pending)}
            )
        )
        return reward

    def save(self, path: Union[str, Path]) -> None:
        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise LearnerPersistenceError("learner state path must be a file")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(self._state.model_dump_json(indent=2))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        except (OSError, ValueError) as exc:
            raise LearnerPersistenceError(
                f"could not atomically save learner state to '{destination}'"
            ) from exc

    @classmethod
    def load(cls: type[LearnerT], path: Union[str, Path]) -> LearnerT:
        source = Path(path)
        try:
            state = LearnerState.model_validate_json(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise LearnerPersistenceError(
                f"could not load valid learner state from '{source}'"
            ) from exc
        target_class: type[ContextualBanditLearner]
        if cls is ContextualBanditLearner:
            target_class = (
                EpsilonGreedyLearner
                if state.algorithm is BanditAlgorithm.EPSILON_GREEDY
                else UCBLearner
            )
        else:
            target_class = cls
        learner = target_class.__new__(target_class)
        learner._state = state
        learner._validate_expected_algorithm()
        return learner

    def _validate_expected_algorithm(self) -> None:
        if self.expected_algorithm is not None and self._state.algorithm is not self.expected_algorithm:
            raise LearnerPersistenceError(
                f"saved algorithm '{self._state.algorithm.value}' cannot load as "
                f"'{self.expected_algorithm.value}'"
            )

    def _context_statistics(
        self, context: StrategyContext, *, create: bool
    ) -> ContextStatistics:
        existing = next(
            (item for item in self._state.contexts if item.context.key == context.key),
            None,
        )
        if existing is not None:
            return existing
        empty = ContextStatistics(
            context=context,
            arms=tuple(ArmStatistics(strategy=item) for item in self._state.strategies),
        )
        if create:
            self._state = self._state.model_copy(
                update={"contexts": self._state.contexts + (empty,)}
            )
        return empty

    def _replace_context(self, replacement: ContextStatistics) -> None:
        contexts = tuple(
            replacement if item.context.key == replacement.context.key else item
            for item in self._state.contexts
        )
        if not any(item.context.key == replacement.context.key for item in self._state.contexts):
            contexts = contexts + (replacement,)
        self._state = self._state.model_copy(update={"contexts": contexts})

    def _epsilon_greedy(self, statistics: ContextStatistics) -> str:
        config = self._state.bandit_configuration
        explore_roll = self._deterministic_fraction(statistics.context.key, statistics.selections, "roll")
        if self.mode is LearnerMode.TRAINING and explore_roll < config.epsilon:
            index_fraction = self._deterministic_fraction(
                statistics.context.key, statistics.selections, "arm"
            )
            index = min(len(statistics.arms) - 1, int(index_fraction * len(statistics.arms)))
            return statistics.arms[index].strategy
        return max(
            statistics.arms,
            key=lambda item: (item.mean_reward, -self._state.strategies.index(item.strategy)),
        ).strategy

    def _ucb(self, statistics: ContextStatistics) -> str:
        if self.mode is LearnerMode.EVALUATION:
            return max(
                statistics.arms,
                key=lambda item: (
                    item.mean_reward,
                    -self._state.strategies.index(item.strategy),
                ),
            ).strategy
        unobserved = next((item for item in statistics.arms if item.observations == 0), None)
        if unobserved is not None:
            return unobserved.strategy
        total = sum(item.observations for item in statistics.arms)
        coefficient = self._state.bandit_configuration.ucb_exploration
        return max(
            statistics.arms,
            key=lambda item: (
                item.mean_reward
                + coefficient * math.sqrt(math.log(total) / item.observations),
                -self._state.strategies.index(item.strategy),
            ),
        ).strategy

    def _deterministic_fraction(self, context_key: str, selection: int, purpose: str) -> float:
        payload = f"{self._state.seed}|{context_key}|{selection}|{purpose}"
        integer = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
        return integer / 2**64


class EpsilonGreedyLearner(ContextualBanditLearner):
    expected_algorithm = BanditAlgorithm.EPSILON_GREEDY

    def __init__(
        self,
        strategies: Tuple[str, ...],
        *,
        epsilon: float = 0.10,
        reward_configuration: Optional[RewardConfiguration] = None,
        seed: int = 0,
        mode: LearnerMode = LearnerMode.TRAINING,
    ) -> None:
        super().__init__(
            strategies,
            bandit_configuration=BanditConfiguration(
                algorithm=BanditAlgorithm.EPSILON_GREEDY,
                epsilon=epsilon,
            ),
            reward_configuration=reward_configuration,
            seed=seed,
            mode=mode,
        )


class UCBLearner(ContextualBanditLearner):
    expected_algorithm = BanditAlgorithm.UCB

    def __init__(
        self,
        strategies: Tuple[str, ...],
        *,
        exploration: float = math.sqrt(2.0),
        reward_configuration: Optional[RewardConfiguration] = None,
        seed: int = 0,
        mode: LearnerMode = LearnerMode.TRAINING,
    ) -> None:
        super().__init__(
            strategies,
            bandit_configuration=BanditConfiguration(
                algorithm=BanditAlgorithm.UCB,
                ucb_exploration=exploration,
            ),
            reward_configuration=reward_configuration,
            seed=seed,
            mode=mode,
        )


class LegacyRewardScorer:
    """Compatibility-only CLI score; it is deliberately not described as learning."""

    def score_result(self, evaluation: dict[str, Any]) -> int:
        score = 0
        if evaluation.get("deal_reached"):
            score += 10
            if evaluation.get("fairness_score", 0) >= 60:
                score += 5
            if evaluation.get("rounds_used", 999) <= 3:
                score += 3
        else:
            score -= 5
        if evaluation.get("deal_reached") and evaluation.get("fairness_score", 0) < 40:
            score -= 10
        return score
