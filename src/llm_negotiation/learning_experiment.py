"""Deterministic learning-curve experiment against one fixed opponent policy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from pydantic import Field, field_validator

from .agents import BuyerAgent, SellerAgent
from .communication import MessageBus
from .domain import ParticipantId, price_only_from_legacy
from .domain.legacy import BUYER_ID, SELLER_ID
from .domain.models import DomainModel
from .learning import (
    CompletedEpisodeOutcome,
    EpsilonGreedyLearner,
    LearnerMode,
    RewardConfiguration,
    StrategyContext,
    build_strategy_context,
)
from .memory import AgentMemory, AgentProfile, PublicAgentIdentity
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


EXPERIMENT_STRATEGIES = ("fixed", "boulware", "linear", "conceder")


class LearningCurvePoint(DomainModel):
    episode: int = Field(ge=1)
    selected_strategy: str
    reward: float = Field(allow_inf_nan=False)
    cumulative_mean_reward: float = Field(allow_inf_nan=False)
    agreement: bool


class LearnedArmResult(DomainModel):
    strategy: str
    observations: int = Field(ge=0)
    mean_reward: float = Field(allow_inf_nan=False)


class LearningExperimentResult(DomainModel):
    seed: int
    episodes: int = Field(ge=1)
    fixed_opponent_strategy: str
    context: StrategyContext
    curve: Tuple[LearningCurvePoint, ...]
    arms: Tuple[LearnedArmResult, ...]
    frozen_strategy: str

    @field_validator("curve", "arms", mode="before")
    @classmethod
    def tuples_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class _ExperimentClock:
    def now(self) -> datetime:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return 0.0


def _policy(name: str) -> NegotiationPolicy:
    policies = {
        "fixed": FixedPolicy,
        "boulware": BoulwarePolicy,
        "linear": LinearConcessionPolicy,
        "conceder": ConcederPolicy,
    }
    try:
        return policies[name]()
    except KeyError as exc:  # pragma: no cover - experiment uses a closed registry
        raise ValueError(f"unknown experiment strategy '{name}'") from exc


def _scenario_and_profiles():
    migration = price_only_from_legacy(
        "Learning benchmark item",
        BuyerAgent(
            name="Adaptive buyer",
            role="buyer",
            strategy="neutral",
            max_price=120.0,
            current_offer=40.0,
        ),
        SellerAgent(
            name="Fixed seller",
            role="seller",
            strategy="neutral",
            initial_price=160.0,
            min_acceptable=80.0,
        ),
        rounds=8,
    )
    scenario = migration.scenario
    profiles = {
        BUYER_ID: AgentProfile(
            identity=PublicAgentIdentity(
                participant=scenario.participant(BUYER_ID),
                persona="Adaptive but reservation-respecting buyer",
            ),
            preferences=migration.buyer_preferences,
        ),
        SELLER_ID: AgentProfile(
            identity=PublicAgentIdentity(
                participant=scenario.participant(SELLER_ID),
                persona="Fixed deterministic seller",
            ),
            preferences=migration.seller_preferences,
        ),
    }
    return scenario, profiles


def _run_episode(
    strategy: str,
    scenario,
    profiles,
):
    participant_ids = (BUYER_ID, SELLER_ID)
    policies = {
        BUYER_ID: _policy(strategy),
        SELLER_ID: LinearConcessionPolicy(),
    }
    memories = {
        participant_id: AgentMemory(profiles[participant_id], scenario)
        for participant_id in participant_ids
    }
    return NegotiationOrchestrator(
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
        configuration=OrchestratorConfiguration(maximum_rounds=8),
    ).run()


def run_learning_curve_experiment(
    *,
    episodes: int = 24,
    seed: int = 17,
    epsilon: float = 0.20,
    reward_configuration: Optional[RewardConfiguration] = None,
) -> LearningExperimentResult:
    """Train against one fixed opponent; results are illustrative, not generalization evidence."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    scenario, profiles = _scenario_and_profiles()
    context = build_strategy_context(scenario, profiles[BUYER_ID])
    learner = EpsilonGreedyLearner(
        EXPERIMENT_STRATEGIES,
        epsilon=epsilon,
        reward_configuration=reward_configuration,
        seed=seed,
    )
    curve = []
    reward_sum = 0.0
    for episode_number in range(1, episodes + 1):
        selected = learner.select_strategy(context)
        result = _run_episode(selected, scenario, profiles)
        reward = learner.observe_outcome(
            context,
            selected,
            CompletedEpisodeOutcome(
                terminal_phase=result.final_session.phase,
                evaluation=result.metrics,
            ),
        )
        reward_sum += reward.reward
        curve.append(
            LearningCurvePoint(
                episode=episode_number,
                selected_strategy=selected,
                reward=reward.reward,
                cumulative_mean_reward=reward_sum / episode_number,
                agreement=result.metrics.agreement_valid,
            )
        )

    learner.set_mode(LearnerMode.EVALUATION)
    frozen_strategy = learner.select_strategy(context)
    statistics = learner.state.contexts[0]
    return LearningExperimentResult(
        seed=seed,
        episodes=episodes,
        fixed_opponent_strategy="linear",
        context=context,
        curve=tuple(curve),
        arms=tuple(
            LearnedArmResult(
                strategy=item.strategy,
                observations=item.observations,
                mean_reward=item.mean_reward,
            )
            for item in statistics.arms
        ),
        frozen_strategy=frozen_strategy,
    )
