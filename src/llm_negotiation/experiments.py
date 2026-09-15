"""Offline-first reproducible experiment configuration, execution, and artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Protocol, Sequence, Tuple

from pydantic import Field, field_validator, model_validator

from .communication import MessageBus
from .beliefs import OpponentModelConfiguration, OpponentModelMode
from .domain import (
    AcceptAction,
    ActionType,
    CounterAction,
    NegotiationAction,
    Offer,
    ParticipantId,
    ProposeAction,
    RejectAction,
    WithdrawAction,
    calculate_utility,
)
from .domain.models import DomainModel
from .experiment_dataset import (
    SCENARIO_DATASET_VERSION,
    ExperimentScenario,
    scenario_index,
    team_size_case,
)
from .learning_experiment import LearningExperimentResult, run_learning_curve_experiment
from .llm import (
    FakeLLMClient,
    FakeResponse,
    FakeRule,
    LLMProvider,
    LLMRequest,
    LLMClient,
    LLMUsage,
    ModelConfiguration,
    OpenAILLMClient,
    RetryConfiguration,
)
from .llm_policy import (
    AgentVisibleObservation,
    LLMNegotiationPolicy,
    NegotiationPlan,
    ObjectiveConstraints,
    OpponentHypothesis,
    StructuredActionDecision,
)
from .mediation import DeterministicMediator, MediatorConfiguration
from .memory import AgentMemory, AgentProfile, MemoryLimits, MemorySnapshot
from .multiparty import (
    AcceptanceRule,
    DeliberativePolicy,
    MultipartyProtocolConfiguration,
    MultipartyProtocolFactory,
)
from .opponent import HeuristicOpponentModeller
from .orchestration import (
    AUDIT_READER_ID,
    EpisodeResult,
    FailurePolicy,
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
    SeededRandomPolicy,
    generate_offer_for_utility,
    maximum_attainable_utility,
)
from .verification import (
    VerificationConfiguration,
    VerificationCoordinator,
    VerificationMode,
)


EXPERIMENT_SCHEMA_VERSION = "1.0"
ARTIFACT_SCHEMA_VERSION = "1.0"
NORMAL_95 = 1.959963984540054


class ExperimentError(RuntimeError):
    """Raised for invalid experiment execution or artifact reconstruction."""


class BaselineKind(str, Enum):
    DETERMINISTIC = "deterministic"
    SINGLE_LLM = "single_llm"
    SAMPLE_AND_VOTE = "independent_sample_and_vote"
    HOMOGENEOUS_TEAM = "homogeneous_team"
    HETEROGENEOUS_TEAM = "heterogeneous_team"


class EpisodeRecordStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class ExperimentModelSpec(DomainModel):
    provider: LLMProvider = LLMProvider.FAKE
    model: str = Field(default="offline-structured-fake-v1", min_length=1, max_length=200)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    timeout_seconds: float = Field(default=5.0, gt=0.0, le=600.0, allow_inf_nan=False)


class ExperimentBudgets(DomainModel):
    episode_timeout_seconds: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    action_timeout_seconds: float = Field(default=10.0, gt=0.0, allow_inf_nan=False)
    maximum_model_calls: int = Field(default=100, ge=0)
    maximum_actions: Optional[int] = Field(default=None, ge=1)
    maximum_total_tokens: int = Field(default=100_000, ge=0)
    estimated_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class EnabledComponents(DomainModel):
    memory: bool = True
    theory_of_mind: bool = True
    verifier: bool = True
    mediator: bool = False
    communication: bool = True
    team_size: Optional[int] = Field(default=None, ge=2, le=4)
    deliberation_depth: int = Field(default=1, ge=1, le=20)
    confidence_visibility: bool = True


class ExperimentConfiguration(DomainModel):
    """Complete immutable provenance for one evaluation partition."""

    schema_version: Literal["1.0"] = EXPERIMENT_SCHEMA_VERSION
    experiment_id: str = Field(min_length=1, max_length=100)
    dataset_version: Literal["1.0"] = SCENARIO_DATASET_VERSION
    scenario_ids: Tuple[str, ...] = Field(min_length=1)
    baselines: Tuple[BaselineKind, ...] = Field(min_length=1)
    deterministic_policy_names: Tuple[str, ...] = Field(
        default=("linear", "boulware"), min_length=1
    )
    models: Tuple[ExperimentModelSpec, ...] = Field(
        default=(ExperimentModelSpec(),)
    )
    seeds: Tuple[int, ...] = Field(default=(0,), min_length=1)
    rounds: int = Field(default=6, ge=1, le=1000)
    budgets: ExperimentBudgets = Field(default_factory=ExperimentBudgets)
    enabled_components: EnabledComponents = Field(default_factory=EnabledComponents)
    evaluation_partition: str = Field(default="evaluation", pattern="^evaluation$")

    @field_validator(
        "scenario_ids", "baselines", "deterministic_policy_names", "models", "seeds",
        mode="before",
    )
    @classmethod
    def sequences_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def valid_configuration(self) -> "ExperimentConfiguration":
        for name in ("scenario_ids", "baselines", "seeds"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
        model_keys = {
            (item.provider, item.model, item.temperature) for item in self.models
        }
        if len(model_keys) != len(self.models):
            raise ValueError("models must be unique by provider, model, and temperature")
        if any(not item.strip() for item in self.deterministic_policy_names):
            raise ValueError("deterministic policy names must be nonblank")
        supported = {"fixed", "linear", "boulware", "conceder", "seeded_random"}
        unknown_policies = set(self.deterministic_policy_names) - supported
        if unknown_policies:
            raise ValueError(
                f"unsupported deterministic policies: {sorted(unknown_policies)}"
            )
        available = set(scenario_index())
        unknown = set(self.scenario_ids) - available
        if unknown:
            raise ValueError(f"unknown scenario ids: {sorted(unknown)}")
        if BaselineKind.SINGLE_LLM in self.baselines and not self.models:
            raise ValueError("single-LLM baseline requires a model specification")
        if self.enabled_components.team_size is not None and len(self.scenario_ids) != 1:
            raise ValueError("team-size ablations require exactly one source scenario id")
        return self

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EpisodeMetricRecord(DomainModel):
    agreement: float = Field(ge=0.0, le=1.0)
    mean_utility: float = Field(ge=0.0, le=1.0)
    social_welfare: float = Field(ge=0.0)
    fairness: float = Field(ge=0.0, le=1.0)
    rounds: float = Field(ge=0.0)
    invalid_action_rate: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost: float = Field(ge=0.0)


class ExperimentEpisodeRecord(DomainModel):
    schema_version: Literal["1.0"] = ARTIFACT_SCHEMA_VERSION
    configuration_hash: str = Field(pattern="^[0-9a-f]{64}$")
    git_commit: str = Field(min_length=1, max_length=100)
    timestamp_utc: datetime
    experiment_id: str
    dataset_version: Literal["1.0"]
    scenario_id: str
    baseline: BaselineKind
    seed: int
    model: Optional[str] = Field(default=None, min_length=1, max_length=200)
    temperature: float = Field(ge=0.0, le=2.0)
    components: EnabledComponents
    partition: str = Field(pattern="^evaluation$")
    status: EpisodeRecordStatus
    metrics: Optional[EpisodeMetricRecord] = None
    episode_result: Optional[EpisodeResult] = None
    failure_code: Optional[str] = Field(default=None, max_length=100)
    failure_message: Optional[str] = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def payload_matches_status(self) -> "ExperimentEpisodeRecord":
        succeeded = self.status is EpisodeRecordStatus.SUCCESS
        if succeeded != (self.metrics is not None and self.episode_result is not None):
            raise ValueError("successful records require metrics and raw episode result")
        if succeeded and (self.failure_code is not None or self.failure_message is not None):
            raise ValueError("successful records cannot contain failure details")
        if not succeeded and not self.failure_code:
            raise ValueError("non-success records require an explicit failure code")
        return self


class ConfidenceInterval(DomainModel):
    mean: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    sample_size: int = Field(ge=0)
    method: str = "normal_approximation_95_percent"


class PairedComparison(DomainModel):
    configuration_hash: str = Field(pattern="^[0-9a-f]{64}$")
    baseline_a: BaselineKind
    model_a: Optional[str] = None
    temperature_a: float = Field(ge=0.0, le=2.0)
    baseline_b: BaselineKind
    model_b: Optional[str] = None
    temperature_b: float = Field(ge=0.0, le=2.0)
    metric: str
    paired_seeded_cases: int = Field(ge=0)
    difference: ConfidenceInterval


class AggregateResult(DomainModel):
    configuration_hash: str = Field(pattern="^[0-9a-f]{64}$")
    git_commit: str = Field(min_length=1, max_length=100)
    timestamp_utc: datetime
    seeds: Tuple[int, ...]
    baseline: BaselineKind
    model: Optional[str] = None
    temperature: float = Field(ge=0.0, le=2.0)
    successful_episodes: int = Field(ge=0)
    failed_episodes: int = Field(ge=0)
    not_applicable_episodes: int = Field(ge=0)
    intervals: Mapping[str, ConfidenceInterval]

    @field_validator("seeds", mode="before")
    @classmethod
    def seeds_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ExperimentRunResult(DomainModel):
    configuration: ExperimentConfiguration
    output_directory: str
    episode_jsonl: str
    aggregate_csv: str
    paired_csv: str
    ablation_csv: str
    component_jsonl: str
    learning_curve_json: str
    plot_files: Tuple[str, ...]
    records: Tuple[ExperimentEpisodeRecord, ...]

    @field_validator("plot_files", "records", mode="before")
    @classmethod
    def tuples_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class AblationSuiteResult(DomainModel):
    configuration_hashes: Tuple[str, ...]
    episode_jsonl: str
    ablation_csv: str
    cell_directories: Tuple[str, ...]

    @field_validator("configuration_hashes", "cell_directories", mode="before")
    @classmethod
    def tuples_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ExperimentClock:
    """Fixed clock keeps simulated latency and message timestamps deterministic."""

    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("experiment clock requires a timezone-aware instant")
        self.instant = instant

    def now(self) -> datetime:
        return self.instant

    def monotonic(self) -> float:
        return 0.0


class SampleAndVotePolicy:
    """Independent policy samples followed by deterministic action-type majority vote."""

    name = "independent-sample-and-vote"

    def __init__(self, samples: Sequence[NegotiationPolicy]) -> None:
        if len(samples) < 3:
            raise ValueError("sample-and-vote requires at least three independent samples")
        self.samples = tuple(samples)

    def choose_action(self, memory: MemorySnapshot) -> NegotiationAction:
        candidates = tuple(sample.choose_action(memory) for sample in self.samples)
        counts = Counter(candidate.action for candidate in candidates)
        winning_type = max(
            counts,
            key=lambda action_type: (
                counts[action_type],
                -next(
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.action is action_type
                ),
            ),
        )
        return next(candidate for candidate in candidates if candidate.action is winning_type)


class ExperimentLLMPolicy(LLMNegotiationPolicy):
    """LLM policy with an explicit opponent-confidence visibility ablation."""

    def __init__(self, *args: object, confidence_visible: bool = True, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.confidence_visible = confidence_visible

    def _visible_beliefs(self, beliefs: tuple) -> tuple:
        if self.confidence_visible:
            return beliefs
        # A constant neutral value removes calibrated confidence as an experimental
        # signal without pretending the underlying hypotheses are ground-truth facts.
        return tuple(item.model_copy(update={"confidence": 0.5}) for item in beliefs)

    def form_opponent_hypothesis(
        self, observation: object, objective: object, beliefs: tuple, telemetry: object, recovery: object
    ) -> OpponentHypothesis:
        return super().form_opponent_hypothesis(
            observation, objective, self._visible_beliefs(beliefs), telemetry, recovery
        )

    def choose_negotiation_plan(
        self,
        observation: object,
        objective: object,
        hypothesis: OpponentHypothesis,
        beliefs: tuple,
        telemetry: object,
        recovery: object,
    ) -> NegotiationPlan:
        visible_hypothesis = (
            hypothesis
            if self.confidence_visible
            else hypothesis.model_copy(update={"confidence": 0.5})
        )
        return super().choose_negotiation_plan(
            observation,
            objective,
            visible_hypothesis,
            self._visible_beliefs(beliefs),
            telemetry,
            recovery,
        )


def _offline_fake_client(spec: ExperimentModelSpec) -> FakeLLMClient:
    """Return a strict structured fake that exercises the actual LLM policy boundary."""

    def responder(request: LLMRequest) -> FakeResponse:
        payload = json.loads(request.messages[-1].content)
        stage = payload["stage"]
        context = payload["participant_visible_context"]
        common = {
            "schema_version": "1.0",
            "decision_rationale": "Deterministic offline structured decision.",
            "evidence": ("participant-visible experiment context",),
        }
        if stage == "objective_constraints":
            output: DomainModel = ObjectiveConstraints(
                **common,
                objective="Reach a valid agreement without violating reservation utility.",
                constraints=("Use only legal typed actions.",),
            )
        elif stage == "opponent_hypothesis":
            output = OpponentHypothesis(
                **common,
                hypothesis="The counterparty may concede within the public deadline.",
                confidence=0.25,
            )
        elif stage == "planning":
            output = NegotiationPlan(
                **common,
                plan=("Use a reservation-respecting midpoint aspiration.",),
                target_utility=0.6,
            )
        elif stage == "action_generation":
            observation = AgentVisibleObservation.model_validate_json(
                json.dumps(context["observation"])
            )
            preferences = observation.own_preferences
            reservation = preferences.reservation.reservation_utility
            maximum = maximum_attainable_utility(observation.scenario, preferences)
            # Temperature remains a recorded experimental factor while this offline
            # surrogate converts it into deterministic aspiration diversity.
            aspiration_fraction = max(0.1, min(0.9, 0.5 - 0.1 * spec.temperature))
            target = reservation + aspiration_fraction * (maximum - reservation)
            outstanding = observation.outstanding_offer
            if (
                outstanding is not None
                and ActionType.ACCEPT in observation.legal_actions
                and calculate_utility(observation.scenario, outstanding, preferences) >= target
            ):
                action: NegotiationAction = AcceptAction(
                    actor_id=observation.participant_id,
                    recipients=observation.expected_recipients,
                    round_number=observation.round_number,
                    offer_id=outstanding.offer_id,
                )
            elif ActionType.PROPOSE in observation.legal_actions or ActionType.COUNTER in observation.legal_actions:
                if observation.required_offer_id is None:
                    raise ExperimentError("structured fake requires a grounded offer id")
                offer = generate_offer_for_utility(
                    observation.scenario,
                    preferences,
                    target,
                    observation.required_offer_id,
                )
                if outstanding is None:
                    action = ProposeAction(
                        actor_id=observation.participant_id,
                        recipients=observation.expected_recipients,
                        round_number=observation.round_number,
                        offer=offer,
                    )
                else:
                    action = CounterAction(
                        actor_id=observation.participant_id,
                        recipients=observation.expected_recipients,
                        round_number=observation.round_number,
                        offer=offer,
                        responds_to=outstanding.offer_id,
                    )
            elif outstanding is not None and ActionType.REJECT in observation.legal_actions:
                action = RejectAction(
                    actor_id=observation.participant_id,
                    recipients=observation.expected_recipients,
                    round_number=observation.round_number,
                    offer_id=outstanding.offer_id,
                )
            else:
                action = WithdrawAction(
                    actor_id=observation.participant_id,
                    recipients=observation.expected_recipients,
                    round_number=observation.round_number,
                    reason="No safe structured action remained.",
                )
            output = StructuredActionDecision(**common, action=action)
        else:
            raise ExperimentError(f"unsupported offline fake stage '{stage}'")
        return FakeResponse(
            text=output.model_dump_json(),
            usage=LLMUsage(input_tokens=20, output_tokens=10, total_tokens=30),
            model=request.model_configuration.model,
        )

    return FakeLLMClient(
        rules=(FakeRule(predicate=lambda _request: True, responder=responder),),
        clock=lambda: 0.0,
    )


def _model_configuration(spec: ExperimentModelSpec) -> ModelConfiguration:
    return ModelConfiguration(
        provider=spec.provider,
        model=spec.model,
        timeout_seconds=spec.timeout_seconds,
        temperature=spec.temperature,
        retry=RetryConfiguration(max_retries=0),
    )


def _base_policy(index: int, seed: int, heterogeneous: bool) -> NegotiationPolicy:
    if not heterogeneous:
        return LinearConcessionPolicy()
    policies: Tuple[NegotiationPolicy, ...] = (
        BoulwarePolicy(),
        LinearConcessionPolicy(),
        ConcederPolicy(),
        SeededRandomPolicy(seed=seed),
    )
    return policies[index % len(policies)]


def _named_policy(name: str, seed: int) -> NegotiationPolicy:
    factories = {
        "fixed": FixedPolicy,
        "linear": LinearConcessionPolicy,
        "boulware": BoulwarePolicy,
        "conceder": ConcederPolicy,
    }
    if name == "seeded_random":
        return SeededRandomPolicy(seed=seed)
    return factories[name]()


def _policies_for(
    case: ExperimentScenario,
    baseline: BaselineKind,
    seed: int,
    config: ExperimentConfiguration,
    model_client: Optional[LLMClient],
    model_spec: Optional[ExperimentModelSpec],
) -> Mapping[ParticipantId, NegotiationPolicy]:
    participants = case.scenario.participants
    components = config.enabled_components
    policies: Dict[ParticipantId, NegotiationPolicy] = {}
    for index, participant in enumerate(participants):
        if baseline is BaselineKind.DETERMINISTIC:
            policy: NegotiationPolicy = _named_policy(
                config.deterministic_policy_names[
                    index % len(config.deterministic_policy_names)
                ],
                seed,
            )
        elif baseline is BaselineKind.SAMPLE_AND_VOTE:
            policy = SampleAndVotePolicy(
                (BoulwarePolicy(), LinearConcessionPolicy(), ConcederPolicy())
            )
        elif baseline is BaselineKind.HOMOGENEOUS_TEAM:
            policy = _base_policy(index, seed, heterogeneous=False)
        elif baseline is BaselineKind.HETEROGENEOUS_TEAM:
            policy = _base_policy(index, seed, heterogeneous=True)
        elif baseline is BaselineKind.SINGLE_LLM:
            if index == 0:
                if model_client is None or model_spec is None:
                    raise ExperimentError("single-LLM baseline requires a model gateway")
                policy = ExperimentLLMPolicy(
                    model_client,
                    _model_configuration(model_spec),
                    confidence_visible=components.confidence_visibility,
                    fallback_policy=LinearConcessionPolicy(),
                    opponent_configuration=OpponentModelConfiguration(
                        mode=(
                            OpponentModelMode.HEURISTIC
                            if components.theory_of_mind
                            else OpponentModelMode.DISABLED
                        ),
                        minimum_planning_confidence=(
                            0.35 if components.confidence_visibility else 1.0
                        ),
                    ),
                    clock=lambda: 0.0,
                )
            else:
                policy = LinearConcessionPolicy()
        else:  # pragma: no cover - closed enum
            raise ExperimentError(f"unsupported baseline '{baseline.value}'")
        if len(participants) >= 3:
            critique = (
                f"Public issue critique from speaking position {index + 1}."
                if components.communication
                else "Communication ablation: no substantive critique shared."
            )
            policy = DeliberativePolicy(
                policy,
                critique=critique,
                name=f"deliberative-{getattr(policy, 'name', baseline.value)}",
            )
        policies[participant.participant_id] = policy
    return policies


def _resolved_case(
    original: ExperimentScenario,
    baseline: BaselineKind,
    components: EnabledComponents,
) -> ExperimentScenario:
    if components.team_size is None:
        return original
    return team_size_case(
        components.team_size,
        heterogeneous=baseline is not BaselineKind.HOMOGENEOUS_TEAM,
    )


def _run_episode(
    case: ExperimentScenario,
    baseline: BaselineKind,
    seed: int,
    config: ExperimentConfiguration,
    timestamp: datetime,
    model_spec: Optional[ExperimentModelSpec] = None,
) -> EpisodeResult:
    participants = tuple(item.participant_id for item in case.scenario.participants)
    model_client: Optional[LLMClient] = None
    if baseline is BaselineKind.SINGLE_LLM:
        if model_spec is None:
            raise ExperimentError("single-LLM baseline requires a model specification")
        model_client = (
            _offline_fake_client(model_spec)
            if model_spec.provider is LLMProvider.FAKE
            else OpenAILLMClient.from_env()
        )
    policies = _policies_for(case, baseline, seed, config, model_client, model_spec)
    components = config.enabled_components
    memory_limits = (
        MemoryLimits()
        if components.memory
        else MemoryLimits(
            episodic_event_limit=2,
            episodic_character_limit=512,
            working_message_limit=1,
            working_character_limit=512,
            recent_offer_limit=3,
            learned_strategy_limit=0,
            summary_character_limit=32,
        )
    )
    memories = {
        participant_id: AgentMemory(
            case.profiles[participant_id],
            case.scenario,
            strategy_guidance=(
                (f"Use {getattr(policies[participant_id], 'name', baseline.value)}.",)
                if components.memory
                else ()
            ),
            limits=memory_limits,
        )
        for participant_id in participants
    }
    mediator_configuration = MediatorConfiguration()
    mediator = DeterministicMediator(mediator_configuration) if components.mediator else None
    bus_participants = participants + (
        (mediator_configuration.mediator_id,) if components.mediator else ()
    )
    protocol_factory = None
    if len(participants) >= 3:
        protocol_factory = MultipartyProtocolFactory(
            MultipartyProtocolConfiguration(
                speaking_order=participants,
                eligible_proposers=participants,
                eligible_voters=participants,
                required_participants=participants,
                acceptance_rule=AcceptanceRule.UNANIMITY,
                deliberation_rounds=components.deliberation_depth,
                revision_rounds=1,
            )
        )
    verifier = VerificationCoordinator(
        VerificationConfiguration(
            mode=(
                VerificationMode.DETERMINISTIC
                if components.verifier
                else VerificationMode.DISABLED
            )
        )
    )
    opponent_modellers = (
        {participant_id: HeuristicOpponentModeller() for participant_id in participants}
        if components.theory_of_mind and len(participants) == 2
        else {}
    )
    orchestration = OrchestratorConfiguration(
        maximum_rounds=config.rounds,
        episode_timeout_seconds=config.budgets.episode_timeout_seconds,
        action_timeout_seconds=config.budgets.action_timeout_seconds,
        maximum_model_calls=config.budgets.maximum_model_calls,
        maximum_actions=config.budgets.maximum_actions,
        failure_policy=FailurePolicy.FALLBACK,
        mediation_enabled=components.mediator,
    )
    result = NegotiationOrchestrator(
        scenario=case.scenario,
        profiles=case.profiles,
        policies=policies,
        message_bus=MessageBus(
            case.scenario.scenario_id,
            participants=bus_participants,
            mediator_ids=(
                (mediator_configuration.mediator_id,) if components.mediator else ()
            ),
            audit_reader_ids=(AUDIT_READER_ID,),
        ),
        verifier=verifier,
        mediator=mediator,
        model_gateway=model_client,
        memory_store=InMemoryMemoryStore(memories),
        clock=ExperimentClock(timestamp),
        random_seed=seed,
        configuration=orchestration,
        feasible_region=case.feasible_region,
        opponent_modellers=opponent_modellers,
        protocol_factory=protocol_factory,
    ).run()
    if result.usage.tokens.total_tokens > config.budgets.maximum_total_tokens:
        raise ExperimentError("episode exceeded the configured total-token budget")
    return result


def _episode_metrics(
    result: EpisodeResult, budget: ExperimentBudgets
) -> EpisodeMetricRecord:
    participant_count = len(result.metrics.participant_utilities)
    tokens = result.usage.tokens.total_tokens
    return EpisodeMetricRecord(
        agreement=float(result.metrics.agreement_valid),
        mean_utility=result.metrics.social_welfare / participant_count,
        social_welfare=result.metrics.social_welfare,
        fairness=result.metrics.utility_balance or 0.0,
        rounds=float(result.final_session.round_number),
        invalid_action_rate=result.metrics.action_quality.invalid_action_rate,
        latency_ms=result.usage.latency_ms,
        model_calls=result.usage.model_calls,
        total_tokens=tokens,
        estimated_cost=(tokens / 1000.0) * budget.estimated_cost_per_1k_tokens,
    )


def _git_commit(workdir: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = completed.stdout.strip()
        return value if value else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _interval(values: Sequence[float]) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0, sample_size=0)
    mean = math.fsum(values) / len(values)
    if len(values) == 1:
        return ConfidenceInterval(mean=mean, lower=mean, upper=mean, sample_size=1)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = NORMAL_95 * math.sqrt(variance / len(values))
    return ConfidenceInterval(
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        sample_size=len(values),
    )


METRIC_NAMES = (
    "agreement",
    "mean_utility",
    "social_welfare",
    "fairness",
    "rounds",
    "invalid_action_rate",
    "latency_ms",
    "model_calls",
    "total_tokens",
    "estimated_cost",
)


def aggregate_records(
    records: Sequence[ExperimentEpisodeRecord],
) -> Tuple[AggregateResult, ...]:
    grouped: Dict[
        Tuple[str, BaselineKind, Optional[str], float], list[ExperimentEpisodeRecord]
    ] = defaultdict(list)
    for record in records:
        grouped[
            (
                record.configuration_hash,
                record.baseline,
                record.model,
                record.temperature,
            )
        ].append(record)
    results = []
    for configuration_hash, baseline, model, temperature in sorted(
        grouped,
        key=lambda item: (item[0], item[1].value, item[2] or "", item[3]),
    ):
        group = grouped[(configuration_hash, baseline, model, temperature)]
        successful = [item for item in group if item.status is EpisodeRecordStatus.SUCCESS]
        intervals = {
            metric: _interval(
                [float(getattr(item.metrics, metric)) for item in successful if item.metrics]
            )
            for metric in METRIC_NAMES
        }
        results.append(
            AggregateResult(
                configuration_hash=configuration_hash,
                git_commit=group[0].git_commit,
                timestamp_utc=group[0].timestamp_utc,
                seeds=tuple(sorted({item.seed for item in group})),
                baseline=baseline,
                model=model,
                temperature=temperature,
                successful_episodes=len(successful),
                failed_episodes=sum(
                    item.status is EpisodeRecordStatus.FAILED for item in group
                ),
                not_applicable_episodes=sum(
                    item.status is EpisodeRecordStatus.NOT_APPLICABLE for item in group
                ),
                intervals=intervals,
            )
        )
    return tuple(results)


def paired_comparisons(
    records: Sequence[ExperimentEpisodeRecord],
) -> Tuple[PairedComparison, ...]:
    successful = [item for item in records if item.status is EpisodeRecordStatus.SUCCESS]
    series_type = Tuple[BaselineKind, Optional[str], float]
    by_configuration: Dict[
        str, Dict[series_type, Dict[Tuple[str, int], ExperimentEpisodeRecord]]
    ] = defaultdict(lambda: defaultdict(dict))
    for item in successful:
        series = (item.baseline, item.model, item.temperature)
        by_configuration[item.configuration_hash][series][
            (item.scenario_id, item.seed)
        ] = item
    comparisons = []
    for configuration_hash in sorted(by_configuration):
        by_baseline = by_configuration[configuration_hash]
        series_values = sorted(
            by_baseline, key=lambda item: (item[0].value, item[1] or "", item[2])
        )
        for left_index, left in enumerate(series_values):
            for right in series_values[left_index + 1 :]:
                shared = sorted(set(by_baseline[left]) & set(by_baseline[right]))
                for metric in (
                    "agreement", "mean_utility", "social_welfare", "fairness", "rounds"
                ):
                    differences = [
                        float(getattr(by_baseline[left][key].metrics, metric))
                        - float(getattr(by_baseline[right][key].metrics, metric))
                        for key in shared
                        if by_baseline[left][key].metrics and by_baseline[right][key].metrics
                    ]
                    comparisons.append(
                        PairedComparison(
                            configuration_hash=configuration_hash,
                            baseline_a=left[0],
                            model_a=left[1],
                            temperature_a=left[2],
                            baseline_b=right[0],
                            model_b=right[1],
                            temperature_b=right[2],
                            metric=metric,
                            paired_seeded_cases=len(differences),
                            difference=_interval(differences),
                        )
                    )
    return tuple(comparisons)


def _aggregate_csv(aggregates: Sequence[AggregateResult]) -> str:
    output = io.StringIO(newline="")
    fields = [
        "configuration_hash", "git_commit", "timestamp_utc", "seeds", "baseline",
        "model", "temperature",
        "successful_episodes", "failed_episodes", "not_applicable_episodes",
    ]
    for metric in METRIC_NAMES:
        fields.extend((f"{metric}_mean", f"{metric}_ci_lower", f"{metric}_ci_upper"))
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for aggregate in aggregates:
        row: Dict[str, object] = {
            "configuration_hash": aggregate.configuration_hash,
            "git_commit": aggregate.git_commit,
            "timestamp_utc": aggregate.timestamp_utc.isoformat(),
            "seeds": ";".join(str(seed) for seed in aggregate.seeds),
            "baseline": aggregate.baseline.value,
            "model": aggregate.model or "",
            "temperature": aggregate.temperature,
            "successful_episodes": aggregate.successful_episodes,
            "failed_episodes": aggregate.failed_episodes,
            "not_applicable_episodes": aggregate.not_applicable_episodes,
        }
        for metric, interval in aggregate.intervals.items():
            row[f"{metric}_mean"] = interval.mean
            row[f"{metric}_ci_lower"] = interval.lower
            row[f"{metric}_ci_upper"] = interval.upper
        writer.writerow(row)
    return output.getvalue()


def _paired_csv(comparisons: Sequence[PairedComparison]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ("configuration_hash", "baseline_a", "model_a", "temperature_a", "baseline_b", "model_b", "temperature_b", "metric", "paired_seeded_cases", "mean_difference", "ci_lower", "ci_upper", "method")
    )
    for item in comparisons:
        writer.writerow(
            (
                item.configuration_hash,
                item.baseline_a.value,
                item.model_a or "",
                item.temperature_a,
                item.baseline_b.value,
                item.model_b or "",
                item.temperature_b,
                item.metric,
                item.paired_seeded_cases,
                item.difference.mean,
                item.difference.lower,
                item.difference.upper,
                item.difference.method,
            )
        )
    return output.getvalue()


def _ablation_csv(
    configuration: ExperimentConfiguration,
    aggregates: Sequence[AggregateResult],
) -> str:
    output = io.StringIO(newline="")
    fields = (
        "configuration_hash", "baseline", "model", "temperature", "memory", "theory_of_mind", "verifier",
        "mediator", "communication", "team_size", "deliberation_depth",
        "confidence_visibility", "successful_episodes", "agreement", "mean_utility",
        "social_welfare", "fairness", "rounds", "invalid_action_rate",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    components = configuration.enabled_components
    for aggregate in aggregates:
        writer.writerow(
            {
                "configuration_hash": configuration.configuration_hash,
                "baseline": aggregate.baseline.value,
                "model": aggregate.model or "",
                "temperature": aggregate.temperature,
                "memory": components.memory,
                "theory_of_mind": components.theory_of_mind,
                "verifier": components.verifier,
                "mediator": components.mediator,
                "communication": components.communication,
                "team_size": components.team_size or "dataset",
                "deliberation_depth": components.deliberation_depth,
                "confidence_visibility": components.confidence_visibility,
                "successful_episodes": aggregate.successful_episodes,
                "agreement": aggregate.intervals["agreement"].mean,
                "mean_utility": aggregate.intervals["mean_utility"].mean,
                "social_welfare": aggregate.intervals["social_welfare"].mean,
                "fairness": aggregate.intervals["fairness"].mean,
                "rounds": aggregate.intervals["rounds"].mean,
                "invalid_action_rate": aggregate.intervals["invalid_action_rate"].mean,
            }
        )
    return output.getvalue()


def records_from_jsonl(path: Path) -> Tuple[ExperimentEpisodeRecord, ...]:
    records = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                records.append(ExperimentEpisodeRecord.model_validate_json(line))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f"invalid experiment JSONL at line {line_number if 'line_number' in locals() else 0}") from exc
    return tuple(records)


def rebuild_aggregates(episode_jsonl: Path, aggregate_csv: Path) -> None:
    """Reproduce aggregate CSV solely from preserved raw episode records."""

    _atomic_text(aggregate_csv, _aggregate_csv(aggregate_records(records_from_jsonl(episode_jsonl))))


def rebuild_ablation_table(episode_jsonl: Path, ablation_csv: Path) -> None:
    """Rebuild an ablation table solely from saved per-episode records."""

    records = records_from_jsonl(episode_jsonl)
    groups: Dict[
        Tuple[str, BaselineKind, Optional[str], float, str],
        list[ExperimentEpisodeRecord],
    ] = defaultdict(list)
    for record in records:
        component_json = json.dumps(
            record.components.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        groups[
            (
                record.configuration_hash,
                record.baseline,
                record.model,
                record.temperature,
                component_json,
            )
        ].append(record)
    output = io.StringIO(newline="")
    fields = (
        "configuration_hash", "baseline", "model", "temperature", "memory", "theory_of_mind", "verifier",
        "mediator", "communication", "team_size", "deliberation_depth",
        "confidence_visibility", "successful_episodes", "agreement", "mean_utility",
        "social_welfare", "fairness", "rounds", "invalid_action_rate",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for (config_hash, baseline, model, temperature, _component_json), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1].value, item[0][2] or "", item[0][3]),
    ):
        aggregate = aggregate_records(group)[0]
        components = group[0].components
        writer.writerow(
            {
                "configuration_hash": config_hash,
                "baseline": baseline.value,
                "model": model or "",
                "temperature": temperature,
                "memory": components.memory,
                "theory_of_mind": components.theory_of_mind,
                "verifier": components.verifier,
                "mediator": components.mediator,
                "communication": components.communication,
                "team_size": components.team_size or "dataset",
                "deliberation_depth": components.deliberation_depth,
                "confidence_visibility": components.confidence_visibility,
                "successful_episodes": aggregate.successful_episodes,
                "agreement": aggregate.intervals["agreement"].mean,
                "mean_utility": aggregate.intervals["mean_utility"].mean,
                "social_welfare": aggregate.intervals["social_welfare"].mean,
                "fairness": aggregate.intervals["fairness"].mean,
                "rounds": aggregate.intervals["rounds"].mean,
                "invalid_action_rate": aggregate.intervals["invalid_action_rate"].mean,
            }
        )
    _atomic_text(ablation_csv, output.getvalue())


def one_at_a_time_ablation_configurations(
    base: ExperimentConfiguration,
) -> Tuple[ExperimentConfiguration, ...]:
    """Generate explicit one-factor ablations, including team-size and depth cells."""

    configurations = [base]
    toggles = (
        "memory",
        "theory_of_mind",
        "verifier",
        "mediator",
        "communication",
        "confidence_visibility",
    )
    for name in toggles:
        value = not getattr(base.enabled_components, name)
        components = base.enabled_components.model_copy(update={name: value})
        configurations.append(
            base.model_copy(
                update={
                    "experiment_id": f"{base.experiment_id}-ablate-{name}",
                    "enabled_components": components,
                }
            )
        )
    for size in (2, 3, 4):
        team_source = (
            "three-principal-resource-allocation"
            if "three-principal-resource-allocation" in base.scenario_ids
            else base.scenario_ids[0]
        )
        configurations.append(
            base.model_copy(
                update={
                    "experiment_id": f"{base.experiment_id}-team-{size}",
                    "scenario_ids": (team_source,),
                    "enabled_components": base.enabled_components.model_copy(
                        update={"team_size": size}
                    ),
                }
            )
        )
    for depth in (1, 2):
        configurations.append(
            base.model_copy(
                update={
                    "experiment_id": f"{base.experiment_id}-depth-{depth}",
                    "enabled_components": base.enabled_components.model_copy(
                        update={"deliberation_depth": depth}
                    ),
                }
            )
        )
    return tuple(configurations)


def _svg_plot(
    title: str,
    series: Mapping[str, Mapping[str, float]],
    *,
    width: int = 800,
    height: int = 420,
) -> str:
    labels = tuple(series)
    series_names = tuple(next(iter(series.values())).keys()) if series else ()
    values = [value for group in series.values() for value in group.values()]
    maximum = max(values, default=1.0) or 1.0
    plot_left, plot_top, plot_width, plot_height = 70, 55, width - 100, height - 125
    bar_slot = plot_width / max(1, len(labels))
    bar_width = bar_slot / max(1, len(series_names) + 1)
    colours = ("#2563eb", "#16a34a", "#dc2626", "#9333ea")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#111"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#111"/>',
    ]
    for label_index, label in enumerate(labels):
        for series_index, name in enumerate(series_names):
            value = series[label][name]
            bar_height = (value / maximum) * plot_height
            x = plot_left + label_index * bar_slot + 8 + series_index * bar_width
            y = plot_top + plot_height - bar_height
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * .8:.2f}" height="{bar_height:.2f}" fill="{colours[series_index % len(colours)]}"><title>{escape(label)} {escape(name)}: {value:.6g}</title></rect>'
            )
        x_label = plot_left + (label_index + 0.5) * bar_slot
        elements.append(
            f'<text x="{x_label:.2f}" y="{plot_top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{escape(label[:20])}</text>'
        )
    for index, name in enumerate(series_names):
        elements.append(
            f'<rect x="{plot_left + index * 150}" y="{height - 25}" width="12" height="12" fill="{colours[index % len(colours)]}"/>'
        )
        elements.append(
            f'<text x="{plot_left + 17 + index * 150}" y="{height - 14}" font-family="sans-serif" font-size="11">{escape(name)}</text>'
        )
    elements.append("</svg>\n")
    return "".join(elements)


def _write_plots(
    directory: Path,
    aggregates: Sequence[AggregateResult],
    learning: LearningExperimentResult,
) -> Tuple[str, ...]:
    by_baseline = {
        (
            aggregate.baseline.value
            if aggregate.model is None
            else f"{aggregate.baseline.value}:{aggregate.model}@{aggregate.temperature:g}"
        ): aggregate.intervals
        for aggregate in aggregates
    }
    specifications = {
        "agreement.svg": ("Agreement rate", ("agreement",)),
        "utility_welfare.svg": ("Utility and social welfare", ("mean_utility", "social_welfare")),
        "fairness.svg": ("Utility balance fairness", ("fairness",)),
        "rounds.svg": ("Rounds", ("rounds",)),
        "invalid_actions.svg": ("Invalid-action rate", ("invalid_action_rate",)),
        "latency.svg": ("Model latency (ms)", ("latency_ms",)),
        "usage_cost.svg": ("Usage and estimated cost", ("total_tokens", "estimated_cost")),
    }
    paths = []
    for filename, (title, metrics) in specifications.items():
        series = {
            baseline: {metric: intervals[metric].mean for metric in metrics}
            for baseline, intervals in by_baseline.items()
        }
        path = directory / filename
        _atomic_text(path, _svg_plot(title, series))
        paths.append(str(path))
    learning_series = {
        str(item.episode): {"cumulative_mean_reward": item.cumulative_mean_reward}
        for item in learning.curve
    }
    learning_path = directory / "learning_curves.svg"
    _atomic_text(learning_path, _svg_plot("Training-only learning curve", learning_series))
    paths.append(str(learning_path))
    return tuple(paths)


def run_experiment(
    configuration: ExperimentConfiguration,
    output_directory: Path,
    *,
    timestamp: Optional[datetime] = None,
    repository_root: Optional[Path] = None,
    allow_live: bool = False,
) -> ExperimentRunResult:
    """Run every configured cell and always record failures instead of skipping silently."""

    instant = timestamp or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("experiment timestamp must be timezone-aware")
    if any(model.provider is not LLMProvider.FAKE for model in configuration.models) and not allow_live:
        raise ExperimentError(
            "live model configuration requires the explicit live command/allow_live=True"
        )
    root = repository_root or Path.cwd()
    git_commit = _git_commit(root)
    output_directory.mkdir(parents=True, exist_ok=True)
    index = scenario_index()
    records = []
    for scenario_id in configuration.scenario_ids:
        original = index[scenario_id]
        for baseline in configuration.baselines:
            case = _resolved_case(original, baseline, configuration.enabled_components)
            model_specs: Tuple[Optional[ExperimentModelSpec], ...] = (
                tuple(configuration.models)
                if baseline is BaselineKind.SINGLE_LLM
                else (None,)
            )
            for model_spec in model_specs:
                for seed in configuration.seeds:
                    common = {
                        "configuration_hash": configuration.configuration_hash,
                        "git_commit": git_commit,
                        "timestamp_utc": instant,
                        "experiment_id": configuration.experiment_id,
                        "dataset_version": configuration.dataset_version,
                        "scenario_id": case.case_id,
                        "baseline": baseline,
                        "seed": seed,
                        "model": model_spec.model if model_spec is not None else None,
                        "temperature": (
                            model_spec.temperature if model_spec is not None else 0.0
                        ),
                        "components": configuration.enabled_components,
                        "partition": configuration.evaluation_partition,
                    }
                    if (
                        baseline is BaselineKind.SINGLE_LLM
                        and len(case.scenario.participants) != 2
                    ):
                        records.append(
                            ExperimentEpisodeRecord(
                                **common,
                                status=EpisodeRecordStatus.NOT_APPLICABLE,
                                failure_code="baseline_not_applicable",
                                failure_message=(
                                    "Single-policy LLM baseline is bilateral in this harness."
                                ),
                            )
                        )
                        continue
                    try:
                        result = _run_episode(
                            case, baseline, seed, configuration, instant, model_spec
                        )
                        records.append(
                            ExperimentEpisodeRecord(
                                **common,
                                status=EpisodeRecordStatus.SUCCESS,
                                metrics=_episode_metrics(result, configuration.budgets),
                                episode_result=result,
                            )
                        )
                    except Exception as exc:  # failed cells become explicit artifact rows
                        records.append(
                            ExperimentEpisodeRecord(
                                **common,
                                status=EpisodeRecordStatus.FAILED,
                                failure_code=type(exc).__name__[:100],
                                failure_message=(
                                    "Episode execution failed; inspect local debug logs."
                                ),
                            )
                        )
    record_tuple = tuple(records)
    episode_path = output_directory / "episodes.jsonl"
    _atomic_text(
        episode_path,
        "".join(item.model_dump_json() + "\n" for item in record_tuple),
    )
    aggregates = aggregate_records(record_tuple)
    aggregate_path = output_directory / "aggregate.csv"
    _atomic_text(aggregate_path, _aggregate_csv(aggregates))
    paired_path = output_directory / "paired_comparisons.csv"
    _atomic_text(paired_path, _paired_csv(paired_comparisons(record_tuple)))
    ablation_path = output_directory / "ablation_table.csv"
    _atomic_text(ablation_path, _ablation_csv(configuration, aggregates))

    from .experiment_components import run_component_evaluations

    component_path = output_directory / "component_evaluations.jsonl"
    component_records = run_component_evaluations(configuration.enabled_components)
    _atomic_text(
        component_path,
        "".join(item.model_dump_json() + "\n" for item in component_records),
    )

    # Training is intentionally stored separately and never enters evaluation aggregates.
    learning = run_learning_curve_experiment(episodes=6, seed=configuration.seeds[0], epsilon=0.25)
    learning_path = output_directory / "learning_curve_training.json"
    _atomic_text(learning_path, learning.model_dump_json(indent=2) + "\n")
    plot_directory = output_directory / "plots"
    plot_files = _write_plots(plot_directory, aggregates, learning)
    manifest_path = output_directory / "manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "configuration": configuration.model_dump(mode="json"),
                "configuration_hash": configuration.configuration_hash,
                "git_commit": git_commit,
                "timestamp_utc": instant.isoformat(),
                "statistical_method": "two-sided 95% normal approximation; paired differences use identical scenario/seed/temperature keys",
                "evaluation_partition": "episodes.jsonl",
                "training_partition": "learning_curve_training.json",
                "artifacts": {
                    "episodes": episode_path.name,
                    "aggregate": aggregate_path.name,
                    "paired": paired_path.name,
                    "ablation": ablation_path.name,
                    "components": component_path.name,
                    "plots": [str(Path(item).relative_to(output_directory)) for item in plot_files],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return ExperimentRunResult(
        configuration=configuration,
        output_directory=str(output_directory),
        episode_jsonl=str(episode_path),
        aggregate_csv=str(aggregate_path),
        paired_csv=str(paired_path),
        ablation_csv=str(ablation_path),
        component_jsonl=str(component_path),
        learning_curve_json=str(learning_path),
        plot_files=plot_files,
        records=record_tuple,
    )


def run_ablation_suite(
    base_configuration: ExperimentConfiguration,
    output_directory: Path,
    *,
    timestamp: Optional[datetime] = None,
    repository_root: Optional[Path] = None,
    allow_live: bool = False,
) -> AblationSuiteResult:
    """Run the explicit one-factor matrix and build one table from combined raw rows."""

    instant = timestamp or datetime.now(timezone.utc)
    output_directory.mkdir(parents=True, exist_ok=True)
    all_records = []
    cell_directories = []
    configurations = one_at_a_time_ablation_configurations(base_configuration)
    for configuration in configurations:
        cell = output_directory / "cells" / configuration.configuration_hash[:12]
        result = run_experiment(
            configuration,
            cell,
            timestamp=instant,
            repository_root=repository_root,
            allow_live=allow_live,
        )
        all_records.extend(result.records)
        cell_directories.append(str(cell))
    episode_path = output_directory / "episodes.jsonl"
    _atomic_text(
        episode_path,
        "".join(record.model_dump_json() + "\n" for record in all_records),
    )
    ablation_path = output_directory / "ablation_table.csv"
    rebuild_ablation_table(episode_path, ablation_path)
    return AblationSuiteResult(
        configuration_hashes=tuple(item.configuration_hash for item in configurations),
        episode_jsonl=str(episode_path),
        ablation_csv=str(ablation_path),
        cell_directories=tuple(cell_directories),
    )


def small_offline_configuration() -> ExperimentConfiguration:
    return ExperimentConfiguration(
        experiment_id="offline-smoke-v1",
        scenario_ids=(
            "price-feasible-symmetric",
            "price-infeasible",
            "multi-issue-asymmetric",
            "three-principal-resource-allocation",
        ),
        baselines=(
            BaselineKind.DETERMINISTIC,
            BaselineKind.SINGLE_LLM,
            BaselineKind.SAMPLE_AND_VOTE,
            BaselineKind.HOMOGENEOUS_TEAM,
            BaselineKind.HETEROGENEOUS_TEAM,
        ),
        seeds=(7,),
        rounds=6,
    )
