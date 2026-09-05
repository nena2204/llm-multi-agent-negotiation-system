from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Optional, Protocol, Tuple, runtime_checkable

from pydantic import Field, field_validator, model_validator

from .communication import (
    MessageBus,
    MessageEnvelope,
    MessageType,
    MessageVisibility,
    build_episode,
)
from .domain import (
    MediationAccessMode,
    MediationTrigger,
    MediatorIntervention,
    MessageAction,
    NegotiationAction,
    NegotiationScenario,
    ParticipantId,
    RequestMediationAction,
    TerminalOutcome,
    WithdrawAction,
)
from .domain.models import DomainModel
from .evaluation import (
    DeterministicEvaluation,
    EvaluationConfiguration,
    EvaluationInput,
    ModelCallRecord,
    evaluate_episode,
    model_records_from_policy_traces,
    model_records_from_verification,
)
from .llm import (
    LLMClient,
    LLMErrorCode,
    LLMErrorInfo,
    LLMNonRetryableError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    aggregate_usage,
)
from .mediation import (
    ConfidentialPreferenceSummary,
    DeadlockDetector,
    DeterministicMediator,
    MediationService,
    MediatorConfiguration,
    MediatorContext,
)
from .memory import AgentMemory, AgentObservation, AgentProfile, MemorySnapshot
from .opponent import OpponentModeller, evidence_from_memory
from .policies import LinearConcessionPolicy, NegotiationPolicy
from .protocol import (
    ActionAppliedEvent,
    FeasibleRegionStatus,
    MediationEnteredEvent,
    MediatorInterventionEvent,
    NegotiationProtocol,
    NegotiationSession,
    ProtocolEvent,
    ProtocolPhase,
    TERMINAL_PHASES,
)
from .verification import (
    CorrectionProvider,
    VerificationCoordinator,
    VerificationLogEntry,
    VerificationVerdict,
    VerifiedProtocolExecutor,
)


AUDIT_READER_ID = ParticipantId("orchestrator-audit")


class OrchestrationError(RuntimeError):
    """Raised when configured orchestration cannot safely continue."""


class FailurePolicy(str, Enum):
    FALLBACK = "fallback"
    WITHDRAW = "withdraw"
    RAISE = "raise"


class EpisodeStopReason(str, Enum):
    TERMINAL_OUTCOME = "terminal_outcome"
    MEDIATION_PAUSED = "mediation_paused"


class OrchestratorConfiguration(DomainModel):
    maximum_rounds: int = Field(default=6, ge=1)
    episode_timeout_seconds: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    action_timeout_seconds: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    maximum_model_calls: int = Field(default=100, ge=0)
    maximum_actions: Optional[int] = Field(default=None, ge=1)
    failure_policy: FailurePolicy = FailurePolicy.FALLBACK
    mediation_enabled: bool = False
    pause_after_mediator_intervention: bool = False

    @property
    def action_limit(self) -> int:
        return self.maximum_actions or self.maximum_rounds * 8


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Return a timezone-aware wall-clock time for message envelopes."""

    def monotonic(self) -> float:
        """Return monotonic seconds for budgets and elapsed metrics."""


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.perf_counter()


class EpisodeModelGateway:
    """Episode-local hard call budget around the injected provider gateway."""

    def __init__(self, delegate: LLMClient, maximum_calls: int) -> None:
        self.delegate = delegate
        self.maximum_calls = maximum_calls
        self.used_calls = 0
        self.provider = getattr(delegate, "provider", None)
        self._records: list[ModelCallRecord] = []

    @property
    def records(self) -> Tuple[ModelCallRecord, ...]:
        return tuple(self._records)

    def generate(self, request: LLMRequest) -> LLMResponse:
        remaining = self.maximum_calls - self.used_calls
        if remaining <= 0:
            raise LLMNonRetryableError(
                LLMErrorInfo(
                    code=LLMErrorCode.BUDGET_EXCEEDED,
                    message="Episode model-call budget is exhausted; no provider call was made.",
                    retryable=False,
                    provider=self.provider,
                    model=request.model_configuration.model,
                )
            )
        retry = request.model_configuration.retry
        allowed_retries = min(retry.max_retries, remaining - 1)
        bounded_request = request.model_copy(
            update={
                "model_configuration": request.model_configuration.model_copy(
                    update={
                        "retry": retry.model_copy(
                            update={"max_retries": allowed_retries}
                        )
                    }
                )
            }
        )
        started = time.perf_counter()
        try:
            response = self.delegate.generate(bounded_request)
        except Exception as exc:
            attempts = getattr(getattr(exc, "info", None), "attempts", 1)
            actual_attempts = min(remaining, max(1, attempts))
            self.used_calls += actual_attempts
            info = getattr(exc, "info", None)
            self._records.append(
                ModelCallRecord(
                    source=request.request_id[:100],
                    provider=getattr(info, "provider", None) or self.provider,
                    model=getattr(info, "model", None)
                    or request.model_configuration.model,
                    model_calls=actual_attempts,
                    latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
                )
            )
            raise
        self.used_calls += response.attempts
        self._records.append(
            ModelCallRecord(
                source=request.request_id[:100],
                provider=response.provider,
                model=response.model,
                model_calls=response.attempts,
                latency_ms=response.latency.total_ms,
                usage=response.usage,
            )
        )
        return response


@runtime_checkable
class MemoryStore(Protocol):
    def get(self, participant_id: ParticipantId) -> AgentMemory:
        """Return only the named participant's controlled memory."""

    def snapshots(self) -> Tuple[MemorySnapshot, ...]:
        """Return participant-owned snapshots for episode persistence."""


class InMemoryMemoryStore:
    """Episode-scoped store; no module or process global state is used."""

    def __init__(self, memories: Mapping[ParticipantId, AgentMemory]) -> None:
        if not memories:
            raise ValueError("memory store requires at least one participant memory")
        if any(key != value.owner_id for key, value in memories.items()):
            raise ValueError("memory store keys must match memory owners")
        self._memories = dict(memories)

    def get(self, participant_id: ParticipantId) -> AgentMemory:
        try:
            return self._memories[participant_id]
        except KeyError as exc:
            raise KeyError(f"no memory configured for participant '{participant_id}'") from exc

    def snapshots(self) -> Tuple[MemorySnapshot, ...]:
        return tuple(self._memories[key].snapshot() for key in self._memories)


class AgentMemoryReference(DomainModel):
    participant_id: ParticipantId
    schema_version: str = Field(min_length=1, max_length=20)
    episodic_event_count: int = Field(ge=0)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EpisodeUsage(DomainModel):
    model_calls: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    tokens: LLMUsage = Field(default_factory=LLMUsage)
    records: Tuple[ModelCallRecord, ...] = ()

    @field_validator("records", mode="before")
    @classmethod
    def records_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def totals_match_records(self) -> "EpisodeUsage":
        if self.model_calls != sum(item.model_calls for item in self.records):
            raise ValueError("episode model-call total does not match records")
        if abs(self.latency_ms - sum(item.latency_ms for item in self.records)) > 1e-9:
            raise ValueError("episode model latency does not match records")
        if self.tokens != aggregate_usage(item.usage for item in self.records):
            raise ValueError("episode token usage does not match records")
        return self


class EpisodeResult(DomainModel):
    schema_version: str = "1.0"
    scenario_id: str
    random_seed: int
    stop_reason: EpisodeStopReason
    outcome: Optional[TerminalOutcome]
    final_session: NegotiationSession
    public_transcript: Tuple[MessageEnvelope, ...]
    audit_messages: Tuple[MessageEnvelope, ...]
    audit_events: Tuple[ProtocolEvent, ...]
    memory_references: Tuple[AgentMemoryReference, ...]
    metrics: DeterministicEvaluation
    usage: EpisodeUsage
    verification_log: Tuple[VerificationLogEntry, ...]
    elapsed_time_ms: float = Field(ge=0.0, allow_inf_nan=False)

    @field_validator(
        "public_transcript",
        "audit_messages",
        "audit_events",
        "memory_references",
        "verification_log",
        mode="before",
    )
    @classmethod
    def sequences_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def consistent_result(self) -> "EpisodeResult":
        if self.scenario_id != self.final_session.scenario_id:
            raise ValueError("episode scenario and final session must match")
        if self.outcome != self.final_session.outcome:
            raise ValueError("episode outcome must equal the protocol outcome")
        if self.audit_events != self.final_session.event_sequence:
            raise ValueError("episode audit events must equal the session event sequence")
        expected_public = tuple(
            message
            for message in self.audit_messages
            if message.visibility is MessageVisibility.PUBLIC
        )
        if self.public_transcript != expected_public:
            raise ValueError("public transcript must equal the public subset of the audit stream")
        build_episode(self.final_session, self.audit_messages)
        if self.metrics.negotiation_id != self.scenario_id:
            raise ValueError("episode metrics must belong to the result scenario")
        if self.metrics.telemetry.model_call_count != self.usage.model_calls:
            raise ValueError("episode metrics and usage must report the same model-call total")
        if self.metrics.telemetry.model_usage != self.usage.tokens:
            raise ValueError("episode metrics and usage must report the same token totals")
        return self


class Mediator(Protocol):
    configuration: MediatorConfiguration

    def intervene(
        self, context: MediatorContext, trigger: MediationTrigger
    ) -> MediatorIntervention:
        """Return a typed, non-binding mediator intervention."""


@runtime_checkable
class ProtocolFactory(Protocol):
    """Extension boundary for replacing the bilateral turn protocol in later work."""

    supports_multiparty: bool

    def create(
        self,
        *,
        scenario: NegotiationScenario,
        maximum_rounds: int,
        initial_turn: ParticipantId,
        feasible_region: FeasibleRegionStatus,
        mediator_ids: Tuple[ParticipantId, ...],
    ) -> NegotiationProtocol:
        """Create an episode-scoped authoritative protocol."""


class BilateralAlternatingOfferProtocolFactory:
    supports_multiparty = False

    def create(
        self,
        *,
        scenario: NegotiationScenario,
        maximum_rounds: int,
        initial_turn: ParticipantId,
        feasible_region: FeasibleRegionStatus,
        mediator_ids: Tuple[ParticipantId, ...],
    ) -> NegotiationProtocol:
        return NegotiationProtocol(
            scenario=scenario,
            maximum_rounds=maximum_rounds,
            initial_turn=initial_turn,
            feasible_region=feasible_region,
            mediator_ids=mediator_ids,
        )


def _memory_reference(snapshot: MemorySnapshot) -> AgentMemoryReference:
    payload = snapshot.model_dump_json()
    return AgentMemoryReference(
        participant_id=snapshot.owner_id,
        schema_version=snapshot.schema_version,
        episodic_event_count=len(snapshot.episodic.events),
        snapshot_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


class NegotiationOrchestrator:
    """Deterministic bilateral lifecycle coordinator over injected components.

    Protocol transitions remain authoritative. The participant ordering is the extension point for
    future multiparty turn protocols; this implementation intentionally requires two participants.
    """

    def __init__(
        self,
        *,
        scenario: NegotiationScenario,
        profiles: Mapping[ParticipantId, AgentProfile],
        policies: Mapping[ParticipantId, NegotiationPolicy],
        message_bus: MessageBus,
        verifier: VerificationCoordinator,
        mediator: Optional[Mediator],
        model_gateway: Optional[LLMClient],
        memory_store: MemoryStore,
        clock: Clock,
        random_seed: int,
        configuration: Optional[OrchestratorConfiguration] = None,
        feasible_region: FeasibleRegionStatus = FeasibleRegionStatus.UNKNOWN,
        mediation_service: Optional[MediationService] = None,
        correction_providers: Optional[Mapping[ParticipantId, CorrectionProvider]] = None,
        fallback_policies: Optional[Mapping[ParticipantId, NegotiationPolicy]] = None,
        opponent_modellers: Optional[Mapping[ParticipantId, OpponentModeller]] = None,
        evaluation_configuration: Optional[EvaluationConfiguration] = None,
        protocol_factory: Optional[ProtocolFactory] = None,
        audit_reader_id: ParticipantId = AUDIT_READER_ID,
    ) -> None:
        self.scenario = scenario
        self.profiles = dict(profiles)
        self.policies = dict(policies)
        self.message_bus = message_bus
        self.verifier = verifier
        self.mediator = mediator
        self.model_gateway = model_gateway
        self.memory_store = memory_store
        self.clock = clock
        self.random_seed = random_seed
        self.configuration = configuration or OrchestratorConfiguration()
        self.feasible_region = feasible_region
        self.mediation_service = mediation_service or MediationService()
        self.correction_providers = dict(correction_providers or {})
        self.fallback_policies = dict(fallback_policies or {})
        self.opponent_modellers = dict(opponent_modellers or {})
        self.evaluation_configuration = evaluation_configuration
        self.protocol_factory = protocol_factory or BilateralAlternatingOfferProtocolFactory()
        self.audit_reader_id = audit_reader_id
        self._validate_dependencies()
        self.episode_model_gateway = (
            EpisodeModelGateway(model_gateway, self.configuration.maximum_model_calls)
            if model_gateway is not None
            else None
        )
        self._bind_episode_model_gateway()

    @classmethod
    def with_defaults(
        cls,
        *,
        scenario: NegotiationScenario,
        profiles: Mapping[ParticipantId, AgentProfile],
        policies: Mapping[ParticipantId, NegotiationPolicy],
        random_seed: int = 0,
        configuration: Optional[OrchestratorConfiguration] = None,
        feasible_region: FeasibleRegionStatus = FeasibleRegionStatus.UNKNOWN,
        mediator_configuration: Optional[MediatorConfiguration] = None,
        model_gateway: Optional[LLMClient] = None,
        clock: Optional[Clock] = None,
        correction_providers: Optional[Mapping[ParticipantId, CorrectionProvider]] = None,
        fallback_policies: Optional[Mapping[ParticipantId, NegotiationPolicy]] = None,
        opponent_modellers: Optional[Mapping[ParticipantId, OpponentModeller]] = None,
        protocol_factory: Optional[ProtocolFactory] = None,
    ) -> "NegotiationOrchestrator":
        participant_ids = tuple(item.participant_id for item in scenario.participants)
        mediator_config = mediator_configuration or MediatorConfiguration()
        bus_participants = participant_ids + (mediator_config.mediator_id,)
        memories = {
            participant_id: AgentMemory(
                profiles[participant_id],
                scenario,
                strategy_guidance=(f"Use the {policies[participant_id].name} policy.",),
            )
            for participant_id in participant_ids
        }
        return cls(
            scenario=scenario,
            profiles=profiles,
            policies=policies,
            message_bus=MessageBus(
                scenario.scenario_id,
                participants=bus_participants,
                mediator_ids=(mediator_config.mediator_id,),
                audit_reader_ids=(AUDIT_READER_ID,),
            ),
            verifier=VerificationCoordinator(),
            mediator=DeterministicMediator(mediator_config),
            model_gateway=model_gateway,
            memory_store=InMemoryMemoryStore(memories),
            clock=clock or SystemClock(),
            random_seed=random_seed,
            configuration=configuration,
            feasible_region=feasible_region,
            correction_providers=correction_providers,
            fallback_policies=fallback_policies,
            opponent_modellers=opponent_modellers,
            protocol_factory=protocol_factory,
        )

    def _validate_dependencies(self) -> None:
        participant_ids = tuple(item.participant_id for item in self.scenario.participants)
        expected = set(participant_ids)
        if len(participant_ids) != 2 and not self.protocol_factory.supports_multiparty:
            raise ValueError(
                "the alternating-offer orchestrator currently supports exactly two participants"
            )
        if set(self.profiles) != expected or set(self.policies) != expected:
            raise ValueError("profiles and policies must cover every participant exactly once")
        if any(self.profiles[key].identity.participant != self.scenario.participant(key) for key in expected):
            raise ValueError("profile identities must match public scenario participants")
        if self.message_bus.negotiation_id != self.scenario.scenario_id:
            raise ValueError("message bus must belong to the orchestration scenario")
        if not expected.issubset(set(self.message_bus.participants)):
            raise ValueError("message bus must include every scenario participant")
        if self.audit_reader_id not in self.message_bus.audit_reader_ids:
            raise ValueError("the injected audit reader must be authorized by the message bus")
        for participant_id in participant_ids:
            if self.memory_store.get(participant_id).owner_id != participant_id:
                raise ValueError("memory store returned a memory owned by another participant")
        if self.configuration.mediation_enabled and self.mediator is None:
            raise ValueError("enabled mediation requires an injected mediator")
        if self.mediator is not None:
            mediator_id = self.mediator.configuration.mediator_id
            if mediator_id not in self.message_bus.mediator_ids:
                raise ValueError("mediator must be registered with the message bus")
        if set(self.correction_providers) - expected or set(self.fallback_policies) - expected:
            raise ValueError("correction and fallback policies must belong to scenario participants")
        if set(self.opponent_modellers) - expected:
            raise ValueError("opponent modeller keys must belong to scenario participants")
        if any(getattr(policy, "client", None) is not None for policy in self.fallback_policies.values()):
            raise ValueError("failure fallback policies must be deterministic and model-free")
        for label, component in self._model_components():
            client = getattr(component, "client", None)
            if (
                client is not None
                and self.model_gateway is not None
                and self.model_gateway is not client
            ):
                raise ValueError(
                    f"LLM-backed {label} must use the model_gateway injected into the orchestrator"
                )

    def _model_components(self) -> list[tuple[str, object]]:
        components = [
            (f"policy '{participant_id}'", policy)
            for participant_id, policy in self.policies.items()
        ]
        for participant_id, policy in self.policies.items():
            modeller = getattr(policy, "opponent_modeller", None)
            if modeller is not None:
                components.append((f"policy opponent modeller '{participant_id}'", modeller))
        components.extend(
            (f"opponent modeller '{participant_id}'", modeller)
            for participant_id, modeller in self.opponent_modellers.items()
        )
        components.extend(
            (f"correction provider '{participant_id}'", provider)
            for participant_id, provider in self.correction_providers.items()
        )
        if self.mediator is not None:
            components.append(("mediator", self.mediator))
        if self.verifier.llm_verifier is not None:
            components.append(("action verifier", self.verifier.llm_verifier))
        return components

    def _bind_episode_model_gateway(self) -> None:
        if self.episode_model_gateway is None:
            return
        rebound = set()
        for _label, component in self._model_components():
            if id(component) in rebound:
                continue
            if getattr(component, "client", None) is self.model_gateway:
                component.client = self.episode_model_gateway
                rebound.add(id(component))

    def run(self) -> EpisodeResult:
        config = self.configuration
        participant_ids = tuple(item.participant_id for item in self.scenario.participants)
        mediator_ids = tuple(self.message_bus.mediator_ids)
        protocol = self.protocol_factory.create(
            scenario=self.scenario,
            maximum_rounds=config.maximum_rounds,
            initial_turn=participant_ids[0],
            feasible_region=self.feasible_region,
            mediator_ids=mediator_ids,
        )
        executor = VerifiedProtocolExecutor(protocol, self.verifier)
        state = protocol.create_session()
        started = self.clock.monotonic()
        traces = []
        mediator_records = []
        beliefs = {}
        invalid_actor_ids = []
        action_count = 0
        mediation_handled_at = set()
        stop_reason = EpisodeStopReason.TERMINAL_OUTCOME

        self._update_all_memories(protocol, state)
        while state.phase not in TERMINAL_PHASES:
            elapsed = self.clock.monotonic() - started
            if elapsed >= config.episode_timeout_seconds or action_count >= config.action_limit:
                reason = (
                    "Episode timeout reached."
                    if elapsed >= config.episode_timeout_seconds
                    else "Maximum orchestration action count reached."
                )
                state = self._verified_withdraw(executor, state, reason).transition.state
                self._update_all_memories(protocol, state)
                break

            if config.mediation_enabled and state.phase is ProtocolPhase.ACTIVE:
                detector = DeadlockDetector(
                    self.scenario,
                    self.mediator.configuration.deadlock,
                )
                decision = detector.assess(
                    state,
                    invalid_action_actor_ids=tuple(invalid_actor_ids),
                )
                entered = self.mediation_service.enter_if_triggered(protocol, state, decision)
                if entered.events:
                    state = entered.state
                    self._update_all_memories(protocol, state)

            if config.mediation_enabled and state.phase is ProtocolPhase.MEDIATION:
                marker = len(state.event_sequence)
                if marker not in mediation_handled_at and not self._has_current_intervention(state):
                    mediation_handled_at.add(marker)
                    trigger = self._mediation_trigger(state)
                    context = self._mediator_context(state)
                    before_usage = getattr(self.mediator, "last_usage", LLMUsage())
                    try:
                        if (
                            getattr(self.mediator, "client", None) is not None
                            and self._model_call_count(traces, mediator_records)
                            >= config.maximum_model_calls
                        ):
                            raise OrchestrationError("Model-call budget exhausted before mediation.")
                        intervention = self.mediator.intervene(context, trigger)
                    except Exception as exc:
                        if config.failure_policy is FailurePolicy.RAISE:
                            raise OrchestrationError("mediator/provider execution failed") from exc
                        if config.failure_policy is FailurePolicy.WITHDRAW:
                            state = self._verified_withdraw(
                                executor, state, "Mediator/provider failure configured to withdraw."
                            ).transition.state
                            self._update_all_memories(protocol, state)
                            break
                        try:
                            intervention = DeterministicMediator(
                                self.mediator.configuration
                            ).intervene(context, trigger)
                        except Exception:
                            state = self._verified_withdraw(
                                executor,
                                state,
                                "Deterministic mediator fallback failed.",
                            ).transition.state
                            self._update_all_memories(protocol, state)
                            break
                    mediated = self.mediation_service.apply(
                        protocol, state, intervention, self.message_bus, self.clock.now()
                    )
                    state = mediated.transition.state
                    after_usage = getattr(self.mediator, "last_usage", LLMUsage())
                    latency = float(getattr(self.mediator, "last_latency_ms", 0.0))
                    if getattr(self.mediator, "client", None) is not None:
                        mediator_records.append(
                            ModelCallRecord(
                                source="mediator",
                                provider=getattr(
                                    getattr(self.mediator, "client", None), "provider", None
                                ),
                                model=getattr(
                                    getattr(self.mediator, "model_configuration", None),
                                    "model",
                                    None,
                                ),
                                model_calls=1,
                                latency_ms=latency,
                                usage=(
                                    after_usage
                                    if after_usage != before_usage
                                    else LLMUsage()
                                ),
                            )
                        )
                    self._update_all_memories(protocol, state)
                    if config.pause_after_mediator_intervention:
                        stop_reason = EpisodeStopReason.MEDIATION_PAUSED
                        break

            actor_id = state.current_turn
            if actor_id is None:
                raise OrchestrationError("non-terminal protocol state has no current actor")
            memory = self._observe(protocol, state, actor_id)
            policy = self.policies[actor_id]
            if (
                getattr(policy, "client", None) is not None
                and self._model_call_count(traces, mediator_records)
                >= config.maximum_model_calls
            ):
                candidate = self._budget_failure(
                    state, actor_id, "Model-call budget exhausted."
                )
                policy = None
            action_started = self.clock.monotonic()
            if policy is not None:
                candidate = self._choose_with_failure_policy(actor_id, policy, memory, state)
                for belief in getattr(policy, "last_belief_states", ()):
                    self.memory_store.get(actor_id).record_belief_snapshot(belief)
            action_elapsed = self.clock.monotonic() - action_started
            trace = getattr(policy, "last_trace", None) if policy is not None else None
            if trace is not None:
                traces.append(trace)
                if trace.used_fallback and config.failure_policy is not FailurePolicy.FALLBACK:
                    if config.failure_policy is FailurePolicy.RAISE:
                        raise OrchestrationError(
                            f"model policy for '{actor_id}' used fallback: {trace.fallback_reason}"
                        )
                    candidate = self._withdraw_action(
                        state, actor_id, "Model/provider failure configured to withdraw."
                    )
                if self._model_call_count(traces, mediator_records) > config.maximum_model_calls:
                    candidate = self._budget_failure(state, actor_id, "Model-call budget exceeded.")
            if action_elapsed >= config.action_timeout_seconds:
                candidate = self._budget_failure(state, actor_id, "Per-action timeout reached.")

            verified = executor.submit(
                state,
                self.profiles[actor_id],
                candidate,
                declared_strategy=((policy.name,) if policy is not None else ("budget-fallback",)),
                authorized_communication_recipients=tuple(
                    item for item in participant_ids if item != actor_id
                ),
                correction_provider=self.correction_providers.get(actor_id),
            )
            if (
                verified.verification.results
                and verified.verification.results[0].verdict is VerificationVerdict.REJECT
            ):
                invalid_actor_ids.append(actor_id)
            selected = verified.verification.action
            state = verified.transition.state
            action_count += 1
            self._deliver_action_message(selected, verified.transition.events[0])
            self._update_all_memories(protocol, state)
            self._update_external_beliefs(state, beliefs)

            if (
                config.mediation_enabled
                and isinstance(selected, RequestMediationAction)
                and state.phase is ProtocolPhase.MEDIATION
            ):
                # The next loop performs and separately records the intervention.
                continue

        elapsed_ms = max(0.0, (self.clock.monotonic() - started) * 1000.0)
        if self.episode_model_gateway is not None:
            records = self.episode_model_gateway.records
        else:
            policy_records = self._policy_model_records(traces)
            verification_records = model_records_from_verification(self.verifier.audit_log)
            records = policy_records + verification_records + tuple(mediator_records)
        metrics = evaluate_episode(
            EvaluationInput(
                scenario=self.scenario,
                session=state,
                preferences=tuple(self.profiles[item].preferences for item in participant_ids),
                messages=self.message_bus.audit_view(self.audit_reader_id),
                verification_log=self.verifier.audit_log,
                model_calls=records,
                elapsed_time_ms=elapsed_ms,
            ),
            self.evaluation_configuration,
        )
        snapshots = self.memory_store.snapshots()
        return EpisodeResult(
            scenario_id=self.scenario.scenario_id,
            random_seed=self.random_seed,
            stop_reason=stop_reason,
            outcome=state.outcome,
            final_session=state,
            public_transcript=self.message_bus.public_transcript(),
            audit_messages=self.message_bus.audit_view(self.audit_reader_id),
            audit_events=state.event_sequence,
            memory_references=tuple(_memory_reference(item) for item in snapshots),
            metrics=metrics,
            usage=EpisodeUsage(
                model_calls=sum(item.model_calls for item in records),
                latency_ms=sum(item.latency_ms for item in records),
                tokens=aggregate_usage(item.usage for item in records),
                records=records,
            ),
            verification_log=self.verifier.audit_log,
            elapsed_time_ms=elapsed_ms,
        )

    def replay(self, result: EpisodeResult) -> Tuple[NegotiationSession, DeterministicEvaluation]:
        """Reconstruct state and authoritative metrics from immutable episode artifacts."""

        protocol = self.protocol_factory.create(
            scenario=self.scenario,
            maximum_rounds=result.final_session.maximum_rounds,
            initial_turn=self.scenario.participants[0].participant_id,
            feasible_region=self.feasible_region,
            mediator_ids=tuple(self.message_bus.mediator_ids),
        )
        state = protocol.replay(result.audit_events)
        metrics = evaluate_episode(
            EvaluationInput(
                scenario=self.scenario,
                session=state,
                preferences=tuple(
                    self.profiles[item.participant_id].preferences
                    for item in self.scenario.participants
                ),
                messages=result.audit_messages,
                verification_log=result.verification_log,
                model_calls=result.usage.records,
                elapsed_time_ms=result.elapsed_time_ms,
            ),
            self.evaluation_configuration,
        )
        return state, metrics

    def _observe(
        self, protocol: NegotiationProtocol, state: NegotiationSession, actor_id: ParticipantId
    ) -> MemorySnapshot:
        snapshot = self.memory_store.get(actor_id).snapshot()
        if (
            snapshot.working.negotiation_id != state.scenario_id
            or snapshot.working.public_event_count != len(state.event_sequence)
            or snapshot.working.current_turn != state.current_turn
            or snapshot.working.last_visible_message_sequence
            != max(
                (item.sequence_number for item in self.message_bus.inbox(actor_id)),
                default=0,
            )
        ):
            raise OrchestrationError("participant memory is not synchronized with public state")
        if snapshot.working.legal_actions != protocol.legal_action_types(state, actor_id):
            raise OrchestrationError("participant memory contains stale legal actions")
        return snapshot

    def _update_all_memories(
        self, protocol: NegotiationProtocol, state: NegotiationSession
    ) -> None:
        for participant in self.scenario.participants:
            participant_id = participant.participant_id
            self.memory_store.get(participant_id).update(
                AgentObservation(
                    scenario=self.scenario,
                    session=state,
                    own_profile=self.profiles[participant_id],
                ),
                legal_actions=protocol.legal_action_types(state, participant_id),
                visible_messages=self.message_bus.inbox(participant_id),
                active_plan=(
                    (f"Choose one legal {self.policies[participant_id].name} action.",)
                    if state.current_turn == participant_id
                    else ()
                ),
            )

    def _choose_with_failure_policy(
        self,
        actor_id: ParticipantId,
        policy: NegotiationPolicy,
        memory: MemorySnapshot,
        state: NegotiationSession,
    ) -> object:
        try:
            return policy.choose_action(memory)
        except Exception as exc:
            if self.configuration.failure_policy is FailurePolicy.RAISE:
                raise OrchestrationError(
                    f"policy '{policy.name}' failed for participant '{actor_id}'"
                ) from exc
            if self.configuration.failure_policy is FailurePolicy.FALLBACK:
                fallback = self.fallback_policies.get(actor_id, LinearConcessionPolicy())
                try:
                    return fallback.choose_action(memory)
                except Exception as fallback_exc:
                    return self._withdraw_action(
                        state,
                        actor_id,
                        f"Fallback policy failed ({type(fallback_exc).__name__}).",
                    )
            return self._withdraw_action(
                state, actor_id, f"Policy failure ({type(exc).__name__})."
            )

    def _budget_failure(
        self, state: NegotiationSession, actor_id: ParticipantId, reason: str
    ) -> NegotiationAction:
        if self.configuration.failure_policy is FailurePolicy.RAISE:
            raise OrchestrationError(reason)
        if self.configuration.failure_policy is FailurePolicy.FALLBACK:
            memory = self.memory_store.get(actor_id).snapshot()
            fallback = self.fallback_policies.get(actor_id, LinearConcessionPolicy())
            try:
                return fallback.choose_action(memory)
            except Exception as exc:
                return self._withdraw_action(
                    state,
                    actor_id,
                    f"Budget fallback policy failed ({type(exc).__name__}).",
                )
        return self._withdraw_action(state, actor_id, reason)

    @staticmethod
    def _withdraw_action(
        state: NegotiationSession, actor_id: ParticipantId, reason: str
    ) -> WithdrawAction:
        return WithdrawAction(
            actor_id=actor_id,
            recipients=tuple(item for item in state.participants if item != actor_id),
            round_number=state.round_number,
            reason=reason,
        )

    def _verified_withdraw(
        self, executor: VerifiedProtocolExecutor, state: NegotiationSession, reason: str
    ):
        actor_id = state.current_turn
        if actor_id is None:
            raise OrchestrationError("cannot terminate a state without a current actor")
        return executor.submit(
            state,
            self.profiles[actor_id],
            self._withdraw_action(state, actor_id, reason),
            declared_strategy=("orchestrator-failure-policy",),
            authorized_communication_recipients=tuple(
                item for item in state.participants if item != actor_id
            ),
        )

    def _deliver_action_message(self, action: NegotiationAction, event: ProtocolEvent) -> None:
        if isinstance(action, MessageAction):
            all_other_bus_participants = tuple(
                item for item in self.message_bus.participants if item != action.actor_id
            )
            public = set(action.recipients) == set(action_id for action_id in self._other_participants(action.actor_id))
            self.message_bus.send(
                sender=action.actor_id,
                recipients=all_other_bus_participants if public else action.recipients,
                timestamp=self.clock.now(),
                message_type=MessageType.INTENT_SIGNAL,
                visibility=(
                    MessageVisibility.PUBLIC if public else MessageVisibility.DIRECT_PRIVATE
                ),
                content=action.content,
                correlation_id=event.correlation_id,
                referenced_action_id=getattr(event, "action_id", None),
            )
        elif isinstance(action, RequestMediationAction):
            self.message_bus.send(
                sender=action.actor_id,
                recipients=tuple(
                    item for item in self.message_bus.participants if item != action.actor_id
                ),
                timestamp=self.clock.now(),
                message_type=MessageType.MEDIATION_REQUEST,
                visibility=MessageVisibility.PUBLIC,
                content=action.reason or "I request mediation.",
                correlation_id=event.correlation_id,
                referenced_action_id=getattr(event, "action_id", None),
            )

    def _other_participants(self, actor_id: ParticipantId) -> Tuple[ParticipantId, ...]:
        return tuple(
            item.participant_id for item in self.scenario.participants
            if item.participant_id != actor_id
        )

    def _mediator_context(self, state: NegotiationSession) -> MediatorContext:
        mode = self.mediator.configuration.access_mode
        kwargs = {}
        if mode is MediationAccessMode.CONFIDENTIAL_SUMMARY:
            kwargs["confidential_summaries"] = tuple(
                ConfidentialPreferenceSummary(
                    participant_id=item.identity.participant.participant_id,
                    issue_preferences=item.preferences.issue_preferences,
                    reservation_utility=item.preferences.reservation.reservation_utility,
                )
                for item in self.profiles.values()
            )
        elif mode is MediationAccessMode.SIMULATION_ORACLE:
            kwargs["simulation_ground_truth"] = tuple(
                item.preferences for item in self.profiles.values()
            )
        return MediatorContext(
            scenario=self.scenario,
            session=state,
            access_mode=mode,
            public_messages=self.message_bus.public_transcript(),
            **kwargs,
        )

    @staticmethod
    def _has_current_intervention(state: NegotiationSession) -> bool:
        for event in reversed(state.event_sequence):
            if isinstance(event, MediatorInterventionEvent):
                return True
            if isinstance(event, MediationEnteredEvent):
                return False
            if isinstance(event, ActionAppliedEvent) and isinstance(
                event.action, RequestMediationAction
            ):
                return False
        return False

    @staticmethod
    def _mediation_trigger(state: NegotiationSession) -> MediationTrigger:
        for event in reversed(state.event_sequence):
            if event.event_type.value == "mediation_entered":
                return event.trigger
            if (
                event.event_type.value == "action_applied"
                and isinstance(event.action, RequestMediationAction)
            ):
                return MediationTrigger.EXPLICIT_REQUEST
        return MediationTrigger.DEADLOCK

    def _update_external_beliefs(self, state: NegotiationSession, beliefs: dict) -> None:
        for observer_id, modeller in self.opponent_modellers.items():
            snapshot = self.memory_store.get(observer_id).snapshot()
            for opponent_id in self._other_participants(observer_id):
                key = (observer_id, opponent_id)
                belief = modeller.update(
                    beliefs.get(key), evidence_from_memory(snapshot, opponent_id)
                )
                if beliefs.get(key) != belief:
                    self.memory_store.get(observer_id).record_belief_snapshot(belief)
                    beliefs[key] = belief

    def _model_call_count(self, traces: list, mediator_records: list) -> int:
        if self.episode_model_gateway is not None:
            return self.episode_model_gateway.used_calls
        return sum(
            stage.model_calls for trace in traces for stage in trace.stages
        ) + sum(item.model_calls for item in mediator_records)

    def _policy_model_records(self, traces: list) -> Tuple[ModelCallRecord, ...]:
        records = []
        for record in model_records_from_policy_traces(traces):
            policy = self.policies[record.participant_id]
            records.append(
                record.model_copy(
                    update={
                        "provider": getattr(getattr(policy, "client", None), "provider", None),
                        "model": getattr(
                            getattr(policy, "model_configuration", None), "model", None
                        ),
                    }
                )
            )
        return tuple(records)
