from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator

from .beliefs import (
    OpponentBeliefState,
    OpponentModelConfiguration,
    OpponentModelMode,
)
from .domain import (
    AcceptAction,
    ActionType,
    CounterAction,
    MalformedActionError,
    MessageAction,
    NegotiationAction,
    NegotiationScenario,
    Offer,
    OfferId,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    ProposeAction,
    RejectAction,
    RequestMediationAction,
    WithdrawAction,
    calculate_utility,
    parse_action,
)
from .llm import (
    LLMClient,
    LLMClientError,
    LLMErrorCode,
    LLMMessage,
    LLMRequest,
    LLMUsage,
    ModelConfiguration,
)
from .llm_prompts import PROMPT_VERSION, repair_messages, stage_messages
from .memory import (
    ActionMemoryEvent,
    BeliefSnapshotMemoryEvent,
    MemorySnapshot,
    ObservationMemoryEvent,
    OutcomeMemoryEvent,
    ReceivedMessageMemoryEvent,
    RecentOfferMemory,
    SummaryMemoryEvent,
    VisibleMessage,
)
from .policies import LinearConcessionPolicy, NegotiationPolicy, PolicyError
from .opponent import (
    HeuristicOpponentModeller,
    OpponentModeller,
    evidence_from_memory,
)
from .protocol import ProtocolPhase, TERMINAL_PHASES


RATIONALE_LIMIT = 500
EVIDENCE_LIMIT = 10
UTILITY_TOLERANCE = 1e-9


class LLMPolicyError(PolicyError):
    """Raised when neither model output nor the deterministic fallback can act safely."""


class CognitiveStage(str, Enum):
    OBSERVATION = "observation"
    OBJECTIVE_CONSTRAINTS = "objective_constraints"
    OPPONENT_HYPOTHESIS = "opponent_hypothesis"
    PLANNING = "planning"
    ACTION_GENERATION = "action_generation"
    GROUNDING = "grounding"
    FALLBACK = "fallback"


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AgentVisibleObservation(PolicyModel):
    schema_version: Literal["1.0"] = "1.0"
    negotiation_id: str
    participant_id: ParticipantId
    role: ParticipantRole
    public_persona: str
    scenario: NegotiationScenario
    own_preferences: ParticipantPreferences
    goals: Tuple[str, ...]
    protocol_rules: Tuple[str, ...]
    strategy_guidance: Tuple[str, ...]
    learned_strategy_data: Tuple[str, ...]
    phase: ProtocolPhase
    current_turn: Optional[ParticipantId]
    round_number: int = Field(ge=1)
    deadline_round: int = Field(ge=1)
    rounds_remaining: int = Field(ge=0)
    legal_actions: Tuple[ActionType, ...]
    outstanding_offer: Optional[Offer]
    required_offer_id: Optional[OfferId]
    expected_recipients: Tuple[ParticipantId, ...]
    visible_messages: Tuple[VisibleMessage, ...]
    recent_offers: Tuple[RecentOfferMemory, ...]
    active_plan: Tuple[str, ...]
    episodic_facts: Tuple[str, ...]


class _AuditedStageOutput(PolicyModel):
    schema_version: Literal["1.0"] = "1.0"
    decision_rationale: str = Field(min_length=1, max_length=RATIONALE_LIMIT)
    evidence: Tuple[str, ...] = Field(default=(), max_length=EVIDENCE_LIMIT)

    @field_validator("evidence", mode="before")
    @classmethod
    def evidence_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("evidence")
    @classmethod
    def concise_evidence(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not value.strip() or len(value) > RATIONALE_LIMIT for value in values):
            raise ValueError("evidence items must be nonblank and concise")
        return values


class ObjectiveConstraints(_AuditedStageOutput):
    objective: str = Field(min_length=1, max_length=RATIONALE_LIMIT)
    constraints: Tuple[str, ...] = Field(min_length=1, max_length=10)

    @field_validator("constraints", mode="before")
    @classmethod
    def constraints_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class OpponentHypothesis(_AuditedStageOutput):
    hypothesis: str = Field(min_length=1, max_length=RATIONALE_LIMIT)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class NegotiationPlan(_AuditedStageOutput):
    plan: Tuple[str, ...] = Field(min_length=1, max_length=10)
    target_utility: Optional[float] = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("plan", mode="before")
    @classmethod
    def plan_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class StructuredActionDecision(_AuditedStageOutput):
    action: NegotiationAction


class CognitiveStageTelemetry(PolicyModel):
    stage: CognitiveStage
    model_calls: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0, allow_inf_nan=False)
    usage: LLMUsage
    repair_attempted: bool = False
    succeeded: bool
    failure_code: Optional[str] = Field(default=None, max_length=100)


class LLMDecisionTrace(PolicyModel):
    prompt_version: Literal["negotiation-policy-v1"] = PROMPT_VERSION
    negotiation_id: str
    participant_id: ParticipantId
    objective: Optional[ObjectiveConstraints] = None
    opponent_hypothesis: Optional[OpponentHypothesis] = None
    opponent_beliefs: Tuple[OpponentBeliefState, ...] = ()
    plan: Optional[NegotiationPlan] = None
    decision_rationale: Optional[str] = Field(default=None, max_length=RATIONALE_LIMIT)
    evidence: Tuple[str, ...] = ()
    selected_action: NegotiationAction
    used_fallback: bool
    fallback_reason: Optional[str] = Field(default=None, max_length=100)
    stages: Tuple[CognitiveStageTelemetry, ...]


@dataclass
class _MutableTelemetry:
    model_calls: int = 0
    latency_ms: float = 0.0
    usage: LLMUsage = field(default_factory=LLMUsage)
    repair_attempted: bool = False
    succeeded: bool = False
    failure_code: Optional[str] = None

    def freeze(self, stage: CognitiveStage) -> CognitiveStageTelemetry:
        return CognitiveStageTelemetry(
            stage=stage,
            model_calls=self.model_calls,
            latency_ms=self.latency_ms,
            usage=self.usage,
            repair_attempted=self.repair_attempted,
            succeeded=self.succeeded,
            failure_code=self.failure_code,
        )


@dataclass
class _RecoveryBudget:
    available: bool = True

    def consume(self) -> bool:
        if not self.available:
            return False
        self.available = False
        return True


class _StageFailure(RuntimeError):
    def __init__(self, stage: CognitiveStage, code: str, invalid_response: str = "") -> None:
        super().__init__(code)
        self.stage = stage
        self.code = code
        self.invalid_response = invalid_response


_OutputT = TypeVar("_OutputT", bound=BaseModel)


_ACTION_MODELS = {
    ActionType.PROPOSE: ProposeAction,
    ActionType.COUNTER: CounterAction,
    ActionType.ACCEPT: AcceptAction,
    ActionType.REJECT: RejectAction,
    ActionType.MESSAGE: MessageAction,
    ActionType.REQUEST_MEDIATION: RequestMediationAction,
    ActionType.WITHDRAW: WithdrawAction,
}


def _legal_action_response_schema(
    legal_actions: Tuple[ActionType, ...],
) -> Mapping[str, object]:
    action_models = tuple(_ACTION_MODELS[action] for action in legal_actions)
    if not action_models:
        raise LLMPolicyError("cannot build an action schema without legal actions")
    action_union = Union[action_models]  # type: ignore[arg-type]
    discriminated_action = Annotated[action_union, Field(discriminator="action")]
    legal_decision = create_model(
        "LegalStructuredActionDecision",
        __base__=_AuditedStageOutput,
        action=(discriminated_action, ...),
    )
    return legal_decision.model_json_schema()


def _expected_recipients(memory: MemorySnapshot) -> Tuple[ParticipantId, ...]:
    return tuple(
        participant.participant_id
        for participant in memory.long_term.public_participants
        if participant.participant_id != memory.owner_id
    )


def _required_offer_id(memory: MemorySnapshot) -> OfferId:
    source = (
        f"{PROMPT_VERSION}|{memory.working.negotiation_id}|{memory.owner_id}|"
        f"{memory.working.round_number}|{memory.working.public_event_count}"
    )
    return OfferId(f"offer-{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}")


def _episodic_fact(event: object) -> str:
    if isinstance(event, ObservationMemoryEvent):
        return event.fact
    if isinstance(event, ActionMemoryEvent):
        return f"Own action: {event.action.action.value}."
    if isinstance(event, ReceivedMessageMemoryEvent):
        return f"Visible message from {event.message.sender}: {event.message.content}"
    if isinstance(event, OutcomeMemoryEvent):
        return f"Outcome: {event.outcome.status.value}."
    if isinstance(event, BeliefSnapshotMemoryEvent):
        return (
            f"Opponent belief snapshot for {event.belief.opponent_id} with "
            f"confidence {event.belief.confidence:.2f}."
        )
    if isinstance(event, SummaryMemoryEvent):
        return event.content
    raise LLMPolicyError("unsupported participant memory event")


class LLMNegotiationPolicy:
    """Staged LLM policy that returns only grounded domain actions."""

    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        model_configuration: ModelConfiguration,
        *,
        fallback_policy: Optional[NegotiationPolicy] = None,
        opponent_configuration: Optional[OpponentModelConfiguration] = None,
        opponent_modeller: Optional[OpponentModeller] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.client = client
        self.model_configuration = model_configuration
        self.fallback_policy = fallback_policy or LinearConcessionPolicy()
        self.opponent_configuration = opponent_configuration or OpponentModelConfiguration()
        if (
            self.opponent_configuration.mode is OpponentModelMode.LLM
            and opponent_modeller is None
        ):
            raise ValueError("LLM opponent modelling mode requires an explicit LLM modeller")
        self.opponent_modeller = opponent_modeller or HeuristicOpponentModeller()
        self._clock = clock
        self._hypotheses: Dict[Tuple[str, ParticipantId], OpponentHypothesis] = {}
        self._beliefs: Dict[
            Tuple[str, ParticipantId, ParticipantId], OpponentBeliefState
        ] = {}
        self._cumulative: Dict[CognitiveStage, _MutableTelemetry] = {
            stage: _MutableTelemetry() for stage in CognitiveStage
        }
        self.last_trace: Optional[LLMDecisionTrace] = None
        self.last_belief_states: Tuple[OpponentBeliefState, ...] = ()

    @property
    def cumulative_telemetry(self) -> Tuple[CognitiveStageTelemetry, ...]:
        return tuple(self._cumulative[stage].freeze(stage) for stage in CognitiveStage)

    def build_agent_visible_observation(self, memory: MemorySnapshot) -> AgentVisibleObservation:
        self._validate_memory_for_action(memory)
        needs_offer_id = any(
            action in memory.working.legal_actions
            for action in (ActionType.PROPOSE, ActionType.COUNTER)
        )
        return AgentVisibleObservation(
            negotiation_id=memory.working.negotiation_id,
            participant_id=memory.owner_id,
            role=memory.long_term.role,
            public_persona=memory.long_term.public_persona,
            scenario=memory.scenario,
            own_preferences=memory.long_term.own_preferences,
            goals=memory.long_term.goals,
            protocol_rules=memory.long_term.protocol_rules,
            strategy_guidance=memory.long_term.strategy_guidance,
            learned_strategy_data=memory.long_term.learned_strategy_data,
            phase=memory.working.phase,
            current_turn=memory.working.current_turn,
            round_number=memory.working.round_number,
            deadline_round=memory.working.deadline_round,
            rounds_remaining=memory.working.rounds_remaining,
            legal_actions=memory.working.legal_actions,
            outstanding_offer=memory.working.outstanding_offer,
            required_offer_id=_required_offer_id(memory) if needs_offer_id else None,
            expected_recipients=_expected_recipients(memory),
            visible_messages=memory.working.visible_messages,
            recent_offers=memory.working.recent_offers,
            active_plan=memory.working.active_plan,
            episodic_facts=tuple(_episodic_fact(event) for event in memory.episodic.events[-12:]),
        )

    def choose_action(self, memory: MemorySnapshot) -> NegotiationAction:
        self.last_belief_states = ()
        telemetry = {stage: _MutableTelemetry() for stage in CognitiveStage}
        recovery = _RecoveryBudget()
        objective: Optional[ObjectiveConstraints] = None
        hypothesis: Optional[OpponentHypothesis] = None
        plan: Optional[NegotiationPlan] = None
        decision: Optional[StructuredActionDecision] = None

        try:
            observation = self.build_agent_visible_observation(memory)
            telemetry[CognitiveStage.OBSERVATION].succeeded = True
            objective = self.summarize_objective(observation, telemetry, recovery)
            beliefs = self.update_opponent_beliefs(memory)
            self.last_belief_states = beliefs
            hypothesis = self.form_opponent_hypothesis(
                observation, objective, beliefs, telemetry, recovery
            )
            self._hypotheses[(observation.negotiation_id, observation.participant_id)] = hypothesis
            plan = self.choose_negotiation_plan(
                observation, objective, hypothesis, beliefs, telemetry, recovery
            )
            decision, raw_response = self.generate_structured_action(
                observation, objective, hypothesis, plan, telemetry, recovery
            )
            try:
                action = self.ground_action(memory, decision)
                telemetry[CognitiveStage.GROUNDING].succeeded = True
            except (PolicyError, MalformedActionError, ValueError) as error:
                telemetry[CognitiveStage.GROUNDING].failure_code = "illegal_or_ungrounded_action"
                telemetry[CognitiveStage.ACTION_GENERATION].succeeded = False
                telemetry[CognitiveStage.ACTION_GENERATION].failure_code = (
                    "illegal_or_ungrounded_action"
                )
                if not recovery.consume():
                    raise _StageFailure(
                        CognitiveStage.GROUNDING, "illegal_or_ungrounded_action", raw_response
                    ) from error
                telemetry[CognitiveStage.ACTION_GENERATION].repair_attempted = True
                decision, _ = self._repair_structured(
                    CognitiveStage.ACTION_GENERATION,
                    StructuredActionDecision,
                    self._action_context(observation, objective, hypothesis, plan),
                    raw_response,
                    "Return a schema-valid action using exactly the supplied actor, round, legal "
                    "action type, recipients, outstanding-offer reference, and required offer id.",
                    telemetry,
                    response_schema=_legal_action_response_schema(observation.legal_actions),
                )
                try:
                    action = self.ground_action(memory, decision)
                except (PolicyError, MalformedActionError, ValueError) as repair_error:
                    telemetry[CognitiveStage.ACTION_GENERATION].succeeded = False
                    telemetry[CognitiveStage.ACTION_GENERATION].failure_code = "repair_failed"
                    raise _StageFailure(
                        CognitiveStage.ACTION_GENERATION, "repair_failed"
                    ) from repair_error
                telemetry[CognitiveStage.ACTION_GENERATION].succeeded = True
                telemetry[CognitiveStage.ACTION_GENERATION].failure_code = None
                telemetry[CognitiveStage.GROUNDING].succeeded = True
                telemetry[CognitiveStage.GROUNDING].failure_code = None
        except _StageFailure as failure:
            telemetry[failure.stage].failure_code = failure.code
            action = self._safe_fallback(memory, telemetry)
            self._finish_trace(
                memory,
                telemetry,
                action,
                objective,
                hypothesis,
                plan,
                decision,
                beliefs=self.last_belief_states,
                used_fallback=True,
                fallback_reason=failure.code,
            )
            return action

        self._finish_trace(
            memory,
            telemetry,
            action,
            objective,
            hypothesis,
            plan,
            decision,
            beliefs=self.last_belief_states,
            used_fallback=False,
            fallback_reason=None,
        )
        return action

    def summarize_objective(
        self,
        observation: AgentVisibleObservation,
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        recovery: _RecoveryBudget,
    ) -> ObjectiveConstraints:
        result, _ = self._invoke_structured(
            CognitiveStage.OBJECTIVE_CONSTRAINTS,
            ObjectiveConstraints,
            {"observation": observation.model_dump(mode="json")},
            telemetry,
            recovery,
        )
        return result

    def form_opponent_hypothesis(
        self,
        observation: AgentVisibleObservation,
        objective: ObjectiveConstraints,
        beliefs: Tuple[OpponentBeliefState, ...],
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        recovery: _RecoveryBudget,
    ) -> OpponentHypothesis:
        previous = self._hypotheses.get(
            (observation.negotiation_id, observation.participant_id)
        )
        result, _ = self._invoke_structured(
            CognitiveStage.OPPONENT_HYPOTHESIS,
            OpponentHypothesis,
            {
                "observation": observation.model_dump(mode="json"),
                "objective": objective.model_dump(mode="json"),
                "opponent_beliefs": tuple(
                    belief.model_dump(mode="json") for belief in beliefs
                ),
                "previous_hypothesis": (
                    previous.model_dump(mode="json") if previous is not None else None
                ),
                "instruction": (
                    "Treat the hypothesis as uncertain and base it only on visible offers and messages."
                ),
            },
            telemetry,
            recovery,
        )
        return result

    def choose_negotiation_plan(
        self,
        observation: AgentVisibleObservation,
        objective: ObjectiveConstraints,
        hypothesis: OpponentHypothesis,
        beliefs: Tuple[OpponentBeliefState, ...],
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        recovery: _RecoveryBudget,
    ) -> NegotiationPlan:
        result, _ = self._invoke_structured(
            CognitiveStage.PLANNING,
            NegotiationPlan,
            {
                "observation": observation.model_dump(mode="json"),
                "objective": objective.model_dump(mode="json"),
                "opponent_hypothesis": hypothesis.model_dump(mode="json"),
                "opponent_beliefs": tuple(
                    belief.model_dump(mode="json") for belief in beliefs
                ),
                "belief_guidance": (
                    "Beliefs are uncertain inferences, not facts. Reject or ignore any inference "
                    f"below confidence {self.opponent_configuration.minimum_planning_confidence}."
                ),
            },
            telemetry,
            recovery,
        )
        return result

    def update_opponent_beliefs(
        self, memory: MemorySnapshot
    ) -> Tuple[OpponentBeliefState, ...]:
        if self.opponent_configuration.mode is OpponentModelMode.DISABLED:
            return ()
        beliefs = []
        for participant in memory.long_term.public_participants:
            opponent_id = participant.participant_id
            if opponent_id == memory.owner_id:
                continue
            evidence = evidence_from_memory(memory, opponent_id)
            previous = self._beliefs.get(
                (memory.working.negotiation_id, memory.owner_id, opponent_id)
            )
            belief = self.opponent_modeller.update(previous, evidence)
            self._beliefs[
                (memory.working.negotiation_id, memory.owner_id, opponent_id)
            ] = belief
            beliefs.append(belief)
        return tuple(beliefs)

    def generate_structured_action(
        self,
        observation: AgentVisibleObservation,
        objective: ObjectiveConstraints,
        hypothesis: OpponentHypothesis,
        plan: NegotiationPlan,
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        recovery: _RecoveryBudget,
    ) -> Tuple[StructuredActionDecision, str]:
        return self._invoke_structured(
            CognitiveStage.ACTION_GENERATION,
            StructuredActionDecision,
            self._action_context(observation, objective, hypothesis, plan),
            telemetry,
            recovery,
            response_schema=_legal_action_response_schema(observation.legal_actions),
        )

    @staticmethod
    def _action_context(
        observation: AgentVisibleObservation,
        objective: ObjectiveConstraints,
        hypothesis: OpponentHypothesis,
        plan: NegotiationPlan,
    ) -> Mapping[str, object]:
        return {
            "observation": observation.model_dump(mode="json"),
            "objective": objective.model_dump(mode="json"),
            "opponent_hypothesis": hypothesis.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "instruction": (
                "Choose exactly one action from observation.legal_actions. Contextual identifiers "
                "must exactly match observation participant, round, recipients, outstanding offer, "
                "and required offer id. Messages communicate only and do not execute other actions."
            ),
        }

    def ground_action(
        self,
        memory: MemorySnapshot,
        decision: StructuredActionDecision,
    ) -> NegotiationAction:
        try:
            action_data = json.loads(decision.action.model_dump_json())
            action = parse_action(action_data)
        except (json.JSONDecodeError, MalformedActionError) as error:
            raise LLMPolicyError("model action failed domain parsing") from error
        self._validate_grounded_action(memory, action, enforce_required_offer_id=True)
        return action

    def _validate_grounded_action(
        self,
        memory: MemorySnapshot,
        action: NegotiationAction,
        *,
        enforce_required_offer_id: bool,
    ) -> None:
        scenario = memory.scenario
        scenario.validate_action(action)
        if action.actor_id != memory.owner_id:
            raise LLMPolicyError("grounded action actor does not match memory owner")
        if action.round_number != memory.working.round_number:
            raise LLMPolicyError("grounded action round does not match working memory")
        if action.action not in memory.working.legal_actions:
            raise LLMPolicyError("grounded action type is not currently legal")
        if action.recipients != _expected_recipients(memory):
            raise LLMPolicyError("grounded action recipients do not match the public participants")

        outstanding = memory.working.outstanding_offer
        if isinstance(action, ProposeAction):
            if outstanding is not None:
                raise LLMPolicyError("a proposal cannot replace an outstanding offer")
            self._validate_offer_constraints(
                memory, action.offer, enforce_required_offer_id=enforce_required_offer_id
            )
        elif isinstance(action, CounterAction):
            if outstanding is None or action.responds_to != outstanding.offer_id:
                raise LLMPolicyError("counteroffer does not reference the outstanding offer")
            self._validate_offer_constraints(
                memory, action.offer, enforce_required_offer_id=enforce_required_offer_id
            )
        elif isinstance(action, (AcceptAction, RejectAction)):
            if outstanding is None or action.offer_id != outstanding.offer_id:
                raise LLMPolicyError("action does not reference the outstanding offer")
            if isinstance(action, AcceptAction):
                utility = calculate_utility(
                    scenario, outstanding, memory.long_term.own_preferences
                )
                reservation = memory.long_term.own_preferences.reservation.reservation_utility
                if utility + UTILITY_TOLERANCE < reservation:
                    raise LLMPolicyError("acceptance would violate reservation utility")
                violation = memory.long_term.own_preferences.budget_violation(outstanding)
                if violation is not None:
                    raise LLMPolicyError(f"acceptance would violate budget limit: {violation}")
        elif not isinstance(
            action, (MessageAction, RequestMediationAction, WithdrawAction)
        ):
            raise LLMPolicyError("unsupported grounded action")

    def _validate_offer_constraints(
        self,
        memory: MemorySnapshot,
        offer: Offer,
        *,
        enforce_required_offer_id: bool,
    ) -> None:
        memory.scenario.validate_offer(offer)
        if enforce_required_offer_id and offer.offer_id != _required_offer_id(memory):
            raise LLMPolicyError("offer identifier does not match the grounded identifier")
        utility = calculate_utility(
            memory.scenario, offer, memory.long_term.own_preferences
        )
        reservation = memory.long_term.own_preferences.reservation.reservation_utility
        if utility + UTILITY_TOLERANCE < reservation:
            raise LLMPolicyError("offer would violate reservation utility")
        violation = memory.long_term.own_preferences.budget_violation(offer)
        if violation is not None:
            raise LLMPolicyError(f"offer would violate budget limit: {violation}")

    def _invoke_structured(
        self,
        stage: CognitiveStage,
        output_type: Type[_OutputT],
        context: Mapping[str, object],
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        recovery: _RecoveryBudget,
        response_schema: Optional[Mapping[str, object]] = None,
    ) -> Tuple[_OutputT, str]:
        schema = response_schema or output_type.model_json_schema()
        messages = stage_messages(stage.value, schema, context)
        raw = self._call_model(stage, messages, telemetry)
        try:
            result = output_type.model_validate_json(raw)
        except (ValidationError, ValueError):
            telemetry[stage].failure_code = "malformed_structured_output"
            if not recovery.consume():
                raise _StageFailure(stage, "malformed_structured_output", raw)
            telemetry[stage].repair_attempted = True
            return self._repair_structured(
                stage,
                output_type,
                context,
                raw,
                "Return valid JSON matching the response schema exactly.",
                telemetry,
                response_schema=schema,
            )
        telemetry[stage].succeeded = True
        telemetry[stage].failure_code = None
        return result, raw

    def _repair_structured(
        self,
        stage: CognitiveStage,
        output_type: Type[_OutputT],
        context: Mapping[str, object],
        invalid_response: str,
        guidance: str,
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        response_schema: Optional[Mapping[str, object]] = None,
    ) -> Tuple[_OutputT, str]:
        schema = response_schema or output_type.model_json_schema()
        messages = repair_messages(
            stage.value,
            schema,
            context,
            invalid_response,
            guidance,
        )
        raw = self._call_model(stage, messages, telemetry)
        try:
            result = output_type.model_validate_json(raw)
        except (ValidationError, ValueError) as error:
            telemetry[stage].succeeded = False
            telemetry[stage].failure_code = "repair_failed"
            raise _StageFailure(stage, "repair_failed", raw) from error
        telemetry[stage].succeeded = True
        telemetry[stage].failure_code = None
        return result, raw

    def _call_model(
        self,
        stage: CognitiveStage,
        messages: Tuple[LLMMessage, ...],
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
    ) -> str:
        metrics = telemetry[stage]
        request = LLMRequest(
            request_id=self._request_id(stage, metrics.model_calls + 1),
            messages=messages,
            model_configuration=self.model_configuration,
        )
        started = self._clock()
        try:
            response = self.client.generate(request)
        except LLMClientError as error:
            metrics.model_calls += (
                0
                if error.info.code is LLMErrorCode.BUDGET_EXCEEDED
                else max(1, error.info.attempts)
            )
            metrics.latency_ms += max(0.0, (self._clock() - started) * 1000.0)
            metrics.succeeded = False
            metrics.failure_code = f"llm_{error.info.code.value}"
            raise _StageFailure(stage, metrics.failure_code) from error
        metrics.model_calls += response.attempts
        metrics.latency_ms += response.latency.total_ms
        metrics.usage = metrics.usage + response.usage
        return response.text

    def _request_id(self, stage: CognitiveStage, call_number: int) -> str:
        return f"llm-policy-{stage.value}-{call_number}"

    def _safe_fallback(
        self,
        memory: MemorySnapshot,
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
    ) -> NegotiationAction:
        metrics = telemetry[CognitiveStage.FALLBACK]
        try:
            action = self.fallback_policy.choose_action(memory)
            self._validate_grounded_action(
                memory, action, enforce_required_offer_id=False
            )
        except (PolicyError, ValueError) as error:
            if ActionType.WITHDRAW not in memory.working.legal_actions:
                metrics.failure_code = "no_safe_fallback"
                raise LLMPolicyError("no deterministic protocol-valid fallback is available") from error
            action = WithdrawAction(
                actor_id=memory.owner_id,
                recipients=_expected_recipients(memory),
                round_number=memory.working.round_number,
                reason="Deterministic fallback after model failure.",
            )
            self._validate_grounded_action(
                memory, action, enforce_required_offer_id=False
            )
        metrics.succeeded = True
        return action

    @staticmethod
    def _validate_memory_for_action(memory: MemorySnapshot) -> None:
        if memory.working.phase in TERMINAL_PHASES:
            raise LLMPolicyError("an LLM policy cannot act in a terminal session")
        if memory.working.current_turn != memory.owner_id:
            raise LLMPolicyError("LLM policy participant does not hold the current turn")
        if not memory.working.legal_actions:
            raise LLMPolicyError("working memory contains no legal actions")

    def _finish_trace(
        self,
        memory: MemorySnapshot,
        telemetry: Dict[CognitiveStage, _MutableTelemetry],
        action: NegotiationAction,
        objective: Optional[ObjectiveConstraints],
        hypothesis: Optional[OpponentHypothesis],
        plan: Optional[NegotiationPlan],
        decision: Optional[StructuredActionDecision],
        beliefs: Tuple[OpponentBeliefState, ...],
        *,
        used_fallback: bool,
        fallback_reason: Optional[str],
    ) -> None:
        frozen = tuple(telemetry[stage].freeze(stage) for stage in CognitiveStage)
        self.last_trace = LLMDecisionTrace(
            negotiation_id=memory.working.negotiation_id,
            participant_id=memory.owner_id,
            objective=objective,
            opponent_hypothesis=hypothesis,
            opponent_beliefs=beliefs,
            plan=plan,
            decision_rationale=(decision.decision_rationale if decision is not None else None),
            evidence=(decision.evidence if decision is not None else ()),
            selected_action=action,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            stages=frozen,
        )
        for stage, current in telemetry.items():
            cumulative = self._cumulative[stage]
            cumulative.model_calls += current.model_calls
            cumulative.latency_ms += current.latency_ms
            cumulative.usage = cumulative.usage + current.usage
            cumulative.repair_attempted = (
                cumulative.repair_attempted or current.repair_attempted
            )
            cumulative.succeeded = cumulative.succeeded or current.succeeded
            cumulative.failure_code = current.failure_code or cumulative.failure_code
