from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, Tuple

from pydantic import Field, ValidationError, field_validator, model_validator

from .domain import (
    AcceptAction,
    ActionType,
    CounterAction,
    DomainError,
    IllegalActorError,
    IncompleteOfferError,
    IssueValueError,
    MalformedActionError,
    MessageAction,
    NegotiationAction,
    OfferValidationError,
    ParticipantId,
    ProtocolError,
    ProposeAction,
    RoundMismatchError,
    StaleOfferError,
    UnknownIssueError,
    WithdrawAction,
    calculate_utility,
    parse_action,
)
from .domain.models import DomainModel
from .llm import LLMClient, LLMClientError, LLMProvider, LLMRequest, LLMUsage, ModelConfiguration
from .llm_prompts.action_verifier_v1 import PROMPT_VERSION, action_verifier_messages
from .llm_prompts.action_correction_v1 import (
    PROMPT_VERSION as CORRECTION_PROMPT_VERSION,
    action_correction_messages,
)
from .memory import AgentProfile, MemorySnapshot
from .protocol import (
    NegotiationProtocol,
    NegotiationSession,
    TERMINAL_PHASES,
    TransitionResult,
    legal_action_types_for,
)


UTILITY_TOLERANCE = 1e-9
VERIFIER_VERSION = "1.0"


class VerificationVerdict(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    SKIPPED = "skipped"


class VerificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class VerificationReasonCode(str, Enum):
    SCHEMA_INVALID = "schema_invalid"
    ACTOR_INVALID = "actor_invalid"
    TURN_INVALID = "turn_invalid"
    ISSUE_BOUNDS = "issue_bounds"
    OFFER_INCOMPLETE = "offer_incomplete"
    UNKNOWN_ISSUE = "unknown_issue"
    STALE_ACCEPTANCE = "stale_acceptance"
    STALE_COUNTEROFFER = "stale_counteroffer"
    FABRICATED_OFFER_ID = "fabricated_offer_id"
    RESERVATION_UTILITY_VIOLATION = "reservation_utility_violation"
    BUDGET_LIMIT_VIOLATION = "budget_limit_violation"
    DEADLINE_VIOLATION = "deadline_violation"
    UNAUTHORIZED_COMMUNICATION = "unauthorized_communication"
    PROMPT_INJECTION = "prompt_injection"
    PROTOCOL_ILLEGAL = "protocol_illegal"
    PUBLIC_RULE_INCONSISTENCY = "public_rule_inconsistency"
    DECLARED_STRATEGY_INCONSISTENCY = "declared_strategy_inconsistency"
    VISIBLE_EVIDENCE_INCONSISTENCY = "visible_evidence_inconsistency"
    SAFETY_RISK = "safety_risk"
    VERIFIER_FAILURE = "verifier_failure"
    CORRECTION_FAILURE = "correction_failure"
    VERIFICATION_DISABLED = "verification_disabled"


class VerifierKind(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    ABLATION = "ablation"


class VerificationMode(str, Enum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    DETERMINISTIC_AND_LLM = "deterministic_and_llm"


class VerificationReason(DomainModel):
    code: VerificationReasonCode
    message: str = Field(min_length=1, max_length=300)


class VerifierMetadata(DomainModel):
    verifier_name: str = Field(min_length=1, max_length=100)
    verifier_version: str = Field(min_length=1, max_length=30)
    kind: VerifierKind
    provider: Optional[LLMProvider] = None
    model: Optional[str] = Field(default=None, min_length=1, max_length=200)
    latency_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    model_calls: int = Field(default=0, ge=0)


class VerificationResult(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    verdict: VerificationVerdict
    reasons: Tuple[VerificationReason, ...] = ()
    severity: VerificationSeverity
    checked_action_id: str = Field(min_length=1, max_length=100)
    verifier_metadata: Tuple[VerifierMetadata, ...]

    @field_validator("reasons", "verifier_metadata", mode="before")
    @classmethod
    def tuples_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def verdict_matches_reasons(self) -> VerificationResult:
        if self.verdict is VerificationVerdict.PASS and self.reasons:
            raise ValueError("passing verification cannot contain rejection reasons")
        if self.verdict is VerificationVerdict.REJECT and not self.reasons:
            raise ValueError("rejected verification requires machine-readable reasons")
        if not self.verifier_metadata:
            raise ValueError("verification requires verifier metadata")
        if self.verdict in {VerificationVerdict.PASS, VerificationVerdict.SKIPPED}:
            if self.severity is not VerificationSeverity.INFO:
                raise ValueError("passing or skipped verification must use info severity")
        elif self.severity is VerificationSeverity.INFO:
            raise ValueError("rejected verification must not use info severity")
        reason_codes = tuple(item.code for item in self.reasons)
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("verification reason codes must be unique")
        return self


class VerificationConfiguration(DomainModel):
    mode: VerificationMode = VerificationMode.DETERMINISTIC
    maximum_corrections: int = Field(default=1, ge=0, le=1)


class CorrectionFeedback(DomainModel):
    checked_action_id: str = Field(min_length=1, max_length=100)
    reasons: Tuple[VerificationReason, ...]
    correction_attempt: int = Field(ge=1, le=1)

    @field_validator("reasons", mode="before")
    @classmethod
    def reasons_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class VerificationLogEntry(DomainModel):
    negotiation_id: str = Field(min_length=1, max_length=100)
    sequence_number: int = Field(ge=1)
    results: Tuple[VerificationResult, ...]
    correction_count: int = Field(ge=0, le=1)
    used_fallback: bool
    selected_action_type: ActionType

    @field_validator("results", mode="before")
    @classmethod
    def results_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True)
class VerificationContext:
    protocol: NegotiationProtocol
    state: NegotiationSession
    actor_profile: AgentProfile
    declared_strategy: Tuple[str, ...] = ()
    visible_evidence: Tuple[str, ...] = ()
    authorized_communication_recipients: Optional[Tuple[ParticipantId, ...]] = None

    def __post_init__(self) -> None:
        actor_id = self.actor_profile.identity.participant.participant_id
        if self.state.scenario_id != self.protocol.scenario.scenario_id:
            raise ValueError("verification state does not belong to its protocol scenario")
        if self.state.maximum_rounds != self.protocol.maximum_rounds:
            raise ValueError("verification state deadline does not match its protocol")
        self.protocol.scenario.participant(actor_id)
        if self.actor_profile.preferences.participant_id != actor_id:
            raise ValueError("verification profile identity and preferences must match")
        expected_issues = {issue.issue_id for issue in self.protocol.scenario.issues}
        preference_issues = {
            preference.issue_id
            for preference in self.actor_profile.preferences.issue_preferences
        }
        if preference_issues != expected_issues:
            raise ValueError("verification profile preferences must cover the scenario issues")
        recipients = self.authorized_communication_recipients
        if recipients is not None:
            if len(set(recipients)) != len(recipients):
                raise ValueError("authorized communication recipients must be unique")
            for participant_id in recipients:
                self.protocol.scenario.participant(participant_id)

    @property
    def actor_id(self) -> ParticipantId:
        return self.actor_profile.identity.participant.participant_id

    @property
    def communication_recipients(self) -> Tuple[ParticipantId, ...]:
        if self.authorized_communication_recipients is not None:
            return self.authorized_communication_recipients
        return tuple(item for item in self.state.participants if item != self.actor_id)


class ActionVerifier(Protocol):
    def verify(
        self,
        context: VerificationContext,
        candidate: object,
        checked_action_id: str,
    ) -> VerificationResult:
        """Return a side-effect-free verification result."""


def _parse_candidate(candidate: object) -> NegotiationAction:
    if hasattr(candidate, "action") and hasattr(candidate, "model_dump"):
        return parse_action(candidate.model_dump(mode="python"))
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise MalformedActionError("action is not valid JSON") from error
    return parse_action(candidate)


def _reason(
    code: VerificationReasonCode, message: str
) -> VerificationReason:
    return VerificationReason(code=code, message=message)


def _severity(reasons: Tuple[VerificationReason, ...]) -> VerificationSeverity:
    critical = {
        VerificationReasonCode.PROMPT_INJECTION,
        VerificationReasonCode.UNAUTHORIZED_COMMUNICATION,
        VerificationReasonCode.SAFETY_RISK,
    }
    if any(item.code in critical for item in reasons):
        return VerificationSeverity.CRITICAL
    return VerificationSeverity.ERROR if reasons else VerificationSeverity.INFO


def _append_reason(
    reasons: list[VerificationReason], code: VerificationReasonCode, message: str
) -> None:
    if code not in {item.code for item in reasons}:
        reasons.append(_reason(code, message))


class DeterministicActionVerifier:
    """Authoritative preflight checks using domain rules and the pure protocol transition."""

    metadata = VerifierMetadata(
        verifier_name="deterministic-action-verifier",
        verifier_version=VERIFIER_VERSION,
        kind=VerifierKind.DETERMINISTIC,
    )

    def verify(
        self,
        context: VerificationContext,
        candidate: object,
        checked_action_id: str,
    ) -> VerificationResult:
        try:
            action = _parse_candidate(candidate)
        except (MalformedActionError, ValidationError, TypeError, ValueError):
            reasons = (_reason(
                VerificationReasonCode.SCHEMA_INVALID,
                "Candidate action does not match the typed negotiation-action schema.",
            ),)
            return self._result(checked_action_id, reasons)

        reasons: list[VerificationReason] = []
        if action.actor_id != context.actor_id:
            _append_reason(
                reasons,
                VerificationReasonCode.ACTOR_INVALID,
                "Action actor does not match the proposing participant.",
            )
        if action.round_number != context.state.round_number:
            code = (
                VerificationReasonCode.DEADLINE_VIOLATION
                if action.round_number > context.state.maximum_rounds
                or context.state.phase in TERMINAL_PHASES
                else VerificationReasonCode.TURN_INVALID
            )
            _append_reason(reasons, code, "Action round is not valid for the current session.")
        if context.state.phase in TERMINAL_PHASES:
            _append_reason(
                reasons,
                VerificationReasonCode.DEADLINE_VIOLATION,
                "No action is legal after the negotiation has terminated.",
            )
        if (
            context.state.round_number == context.state.maximum_rounds
            and isinstance(action, MessageAction)
        ):
            _append_reason(
                reasons,
                VerificationReasonCode.DEADLINE_VIOLATION,
                "Non-progressing communication is not safe on the final negotiation round.",
            )

        try:
            legal = legal_action_types_for(context.state, action.actor_id)
            if action.action not in legal:
                _append_reason(
                    reasons,
                    VerificationReasonCode.TURN_INVALID,
                    "Action type is not legal for this actor in the current turn and phase.",
                )
        except (IllegalActorError, ProtocolError):
            _append_reason(
                reasons,
                VerificationReasonCode.ACTOR_INVALID,
                "Action actor is not authorized for this negotiation.",
            )

        if isinstance(action, MessageAction):
            authorized = set(context.communication_recipients)
            if not action.recipients or not set(action.recipients).issubset(authorized):
                _append_reason(
                    reasons,
                    VerificationReasonCode.UNAUTHORIZED_COMMUNICATION,
                    "Message recipients are outside the proposer-authorized communication view.",
                )
            normalized = action.content.casefold()
            injection_markers = (
                "ignore previous instructions",
                "ignore all previous",
                "reveal your system prompt",
                "show your system prompt",
                "reveal private preferences",
                "reveal chain-of-thought",
                "api key",
            )
            if any(marker in normalized for marker in injection_markers):
                _append_reason(
                    reasons,
                    VerificationReasonCode.PROMPT_INJECTION,
                    "Message contains a disallowed instruction or secret-exfiltration pattern.",
                )

        offer = action.offer if isinstance(action, (ProposeAction, CounterAction)) else None
        if offer is not None:
            try:
                context.protocol.scenario.validate_offer(offer)
            except IncompleteOfferError:
                _append_reason(
                    reasons,
                    VerificationReasonCode.OFFER_INCOMPLETE,
                    "Offer does not contain every public scenario issue.",
                )
            except UnknownIssueError:
                _append_reason(
                    reasons,
                    VerificationReasonCode.UNKNOWN_ISSUE,
                    "Offer references an issue outside the public scenario.",
                )
            except (IssueValueError, OfferValidationError):
                _append_reason(
                    reasons,
                    VerificationReasonCode.ISSUE_BOUNDS,
                    "Offer contains a value outside its public issue domain.",
                )
            else:
                utility = calculate_utility(
                    context.protocol.scenario,
                    offer,
                    context.actor_profile.preferences,
                )
                reservation = context.actor_profile.preferences.reservation.reservation_utility
                if utility + UTILITY_TOLERANCE < reservation:
                    _append_reason(
                        reasons,
                        VerificationReasonCode.RESERVATION_UTILITY_VIOLATION,
                        "Candidate offer is below the proposer's reservation utility.",
                    )
                if context.actor_profile.preferences.budget_violation(offer) is not None:
                    _append_reason(
                        reasons,
                        VerificationReasonCode.BUDGET_LIMIT_VIOLATION,
                        "Candidate offer price is outside the proposer's budget limit.",
                    )

        if isinstance(action, AcceptAction):
            outstanding = context.state.latest_valid_offer
            if outstanding is None or action.offer_id != outstanding.offer_id:
                _append_reason(
                    reasons,
                    VerificationReasonCode.STALE_ACCEPTANCE,
                    "Acceptance does not reference the outstanding offer.",
                )
                _append_reason(
                    reasons,
                    VerificationReasonCode.FABRICATED_OFFER_ID,
                    "Acceptance references an offer identifier not currently outstanding.",
                )
            else:
                utility = calculate_utility(
                    context.protocol.scenario,
                    outstanding,
                    context.actor_profile.preferences,
                )
                reservation = context.actor_profile.preferences.reservation.reservation_utility
                if utility + UTILITY_TOLERANCE < reservation:
                    _append_reason(
                        reasons,
                        VerificationReasonCode.RESERVATION_UTILITY_VIOLATION,
                        "Acceptance would cross the proposer's reservation utility.",
                    )
                if context.actor_profile.preferences.budget_violation(outstanding) is not None:
                    _append_reason(
                        reasons,
                        VerificationReasonCode.BUDGET_LIMIT_VIOLATION,
                        "Acceptance price is outside the acceptor's budget limit.",
                    )
        elif isinstance(action, CounterAction):
            outstanding = context.state.latest_valid_offer
            if outstanding is None or action.responds_to != outstanding.offer_id:
                _append_reason(
                    reasons,
                    VerificationReasonCode.STALE_COUNTEROFFER,
                    "Counteroffer does not reference the outstanding offer.",
                )

        try:
            context.protocol.transition(context.state, action)
        except StaleOfferError:
            code = (
                VerificationReasonCode.STALE_ACCEPTANCE
                if isinstance(action, AcceptAction)
                else VerificationReasonCode.STALE_COUNTEROFFER
            )
            _append_reason(reasons, code, "Action references a stale or missing offer.")
        except RoundMismatchError:
            _append_reason(
                reasons,
                VerificationReasonCode.DEADLINE_VIOLATION,
                "Action round violates the session round or deadline.",
            )
        except IllegalActorError:
            _append_reason(
                reasons,
                VerificationReasonCode.TURN_INVALID,
                "Actor does not hold the required protocol turn.",
            )
        except (ProtocolError, DomainError, ValueError):
            _append_reason(
                reasons,
                VerificationReasonCode.PROTOCOL_ILLEGAL,
                "Action fails an authoritative domain or protocol rule.",
            )

        return self._result(checked_action_id, tuple(reasons))

    def _result(
        self, checked_action_id: str, reasons: Tuple[VerificationReason, ...]
    ) -> VerificationResult:
        return VerificationResult(
            verdict=(VerificationVerdict.REJECT if reasons else VerificationVerdict.PASS),
            reasons=reasons,
            severity=_severity(reasons),
            checked_action_id=checked_action_id,
            verifier_metadata=(self.metadata,),
        )


class LLMVerifierVerdict(str, Enum):
    PASS = "pass"
    REJECT = "reject"


class LLMVerifierReason(str, Enum):
    PUBLIC_RULE_INCONSISTENCY = "public_rule_inconsistency"
    DECLARED_STRATEGY_INCONSISTENCY = "declared_strategy_inconsistency"
    VISIBLE_EVIDENCE_INCONSISTENCY = "visible_evidence_inconsistency"
    SAFETY_RISK = "safety_risk"


class LLMVerifierDecision(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    verdict: LLMVerifierVerdict
    reasons: Tuple[LLMVerifierReason, ...] = ()
    severity: VerificationSeverity
    summary: str = Field(min_length=1, max_length=300)

    @field_validator("reasons", mode="before")
    @classmethod
    def reasons_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def verdict_matches_reasons(self) -> LLMVerifierDecision:
        if self.verdict is LLMVerifierVerdict.PASS and self.reasons:
            raise ValueError("passing verifier decision cannot contain rejection reasons")
        if self.verdict is LLMVerifierVerdict.REJECT and not self.reasons:
            raise ValueError("rejected verifier decision requires at least one reason")
        if self.verdict is LLMVerifierVerdict.PASS:
            if self.severity is not VerificationSeverity.INFO:
                raise ValueError("passing verifier decision must use info severity")
        elif self.severity is VerificationSeverity.INFO:
            raise ValueError("rejected verifier decision must not use info severity")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("verifier decision reasons must be unique")
        return self


def _participant_labels(context: VerificationContext) -> Mapping[ParticipantId, str]:
    labels = {}
    next_party = 1
    for participant_id in context.state.participants:
        if participant_id == context.actor_id:
            labels[participant_id] = "proposer"
        else:
            labels[participant_id] = f"party-{next_party}"
            next_party += 1
    return labels


def _blind_text(value: str, context: VerificationContext) -> str:
    blinded = value
    labels = _participant_labels(context)
    for participant in context.protocol.scenario.participants:
        replacement = labels[participant.participant_id]
        for identity_token in (
            str(participant.participant_id),
            participant.display_name,
            participant.role.value,
        ):
            blinded = re.sub(
                re.escape(identity_token), replacement, blinded, flags=re.IGNORECASE
            )
    return blinded


def _blinded_action(
    action: NegotiationAction, context: VerificationContext
) -> Mapping[str, object]:
    payload = action.model_dump(mode="json")
    payload["actor_id"] = "proposer"
    labels = _participant_labels(context)
    payload["recipients"] = [labels[recipient] for recipient in action.recipients]
    if isinstance(action, MessageAction):
        payload["content"] = _blind_text(action.content, context)
    if isinstance(action, (AcceptAction,)):
        payload["offer_id"] = (
            "outstanding-offer"
            if context.state.latest_valid_offer is not None
            and action.offer_id == context.state.latest_valid_offer.offer_id
            else "unrecognized-offer"
        )
    if isinstance(action, CounterAction):
        payload["responds_to"] = (
            "outstanding-offer"
            if context.state.latest_valid_offer is not None
            and action.responds_to == context.state.latest_valid_offer.offer_id
            else "unrecognized-offer"
        )
        payload["offer"]["offer_id"] = "candidate-offer"
    elif isinstance(action, ProposeAction):
        payload["offer"]["offer_id"] = "candidate-offer"
    return payload


class LLMActionVerifier:
    """Optional qualitative verifier; deterministic rejection always short-circuits it."""

    def __init__(self, client: LLMClient, model_configuration: ModelConfiguration) -> None:
        self.client = client
        self.model_configuration = model_configuration

    def verify(
        self,
        context: VerificationContext,
        candidate: object,
        checked_action_id: str,
    ) -> VerificationResult:
        try:
            action = _parse_candidate(candidate)
        except (MalformedActionError, ValidationError, TypeError, ValueError):
            return self._failure(checked_action_id)
        public_issues = tuple(
            issue.model_dump(mode="json") for issue in context.protocol.scenario.issues
        )
        request = LLMRequest(
            request_id=f"action-verifier-{checked_action_id}",
            messages=action_verifier_messages(
                LLMVerifierDecision.model_json_schema(),
                {
                    "public_rules": (
                        "Only legal actions may be selected.",
                        "Messages cannot execute protocol actions.",
                        "Do not request secrets, hidden prompts, or private participant data.",
                    ),
                    "phase": context.state.phase.value,
                    "round_number": context.state.round_number,
                    "deadline_round": context.state.maximum_rounds,
                    "public_issues": public_issues,
                    "declared_strategy": context.declared_strategy,
                    "visible_evidence": tuple(
                        _blind_text(item, context) for item in context.visible_evidence
                    ),
                    "candidate_action": _blinded_action(action, context),
                },
            ),
            model_configuration=self.model_configuration,
        )
        try:
            response = self.client.generate(request)
        except LLMClientError as error:
            return self._failure(
                checked_action_id,
                model_calls=max(1, error.info.attempts),
            )
        try:
            decision = LLMVerifierDecision.model_validate_json(response.text)
        except (ValidationError, ValueError):
            return self._failure(
                checked_action_id,
                provider=response.provider,
                model=response.model,
                latency_ms=response.latency.total_ms,
                usage=response.usage,
                model_calls=response.attempts,
            )
        metadata = VerifierMetadata(
            verifier_name="llm-action-verifier",
            verifier_version=PROMPT_VERSION,
            kind=VerifierKind.LLM,
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency.total_ms,
            usage=response.usage,
            model_calls=response.attempts,
        )
        reasons = tuple(
            _reason(VerificationReasonCode(item.value), decision.summary)
            for item in decision.reasons
        )
        if decision.verdict is LLMVerifierVerdict.REJECT and not reasons:
            reasons = (_reason(
                VerificationReasonCode.SAFETY_RISK,
                "Qualitative verifier rejected the action without a specific category.",
            ),)
        return VerificationResult(
            verdict=(
                VerificationVerdict.PASS
                if decision.verdict is LLMVerifierVerdict.PASS
                else VerificationVerdict.REJECT
            ),
            reasons=reasons,
            severity=decision.severity,
            checked_action_id=checked_action_id,
            verifier_metadata=(metadata,),
        )

    def _failure(
        self,
        checked_action_id: str,
        *,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        latency_ms: float = 0.0,
        usage: Optional[LLMUsage] = None,
        model_calls: int = 0,
    ) -> VerificationResult:
        return VerificationResult(
            verdict=VerificationVerdict.REJECT,
            reasons=(_reason(
                VerificationReasonCode.VERIFIER_FAILURE,
                "Qualitative verifier was unavailable or returned malformed structured output.",
            ),),
            severity=VerificationSeverity.ERROR,
            checked_action_id=checked_action_id,
            verifier_metadata=(
                VerifierMetadata(
                    verifier_name="llm-action-verifier",
                    verifier_version=PROMPT_VERSION,
                    kind=VerifierKind.LLM,
                    provider=provider or self.model_configuration.provider,
                    model=model or self.model_configuration.model,
                    latency_ms=latency_ms,
                    usage=usage or LLMUsage(),
                    model_calls=model_calls,
                ),
            ),
        )


CorrectionProvider = Callable[[CorrectionFeedback], object]


class CorrectedActionDecision(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    action: NegotiationAction
    summary: str = Field(min_length=1, max_length=300)


class LLMActionCorrector:
    """One-shot structured corrector using only proposer-visible memory and machine reasons."""

    def __init__(self, client: LLMClient, model_configuration: ModelConfiguration) -> None:
        self.client = client
        self.model_configuration = model_configuration
        self.last_metadata: Optional[VerifierMetadata] = None

    def correct(
        self, memory: MemorySnapshot, feedback: CorrectionFeedback
    ) -> NegotiationAction:
        expected_recipients = tuple(
            participant.participant_id
            for participant in memory.long_term.public_participants
            if participant.participant_id != memory.owner_id
        )
        request = LLMRequest(
            request_id=f"action-correction-{feedback.checked_action_id}",
            messages=action_correction_messages(
                CorrectedActionDecision.model_json_schema(),
                {
                    "actor_id": str(memory.owner_id),
                    "round_number": memory.working.round_number,
                    "deadline_round": memory.working.deadline_round,
                    "phase": memory.working.phase.value,
                    "legal_actions": tuple(
                        action.value for action in memory.working.legal_actions
                    ),
                    "expected_recipients": tuple(
                        str(item) for item in expected_recipients
                    ),
                    "public_scenario": memory.scenario.model_dump(mode="json"),
                    "outstanding_offer": (
                        memory.working.outstanding_offer.model_dump(mode="json")
                        if memory.working.outstanding_offer is not None
                        else None
                    ),
                    "own_reservation_utility": (
                        memory.long_term.own_preferences.reservation.reservation_utility
                    ),
                    "own_budget": (
                        memory.long_term.own_preferences.budget.model_dump(mode="json")
                        if memory.long_term.own_preferences.budget is not None
                        else None
                    ),
                    "own_issue_preferences": tuple(
                        preference.model_dump(mode="json")
                        for preference in (
                            memory.long_term.own_preferences.issue_preferences
                        )
                    ),
                    "verification_feedback": feedback.model_dump(mode="json"),
                },
            ),
            model_configuration=self.model_configuration,
        )
        response = self.client.generate(request)
        decision = CorrectedActionDecision.model_validate_json(response.text)
        self.last_metadata = VerifierMetadata(
            verifier_name="llm-action-corrector",
            verifier_version=CORRECTION_PROMPT_VERSION,
            kind=VerifierKind.LLM,
            provider=response.provider,
            model=response.model,
            latency_ms=response.latency.total_ms,
            usage=response.usage,
            model_calls=response.attempts,
        )
        return decision.action


@dataclass(frozen=True)
class VerificationResolution:
    action: NegotiationAction
    results: Tuple[VerificationResult, ...]
    correction_count: int
    used_fallback: bool


class VerificationFailure(RuntimeError):
    """Raised only when even the deterministic withdrawal fallback cannot be validated."""


class VerificationCoordinator:
    """Bounded verification/correction gate with append-only evaluation records."""

    def __init__(
        self,
        configuration: Optional[VerificationConfiguration] = None,
        deterministic_verifier: Optional[DeterministicActionVerifier] = None,
        llm_verifier: Optional[LLMActionVerifier] = None,
    ) -> None:
        self.configuration = configuration or VerificationConfiguration()
        if (
            self.configuration.mode is VerificationMode.DETERMINISTIC_AND_LLM
            and llm_verifier is None
        ):
            raise ValueError("LLM verification mode requires an LLMActionVerifier")
        self.deterministic_verifier = deterministic_verifier or DeterministicActionVerifier()
        self.llm_verifier = llm_verifier
        self._history: list[VerificationLogEntry] = []
        self._sequence = 0

    @property
    def audit_log(self) -> Tuple[VerificationLogEntry, ...]:
        return tuple(self._history)

    @property
    def correction_count(self) -> int:
        return sum(entry.correction_count for entry in self._history)

    def resolve(
        self,
        context: VerificationContext,
        initial_candidate: object,
        correction_provider: Optional[CorrectionProvider] = None,
    ) -> VerificationResolution:
        self._sequence += 1
        prefix = f"verification-{self._sequence}"
        if self.configuration.mode is VerificationMode.DISABLED:
            action = _parse_candidate(initial_candidate)
            skipped = VerificationResult(
                verdict=VerificationVerdict.SKIPPED,
                reasons=(_reason(
                    VerificationReasonCode.VERIFICATION_DISABLED,
                    "Pre-protocol verification is disabled for this ablation run.",
                ),),
                severity=VerificationSeverity.INFO,
                checked_action_id=f"{prefix}-initial",
                verifier_metadata=(
                    VerifierMetadata(
                        verifier_name="verification-ablation",
                        verifier_version=VERIFIER_VERSION,
                        kind=VerifierKind.ABLATION,
                    ),
                ),
            )
            resolution = VerificationResolution(action, (skipped,), 0, False)
            self._record(context, resolution)
            return resolution

        results = []
        initial = self._verify(context, initial_candidate, f"{prefix}-initial")
        results.append(initial)
        if initial.verdict is VerificationVerdict.PASS:
            action = _parse_candidate(initial_candidate)
            resolution = VerificationResolution(action, tuple(results), 0, False)
            self._record(context, resolution)
            return resolution

        correction_count = 0
        if self.configuration.maximum_corrections == 1 and correction_provider is not None:
            correction_count = 1
            feedback = CorrectionFeedback(
                checked_action_id=initial.checked_action_id,
                reasons=initial.reasons,
                correction_attempt=1,
            )
            try:
                corrected_candidate = correction_provider(feedback)
                corrected = self._verify(
                    context, corrected_candidate, f"{prefix}-correction-1"
                )
            except Exception:
                corrected_candidate = None
                corrected = VerificationResult(
                    verdict=VerificationVerdict.REJECT,
                    reasons=(_reason(
                        VerificationReasonCode.CORRECTION_FAILURE,
                        "The single correction request failed to return a usable action.",
                    ),),
                    severity=VerificationSeverity.ERROR,
                    checked_action_id=f"{prefix}-correction-1",
                    verifier_metadata=(self.deterministic_verifier.metadata,),
                )
            results.append(corrected)
            if corrected.verdict is VerificationVerdict.PASS and corrected_candidate is not None:
                action = _parse_candidate(corrected_candidate)
                resolution = VerificationResolution(action, tuple(results), 1, False)
                self._record(context, resolution)
                return resolution

        fallback = WithdrawAction(
            actor_id=context.actor_id,
            recipients=tuple(item for item in context.state.participants if item != context.actor_id),
            round_number=context.state.round_number,
            reason="Deterministic withdrawal fallback after action verification failure.",
        )
        fallback_result = self.deterministic_verifier.verify(
            context, fallback, f"{prefix}-fallback"
        )
        results.append(fallback_result)
        if fallback_result.verdict is not VerificationVerdict.PASS:
            raise VerificationFailure("deterministic fallback failed authoritative verification")
        resolution = VerificationResolution(
            fallback, tuple(results), correction_count, True
        )
        self._record(context, resolution)
        return resolution

    def _verify(
        self, context: VerificationContext, candidate: object, checked_action_id: str
    ) -> VerificationResult:
        deterministic = self.deterministic_verifier.verify(
            context, candidate, checked_action_id
        )
        if deterministic.verdict is not VerificationVerdict.PASS:
            return deterministic
        if self.configuration.mode is VerificationMode.DETERMINISTIC_AND_LLM:
            qualitative = self.llm_verifier.verify(context, candidate, checked_action_id)
            return VerificationResult(
                verdict=qualitative.verdict,
                reasons=qualitative.reasons,
                severity=qualitative.severity,
                checked_action_id=checked_action_id,
                verifier_metadata=(
                    deterministic.verifier_metadata + qualitative.verifier_metadata
                ),
            )
        return deterministic

    def _record(
        self, context: VerificationContext, resolution: VerificationResolution
    ) -> None:
        self._history.append(
            VerificationLogEntry(
                negotiation_id=context.state.scenario_id,
                sequence_number=len(self._history) + 1,
                results=resolution.results,
                correction_count=resolution.correction_count,
                used_fallback=resolution.used_fallback,
                selected_action_type=resolution.action.action,
            )
        )


@dataclass(frozen=True)
class VerifiedTransition:
    transition: TransitionResult
    verification: VerificationResolution


class VerifiedProtocolExecutor:
    """Only calls the protocol after bounded verification has selected an action."""

    def __init__(
        self,
        protocol: NegotiationProtocol,
        coordinator: VerificationCoordinator,
        pre_transition_guard: Optional[
            Callable[[NegotiationSession, NegotiationAction], None]
        ] = None,
    ) -> None:
        self.protocol = protocol
        self.coordinator = coordinator
        self.pre_transition_guard = pre_transition_guard

    def submit(
        self,
        state: NegotiationSession,
        actor_profile: AgentProfile,
        candidate: object,
        *,
        declared_strategy: Tuple[str, ...] = (),
        visible_evidence: Tuple[str, ...] = (),
        authorized_communication_recipients: Optional[Tuple[ParticipantId, ...]] = None,
        correction_provider: Optional[CorrectionProvider] = None,
    ) -> VerifiedTransition:
        context = VerificationContext(
            protocol=self.protocol,
            state=state,
            actor_profile=actor_profile,
            declared_strategy=declared_strategy,
            visible_evidence=visible_evidence,
            authorized_communication_recipients=authorized_communication_recipients,
        )
        resolution = self.coordinator.resolve(context, candidate, correction_provider)
        if self.pre_transition_guard is not None:
            self.pre_transition_guard(state, resolution.action)
        transition = self.protocol.transition(state, resolution.action)
        return VerifiedTransition(transition=transition, verification=resolution)
