"""Layered communication safety, incident auditing, and tamper-evident chains.

The deterministic rules in this module reduce observable risks. They deliberately do not
claim to detect arbitrary semantic steganography or prove that collusion is absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import Enum
from typing import Any, Literal, Optional, Protocol, Sequence, Tuple, runtime_checkable

from pydantic import Field, field_validator, model_validator

from .communication import (
    SYSTEM_SENDER,
    MessageEnvelope,
    MessageType,
    MessageVisibility,
)
from .domain import ParticipantId
from .domain.models import DomainModel
from .protocol import ProtocolEvent


SAFETY_SCHEMA_VERSION = "1.0"
GENESIS_HASH = "0" * 64


class ThreatCategory(str, Enum):
    PRIVATE_PREFERENCE_LEAKAGE = "private_preference_leakage"
    UNAUTHORIZED_DIRECT_MESSAGE = "unauthorized_direct_message"
    PROMPT_INJECTION = "prompt_injection"
    COVERT_OR_ENCODED_CONTENT = "covert_or_encoded_content"
    JUDGE_MANIPULATION = "judge_manipulation"
    MEDIATOR_MANIPULATION = "mediator_manipulation"
    REWARD_HACKING = "reward_hacking"
    FAKE_SYSTEM_MESSAGE = "fake_system_message"
    POLICY_VIOLATION = "policy_violation"
    COORDINATED_ANOMALY = "coordinated_anomaly"


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SuspiciousMessageDisposition(str, Enum):
    QUARANTINE = "quarantine"
    REJECT = "reject"


class CommunicationRole(str, Enum):
    PRINCIPAL = "principal"
    MEDIATOR = "mediator"
    SYSTEM = "system"


class ThreatDefinition(DomainModel):
    category: ThreatCategory
    protected_asset: str = Field(min_length=1, max_length=200)
    attack_surface: str = Field(min_length=1, max_length=300)
    deterministic_controls: Tuple[str, ...] = Field(min_length=1)
    residual_risk: str = Field(min_length=1, max_length=500)

    @field_validator("deterministic_controls", mode="before")
    @classmethod
    def controls_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class CommunicationThreatModel(DomainModel):
    schema_version: Literal["1.0"] = SAFETY_SCHEMA_VERSION
    threats: Tuple[ThreatDefinition, ...]
    trust_boundary: str = Field(min_length=1, max_length=500)
    limitation: str = Field(min_length=1, max_length=500)

    @field_validator("threats", mode="before")
    @classmethod
    def threats_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def complete_unique_categories(self) -> "CommunicationThreatModel":
        categories = tuple(item.category for item in self.threats)
        required = {
            ThreatCategory.PRIVATE_PREFERENCE_LEAKAGE,
            ThreatCategory.UNAUTHORIZED_DIRECT_MESSAGE,
            ThreatCategory.PROMPT_INJECTION,
            ThreatCategory.COVERT_OR_ENCODED_CONTENT,
            ThreatCategory.JUDGE_MANIPULATION,
            ThreatCategory.MEDIATOR_MANIPULATION,
            ThreatCategory.REWARD_HACKING,
        }
        if not required.issubset(categories) or len(set(categories)) != len(categories):
            raise ValueError("threat model must cover each required threat exactly once")
        return self


def default_threat_model() -> CommunicationThreatModel:
    controls = {
        ThreatCategory.PRIVATE_PREFERENCE_LEAKAGE: (
            "private preferences",
            "participant-authored message content",
            ("prohibited-field detection", "participant-specific observations"),
        ),
        ThreatCategory.UNAUTHORIZED_DIRECT_MESSAGE: (
            "channel authorization",
            "direct and mediator-only routing",
            ("role capabilities", "channel allowlist"),
        ),
        ThreatCategory.PROMPT_INJECTION: (
            "agent instruction hierarchy",
            "natural-language messages copied into model context",
            ("deterministic pattern monitoring", "untrusted-data prompt framing"),
        ),
        ThreatCategory.COVERT_OR_ENCODED_CONTENT: (
            "observable communication policy",
            "encoded text, unusual characters, and coordinated anomalies",
            ("character limits", "encoding-pattern monitor", "incident correlation"),
        ),
        ThreatCategory.JUDGE_MANIPULATION: (
            "evaluation independence",
            "transcript attempts to influence qualitative scoring",
            ("judge-manipulation rules", "objective metrics remain authoritative"),
        ),
        ThreatCategory.MEDIATOR_MANIPULATION: (
            "mediator independence",
            "messages attempting to override non-binding mediation rules",
            ("mediator-manipulation rules", "typed non-binding interventions"),
        ),
        ThreatCategory.REWARD_HACKING: (
            "reward integrity",
            "messages targeting metrics, scores, or learning updates",
            ("reward-hacking rules", "post-episode authoritative evaluation"),
        ),
    }
    return CommunicationThreatModel(
        threats=tuple(
            ThreatDefinition(
                category=category,
                protected_asset=values[0],
                attack_surface=values[1],
                deterministic_controls=values[2],
                residual_risk=(
                    "Rules can detect declared fields and known patterns but cannot prove the "
                    "absence of implicit meaning or semantic steganography."
                ),
            )
            for category, values in controls.items()
        ),
        trust_boundary=(
            "All participant and model-authored natural language is untrusted until channel, "
            "content, monitoring, and visibility checks pass."
        ),
        limitation=(
            "Monitoring reduces known risks; it cannot prove that semantic steganography, "
            "emergent coordination, or novel prompt attacks are absent."
        ),
    )


class ChannelRule(DomainModel):
    sender_role: CommunicationRole
    visibility: MessageVisibility
    message_types: Tuple[MessageType, ...] = Field(min_length=1)
    recipient_roles: Tuple[CommunicationRole, ...] = Field(min_length=1)

    @field_validator("message_types", "recipient_roles", mode="before")
    @classmethod
    def tuples_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ChannelGrant(DomainModel):
    """Explicit sender-to-recipient grant for a non-public communication channel."""

    sender: ParticipantId
    recipient: ParticipantId
    visibility: Literal[
        MessageVisibility.DIRECT_PRIVATE,
        MessageVisibility.MEDIATOR_ONLY,
    ]
    message_types: Tuple[MessageType, ...] = Field(min_length=1)

    @field_validator("message_types", mode="before")
    @classmethod
    def types_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def distinct_endpoints(self) -> "ChannelGrant":
        if self.sender == self.recipient:
            raise ValueError("channel grant sender and recipient must differ")
        return self


class CommunicationSafetyConfiguration(DomainModel):
    enabled: bool = True
    disposition: SuspiciousMessageDisposition = SuspiciousMessageDisposition.QUARANTINE
    maximum_content_characters: int = Field(default=2000, ge=32, le=4000)
    maximum_recipients: int = Field(default=16, ge=1, le=100)
    ascii_text_only: bool = True
    coordinated_anomaly_threshold: int = Field(default=2, ge=2, le=20)
    prohibited_field_names: Tuple[str, ...] = (
        "private_preferences",
        "reservation_utility",
        "batna_utility",
        "batna_description",
        "batna",
        "limit_price",
        "maximum_concession_fraction",
        "required_attributes",
        "private_preference",
        "confidential_preference",
        "api_key",
        "system_prompt",
    )
    channel_rules: Optional[Tuple[ChannelRule, ...]] = None
    channel_allowlist: Optional[Tuple[ChannelGrant, ...]] = None

    @field_validator(
        "prohibited_field_names", "channel_rules", "channel_allowlist", mode="before"
    )
    @classmethod
    def tuples_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def default_channel_rules() -> Tuple[ChannelRule, ...]:
    participant_types = (
        MessageType.PROPOSAL_EXPLANATION,
        MessageType.QUESTION,
        MessageType.ANSWER,
        MessageType.INTENT_SIGNAL,
        MessageType.MEDIATION_REQUEST,
        MessageType.CRITIQUE,
    )
    mediator_types = (
        MessageType.MEDIATOR_PROPOSAL,
        MessageType.MEDIATOR_QUESTION,
        MessageType.MEDIATOR_NOTICE,
        MessageType.QUESTION,
        MessageType.ANSWER,
    )
    return (
        ChannelRule(
            sender_role=CommunicationRole.PRINCIPAL,
            visibility=MessageVisibility.PUBLIC,
            message_types=participant_types,
            recipient_roles=(CommunicationRole.PRINCIPAL, CommunicationRole.MEDIATOR),
        ),
        ChannelRule(
            sender_role=CommunicationRole.PRINCIPAL,
            visibility=MessageVisibility.DIRECT_PRIVATE,
            message_types=participant_types,
            recipient_roles=(CommunicationRole.PRINCIPAL, CommunicationRole.MEDIATOR),
        ),
        ChannelRule(
            sender_role=CommunicationRole.PRINCIPAL,
            visibility=MessageVisibility.MEDIATOR_ONLY,
            message_types=participant_types,
            recipient_roles=(CommunicationRole.MEDIATOR,),
        ),
        ChannelRule(
            sender_role=CommunicationRole.MEDIATOR,
            visibility=MessageVisibility.PUBLIC,
            message_types=mediator_types,
            recipient_roles=(CommunicationRole.PRINCIPAL, CommunicationRole.MEDIATOR),
        ),
        ChannelRule(
            sender_role=CommunicationRole.MEDIATOR,
            visibility=MessageVisibility.DIRECT_PRIVATE,
            message_types=mediator_types,
            recipient_roles=(CommunicationRole.PRINCIPAL,),
        ),
        ChannelRule(
            sender_role=CommunicationRole.SYSTEM,
            visibility=MessageVisibility.SYSTEM_AUDIT,
            message_types=(MessageType.SYSTEM_NOTICE,),
            recipient_roles=(CommunicationRole.SYSTEM,),
        ),
    )


class MonitorFinding(DomainModel):
    category: ThreatCategory
    reason_code: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    severity: IncidentSeverity


@runtime_checkable
class MessageMonitor(Protocol):
    def inspect(
        self,
        message: MessageEnvelope,
        prior_incidents: Tuple["SecurityIncident", ...],
        configuration: CommunicationSafetyConfiguration,
    ) -> Tuple[MonitorFinding, ...]:
        """Return deterministic findings from authorized observable message fields."""


_BASE64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_HEX = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{32,}(?![A-Fa-f0-9])")
_INJECTION = re.compile(
    r"\b(ignore (all |any )?(previous|prior)|developer message|system prompt|"
    r"follow these instructions|reveal (your|the) prompt|act as system)\b",
    re.IGNORECASE,
)
_FAKE_SYSTEM = re.compile(r"^\s*(system|developer|assistant)\s*[:>]", re.IGNORECASE)
_JUDGE = re.compile(
    r"\b(judge|evaluator|rubric)\b.{0,80}\b(score|rate|award|favor|favour)\b",
    re.IGNORECASE | re.DOTALL,
)
_MEDIATOR = re.compile(
    r"\bmediator\b.{0,80}\b(ignore|override|force|accept for|reveal|side with)\b",
    re.IGNORECASE | re.DOTALL,
)
_REWARD = re.compile(
    r"\b(reward|metric|score|benchmark)\b.{0,80}\b(game|hack|inflate|fake|manipulate)\b",
    re.IGNORECASE | re.DOTALL,
)


class DeterministicMessageMonitor:
    """Known-pattern baseline; deliberately bounded and explainable."""

    def inspect(
        self,
        message: MessageEnvelope,
        prior_incidents: Tuple["SecurityIncident", ...],
        configuration: CommunicationSafetyConfiguration,
    ) -> Tuple[MonitorFinding, ...]:
        text = message.content
        lower = text.casefold()
        findings: list[MonitorFinding] = []

        def add(category: ThreatCategory, code: str, severity: IncidentSeverity) -> None:
            if code not in {item.reason_code for item in findings}:
                findings.append(
                    MonitorFinding(category=category, reason_code=code, severity=severity)
                )

        base64_candidate = any(
            len(set(match.group(0).rstrip("="))) >= 8
            for match in _BASE64.finditer(text)
        )
        hex_candidate = any(
            len(set(match.group(0))) >= 6 for match in _HEX.finditer(text)
        )
        if base64_candidate or hex_candidate:
            add(
                ThreatCategory.COVERT_OR_ENCODED_CONTENT,
                "encoded_payload_pattern",
                IncidentSeverity.HIGH,
            )
        if any(character in text for character in ("\u200b", "\u200c", "\u200d", "\ufeff")):
            add(
                ThreatCategory.COVERT_OR_ENCODED_CONTENT,
                "invisible_character",
                IncidentSeverity.HIGH,
            )
        if _INJECTION.search(text):
            add(ThreatCategory.PROMPT_INJECTION, "instruction_injection", IncidentSeverity.HIGH)
        if message.sender != SYSTEM_SENDER and _FAKE_SYSTEM.search(text):
            add(ThreatCategory.FAKE_SYSTEM_MESSAGE, "fake_privileged_role", IncidentSeverity.HIGH)
        prohibited = tuple(
            field for field in configuration.prohibited_field_names
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(field.casefold())}(?![A-Za-z0-9_])", lower)
        )
        private_concept = re.search(
            r"\b(batna|reservation utility|private preferences?|"
            r"confidential preferences?)\b",
            lower,
        )
        if prohibited or private_concept:
            add(
                ThreatCategory.PRIVATE_PREFERENCE_LEAKAGE,
                "prohibited_private_field",
                IncidentSeverity.CRITICAL,
            )
        if _JUDGE.search(text):
            add(ThreatCategory.JUDGE_MANIPULATION, "judge_persuasion", IncidentSeverity.HIGH)
        if _MEDIATOR.search(text):
            add(
                ThreatCategory.MEDIATOR_MANIPULATION,
                "mediator_override_attempt",
                IncidentSeverity.HIGH,
            )
        if _REWARD.search(text):
            add(ThreatCategory.REWARD_HACKING, "reward_manipulation", IncidentSeverity.HIGH)

        fingerprint = _content_fingerprint(text)
        current_categories = {item.category for item in findings}
        related_senders = {
            incident.sender
            for incident in prior_incidents
            if (
                incident.content_fingerprint == fingerprint
                or incident.category in current_categories
            )
            and incident.sender != message.sender
        }
        if len(related_senders) + 1 >= configuration.coordinated_anomaly_threshold:
            add(
                ThreatCategory.COORDINATED_ANOMALY,
                "repeated_cross_sender_anomaly",
                IncidentSeverity.HIGH,
            )
        return tuple(findings)


class SecurityIncident(DomainModel):
    schema_version: Literal["1.0"] = SAFETY_SCHEMA_VERSION
    incident_id: str = Field(pattern=r"^incident-[1-9][0-9]*$")
    negotiation_id: str = Field(min_length=1, max_length=100)
    attempt_sequence: int = Field(ge=1)
    sender: ParticipantId
    recipients: Tuple[ParticipantId, ...]
    message_type: MessageType
    visibility: MessageVisibility
    category: ThreatCategory
    severity: IncidentSeverity
    reason_code: str = Field(pattern=r"^[a-z0-9_]+$")
    disposition: SuspiciousMessageDisposition
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("recipients", mode="before")
    @classmethod
    def recipients_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class ProtectedMessageRecord(DomainModel):
    """Raw quarantined content; accessible only through an authorized audit-reader method."""

    record_id: str = Field(pattern=r"^protected-[1-9][0-9]*$")
    message: MessageEnvelope
    incident_ids: Tuple[str, ...] = Field(min_length=1)
    disposition: SuspiciousMessageDisposition

    @field_validator("incident_ids", mode="before")
    @classmethod
    def ids_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class SafetyMetrics(DomainModel):
    attempted_messages: int = Field(default=0, ge=0)
    delivered_messages: int = Field(default=0, ge=0)
    quarantined_messages: int = Field(default=0, ge=0)
    rejected_messages: int = Field(default=0, ge=0)
    incident_count: int = Field(default=0, ge=0)
    incidents_by_category: Tuple[Tuple[ThreatCategory, int], ...] = ()

    @field_validator("incidents_by_category", mode="before")
    @classmethod
    def categories_from_json(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(item) for item in value)
        return value

    @model_validator(mode="after")
    def consistent_counts(self) -> "SafetyMetrics":
        if self.attempted_messages != (
            self.delivered_messages + self.quarantined_messages + self.rejected_messages
        ):
            raise ValueError("safety message counters must account for every attempted message")
        if self.incident_count != sum(count for _, count in self.incidents_by_category):
            raise ValueError("safety incident total must match category counts")
        return self


class MessageSafetyError(RuntimeError):
    def __init__(self, message: str, incident_ids: Tuple[str, ...]) -> None:
        super().__init__(message)
        self.incident_ids = incident_ids


class MessageQuarantinedError(MessageSafetyError):
    pass


class MessageRejectedError(MessageSafetyError):
    pass


class CommunicationSafetyController:
    """Episode-local least-privilege enforcement and protected incident storage."""

    def __init__(
        self,
        *,
        participants: Tuple[ParticipantId, ...],
        mediator_ids: Tuple[ParticipantId, ...],
        audit_reader_ids: Tuple[ParticipantId, ...],
        configuration: Optional[CommunicationSafetyConfiguration] = None,
        monitor: Optional[MessageMonitor] = None,
    ) -> None:
        self.participants = participants
        self.mediator_ids = mediator_ids
        self.audit_reader_ids = audit_reader_ids
        supplied = configuration or CommunicationSafetyConfiguration()
        self.configuration = (
            supplied
            if supplied.channel_rules is not None
            else supplied.model_copy(update={"channel_rules": default_channel_rules()})
        )
        self.monitor = monitor or DeterministicMessageMonitor()
        self._incidents: Tuple[SecurityIncident, ...] = ()
        self._protected: Tuple[ProtectedMessageRecord, ...] = ()
        self._attempted = 0
        self._delivered = 0
        self._quarantined = 0
        self._rejected = 0

    @property
    def incidents(self) -> Tuple[SecurityIncident, ...]:
        return self._incidents

    @property
    def metrics(self) -> SafetyMetrics:
        counts = Counter(item.category for item in self._incidents)
        return SafetyMetrics(
            attempted_messages=self._attempted,
            delivered_messages=self._delivered,
            quarantined_messages=self._quarantined,
            rejected_messages=self._rejected,
            incident_count=len(self._incidents),
            incidents_by_category=tuple(sorted(counts.items(), key=lambda item: item[0].value)),
        )

    def protected_audit_view(
        self, viewer_id: ParticipantId
    ) -> Tuple[ProtectedMessageRecord, ...]:
        if viewer_id not in self.audit_reader_ids:
            raise PermissionError("viewer is not authorized for the protected safety audit")
        return self._protected

    def preflight(self, message: MessageEnvelope) -> None:
        """Block and record suspicious content before any protocol transition."""

        self._assess(message, count_safe_delivery=False)

    def deliver(self, message: MessageEnvelope) -> None:
        """Authorize one boundary delivery or raise without exposing its content."""

        self._assess(message, count_safe_delivery=True)

    def record_delivery(self) -> None:
        """Record a delivery after preflight and structural routing both succeeded."""

        self._attempted += 1
        self._delivered += 1

    def record_routing_violation(self, message: MessageEnvelope) -> None:
        """Audit a structural boundary rejection without replacing its routing exception."""

        try:
            self._record_blocked(
                message,
                (
                    MonitorFinding(
                        category=ThreatCategory.POLICY_VIOLATION,
                        reason_code="structural_routing_violation",
                        severity=IncidentSeverity.HIGH,
                    ),
                ),
            )
        except MessageSafetyError:
            return

    def _assess(self, message: MessageEnvelope, *, count_safe_delivery: bool) -> None:
        if not self.configuration.enabled:
            if count_safe_delivery:
                self._attempted += 1
                self._delivered += 1
            return
        findings = list(self._policy_findings(message))
        findings.extend(self.monitor.inspect(message, self._incidents, self.configuration))
        unique = {
            (item.category, item.reason_code): item for item in findings
        }
        findings = list(unique.values())
        if not findings:
            if count_safe_delivery:
                self._attempted += 1
                self._delivered += 1
            return

        self._record_blocked(message, tuple(findings))

    def _record_blocked(
        self, message: MessageEnvelope, findings: Tuple[MonitorFinding, ...]
    ) -> None:

        self._attempted += 1
        disposition = self.configuration.disposition
        if disposition is SuspiciousMessageDisposition.QUARANTINE:
            self._quarantined += 1
        else:
            self._rejected += 1
        incident_ids = []
        for finding in findings:
            incident_id = f"incident-{len(self._incidents) + 1}"
            incident_ids.append(incident_id)
            self._incidents += (
                SecurityIncident(
                    incident_id=incident_id,
                    negotiation_id=message.negotiation_id,
                    attempt_sequence=self._attempted,
                    sender=message.sender,
                    recipients=message.recipients,
                    message_type=message.message_type,
                    visibility=message.visibility,
                    category=finding.category,
                    severity=finding.severity,
                    reason_code=finding.reason_code,
                    disposition=disposition,
                    content_sha256=_sha256(message.content),
                    content_fingerprint=_content_fingerprint(message.content),
                ),
            )
        self._protected += (
            ProtectedMessageRecord(
                record_id=f"protected-{len(self._protected) + 1}",
                message=message,
                incident_ids=tuple(incident_ids),
                disposition=disposition,
            ),
        )
        error_type = (
            MessageQuarantinedError
            if disposition is SuspiciousMessageDisposition.QUARANTINE
            else MessageRejectedError
        )
        raise error_type("message blocked by communication safety policy", tuple(incident_ids))

    def _policy_findings(self, message: MessageEnvelope) -> Tuple[MonitorFinding, ...]:
        findings = []
        config = self.configuration
        if len(message.content) > config.maximum_content_characters:
            findings.append(MonitorFinding(
                category=ThreatCategory.POLICY_VIOLATION,
                reason_code="content_size_limit",
                severity=IncidentSeverity.MEDIUM,
            ))
        if len(message.recipients) > config.maximum_recipients:
            findings.append(MonitorFinding(
                category=ThreatCategory.POLICY_VIOLATION,
                reason_code="recipient_count_limit",
                severity=IncidentSeverity.MEDIUM,
            ))
        if config.ascii_text_only and any(
            character not in "\t\n\r" and not (" " <= character <= "~")
            for character in message.content
        ):
            findings.append(MonitorFinding(
                category=ThreatCategory.COVERT_OR_ENCODED_CONTENT,
                reason_code="character_set_violation",
                severity=IncidentSeverity.HIGH,
            ))

        sender_role = self._role(message.sender)
        recipient_roles = {self._role(item) for item in message.recipients}
        participant_set = set(self.participants)
        if message.sender != SYSTEM_SENDER and message.sender not in participant_set:
            findings.append(MonitorFinding(
                category=ThreatCategory.POLICY_VIOLATION,
                reason_code="unknown_sender",
                severity=IncidentSeverity.HIGH,
            ))
        if set(message.recipients) - participant_set:
            findings.append(MonitorFinding(
                category=ThreatCategory.POLICY_VIOLATION,
                reason_code="unknown_recipient",
                severity=IncidentSeverity.HIGH,
            ))
        matching = tuple(
            rule for rule in (config.channel_rules or ())
            if rule.sender_role is sender_role
            and rule.visibility is message.visibility
            and message.message_type in rule.message_types
            and recipient_roles.issubset(set(rule.recipient_roles))
        )
        if not matching:
            category = (
                ThreatCategory.UNAUTHORIZED_DIRECT_MESSAGE
                if message.visibility in {
                    MessageVisibility.DIRECT_PRIVATE,
                    MessageVisibility.MEDIATOR_ONLY,
                }
                else ThreatCategory.POLICY_VIOLATION
            )
            findings.append(MonitorFinding(
                category=category,
                reason_code="channel_not_allowlisted",
                severity=IncidentSeverity.HIGH,
            ))
        if (
            message.visibility in {
                MessageVisibility.DIRECT_PRIVATE,
                MessageVisibility.MEDIATOR_ONLY,
            }
            and config.channel_allowlist is not None
        ):
            grants = config.channel_allowlist
            unauthorized = tuple(
                recipient
                for recipient in message.recipients
                if not any(
                    grant.sender == message.sender
                    and grant.recipient == recipient
                    and grant.visibility is message.visibility
                    and message.message_type in grant.message_types
                    for grant in grants
                )
            )
            if unauthorized:
                findings.append(MonitorFinding(
                    category=ThreatCategory.UNAUTHORIZED_DIRECT_MESSAGE,
                    reason_code="private_route_not_allowlisted",
                    severity=IncidentSeverity.HIGH,
                ))
        return tuple(findings)

    def _role(self, participant_id: ParticipantId) -> CommunicationRole:
        if participant_id == SYSTEM_SENDER:
            return CommunicationRole.SYSTEM
        if participant_id in self.mediator_ids:
            return CommunicationRole.MEDIATOR
        return CommunicationRole.PRINCIPAL


class AuditRecordKind(str, Enum):
    PROTOCOL_EVENT = "protocol_event"
    MESSAGE = "message"
    SECURITY_INCIDENT = "security_incident"


class AuditChainEntry(DomainModel):
    sequence_number: int = Field(ge=1)
    record_kind: AuditRecordKind
    record_id: str = Field(min_length=1, max_length=100)
    previous_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_audit_chain(
    protocol_events: Sequence[ProtocolEvent],
    messages: Sequence[MessageEnvelope],
    incidents: Sequence[SecurityIncident],
) -> Tuple[AuditChainEntry, ...]:
    records: list[tuple[AuditRecordKind, str, object]] = []
    records.extend(
        (AuditRecordKind.PROTOCOL_EVENT, f"event-{event.sequence_number}", event)
        for event in protocol_events
    )
    records.extend(
        (AuditRecordKind.MESSAGE, str(message.message_id), message) for message in messages
    )
    records.extend(
        (AuditRecordKind.SECURITY_INCIDENT, incident.incident_id, incident)
        for incident in incidents
    )
    previous = GENESIS_HASH
    entries = []
    for sequence_number, (kind, record_id, record) in enumerate(records, 1):
        payload_hash = _sha256(_canonical_json(record))
        chain_hash = _sha256(
            f"{sequence_number}|{kind.value}|{record_id}|{previous}|{payload_hash}"
        )
        entries.append(AuditChainEntry(
            sequence_number=sequence_number,
            record_kind=kind,
            record_id=record_id,
            previous_hash=previous,
            payload_hash=payload_hash,
            chain_hash=chain_hash,
        ))
        previous = chain_hash
    return tuple(entries)


def verify_audit_chain(
    chain: Sequence[AuditChainEntry],
    protocol_events: Sequence[ProtocolEvent],
    messages: Sequence[MessageEnvelope],
    incidents: Sequence[SecurityIncident],
) -> bool:
    return tuple(chain) == build_audit_chain(protocol_events, messages, incidents)


def verify_protected_records(
    records: Sequence[ProtectedMessageRecord],
    incidents: Sequence[SecurityIncident],
) -> bool:
    """Verify that protected raw records still match their sanitized hash commitments."""

    by_id = {item.incident_id: item for item in incidents}
    if len(by_id) != len(incidents):
        return False
    for record in records:
        linked = tuple(by_id.get(incident_id) for incident_id in record.incident_ids)
        if any(item is None for item in linked):
            return False
        content_hash = _sha256(record.message.content)
        fingerprint = _content_fingerprint(record.message.content)
        if any(
            item.content_sha256 != content_hash
            or item.content_fingerprint != fingerprint
            or item.sender != record.message.sender
            or item.recipients != record.message.recipients
            or item.message_type is not record.message.message_type
            or item.visibility is not record.message.visibility
            for item in linked
            if item is not None
        ):
            return False
    return True


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        payload: Any = value.model_dump(mode="json")
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content_fingerprint(content: str) -> str:
    normalized = re.sub(r"\s+", " ", content.strip().casefold())
    return _sha256(normalized)
