from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple

from pydantic import Field, TypeAdapter, field_validator, model_validator

from .domain import (
    ActionId,
    CorrelationId,
    CounterAction,
    MessageAccessError,
    MessageId,
    MessageReplayError,
    MessageRoutingError,
    OfferId,
    ParticipantId,
    ProposeAction,
)
from .domain.models import DomainModel
from .protocol import (
    ActionAppliedEvent,
    MediatorInterventionEvent,
    NegotiationSession,
    ProtocolEvent,
)

if TYPE_CHECKING:
    from .safety import (
        CommunicationSafetyController,
        ProtectedMessageRecord,
        SafetyMetrics,
        SecurityIncident,
    )


SYSTEM_SENDER = ParticipantId("system")


class MessageType(str, Enum):
    PROPOSAL_EXPLANATION = "proposal_explanation"
    QUESTION = "question"
    ANSWER = "answer"
    INTENT_SIGNAL = "intent_signal"
    MEDIATION_REQUEST = "mediation_request"
    CRITIQUE = "critique"
    SYSTEM_NOTICE = "system_notice"
    MEDIATOR_PROPOSAL = "mediator_proposal"
    MEDIATOR_QUESTION = "mediator_question"
    MEDIATOR_NOTICE = "mediator_notice"


class MessageVisibility(str, Enum):
    PUBLIC = "public"
    DIRECT_PRIVATE = "direct_private"
    MEDIATOR_ONLY = "mediator_only"
    SYSTEM_AUDIT = "system_audit"


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


class MessageEnvelope(DomainModel):
    """Serializable communication data containing no arbitrary metadata or object references."""

    message_id: MessageId
    sender: ParticipantId
    recipients: Tuple[ParticipantId, ...]
    timestamp: datetime = Field(strict=False)
    sequence_number: int = Field(ge=1)
    negotiation_id: str = Field(min_length=1, max_length=100)
    message_type: MessageType
    visibility: MessageVisibility
    content: str = Field(min_length=1, max_length=4000)
    correlation_id: CorrelationId
    referenced_offer_id: Optional[OfferId] = None
    referenced_action_id: Optional[ActionId] = None

    @field_validator("recipients", mode="before")
    @classmethod
    def recipients_from_json_array(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("timestamp")
    @classmethod
    def timestamp_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("message timestamp must include a timezone")
        return value

    @field_validator("negotiation_id", "content")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message text fields must not be blank")
        return value

    @model_validator(mode="after")
    def valid_recipients(self) -> MessageEnvelope:
        if len(set(self.recipients)) != len(self.recipients):
            raise ValueError("message recipients must be unique")
        if self.sender in self.recipients:
            raise ValueError("message sender cannot also be a recipient")
        if self.visibility is MessageVisibility.SYSTEM_AUDIT:
            if self.recipients:
                raise ValueError("system/audit messages cannot have participant recipients")
        elif not self.recipients:
            raise ValueError("non-audit messages require at least one recipient")
        return self


class MessageBus:
    """In-memory append-only router with deterministic sequence ordering."""

    def __init__(
        self,
        negotiation_id: str,
        participants: Tuple[ParticipantId, ...],
        mediator_ids: Tuple[ParticipantId, ...],
        audit_reader_ids: Tuple[ParticipantId, ...],
        safety_controller: Optional["CommunicationSafetyController"] = None,
    ) -> None:
        if not negotiation_id or not negotiation_id.strip():
            raise ValueError("negotiation_id must not be blank")
        if len(set(participants)) != len(participants):
            raise ValueError("message bus participants must be unique")
        if not participants:
            raise ValueError("message bus requires at least one participant")
        if SYSTEM_SENDER in participants:
            raise ValueError("the reserved system sender cannot be a negotiation participant")
        if not set(mediator_ids).issubset(participants):
            raise ValueError("mediators must be message bus participants")
        if len(set(audit_reader_ids)) != len(audit_reader_ids):
            raise ValueError("audit readers must be unique")
        if set(audit_reader_ids).intersection(participants):
            raise ValueError("audit readers must be separate from negotiation participants")
        self.negotiation_id = negotiation_id
        self.participants = tuple(participants)
        self.mediator_ids = tuple(mediator_ids)
        self.audit_reader_ids = tuple(audit_reader_ids)
        if safety_controller is None:
            from .safety import CommunicationSafetyController

            safety_controller = CommunicationSafetyController(
                participants=self.participants,
                mediator_ids=self.mediator_ids,
                audit_reader_ids=self.audit_reader_ids,
            )
        elif (
            tuple(safety_controller.participants) != self.participants
            or tuple(safety_controller.mediator_ids) != self.mediator_ids
            or tuple(safety_controller.audit_reader_ids) != self.audit_reader_ids
        ):
            raise ValueError("safety controller scope must match the message bus")
        self.safety_controller = safety_controller
        self._messages: Tuple[MessageEnvelope, ...] = ()
        self._message_ids: set[MessageId] = set()

    def send(
        self,
        *,
        sender: ParticipantId,
        recipients: Tuple[ParticipantId, ...],
        timestamp: datetime,
        message_type: MessageType,
        visibility: MessageVisibility,
        content: str,
        correlation_id: Optional[CorrelationId] = None,
        referenced_offer_id: Optional[OfferId] = None,
        referenced_action_id: Optional[ActionId] = None,
    ) -> MessageEnvelope:
        sequence_number = len(self._messages) + 1
        envelope = MessageEnvelope(
            message_id=MessageId(f"message-{sequence_number}"),
            sender=sender,
            recipients=recipients,
            timestamp=timestamp,
            sequence_number=sequence_number,
            negotiation_id=self.negotiation_id,
            message_type=message_type,
            visibility=visibility,
            content=content,
            correlation_id=correlation_id or CorrelationId(f"correlation-message-{sequence_number}"),
            referenced_offer_id=referenced_offer_id,
            referenced_action_id=referenced_action_id,
        )
        return self.publish(envelope)

    def publish(self, envelope: MessageEnvelope) -> MessageEnvelope:
        try:
            self.safety_controller.preflight(envelope)
        except Exception as safety_error:
            from .safety import MessageSafetyError

            if not isinstance(safety_error, MessageSafetyError):
                raise
            # Preserve the established structural routing error API while the
            # safety controller retains the protected incident attempt.
            self._validate_for_delivery(envelope)
            raise
        try:
            self._validate_for_delivery(envelope)
        except MessageRoutingError:
            self.safety_controller.record_routing_violation(envelope)
            raise
        self.safety_controller.record_delivery()
        self._messages = self._messages + (envelope,)
        self._message_ids.add(envelope.message_id)
        return envelope

    def preflight(
        self,
        *,
        sender: ParticipantId,
        recipients: Tuple[ParticipantId, ...],
        timestamp: datetime,
        message_type: MessageType,
        visibility: MessageVisibility,
        content: str,
        correlation_id: Optional[CorrelationId] = None,
        referenced_offer_id: Optional[OfferId] = None,
        referenced_action_id: Optional[ActionId] = None,
    ) -> MessageEnvelope:
        """Validate a prospective message without delivering or counting safe content."""

        sequence_number = len(self._messages) + 1
        envelope = MessageEnvelope(
            message_id=MessageId(f"message-{sequence_number}"),
            sender=sender,
            recipients=recipients,
            timestamp=timestamp,
            sequence_number=sequence_number,
            negotiation_id=self.negotiation_id,
            message_type=message_type,
            visibility=visibility,
            content=content,
            correlation_id=correlation_id or CorrelationId(f"correlation-message-{sequence_number}"),
            referenced_offer_id=referenced_offer_id,
            referenced_action_id=referenced_action_id,
        )
        try:
            self.safety_controller.preflight(envelope)
        except Exception as safety_error:
            from .safety import MessageSafetyError

            if not isinstance(safety_error, MessageSafetyError):
                raise
            self._validate_for_delivery(envelope)
            raise
        try:
            self._validate_for_delivery(envelope)
        except MessageRoutingError:
            self.safety_controller.record_routing_violation(envelope)
            raise
        return envelope

    @property
    def safety_incidents(self) -> Tuple["SecurityIncident", ...]:
        return self.safety_controller.incidents

    @property
    def safety_metrics(self) -> "SafetyMetrics":
        return self.safety_controller.metrics

    def protected_safety_audit(
        self, viewer_id: ParticipantId
    ) -> Tuple["ProtectedMessageRecord", ...]:
        return self.safety_controller.protected_audit_view(viewer_id)

    def verify_protected_safety_audit(self, viewer_id: ParticipantId) -> bool:
        from .safety import verify_protected_records

        records = self.safety_controller.protected_audit_view(viewer_id)
        return verify_protected_records(records, self.safety_controller.incidents)

    def _validate_for_delivery(self, envelope: MessageEnvelope) -> None:
        if envelope.negotiation_id != self.negotiation_id:
            raise MessageRoutingError("message negotiation id does not match this bus")
        expected_sequence = len(self._messages) + 1
        if envelope.sequence_number != expected_sequence:
            raise MessageRoutingError(
                f"message sequence must be {expected_sequence}; got {envelope.sequence_number}"
            )
        if envelope.message_id in self._message_ids:
            raise MessageRoutingError(f"message id '{envelope.message_id}' has already been used")
        if self._messages and envelope.timestamp < self._messages[-1].timestamp:
            raise MessageRoutingError("message timestamps must be nondecreasing")

        participant_set = set(self.participants)
        recipient_set = set(envelope.recipients)
        if envelope.visibility is MessageVisibility.SYSTEM_AUDIT:
            if envelope.sender != SYSTEM_SENDER:
                raise MessageRoutingError("only the system sender may emit system/audit messages")
            if envelope.message_type is not MessageType.SYSTEM_NOTICE:
                raise MessageRoutingError("system/audit messages must use the system_notice type")
            return

        if envelope.sender not in participant_set:
            raise MessageRoutingError(f"unknown message sender '{envelope.sender}'")
        unknown = recipient_set - participant_set
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise MessageRoutingError(f"unknown message recipients: {names}")

        if envelope.visibility is MessageVisibility.PUBLIC:
            expected = participant_set - {envelope.sender}
            if recipient_set != expected:
                raise MessageRoutingError("public messages must address every other participant")
        elif envelope.visibility is MessageVisibility.MEDIATOR_ONLY:
            if not recipient_set.issubset(set(self.mediator_ids)):
                raise MessageRoutingError("mediator-only messages may address only configured mediators")

    def inbox(self, participant_id: ParticipantId) -> Tuple[MessageEnvelope, ...]:
        if participant_id not in self.participants:
            raise MessageAccessError(f"unknown inbox participant '{participant_id}'")
        return tuple(
            message
            for message in self._messages
            if message.visibility is MessageVisibility.PUBLIC or participant_id in message.recipients
        )

    def public_transcript(self) -> Tuple[MessageEnvelope, ...]:
        return tuple(
            message for message in self._messages if message.visibility is MessageVisibility.PUBLIC
        )

    def read(self, message_id: MessageId, viewer_id: ParticipantId) -> MessageEnvelope:
        message = next(
            (candidate for candidate in self._messages if candidate.message_id == message_id),
            None,
        )
        if message is None:
            raise MessageAccessError(f"unknown message '{message_id}'")
        if viewer_id in self.audit_reader_ids:
            return message
        if message.visibility is MessageVisibility.PUBLIC:
            return message
        if viewer_id == message.sender or viewer_id in message.recipients:
            return message
        raise MessageAccessError(
            f"viewer '{viewer_id}' is not authorized to read message '{message_id}'"
        )

    def audit_view(self, viewer_id: ParticipantId) -> Tuple[MessageEnvelope, ...]:
        if viewer_id not in self.audit_reader_ids:
            raise MessageAccessError(f"viewer '{viewer_id}' is not an authorized audit reader")
        return tuple(self._messages)

    def export_audit_json(self, viewer_id: ParticipantId) -> str:
        return _MESSAGE_SEQUENCE_ADAPTER.dump_json(self.audit_view(viewer_id)).decode("utf-8")

    @classmethod
    def replay(
        cls,
        *,
        negotiation_id: str,
        participants: Tuple[ParticipantId, ...],
        mediator_ids: Tuple[ParticipantId, ...],
        audit_reader_ids: Tuple[ParticipantId, ...],
        messages: Iterable[MessageEnvelope],
    ) -> MessageBus:
        bus = cls(negotiation_id, participants, mediator_ids, audit_reader_ids)
        try:
            for message in messages:
                bus.publish(message)
        except MessageRoutingError as exc:
            raise MessageReplayError(f"invalid message audit sequence: {exc}") from exc
        return bus

    @classmethod
    def replay_json(
        cls,
        *,
        negotiation_id: str,
        participants: Tuple[ParticipantId, ...],
        mediator_ids: Tuple[ParticipantId, ...],
        audit_reader_ids: Tuple[ParticipantId, ...],
        payload: str,
    ) -> MessageBus:
        messages = _MESSAGE_SEQUENCE_ADAPTER.validate_json(payload)
        return cls.replay(
            negotiation_id=negotiation_id,
            participants=participants,
            mediator_ids=mediator_ids,
            audit_reader_ids=audit_reader_ids,
            messages=messages,
        )


class EpisodeRecord(DomainModel):
    negotiation_id: str = Field(min_length=1, max_length=100)
    protocol_events: Tuple[ProtocolEvent, ...]
    messages: Tuple[MessageEnvelope, ...]

    @field_validator("protocol_events", "messages", mode="before")
    @classmethod
    def sequences_from_json_arrays(cls, value: Any) -> Any:
        return _as_tuple(value)

    @model_validator(mode="after")
    def consistent_episode(self) -> EpisodeRecord:
        event_numbers = tuple(event.sequence_number for event in self.protocol_events)
        if event_numbers != tuple(range(1, len(self.protocol_events) + 1)):
            raise ValueError("episode protocol event sequence must be contiguous starting at one")
        message_numbers = tuple(message.sequence_number for message in self.messages)
        if message_numbers != tuple(range(1, len(self.messages) + 1)):
            raise ValueError("episode message sequence must be contiguous starting at one")
        if any(message.negotiation_id != self.negotiation_id for message in self.messages):
            raise ValueError("episode messages must match the episode negotiation id")
        message_ids = tuple(message.message_id for message in self.messages)
        if len(set(message_ids)) != len(message_ids):
            raise ValueError("episode message identifiers must be unique")

        action_events = tuple(
            event for event in self.protocol_events if isinstance(event, ActionAppliedEvent)
        )
        action_ids = tuple(event.action_id for event in action_events)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("episode action identifiers must be unique")
        event_correlations = tuple(event.correlation_id for event in self.protocol_events)
        if len(set(event_correlations)) != len(event_correlations):
            raise ValueError("episode protocol event correlation identifiers must be unique")
        actions = {event.action_id: event for event in action_events}
        offers = {
            event.action.offer.offer_id
            for event in actions.values()
            if isinstance(event.action, (ProposeAction, CounterAction))
        }
        offers.update(
            event.intervention.offer.offer_id
            for event in self.protocol_events
            if isinstance(event, MediatorInterventionEvent)
            and event.intervention.offer is not None
        )
        for message in self.messages:
            if message.referenced_action_id is not None:
                event = actions.get(message.referenced_action_id)
                if event is None:
                    raise ValueError(
                        f"message references unknown action '{message.referenced_action_id}'"
                    )
                if message.correlation_id != event.correlation_id:
                    raise ValueError("message and referenced action must share a correlation id")
            if message.referenced_offer_id is not None and message.referenced_offer_id not in offers:
                raise ValueError(
                    f"message references unknown offer '{message.referenced_offer_id}'"
                )
        return self


def build_episode(
    session: NegotiationSession,
    messages: Tuple[MessageEnvelope, ...],
) -> EpisodeRecord:
    return EpisodeRecord(
        negotiation_id=session.scenario_id,
        protocol_events=session.event_sequence,
        messages=messages,
    )


def format_message(
    message: MessageEnvelope,
    participant_names: Optional[Dict[ParticipantId, str]] = None,
) -> str:
    names = participant_names or {}
    sender = names.get(message.sender, str(message.sender))
    content = " ".join(message.content.split())
    references = []
    if message.referenced_offer_id is not None:
        references.append(f"offer={message.referenced_offer_id}")
    if message.referenced_action_id is not None:
        references.append(f"action={message.referenced_action_id}")
    suffix = f" ({', '.join(references)})" if references else ""
    return (
        f"[{message.sequence_number:03d}] {sender} "
        f"[{message.visibility.value}/{message.message_type.value}]: {content}{suffix}"
    )


def format_transcript(
    messages: Iterable[MessageEnvelope],
    participant_names: Optional[Dict[ParticipantId, str]] = None,
) -> str:
    return "\n".join(format_message(message, participant_names) for message in messages)


class TranscriptSummaryItem(DomainModel):
    content: str = Field(min_length=1, max_length=4000)
    source_event_ids: Tuple[str, ...] = Field(min_length=1)
    summarized: bool

    @field_validator("source_event_ids", mode="before")
    @classmethod
    def source_ids_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class TranscriptSummary(DomainModel):
    items: Tuple[TranscriptSummaryItem, ...]
    source_message_count: int = Field(ge=0)

    @field_validator("items", mode="before")
    @classmethod
    def items_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def sources_are_preserved(self) -> "TranscriptSummary":
        source_count = sum(len(item.source_event_ids) for item in self.items)
        if source_count != self.source_message_count:
            raise ValueError("transcript summary must preserve one source id per message")
        return self


def summarize_transcript(
    messages: Iterable[MessageEnvelope],
    *,
    maximum_items: int = 8,
    maximum_characters: int = 2000,
) -> TranscriptSummary:
    """Deterministically compact a visible transcript while retaining source event ids."""

    ordered = tuple(sorted(messages, key=lambda item: item.sequence_number))
    if maximum_items < 1:
        raise ValueError("maximum_items must be positive")
    if maximum_characters < 64:
        raise ValueError("maximum_characters must be at least 64")
    if maximum_items > maximum_characters:
        raise ValueError("maximum_items cannot exceed the character budget")

    def bound_contents(
        items: Tuple[TranscriptSummaryItem, ...]
    ) -> Tuple[TranscriptSummaryItem, ...]:
        remaining = maximum_characters
        bounded = []
        for index, item in enumerate(items):
            items_left = len(items) - index
            allowance = max(1, remaining // items_left)
            content = item.content[:allowance]
            bounded.append(item.model_copy(update={"content": content}))
            remaining -= len(content)
        return tuple(bounded)

    def source_id(message: MessageEnvelope) -> str:
        return str(message.referenced_action_id or message.correlation_id)

    if len(ordered) <= maximum_items:
        items = tuple(
            TranscriptSummaryItem(
                content=format_message(message)[:maximum_characters],
                source_event_ids=(source_id(message),),
                summarized=False,
            )
            for message in ordered
        )
        return TranscriptSummary(items=bound_contents(items), source_message_count=len(ordered))

    recent_count = max(0, maximum_items - 1)
    compacted = ordered[:-recent_count] if recent_count else ordered
    recent = ordered[-recent_count:] if recent_count else ()
    source_ids = tuple(source_id(message) for message in compacted)
    senders = ", ".join(dict.fromkeys(str(message.sender) for message in compacted))
    summary_text = (
        f"{len(compacted)} earlier messages from {senders}; "
        f"types: {', '.join(message.message_type.value for message in compacted)}."
    )[:maximum_characters]
    items = (
        TranscriptSummaryItem(
            content=summary_text,
            source_event_ids=source_ids,
            summarized=True,
        ),
    ) + tuple(
        TranscriptSummaryItem(
            content=format_message(message)[:maximum_characters],
            source_event_ids=(source_id(message),),
            summarized=False,
        )
        for message in recent
    )
    return TranscriptSummary(items=bound_contents(items), source_message_count=len(ordered))


_MESSAGE_SEQUENCE_ADAPTER = TypeAdapter(Tuple[MessageEnvelope, ...])
