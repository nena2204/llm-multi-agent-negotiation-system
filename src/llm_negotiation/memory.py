from __future__ import annotations

import hashlib
from math import ceil, isclose
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Optional, Protocol, Tuple, Union

from pydantic import Field, ValidationError, field_validator, model_validator

from .communication import MessageEnvelope, MessageType, MessageVisibility
from .domain import (
    ActionType,
    ActionId,
    CategoricalIssue,
    CategoricalIssueValue,
    CorrelationId,
    CounterAction,
    MessageId,
    NegotiationAction,
    NegotiationScenario,
    Offer,
    OfferId,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    ProposeAction,
    TerminalOutcome,
    NumericIssue,
    NumericIssueValue,
    calculate_utility,
)
from .domain.models import DomainModel, Issue
from .protocol import (
    ActionAppliedEvent,
    NegotiationSession,
    ProtocolEvent,
    ProtocolPhase,
    SystemTerminatedEvent,
    legal_action_types_for,
)


MEMORY_SCHEMA_VERSION = "1.0"


class MemoryError(ValueError):
    """Base error for participant memory invariants."""


class MemoryIsolationError(MemoryError):
    """Raised when memory ingestion would cross a participant visibility boundary."""


class MemoryLimitError(MemoryError):
    """Raised when configured bounds cannot hold even a compacted memory event."""


class MemoryPersistenceError(MemoryError):
    """Raised when a persisted snapshot has an unsupported or invalid schema."""


class PublicAgentIdentity(DomainModel):
    participant: Participant
    persona: str = Field(default="", max_length=1000)


class AgentProfile(DomainModel):
    """Immutable owner configuration: public identity plus that owner's private preferences."""

    identity: PublicAgentIdentity
    preferences: ParticipantPreferences

    @model_validator(mode="after")
    def matching_identity(self) -> AgentProfile:
        if self.identity.participant.participant_id != self.preferences.participant_id:
            raise ValueError("agent identity and private preferences must belong to the same participant")
        return self


class AgentObservation(DomainModel):
    """Participant-specific ingestion input; never passed directly to a policy or LLM."""

    scenario: NegotiationScenario
    session: NegotiationSession
    own_profile: AgentProfile

    @model_validator(mode="after")
    def consistent_observation(self) -> AgentObservation:
        participant_id = self.own_profile.identity.participant.participant_id
        if self.session.scenario_id != self.scenario.scenario_id:
            raise ValueError("observation session and scenario identifiers must match")
        scenario_participants = tuple(item.participant_id for item in self.scenario.participants)
        if self.session.participants != scenario_participants:
            raise ValueError("observation session participants must match the scenario")
        if participant_id not in self.session.participants:
            raise ValueError("observing participant must belong to the session")
        return self

    @property
    def participant_id(self) -> ParticipantId:
        return self.own_profile.identity.participant.participant_id


class MemoryLimits(DomainModel):
    episodic_event_limit: int = Field(default=100, ge=2)
    episodic_character_limit: int = Field(default=20_000, ge=512)
    episodic_token_limit: Optional[int] = Field(default=None, ge=128)
    working_message_limit: int = Field(default=20, ge=1)
    working_character_limit: int = Field(default=8_000, ge=512)
    working_token_limit: Optional[int] = Field(default=None, ge=128)
    recent_offer_limit: int = Field(default=6, ge=3, le=100)
    learned_strategy_limit: int = Field(default=50, ge=0, le=1000)
    approximate_characters_per_token: int = Field(default=4, ge=1, le=16)
    summary_character_limit: int = Field(default=500, ge=32, le=4000)

    @property
    def effective_episodic_characters(self) -> int:
        token_characters = (
            self.episodic_token_limit * self.approximate_characters_per_token
            if self.episodic_token_limit is not None
            else self.episodic_character_limit
        )
        return min(self.episodic_character_limit, token_characters)

    @property
    def effective_working_characters(self) -> int:
        token_characters = (
            self.working_token_limit * self.approximate_characters_per_token
            if self.working_token_limit is not None
            else self.working_character_limit
        )
        return min(self.working_character_limit, token_characters)


class LongTermMemory(DomainModel):
    owner_id: ParticipantId
    negotiation_id: str = Field(min_length=1, max_length=100)
    scenario_title: str = Field(min_length=1, max_length=200)
    public_participants: Tuple[Participant, ...] = Field(min_length=2)
    role: ParticipantRole
    goals: Tuple[str, ...] = Field(min_length=1)
    protocol_rules: Tuple[str, ...] = Field(min_length=1)
    issue_descriptions: Tuple[Issue, ...] = Field(min_length=1)
    strategy_guidance: Tuple[str, ...] = ()
    learned_strategy_data: Tuple[str, ...] = ()
    public_persona: str = Field(default="", max_length=1000)
    own_preferences: ParticipantPreferences

    @field_validator(
        "public_participants",
        "goals",
        "protocol_rules",
        "issue_descriptions",
        "strategy_guidance",
        "learned_strategy_data",
        mode="before",
    )
    @classmethod
    def tuples_from_json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("goals", "protocol_rules", "strategy_guidance", "learned_strategy_data")
    @classmethod
    def concise_nonblank_facts(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("long-term memory facts must not be blank")
        return values

    @model_validator(mode="after")
    def owner_is_consistent(self) -> LongTermMemory:
        participant_ids = tuple(item.participant_id for item in self.public_participants)
        if self.owner_id not in participant_ids:
            raise ValueError("long-term memory owner must be a public scenario participant")
        if self.own_preferences.participant_id != self.owner_id:
            raise ValueError("long-term memory may contain only its owner's private preferences")
        participant = next(item for item in self.public_participants if item.participant_id == self.owner_id)
        if participant.role is not self.role:
            raise ValueError("long-term role must match the owner's public participant role")
        return self

    def scenario(self) -> NegotiationScenario:
        return NegotiationScenario(
            scenario_id=self.negotiation_id,
            title=self.scenario_title,
            participants=self.public_participants,
            issues=self.issue_descriptions,
        )


class OfferUtilityEstimate(DomainModel):
    offer_id: OfferId
    utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class VisibleMessage(DomainModel):
    message_id: MessageId
    sender: ParticipantId
    message_type: MessageType
    visibility: MessageVisibility
    sequence_number: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=1000)
    correlation_id: CorrelationId
    referenced_offer_id: Optional[OfferId] = None
    referenced_action_id: Optional[ActionId] = None


class RecentOfferMemory(DomainModel):
    actor_id: ParticipantId
    offer: Offer
    owner_utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class WorkingMemory(DomainModel):
    owner_id: ParticipantId
    negotiation_id: str = Field(min_length=1, max_length=100)
    legal_actions: Tuple[ActionType, ...]
    outstanding_offer: Optional[Offer] = None
    round_number: int = Field(ge=1)
    deadline_round: int = Field(ge=1)
    rounds_remaining: int = Field(ge=0)
    current_turn: Optional[ParticipantId]
    phase: ProtocolPhase
    visible_messages: Tuple[VisibleMessage, ...] = ()
    recent_offers: Tuple[RecentOfferMemory, ...] = ()
    current_utility_estimates: Tuple[OfferUtilityEstimate, ...] = ()
    active_plan: Tuple[str, ...] = Field(default=(), max_length=10)
    public_event_count: int = Field(ge=0)
    last_protocol_correlation_id: Optional[CorrelationId] = None
    protocol_history_digest: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    last_visible_message_sequence: int = Field(default=0, ge=0)
    outcome: Optional[TerminalOutcome] = None
    approximate_characters: int = Field(ge=0)
    approximate_tokens: int = Field(ge=0)

    @field_validator(
        "legal_actions",
        "visible_messages",
        "recent_offers",
        "current_utility_estimates",
        "active_plan",
        mode="before",
    )
    @classmethod
    def tuples_from_json_arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("active_plan")
    @classmethod
    def plan_is_concise(cls, values: Tuple[str, ...]) -> Tuple[str, ...]:
        if any(not value.strip() or len(value) > 500 for value in values):
            raise ValueError("active plan items must be nonblank and at most 500 characters")
        return values

    @model_validator(mode="after")
    def consistent_deadline(self) -> WorkingMemory:
        if self.round_number > self.deadline_round:
            raise ValueError("working-memory round cannot exceed its deadline")
        if self.rounds_remaining != self.deadline_round - self.round_number:
            raise ValueError("working-memory rounds_remaining does not match the deadline")
        terminal = self.phase in {
            ProtocolPhase.AGREED,
            ProtocolPhase.FAILED,
            ProtocolPhase.WITHDRAWN,
            ProtocolPhase.EXPIRED,
        }
        if terminal and self.outcome is None:
            raise ValueError("terminal working memory requires an outcome")
        if terminal and self.current_turn is not None:
            raise ValueError("terminal working memory cannot have a current turn")
        if not terminal and self.outcome is not None:
            raise ValueError("non-terminal working memory cannot have an outcome")
        if not terminal and self.current_turn is None:
            raise ValueError("non-terminal working memory requires a current turn")
        return self


class ObservationMemoryEvent(DomainModel):
    event_type: Literal["observation"] = "observation"
    sequence_number: int = Field(ge=1)
    round_number: int = Field(ge=1)
    phase: ProtocolPhase
    current_turn: Optional[ParticipantId]
    outstanding_offer_id: Optional[OfferId] = None
    observed_action: Optional[NegotiationAction] = None
    fact: str = Field(min_length=1, max_length=500)


class ActionMemoryEvent(DomainModel):
    event_type: Literal["action"] = "action"
    sequence_number: int = Field(ge=1)
    action: NegotiationAction


class ReceivedMessageMemoryEvent(DomainModel):
    event_type: Literal["received_message"] = "received_message"
    sequence_number: int = Field(ge=1)
    message: MessageEnvelope


class OutcomeMemoryEvent(DomainModel):
    event_type: Literal["outcome"] = "outcome"
    sequence_number: int = Field(ge=1)
    outcome: TerminalOutcome


class SummaryMemoryEvent(DomainModel):
    event_type: Literal["summary"] = "summary"
    sequence_number: int = Field(ge=1)
    first_sequence_number: int = Field(ge=1)
    last_sequence_number: int = Field(ge=1)
    compacted_event_count: int = Field(ge=1)
    content: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def valid_range(self) -> SummaryMemoryEvent:
        if self.first_sequence_number > self.last_sequence_number:
            raise ValueError("summary event range must be ordered")
        if self.sequence_number != self.last_sequence_number:
            raise ValueError("summary sequence number must equal its summarized range end")
        return self


MemoryEvent = Annotated[
    Union[
        ObservationMemoryEvent,
        ActionMemoryEvent,
        ReceivedMessageMemoryEvent,
        OutcomeMemoryEvent,
        SummaryMemoryEvent,
    ],
    Field(discriminator="event_type"),
]


class EpisodicMemory(DomainModel):
    owner_id: ParticipantId
    negotiation_id: str = Field(min_length=1, max_length=100)
    events: Tuple[MemoryEvent, ...] = ()
    compaction_count: int = Field(default=0, ge=0)
    approximate_characters: int = Field(default=0, ge=0)
    approximate_tokens: int = Field(default=0, ge=0)

    @field_validator("events", mode="before")
    @classmethod
    def events_from_json_array(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def ordered_events(self) -> EpisodicMemory:
        numbers = tuple(event.sequence_number for event in self.events)
        if any(later <= earlier for earlier, later in zip(numbers, numbers[1:])):
            raise ValueError("episodic memory event sequence must be strictly increasing")
        return self


class MemorySnapshot(DomainModel):
    schema_version: Literal["1.0"] = MEMORY_SCHEMA_VERSION
    owner_id: ParticipantId
    limits: MemoryLimits
    long_term: LongTermMemory
    working: WorkingMemory
    episodic: EpisodicMemory

    @model_validator(mode="after")
    def consistent_owner_and_negotiation(self) -> MemorySnapshot:
        if {self.long_term.owner_id, self.working.owner_id, self.episodic.owner_id} != {
            self.owner_id
        }:
            raise ValueError("all memory layers must belong to the snapshot owner")
        negotiation_ids = {
            self.long_term.negotiation_id,
            self.working.negotiation_id,
            self.episodic.negotiation_id,
        }
        if len(negotiation_ids) != 1:
            raise ValueError("all memory layers must belong to one negotiation")
        scenario = self.long_term.scenario()
        validation_offer = Offer(
            offer_id=OfferId("snapshot-validation-offer"),
            values=tuple(_validation_issue_value(issue) for issue in scenario.issues),
        )
        calculate_utility(scenario, validation_offer, self.long_term.own_preferences)
        if self.working.current_turn is not None:
            scenario.participant(self.working.current_turn)
        if self.working.outstanding_offer is not None:
            scenario.validate_offer(self.working.outstanding_offer)
        expected_estimates = {}
        for recent in self.working.recent_offers:
            scenario.participant(recent.actor_id)
            scenario.validate_offer(recent.offer)
            expected_utility = calculate_utility(
                scenario,
                recent.offer,
                self.long_term.own_preferences,
            )
            if not isclose(recent.owner_utility, expected_utility, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("recent-offer utility does not match the owner's preferences")
            expected_estimates[recent.offer.offer_id] = expected_utility
        if self.working.outstanding_offer is not None:
            expected_estimates[self.working.outstanding_offer.offer_id] = calculate_utility(
                scenario,
                self.working.outstanding_offer,
                self.long_term.own_preferences,
            )
        supplied_estimates = {
            estimate.offer_id: estimate.utility
            for estimate in self.working.current_utility_estimates
        }
        if len(supplied_estimates) != len(self.working.current_utility_estimates):
            raise ValueError("working-memory utility estimates must have unique offer identifiers")
        if set(supplied_estimates) != set(expected_estimates):
            raise ValueError("working-memory utility estimates must cover current retained offers")
        if any(
            not isclose(supplied_estimates[offer_id], utility, rel_tol=0.0, abs_tol=1e-9)
            for offer_id, utility in expected_estimates.items()
        ):
            raise ValueError("working-memory utility estimate does not match owner preferences")
        reconstructed_state = NegotiationSession(
            scenario_id=self.working.negotiation_id,
            participants=tuple(
                participant.participant_id
                for participant in self.long_term.public_participants
            ),
            current_turn=self.working.current_turn,
            round_number=self.working.round_number,
            maximum_rounds=self.working.deadline_round,
            latest_valid_offer=self.working.outstanding_offer,
            phase=self.working.phase,
            outcome=self.working.outcome,
        )
        expected_legal_actions = legal_action_types_for(reconstructed_state, self.owner_id)
        if self.working.legal_actions != expected_legal_actions:
            raise ValueError("working-memory legal actions do not match its protocol facts")
        episodic_characters = _event_characters(self.episodic.events)
        if len(self.episodic.events) > self.limits.episodic_event_limit:
            raise ValueError("episodic memory exceeds its configured event limit")
        if episodic_characters > self.limits.effective_episodic_characters:
            raise ValueError("episodic memory exceeds its configured character/token limit")
        if self.episodic.approximate_characters != episodic_characters:
            raise ValueError("episodic memory character estimate does not match its events")
        expected_episodic_tokens = ceil(
            episodic_characters / self.limits.approximate_characters_per_token
        )
        if self.episodic.approximate_tokens != expected_episodic_tokens:
            raise ValueError("episodic memory token estimate does not match its events")
        working_characters = sum(
            len(message.model_dump_json()) for message in self.working.visible_messages
        )
        if len(self.working.visible_messages) > self.limits.working_message_limit:
            raise ValueError("working memory exceeds its configured message limit")
        if len(self.working.recent_offers) > self.limits.recent_offer_limit:
            raise ValueError("working memory exceeds its configured recent-offer limit")
        if len(self.long_term.learned_strategy_data) > self.limits.learned_strategy_limit:
            raise ValueError("long-term memory exceeds its configured learned-strategy limit")
        if working_characters > self.limits.effective_working_characters:
            raise ValueError("working memory exceeds its configured character/token limit")
        if self.working.approximate_characters != working_characters:
            raise ValueError("working memory character estimate does not match visible messages")
        expected_working_tokens = ceil(
            working_characters / self.limits.approximate_characters_per_token
        )
        if self.working.approximate_tokens != expected_working_tokens:
            raise ValueError("working memory token estimate does not match visible messages")
        return self

    @property
    def participant_id(self) -> ParticipantId:
        return self.owner_id

    @property
    def scenario(self) -> NegotiationScenario:
        return self.long_term.scenario()


class MemorySummarizer(Protocol):
    def summarize(self, events: Tuple[MemoryEvent, ...], max_characters: int) -> str:
        """Produce a concise factual summary without hidden reasoning."""


class DeterministicMemorySummarizer:
    """Non-LLM compactor that records only event types and protocol-visible facts."""

    def summarize(self, events: Tuple[MemoryEvent, ...], max_characters: int) -> str:
        facts = []
        for event in events:
            if isinstance(event, SummaryMemoryEvent):
                facts.append(
                    f"prior summary covering {event.first_sequence_number}-{event.last_sequence_number}"
                )
            elif isinstance(event, ObservationMemoryEvent):
                if event.observed_action is not None:
                    facts.append(
                        f"observed {event.observed_action.actor_id} {event.observed_action.action.value}"
                    )
                else:
                    facts.append(f"round {event.round_number} phase {event.phase.value}")
            elif isinstance(event, ActionMemoryEvent):
                facts.append(f"owner action {event.action.action.value}")
            elif isinstance(event, ReceivedMessageMemoryEvent):
                facts.append(
                    f"received {event.message.message_type.value} from {event.message.sender}"
                )
            else:
                facts.append(f"outcome {event.outcome.status.value}")
        summary = "; ".join(facts) or "No retained event details."
        if len(summary) > max_characters:
            summary = summary[: max(1, max_characters - 1)].rstrip() + "…"
        return summary


def _event_characters(events: Tuple[MemoryEvent, ...]) -> int:
    return sum(len(event.model_dump_json()) for event in events)


def _protocol_history_digest(events: Tuple[ProtocolEvent, ...]) -> Optional[str]:
    if not events:
        return None
    payload = "\n".join(event.model_dump_json() for event in events)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _represented_event_count(events: Tuple[MemoryEvent, ...]) -> int:
    return sum(
        event.compacted_event_count if isinstance(event, SummaryMemoryEvent) else 1
        for event in events
    )


def _compact_events(
    events: Tuple[MemoryEvent, ...],
    limits: MemoryLimits,
    summarizer: MemorySummarizer,
) -> Tuple[Tuple[MemoryEvent, ...], int]:
    compacted = list(events)
    compactions = 0
    character_limit = limits.effective_episodic_characters
    while (
        len(compacted) > limits.episodic_event_limit
        or _event_characters(tuple(compacted)) > character_limit
    ):
        if len(compacted) > 1:
            prefix_length = max(2, len(compacted) - limits.episodic_event_limit + 1)
        else:
            prefix_length = 1
        prefix = tuple(compacted[:prefix_length])
        first = (
            prefix[0].first_sequence_number
            if isinstance(prefix[0], SummaryMemoryEvent)
            else prefix[0].sequence_number
        )
        last = (
            prefix[-1].last_sequence_number
            if isinstance(prefix[-1], SummaryMemoryEvent)
            else prefix[-1].sequence_number
        )
        content_limit = min(limits.summary_character_limit, max(32, character_limit - 300))
        summary = SummaryMemoryEvent(
            sequence_number=last,
            first_sequence_number=first,
            last_sequence_number=last,
            compacted_event_count=_represented_event_count(prefix),
            content=summarizer.summarize(prefix, content_limit),
        )
        while len(summary.model_dump_json()) > character_limit and len(summary.content) > 1:
            summary = summary.model_copy(update={"content": summary.content[: max(1, len(summary.content) // 2)]})
        if len(summary.model_dump_json()) > character_limit:
            raise MemoryLimitError("episodic character limit is too small for a summary event")
        compacted = [summary] + compacted[prefix_length:]
        compactions += 1
    return tuple(compacted), compactions


def _visible_messages(
    messages: Tuple[MessageEnvelope, ...],
    limits: MemoryLimits,
) -> Tuple[Tuple[VisibleMessage, ...], int]:
    retained = []
    character_limit = limits.effective_working_characters
    for message in reversed(messages[-limits.working_message_limit :]):
        content = message.content[:1000]
        digest = VisibleMessage(
            message_id=message.message_id,
            sender=message.sender,
            message_type=message.message_type,
            visibility=message.visibility,
            sequence_number=message.sequence_number,
            content=content,
            correlation_id=message.correlation_id,
            referenced_offer_id=message.referenced_offer_id,
            referenced_action_id=message.referenced_action_id,
        )
        while len(digest.model_dump_json()) > character_limit and len(digest.content) > 1:
            digest = digest.model_copy(update={"content": digest.content[: max(1, len(digest.content) // 2)]})
        candidate = [digest] + retained
        if sum(len(item.model_dump_json()) for item in candidate) > character_limit:
            break
        retained = candidate
    characters = sum(len(item.model_dump_json()) for item in retained)
    return tuple(retained), characters


class AgentMemory:
    """Controlled, participant-owned memory; the raw protocol/audit trails remain external."""

    DEFAULT_PROTOCOL_RULES: ClassVar[Tuple[str, ...]] = (
        "Only actions listed in working memory are currently legal.",
        "Acceptance must reference the outstanding offer.",
        "Messages communicate but never execute protocol actions.",
        "Do not disclose private preference or BATNA information.",
    )

    def __init__(
        self,
        profile: AgentProfile,
        scenario: NegotiationScenario,
        *,
        goals: Optional[Tuple[str, ...]] = None,
        strategy_guidance: Tuple[str, ...] = (),
        learned_strategy_data: Tuple[str, ...] = (),
        limits: Optional[MemoryLimits] = None,
        summarizer: Optional[MemorySummarizer] = None,
    ) -> None:
        self.profile = profile
        self.limits = limits or MemoryLimits()
        self.summarizer = summarizer or DeterministicMemorySummarizer()
        self.long_term = self._build_long_term(
            scenario,
            goals=goals,
            strategy_guidance=strategy_guidance,
            learned_strategy_data=learned_strategy_data,
        )
        self.working: Optional[WorkingMemory] = None
        self.episodic = EpisodicMemory(owner_id=self.owner_id, negotiation_id=scenario.scenario_id)
        self._next_memory_sequence = 1
        self._seen_protocol_event_count = 0
        self._last_protocol_correlation_id: Optional[CorrelationId] = None
        self._protocol_history_digest: Optional[str] = None
        self._last_visible_message_sequence = 0
        self._outcome_recorded = False

    @property
    def owner_id(self) -> ParticipantId:
        return self.profile.identity.participant.participant_id

    def _build_long_term(
        self,
        scenario: NegotiationScenario,
        *,
        goals: Optional[Tuple[str, ...]],
        strategy_guidance: Tuple[str, ...],
        learned_strategy_data: Tuple[str, ...],
    ) -> LongTermMemory:
        public_participant = scenario.participant(self.owner_id)
        if public_participant != self.profile.identity.participant:
            raise MemoryIsolationError("agent public identity must match the scenario participant")
        calculate_utility(
            scenario,
            Offer(
                offer_id=OfferId("memory-validation-offer"),
                values=tuple(
                    issue_value
                    for issue in scenario.issues
                    for issue_value in [_validation_issue_value(issue)]
                ),
            ),
            self.profile.preferences,
        )
        resolved_goals = goals or (
            f"Negotiate consistently with the {public_participant.role.value} role.",
            "Do not knowingly accept below reservation utility.",
        )
        retained_learning = (
            learned_strategy_data[-self.limits.learned_strategy_limit :]
            if self.limits.learned_strategy_limit
            else ()
        )
        return LongTermMemory(
            owner_id=self.owner_id,
            negotiation_id=scenario.scenario_id,
            scenario_title=scenario.title,
            public_participants=scenario.participants,
            role=public_participant.role,
            goals=resolved_goals,
            protocol_rules=self.DEFAULT_PROTOCOL_RULES,
            issue_descriptions=scenario.issues,
            strategy_guidance=strategy_guidance,
            learned_strategy_data=retained_learning,
            public_persona=self.profile.identity.persona,
            own_preferences=self.profile.preferences,
        )

    def update(
        self,
        observation: AgentObservation,
        *,
        legal_actions: Optional[Tuple[ActionType, ...]] = None,
        visible_messages: Tuple[MessageEnvelope, ...] = (),
        active_plan: Tuple[str, ...] = (),
    ) -> MemorySnapshot:
        checkpoint = (
            self.working,
            self.episodic,
            self._next_memory_sequence,
            self._seen_protocol_event_count,
            self._last_protocol_correlation_id,
            self._protocol_history_digest,
            self._last_visible_message_sequence,
            self._outcome_recorded,
        )
        try:
            return self._update(
                observation,
                legal_actions=legal_actions,
                visible_messages=visible_messages,
                active_plan=active_plan,
            )
        except Exception:
            (
                self.working,
                self.episodic,
                self._next_memory_sequence,
                self._seen_protocol_event_count,
                self._last_protocol_correlation_id,
                self._protocol_history_digest,
                self._last_visible_message_sequence,
                self._outcome_recorded,
            ) = checkpoint
            raise

    def _update(
        self,
        observation: AgentObservation,
        *,
        legal_actions: Optional[Tuple[ActionType, ...]] = None,
        visible_messages: Tuple[MessageEnvelope, ...] = (),
        active_plan: Tuple[str, ...] = (),
    ) -> MemorySnapshot:
        self._validate_observation(observation)
        derived_legal_actions = legal_action_types_for(observation.session, self.owner_id)
        if legal_actions is not None and legal_actions != derived_legal_actions:
            raise MemoryIsolationError("supplied legal actions do not match authoritative protocol state")
        ordered_messages = tuple(sorted(visible_messages, key=lambda item: item.sequence_number))
        self._validate_visible_messages(ordered_messages, observation.scenario.scenario_id)
        self._ingest_protocol_events(observation.session)
        self._ingest_messages(ordered_messages)
        self._append_event(
            ObservationMemoryEvent(
                sequence_number=self._take_sequence(),
                round_number=observation.session.round_number,
                phase=observation.session.phase,
                current_turn=observation.session.current_turn,
                outstanding_offer_id=(
                    observation.session.latest_valid_offer.offer_id
                    if observation.session.latest_valid_offer is not None
                    else None
                ),
                fact=(
                    f"Round {observation.session.round_number} is "
                    f"{observation.session.phase.value}; current turn is "
                    f"{observation.session.current_turn}."
                ),
            )
        )
        if observation.session.outcome is not None and not self._outcome_recorded:
            self._append_event(
                OutcomeMemoryEvent(
                    sequence_number=self._take_sequence(),
                    outcome=observation.session.outcome,
                )
            )
            self._outcome_recorded = True

        visible, working_characters = _visible_messages(ordered_messages, self.limits)
        recent_offers = []
        for event in observation.session.event_sequence:
            if isinstance(event, ActionAppliedEvent) and isinstance(
                event.action, (ProposeAction, CounterAction)
            ):
                recent_offers.append(
                    RecentOfferMemory(
                        actor_id=event.action.actor_id,
                        offer=event.action.offer,
                        owner_utility=calculate_utility(
                            observation.scenario,
                            event.action.offer,
                            self.profile.preferences,
                        ),
                    )
                )
        recent = tuple(recent_offers[-self.limits.recent_offer_limit :])
        estimates = tuple(
            OfferUtilityEstimate(offer_id=item.offer.offer_id, utility=item.owner_utility)
            for item in recent
        )
        if observation.session.latest_valid_offer is not None:
            if all(
                estimate.offer_id != observation.session.latest_valid_offer.offer_id
                for estimate in estimates
            ):
                estimates = estimates + (
                    OfferUtilityEstimate(
                        offer_id=observation.session.latest_valid_offer.offer_id,
                        utility=calculate_utility(
                            observation.scenario,
                            observation.session.latest_valid_offer,
                            self.profile.preferences,
                        ),
                    ),
                )
        self.working = WorkingMemory(
            owner_id=self.owner_id,
            negotiation_id=observation.scenario.scenario_id,
            legal_actions=derived_legal_actions,
            outstanding_offer=observation.session.latest_valid_offer,
            round_number=observation.session.round_number,
            deadline_round=observation.session.maximum_rounds,
            rounds_remaining=observation.session.maximum_rounds - observation.session.round_number,
            current_turn=observation.session.current_turn,
            phase=observation.session.phase,
            visible_messages=visible,
            recent_offers=recent,
            current_utility_estimates=estimates,
            active_plan=active_plan,
            public_event_count=len(observation.session.event_sequence),
            last_protocol_correlation_id=self._last_protocol_correlation_id,
            protocol_history_digest=self._protocol_history_digest,
            last_visible_message_sequence=self._last_visible_message_sequence,
            outcome=observation.session.outcome,
            approximate_characters=working_characters,
            approximate_tokens=ceil(
                working_characters / self.limits.approximate_characters_per_token
            ),
        )
        return self.snapshot()

    def _validate_observation(self, observation: AgentObservation) -> None:
        if observation.participant_id != self.owner_id:
            raise MemoryIsolationError("agent memory cannot ingest another participant's observation")
        if observation.own_profile != self.profile:
            raise MemoryIsolationError("observation profile does not match this memory owner")
        if observation.scenario != self.long_term.scenario():
            raise MemoryIsolationError("observation scenario does not match active long-term memory")
        if len(observation.session.event_sequence) < self._seen_protocol_event_count:
            raise MemoryIsolationError("protocol history cannot move backwards within agent memory")
        if self._seen_protocol_event_count and (
            _protocol_history_digest(
                observation.session.event_sequence[: self._seen_protocol_event_count]
            )
            != self._protocol_history_digest
        ):
            raise MemoryIsolationError("protocol history does not continue the previously observed episode")

    def _validate_visible_messages(
        self,
        messages: Tuple[MessageEnvelope, ...],
        negotiation_id: str,
    ) -> None:
        if len({message.message_id for message in messages}) != len(messages):
            raise MemoryIsolationError("visible message view contains duplicate identifiers")
        if any(later.sequence_number <= earlier.sequence_number for earlier, later in zip(messages, messages[1:])):
            raise MemoryIsolationError("visible messages must have deterministic increasing order")
        for message in messages:
            if message.negotiation_id != negotiation_id:
                raise MemoryIsolationError("visible message belongs to a different negotiation")
            readable = (
                message.visibility is MessageVisibility.PUBLIC
                or message.sender == self.owner_id
                or self.owner_id in message.recipients
            )
            if not readable or message.visibility is MessageVisibility.SYSTEM_AUDIT:
                raise MemoryIsolationError(
                    f"message '{message.message_id}' is outside participant '{self.owner_id}' visibility"
                )

    def _ingest_protocol_events(self, session: NegotiationSession) -> None:
        for event in session.event_sequence[self._seen_protocol_event_count :]:
            if isinstance(event, ActionAppliedEvent):
                if event.action.actor_id == self.owner_id:
                    memory_event: MemoryEvent = ActionMemoryEvent(
                        sequence_number=self._take_sequence(),
                        action=event.action,
                    )
                else:
                    memory_event = ObservationMemoryEvent(
                        sequence_number=self._take_sequence(),
                        round_number=event.round_after,
                        phase=event.phase_after,
                        current_turn=session.current_turn,
                        outstanding_offer_id=(
                            event.action.offer.offer_id
                            if isinstance(event.action, (ProposeAction, CounterAction))
                            else None
                        ),
                        observed_action=event.action,
                        fact=f"Observed {event.action.actor_id} perform {event.action.action.value}.",
                    )
                self._append_event(memory_event)
            elif isinstance(event, SystemTerminatedEvent):
                self._append_event(
                    ObservationMemoryEvent(
                        sequence_number=self._take_sequence(),
                        round_number=event.round_number,
                        phase=event.phase_after,
                        current_turn=None,
                        fact=f"System terminated the negotiation: {event.reason.value}.",
                    )
                )
            self._last_protocol_correlation_id = event.correlation_id
        self._seen_protocol_event_count = len(session.event_sequence)
        self._protocol_history_digest = _protocol_history_digest(session.event_sequence)

    def _ingest_messages(self, messages: Tuple[MessageEnvelope, ...]) -> None:
        for message in messages:
            if message.sequence_number <= self._last_visible_message_sequence:
                continue
            if message.sender != self.owner_id:
                self._append_event(
                    ReceivedMessageMemoryEvent(
                        sequence_number=self._take_sequence(),
                        message=message,
                    )
                )
            self._last_visible_message_sequence = message.sequence_number

    def _take_sequence(self) -> int:
        sequence_number = self._next_memory_sequence
        self._next_memory_sequence += 1
        return sequence_number

    def _append_event(self, event: MemoryEvent) -> None:
        events, new_compactions = _compact_events(
            self.episodic.events + (event,),
            self.limits,
            self.summarizer,
        )
        characters = _event_characters(events)
        self.episodic = EpisodicMemory(
            owner_id=self.owner_id,
            negotiation_id=self.long_term.negotiation_id,
            events=events,
            compaction_count=self.episodic.compaction_count + new_compactions,
            approximate_characters=characters,
            approximate_tokens=ceil(
                characters / self.limits.approximate_characters_per_token
            ),
        )

    def snapshot(self) -> MemorySnapshot:
        if self.working is None:
            raise MemoryError("working memory has not been initialized from an observation")
        return MemorySnapshot(
            owner_id=self.owner_id,
            limits=self.limits,
            long_term=self.long_term,
            working=self.working,
            episodic=self.episodic,
        )

    def reset_for_negotiation(
        self,
        scenario: NegotiationScenario,
        *,
        retain_learned_strategy: bool = True,
    ) -> None:
        learned = self.long_term.learned_strategy_data if retain_learned_strategy else ()
        guidance = self.long_term.strategy_guidance
        goals = self.long_term.goals
        self.long_term = self._build_long_term(
            scenario,
            goals=goals,
            strategy_guidance=guidance,
            learned_strategy_data=learned,
        )
        self.working = None
        self.episodic = EpisodicMemory(owner_id=self.owner_id, negotiation_id=scenario.scenario_id)
        self._next_memory_sequence = 1
        self._seen_protocol_event_count = 0
        self._last_protocol_correlation_id = None
        self._protocol_history_digest = None
        self._last_visible_message_sequence = 0
        self._outcome_recorded = False

    def remember_strategy_fact(self, fact: str) -> None:
        """Store one concise learned fact, bounded independently from episodic memory."""

        if not isinstance(fact, str) or not fact.strip() or len(fact) > 500:
            raise ValueError("learned strategy facts must be nonblank and at most 500 characters")
        learned = self.long_term.learned_strategy_data
        if fact in learned:
            return
        limit = self.limits.learned_strategy_limit
        retained = (learned + (fact,))[-limit:] if limit else ()
        values = {
            field_name: getattr(self.long_term, field_name)
            for field_name in LongTermMemory.model_fields
        }
        values["learned_strategy_data"] = retained
        self.long_term = LongTermMemory.model_validate(values)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: MemorySnapshot,
        *,
        limits: Optional[MemoryLimits] = None,
        summarizer: Optional[MemorySummarizer] = None,
    ) -> AgentMemory:
        profile = AgentProfile(
            identity=PublicAgentIdentity(
                participant=next(
                    participant
                    for participant in snapshot.long_term.public_participants
                    if participant.participant_id == snapshot.owner_id
                ),
                persona=snapshot.long_term.public_persona,
            ),
            preferences=snapshot.long_term.own_preferences,
        )
        memory = cls.__new__(cls)
        memory.profile = profile
        memory.limits = limits or snapshot.limits
        memory.summarizer = summarizer or DeterministicMemorySummarizer()
        memory.long_term = snapshot.long_term
        memory.working = snapshot.working
        memory.episodic = snapshot.episodic
        memory._next_memory_sequence = (
            max((event.sequence_number for event in snapshot.episodic.events), default=0) + 1
        )
        memory._seen_protocol_event_count = snapshot.working.public_event_count
        memory._last_protocol_correlation_id = snapshot.working.last_protocol_correlation_id
        memory._protocol_history_digest = snapshot.working.protocol_history_digest
        memory._last_visible_message_sequence = snapshot.working.last_visible_message_sequence
        memory._outcome_recorded = snapshot.working.outcome is not None
        return memory


def _validation_issue_value(issue: Issue):
    if isinstance(issue, NumericIssue):
        return NumericIssueValue(issue_id=issue.issue_id, value=issue.minimum)
    if isinstance(issue, CategoricalIssue):
        return CategoricalIssueValue(issue_id=issue.issue_id, value=issue.choices[0])
    raise TypeError(f"unsupported issue type: {type(issue).__name__}")


def save_memory_snapshot(snapshot: MemorySnapshot, path: Union[str, Path]) -> None:
    Path(path).write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def load_memory_snapshot(path: Union[str, Path]) -> MemorySnapshot:
    payload = Path(path).read_text(encoding="utf-8")
    try:
        return MemorySnapshot.model_validate_json(payload)
    except ValidationError as exc:
        raise MemoryPersistenceError(f"invalid or unsupported memory snapshot: {exc}") from exc
