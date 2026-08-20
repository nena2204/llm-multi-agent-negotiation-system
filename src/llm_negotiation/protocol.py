from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from .domain.errors import (
    IllegalActionError,
    IllegalActorError,
    IllegalPhaseError,
    ReplayError,
    RoundMismatchError,
    StaleOfferError,
)
from .domain.identifiers import AgreementId, OfferId, ParticipantId
from .domain.models import (
    AcceptAction,
    Agreement,
    CounterAction,
    DomainModel,
    MessageAction,
    NegotiationAction,
    NegotiationScenario,
    Offer,
    OutcomeStatus,
    ProposeAction,
    RejectAction,
    RequestMediationAction,
    TerminalOutcome,
    WithdrawAction,
)


class ProtocolPhase(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    MEDIATION = "mediation"
    AGREED = "agreed"
    FAILED = "failed"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class FeasibleRegionStatus(str, Enum):
    UNKNOWN = "unknown"
    FEASIBLE = "feasible"
    IMPOSSIBLE = "impossible"


class ProtocolEventType(str, Enum):
    ACTION_APPLIED = "action_applied"
    SYSTEM_TERMINATED = "system_terminated"


class SystemTerminationReason(str, Enum):
    IMPOSSIBLE_FEASIBLE_REGION = "impossible_feasible_region"


TERMINAL_PHASES = frozenset(
    {
        ProtocolPhase.AGREED,
        ProtocolPhase.FAILED,
        ProtocolPhase.WITHDRAWN,
        ProtocolPhase.EXPIRED,
    }
)


def _list_to_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


class ActionAppliedEvent(DomainModel):
    event_type: Literal[ProtocolEventType.ACTION_APPLIED] = ProtocolEventType.ACTION_APPLIED
    sequence_number: int = Field(ge=1)
    action: NegotiationAction
    phase_before: ProtocolPhase
    phase_after: ProtocolPhase
    round_before: int = Field(ge=1)
    round_after: int = Field(ge=1)


class SystemTerminatedEvent(DomainModel):
    event_type: Literal[ProtocolEventType.SYSTEM_TERMINATED] = ProtocolEventType.SYSTEM_TERMINATED
    sequence_number: int = Field(ge=1)
    reason: SystemTerminationReason
    phase_before: ProtocolPhase
    phase_after: ProtocolPhase
    round_number: int = Field(ge=1)


ProtocolEvent = Annotated[
    Union[ActionAppliedEvent, SystemTerminatedEvent],
    Field(discriminator="event_type"),
]


class NegotiationSession(DomainModel):
    scenario_id: str = Field(min_length=1, max_length=100)
    participants: Tuple[ParticipantId, ...] = Field(min_length=2)
    current_turn: Optional[ParticipantId]
    round_number: int = Field(ge=1)
    maximum_rounds: int = Field(ge=1)
    latest_valid_offer: Optional[Offer] = None
    offer_acceptances: Tuple[ParticipantId, ...] = ()
    event_sequence: Tuple[ProtocolEvent, ...] = ()
    phase: ProtocolPhase = ProtocolPhase.CREATED
    outcome: Optional[TerminalOutcome] = None

    @field_validator("participants", "offer_acceptances", "event_sequence", mode="before")
    @classmethod
    def tuples_from_json_arrays(cls, value: Any) -> Any:
        return _list_to_tuple(value)

    @model_validator(mode="after")
    def consistent_state(self) -> NegotiationSession:
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("session participants must be unique")
        if any(participant not in self.participants for participant in self.offer_acceptances):
            raise ValueError("offer acceptances must belong to session participants")
        if self.latest_valid_offer is None and self.offer_acceptances:
            raise ValueError("offer acceptances require an outstanding offer")
        if self.phase in TERMINAL_PHASES:
            if self.outcome is None:
                raise ValueError("terminal protocol phases require an outcome")
            if self.current_turn is not None:
                raise ValueError("terminal protocol phases cannot have a current turn")
            expected_status = {
                ProtocolPhase.AGREED: OutcomeStatus.AGREEMENT,
                ProtocolPhase.FAILED: OutcomeStatus.NO_AGREEMENT,
                ProtocolPhase.WITHDRAWN: OutcomeStatus.WITHDRAWN,
                ProtocolPhase.EXPIRED: OutcomeStatus.NO_AGREEMENT,
            }[self.phase]
            if self.outcome.status is not expected_status:
                raise ValueError("terminal outcome status does not match protocol phase")
            if self.outcome.final_round != self.round_number:
                raise ValueError("terminal outcome round must match the session round")
            if self.phase is ProtocolPhase.AGREED and self.outcome.agreement.offer != self.latest_valid_offer:
                raise ValueError("agreed outcome must reference the latest valid offer")
        else:
            if self.outcome is not None:
                raise ValueError("non-terminal protocol phases cannot have an outcome")
            if self.current_turn is None or self.current_turn not in self.participants:
                raise ValueError("non-terminal protocol phases require a valid current turn")
        if self.phase is ProtocolPhase.CREATED:
            if self.latest_valid_offer is not None or self.offer_acceptances or self.event_sequence:
                raise ValueError("created sessions cannot contain offers, acceptances, or events")
        if self.event_sequence:
            expected = tuple(range(1, len(self.event_sequence) + 1))
            actual = tuple(event.sequence_number for event in self.event_sequence)
            if actual != expected:
                raise ValueError("event sequence numbers must be contiguous starting at one")
            final_event = self.event_sequence[-1]
            if final_event.phase_after is not self.phase:
                raise ValueError("final event phase must match the session phase")
            final_round = (
                final_event.round_after
                if isinstance(final_event, ActionAppliedEvent)
                else final_event.round_number
            )
            if final_round != self.round_number:
                raise ValueError("final event round must match the session round")
        return self


def _replace_session(state: NegotiationSession, updates: dict[str, Any]) -> NegotiationSession:
    values = {field_name: getattr(state, field_name) for field_name in NegotiationSession.model_fields}
    values.update(updates)
    return NegotiationSession.model_validate(values)


@dataclass(frozen=True)
class TransitionResult:
    state: NegotiationSession
    events: Tuple[ProtocolEvent, ...]


class NegotiationProtocol:
    """Deterministic protocol engine independent of agent implementations."""

    def __init__(
        self,
        scenario: NegotiationScenario,
        maximum_rounds: int,
        initial_turn: Optional[ParticipantId] = None,
        feasible_region: FeasibleRegionStatus = FeasibleRegionStatus.UNKNOWN,
    ) -> None:
        if not isinstance(maximum_rounds, int) or isinstance(maximum_rounds, bool) or maximum_rounds <= 0:
            raise ValueError("maximum_rounds must be a positive integer")
        self.scenario = scenario
        self.maximum_rounds = maximum_rounds
        self.participants = tuple(participant.participant_id for participant in scenario.participants)
        self.initial_turn = initial_turn or self.participants[0]
        if self.initial_turn not in self.participants:
            raise IllegalActorError(f"initial turn participant '{self.initial_turn}' is not in the scenario")
        initial_index = self.participants.index(self.initial_turn)
        self._turn_order = self.participants[initial_index:] + self.participants[:initial_index]
        self.feasible_region = feasible_region

    def _created_session(self) -> NegotiationSession:
        return NegotiationSession(
            scenario_id=self.scenario.scenario_id,
            participants=self.participants,
            current_turn=self.initial_turn,
            round_number=1,
            maximum_rounds=self.maximum_rounds,
        )

    def create_session(self) -> NegotiationSession:
        state = self._created_session()
        if self.feasible_region is FeasibleRegionStatus.IMPOSSIBLE:
            return self._terminate_impossible(state).state
        return state

    def _terminate_impossible(self, state: NegotiationSession) -> TransitionResult:
        if state.phase is not ProtocolPhase.CREATED or state.event_sequence:
            raise IllegalPhaseError("impossible-region termination is only legal for a newly created session")
        outcome = TerminalOutcome(
            status=OutcomeStatus.NO_AGREEMENT,
            final_round=state.round_number,
            reason="The feasible region is empty.",
        )
        event = SystemTerminatedEvent(
            sequence_number=1,
            reason=SystemTerminationReason.IMPOSSIBLE_FEASIBLE_REGION,
            phase_before=ProtocolPhase.CREATED,
            phase_after=ProtocolPhase.FAILED,
            round_number=state.round_number,
        )
        new_state = _replace_session(
            state,
            {
                "current_turn": None,
                "event_sequence": (event,),
                "phase": ProtocolPhase.FAILED,
                "outcome": outcome,
            },
        )
        return TransitionResult(state=new_state, events=(event,))

    def transition(self, state: NegotiationSession, action: NegotiationAction) -> TransitionResult:
        self._validate_state_belongs_to_protocol(state)
        if state.phase in TERMINAL_PHASES:
            raise IllegalPhaseError(f"no actions are legal after session phase '{state.phase.value}'")
        self.scenario.validate_action(action)
        if action.round_number != state.round_number:
            raise RoundMismatchError(
                f"action round {action.round_number} does not match session round {state.round_number}"
            )

        phase_before = state.phase
        updates = self._apply_action(state, action)
        phase_after = updates.get("phase", state.phase)
        round_after = updates.get("round_number", state.round_number)
        event = ActionAppliedEvent(
            sequence_number=len(state.event_sequence) + 1,
            action=action,
            phase_before=phase_before,
            phase_after=phase_after,
            round_before=state.round_number,
            round_after=round_after,
        )
        updates["event_sequence"] = state.event_sequence + (event,)
        new_state = _replace_session(state, updates)
        return TransitionResult(state=new_state, events=(event,))

    def _validate_state_belongs_to_protocol(self, state: NegotiationSession) -> None:
        if state.scenario_id != self.scenario.scenario_id:
            raise IllegalActionError("session scenario does not match this protocol")
        if state.participants != self.participants:
            raise IllegalActionError("session participants do not match this protocol")
        if state.maximum_rounds != self.maximum_rounds:
            raise IllegalActionError("session round limit does not match this protocol")

    def _apply_action(self, state: NegotiationSession, action: NegotiationAction) -> dict[str, Any]:
        if isinstance(action, WithdrawAction):
            return self._withdraw(state, action)
        if state.phase is ProtocolPhase.CREATED:
            if not isinstance(action, ProposeAction):
                raise IllegalActionError("the first protocol action must be a proposal")
            self._require_turn(state, action.actor_id)
            self._ensure_new_offer_id(state, action.offer.offer_id)
            return self._record_offer_and_advance(state, action.actor_id, action.offer, ProtocolPhase.ACTIVE)
        if isinstance(action, MessageAction):
            if state.phase not in {ProtocolPhase.ACTIVE, ProtocolPhase.MEDIATION}:
                raise IllegalActionError("messages are legal only in active or mediation phases")
            return {}
        if isinstance(action, RequestMediationAction):
            if state.phase is not ProtocolPhase.ACTIVE:
                raise IllegalActionError("mediation may be requested only during the active phase")
            return {"phase": ProtocolPhase.MEDIATION}

        if state.phase not in {ProtocolPhase.ACTIVE, ProtocolPhase.MEDIATION}:
            raise IllegalPhaseError(f"offer actions are not legal in phase '{state.phase.value}'")
        self._require_turn(state, action.actor_id)

        if isinstance(action, ProposeAction):
            if state.latest_valid_offer is not None:
                raise IllegalActionError("a new proposal is legal only when no offer is outstanding")
            self._ensure_new_offer_id(state, action.offer.offer_id)
            return self._record_offer_and_advance(state, action.actor_id, action.offer, state.phase)
        if isinstance(action, CounterAction):
            outstanding = self._require_outstanding_offer(state, action.responds_to)
            self._ensure_new_offer_id(state, action.offer.offer_id)
            if action.offer.offer_id == outstanding.offer_id:
                raise IllegalActionError("a counteroffer must have a new offer identifier")
            return self._record_offer_and_advance(state, action.actor_id, action.offer, state.phase)
        if isinstance(action, AcceptAction):
            outstanding = self._require_outstanding_offer(state, action.offer_id)
            accepted = state.offer_acceptances + (action.actor_id,)
            if set(accepted) == set(state.participants):
                agreement = Agreement(
                    agreement_id=AgreementId(f"agreement-{len(state.event_sequence) + 1}"),
                    offer=outstanding,
                    accepted_by=accepted,
                    round_number=state.round_number,
                )
                self.scenario.validate_agreement(agreement)
                outcome = TerminalOutcome(
                    status=OutcomeStatus.AGREEMENT,
                    final_round=state.round_number,
                    agreement=agreement,
                )
                return {
                    "current_turn": None,
                    "offer_acceptances": accepted,
                    "phase": ProtocolPhase.AGREED,
                    "outcome": outcome,
                }
            return self._advance(state, accepted=accepted)
        if isinstance(action, RejectAction):
            self._require_outstanding_offer(state, action.offer_id)
            updates = self._advance(state, accepted=())
            updates["latest_valid_offer"] = None
            return updates
        raise IllegalActionError(f"action '{action.action.value}' is not legal in phase '{state.phase.value}'")

    def _withdraw(self, state: NegotiationSession, action: WithdrawAction) -> dict[str, Any]:
        if state.phase not in {ProtocolPhase.CREATED, ProtocolPhase.ACTIVE, ProtocolPhase.MEDIATION}:
            raise IllegalPhaseError(f"withdrawal is not legal in phase '{state.phase.value}'")
        outcome = TerminalOutcome(
            status=OutcomeStatus.WITHDRAWN,
            final_round=state.round_number,
            reason=action.reason or f"Participant '{action.actor_id}' withdrew.",
        )
        return {
            "current_turn": None,
            "phase": ProtocolPhase.WITHDRAWN,
            "outcome": outcome,
        }

    @staticmethod
    def _require_turn(state: NegotiationSession, actor_id: ParticipantId) -> None:
        if actor_id != state.current_turn:
            raise IllegalActorError(
                f"participant '{actor_id}' cannot act; current turn belongs to '{state.current_turn}'"
            )

    @staticmethod
    def _require_outstanding_offer(state: NegotiationSession, offer_id: OfferId) -> Offer:
        if state.latest_valid_offer is None:
            raise StaleOfferError("there is no outstanding offer")
        if state.latest_valid_offer.offer_id != offer_id:
            raise StaleOfferError(
                f"offer '{offer_id}' is stale; outstanding offer is '{state.latest_valid_offer.offer_id}'"
            )
        return state.latest_valid_offer

    @staticmethod
    def _ensure_new_offer_id(state: NegotiationSession, offer_id: OfferId) -> None:
        for event in state.event_sequence:
            if isinstance(event, ActionAppliedEvent) and isinstance(event.action, (ProposeAction, CounterAction)):
                if event.action.offer.offer_id == offer_id:
                    raise IllegalActionError(f"offer identifier '{offer_id}' has already been used")

    def _record_offer_and_advance(
        self,
        state: NegotiationSession,
        actor_id: ParticipantId,
        offer: Offer,
        phase: ProtocolPhase,
    ) -> dict[str, Any]:
        updates = self._advance(state, accepted=(actor_id,))
        updates["latest_valid_offer"] = offer
        if updates.get("phase") is None:
            updates["phase"] = phase
        elif "phase" not in updates:
            updates["phase"] = phase
        return updates

    def _advance(
        self,
        state: NegotiationSession,
        accepted: Tuple[ParticipantId, ...],
    ) -> dict[str, Any]:
        current_index = self._turn_order.index(state.current_turn)
        for offset in range(1, len(self._turn_order) + 1):
            next_index = (current_index + offset) % len(self._turn_order)
            candidate = self._turn_order[next_index]
            if candidate not in accepted:
                wrapped = next_index <= current_index
                next_round = state.round_number + (1 if wrapped else 0)
                if next_round > state.maximum_rounds:
                    outcome = TerminalOutcome(
                        status=OutcomeStatus.NO_AGREEMENT,
                        final_round=state.round_number,
                        reason="Maximum negotiation rounds reached.",
                    )
                    return {
                        "current_turn": None,
                        "round_number": state.round_number,
                        "offer_acceptances": accepted,
                        "phase": ProtocolPhase.EXPIRED,
                        "outcome": outcome,
                    }
                return {
                    "current_turn": candidate,
                    "round_number": next_round,
                    "offer_acceptances": accepted,
                }
        raise IllegalActionError("no participant remains available for the next turn")

    def replay(self, events: Tuple[ProtocolEvent, ...]) -> NegotiationSession:
        state = self._created_session()
        for recorded in events:
            if isinstance(recorded, SystemTerminatedEvent):
                if recorded.reason is not SystemTerminationReason.IMPOSSIBLE_FEASIBLE_REGION:
                    raise ReplayError(f"unsupported system event '{recorded.reason.value}'")
                if self.feasible_region is not FeasibleRegionStatus.IMPOSSIBLE:
                    raise ReplayError("impossible-region event does not match protocol configuration")
                result = self._terminate_impossible(state)
            else:
                result = self.transition(state, recorded.action)
            generated = result.events[0]
            if generated != recorded:
                raise ReplayError(
                    f"event {recorded.sequence_number} does not match deterministic transition output"
                )
            state = result.state
        return state
