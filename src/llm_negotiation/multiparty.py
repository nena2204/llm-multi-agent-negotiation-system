"""Explicit multiparty protocol, example scenarios, and comparison experiments."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from .domain import (
    AcceptAction,
    Agreement,
    AgreementId,
    CategoryUtility,
    CategoricalIssue,
    CategoricalPreference,
    CounterAction,
    IllegalActionError,
    IllegalActorError,
    IllegalPhaseError,
    MessageAction,
    NegotiationAction,
    NegotiationScenario,
    NumericIssue,
    NumericPreference,
    OfferId,
    OutcomeStatus,
    Participant,
    ParticipantId,
    ParticipantPreferences,
    ParticipantRole,
    PreferenceDirection,
    ProposeAction,
    RejectAction,
    ReservationPolicy,
    RoundMismatchError,
    TerminalOutcome,
    WithdrawAction,
)
from .domain.models import DomainModel
from .llm import ModelConfiguration
from .memory import AgentProfile, MemorySnapshot, PublicAgentIdentity
from .policies import NegotiationPolicy, PolicyError, generate_offer_for_utility
from .protocol import (
    ActionAppliedEvent,
    AcceptanceSemantics,
    FeasibleRegionStatus,
    NegotiationProtocol,
    NegotiationSession,
    OfferVote,
    ProposalOwnership,
    ProtocolPhase,
    ProtocolStage,
    ReplayError,
    SystemTerminatedEvent,
    TransitionResult,
    _replace_session,
)
from .domain import ActionId, CorrelationId


class AcceptanceRule(str, Enum):
    UNANIMITY = "unanimity"
    MAJORITY = "majority"
    THRESHOLD = "threshold"


class ProposalOwnershipRule(str, Enum):
    PROPOSER_RETAINS = "proposer_retains"


class Coalition(DomainModel):
    members: Tuple[ParticipantId, ...] = Field(min_length=2)

    @field_validator("members", mode="before")
    @classmethod
    def tuple_members(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_members(self) -> "Coalition":
        if len(set(self.members)) != len(self.members):
            raise ValueError("coalition members must be unique")
        return self


class CoalitionRestrictions(DomainModel):
    """Public rules governing private deliberation groups."""

    private_coalitions_allowed: bool = False
    allowed_coalitions: Tuple[Coalition, ...] = ()
    maximum_coalition_size: Optional[int] = Field(default=None, ge=2)

    @field_validator("allowed_coalitions", mode="before")
    @classmethod
    def tuple_coalitions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def consistent_rules(self) -> "CoalitionRestrictions":
        if not self.private_coalitions_allowed and self.allowed_coalitions:
            raise ValueError("allowed coalitions require private_coalitions_allowed")
        keys = tuple(frozenset(item.members) for item in self.allowed_coalitions)
        if len(set(keys)) != len(keys):
            raise ValueError("allowed coalitions must be unique")
        if self.maximum_coalition_size is not None and any(
            len(item.members) > self.maximum_coalition_size for item in self.allowed_coalitions
        ):
            raise ValueError("an allowed coalition exceeds maximum_coalition_size")
        return self

    def permits(self, members: frozenset[ParticipantId]) -> bool:
        if not self.private_coalitions_allowed:
            return False
        if self.maximum_coalition_size is not None and len(members) > self.maximum_coalition_size:
            return False
        return members in {frozenset(item.members) for item in self.allowed_coalitions}


class MultipartyProtocolConfiguration(DomainModel):
    speaking_order: Tuple[ParticipantId, ...] = Field(min_length=3)
    eligible_proposers: Tuple[ParticipantId, ...] = Field(min_length=1)
    eligible_voters: Tuple[ParticipantId, ...] = Field(min_length=2)
    required_participants: Tuple[ParticipantId, ...] = ()
    acceptance_rule: AcceptanceRule = AcceptanceRule.UNANIMITY
    acceptance_threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    proposal_ownership: ProposalOwnershipRule = ProposalOwnershipRule.PROPOSER_RETAINS
    coalition_restrictions: CoalitionRestrictions = Field(default_factory=CoalitionRestrictions)
    deliberation_rounds: int = Field(default=1, ge=1, le=20)
    revision_rounds: int = Field(default=1, ge=1, le=20)

    @field_validator(
        "speaking_order", "eligible_proposers", "eligible_voters", "required_participants",
        mode="before",
    )
    @classmethod
    def tuples_from_json(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def valid_configuration(self) -> "MultipartyProtocolConfiguration":
        for name in ("speaking_order", "eligible_proposers", "eligible_voters", "required_participants"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique participants")
        speakers = set(self.speaking_order)
        if not set(self.eligible_proposers).issubset(speakers):
            raise ValueError("eligible proposers must occur in speaking_order")
        if set(self.eligible_proposers) != speakers:
            raise ValueError("every principal must be an eligible independent proposer")
        if not set(self.eligible_voters).issubset(speakers):
            raise ValueError("eligible voters must occur in speaking_order")
        if not set(self.required_participants).issubset(set(self.eligible_voters)):
            raise ValueError("required participants must be eligible voters")
        if self.eligible_proposers != tuple(
            item for item in self.speaking_order if item in set(self.eligible_proposers)
        ):
            raise ValueError("eligible_proposers must follow speaking_order")
        if self.eligible_voters != tuple(
            item for item in self.speaking_order if item in set(self.eligible_voters)
        ):
            raise ValueError("eligible_voters must follow speaking_order")
        if self.acceptance_rule is AcceptanceRule.THRESHOLD:
            if self.acceptance_threshold is None:
                raise ValueError("threshold acceptance requires acceptance_threshold")
        elif self.acceptance_threshold is not None:
            raise ValueError("acceptance_threshold is valid only for threshold acceptance")
        if self.required_approval_count < 2:
            raise ValueError("an agreement must require at least two approvals")
        coalition_members = {
            member
            for coalition in self.coalition_restrictions.allowed_coalitions
            for member in coalition.members
        }
        if not coalition_members.issubset(speakers):
            raise ValueError("coalitions may contain only configured participants")
        return self

    @property
    def required_approval_count(self) -> int:
        count = len(self.eligible_voters)
        if self.acceptance_rule is AcceptanceRule.UNANIMITY:
            return count
        if self.acceptance_rule is AcceptanceRule.MAJORITY:
            return count // 2 + 1
        return math.ceil((self.acceptance_threshold or 0.0) * count)


class MultipartyNegotiationProtocol(NegotiationProtocol):
    """Staged proposal, deliberation, revision, and voting state machine."""

    def __init__(
        self,
        scenario: NegotiationScenario,
        maximum_rounds: int,
        configuration: MultipartyProtocolConfiguration,
        feasible_region: FeasibleRegionStatus = FeasibleRegionStatus.UNKNOWN,
        mediator_ids: Optional[Tuple[ParticipantId, ...]] = None,
    ) -> None:
        if len(scenario.participants) < 3:
            raise ValueError("multiparty protocol requires at least three principals")
        scenario_ids = tuple(item.participant_id for item in scenario.participants)
        if set(configuration.speaking_order) != set(scenario_ids):
            raise ValueError("speaking_order must contain every scenario participant exactly once")
        self.configuration = configuration
        super().__init__(
            scenario,
            maximum_rounds,
            initial_turn=configuration.eligible_proposers[0],
            feasible_region=feasible_region,
            mediator_ids=mediator_ids,
        )
        self._turn_order = configuration.speaking_order

    def _acceptance_semantics(self) -> AcceptanceSemantics:
        return AcceptanceSemantics(
            rule_name=self.configuration.acceptance_rule.value,
            eligible_voters=self.configuration.eligible_voters,
            required_participants=self.configuration.required_participants,
            approval_count=self.configuration.required_approval_count,
        )

    def _created_session(self) -> NegotiationSession:
        return NegotiationSession(
            scenario_id=self.scenario.scenario_id,
            participants=self.participants,
            current_turn=self.configuration.eligible_proposers[0],
            round_number=1,
            maximum_rounds=self.maximum_rounds,
            protocol_stage=ProtocolStage.INDEPENDENT_PROPOSALS,
            acceptance_semantics=self._acceptance_semantics(),
        )

    def independent_initial_view(
        self, state: NegotiationSession, actor_id: ParticipantId
    ) -> NegotiationSession:
        """Return an empty pre-deliberation view so proposals are genuinely independent."""

        if state.protocol_stage is not ProtocolStage.INDEPENDENT_PROPOSALS:
            return state
        if actor_id != state.current_turn:
            raise IllegalActorError("only the current independent proposer may receive this view")
        return NegotiationSession(
            scenario_id=state.scenario_id,
            participants=state.participants,
            current_turn=actor_id,
            round_number=1,
            maximum_rounds=state.maximum_rounds,
            protocol_stage=ProtocolStage.INDEPENDENT_PROPOSALS,
            acceptance_semantics=state.acceptance_semantics,
        )

    def _validate_state_belongs_to_protocol(self, state: NegotiationSession) -> None:
        super()._validate_state_belongs_to_protocol(state)
        if state.acceptance_semantics != self._acceptance_semantics():
            raise IllegalActionError("session acceptance semantics do not match this protocol")

    def transition(self, state: NegotiationSession, action: NegotiationAction) -> TransitionResult:
        self._validate_state_belongs_to_protocol(state)
        if state.phase in {ProtocolPhase.AGREED, ProtocolPhase.FAILED, ProtocolPhase.WITHDRAWN, ProtocolPhase.EXPIRED}:
            raise IllegalPhaseError("no actions are legal after a multiparty session terminates")
        self.scenario.validate_action(action)
        if action.round_number != state.round_number:
            raise RoundMismatchError("action round does not match the multiparty stage round")
        if isinstance(action, WithdrawAction):
            updates = self._withdraw(state, action)
        else:
            self._require_turn(state, action.actor_id)
            updates = self._apply_stage_action(state, action)

        sequence = len(state.event_sequence) + 1
        phase_after = updates.get("phase", state.phase)
        round_after = updates.get("round_number", state.round_number)
        event = ActionAppliedEvent(
            sequence_number=sequence,
            action_id=ActionId(f"action-{sequence}"),
            correlation_id=CorrelationId(f"correlation-action-{sequence}"),
            action=action,
            phase_before=state.phase,
            phase_after=phase_after,
            round_before=state.round_number,
            round_after=round_after,
        )
        updates["event_sequence"] = state.event_sequence + (event,)
        return TransitionResult(state=_replace_session(state, updates), events=(event,))

    def _apply_stage_action(
        self, state: NegotiationSession, action: NegotiationAction
    ) -> dict[str, Any]:
        stage = state.protocol_stage
        if stage is ProtocolStage.INDEPENDENT_PROPOSALS:
            if not isinstance(action, ProposeAction) or action.actor_id not in self.configuration.eligible_proposers:
                raise IllegalActionError("independent initialization requires an eligible proposer")
            self._require_public_protocol_recipients(action)
            self._ensure_new_offer_id(state, action.offer.offer_id)
            ownership = state.proposal_ownership + (
                ProposalOwnership(offer_id=action.offer.offer_id, owner_id=action.actor_id, initial=True),
            )
            acted = tuple(item.owner_id for item in ownership if item.initial)
            remaining = tuple(item for item in self.configuration.eligible_proposers if item not in acted)
            if remaining:
                return {
                    "phase": ProtocolPhase.ACTIVE,
                    "current_turn": remaining[0],
                    "proposal_ownership": ownership,
                }
            return self._enter_stage(
                state,
                ProtocolStage.DELIBERATION,
                self.configuration.speaking_order[0],
                proposal_ownership=ownership,
                latest_valid_offer=action.offer,
                offer_acceptances=(),
            )

        if stage is ProtocolStage.DELIBERATION:
            if not isinstance(action, MessageAction):
                raise IllegalActionError("deliberation turns require a critique message")
            self._validate_coalition_message(action)
            count = state.deliberation_actions + 1
            limit = self.configuration.deliberation_rounds * len(self.configuration.speaking_order)
            if count < limit:
                return {
                    "current_turn": self._next_in(action.actor_id, self.configuration.speaking_order),
                    "deliberation_actions": count,
                }
            return self._enter_stage(
                state,
                ProtocolStage.REVISION,
                self.configuration.eligible_proposers[0],
                deliberation_actions=count,
            )

        if stage is ProtocolStage.REVISION:
            if not isinstance(action, CounterAction):
                raise IllegalActionError("revision turns require a counteroffer")
            self._require_public_protocol_recipients(action)
            self._require_outstanding_offer(state, action.responds_to)
            self._ensure_new_offer_id(state, action.offer.offer_id)
            ownership = state.proposal_ownership + (
                ProposalOwnership(offer_id=action.offer.offer_id, owner_id=action.actor_id, initial=False),
            )
            count = state.revision_actions + 1
            limit = self.configuration.revision_rounds * len(self.configuration.eligible_proposers)
            if count < limit:
                next_actor = self._next_in(action.actor_id, self.configuration.eligible_proposers)
                return {
                    "current_turn": next_actor,
                    "latest_valid_offer": action.offer,
                    "offer_acceptances": (),
                    "proposal_ownership": ownership,
                    "revision_actions": count,
                }
            return self._enter_stage(
                state,
                ProtocolStage.VOTING,
                self.configuration.eligible_voters[0],
                latest_valid_offer=action.offer,
                offer_acceptances=(),
                proposal_ownership=ownership,
                revision_actions=count,
                votes=(),
            )

        if stage is ProtocolStage.VOTING:
            if not isinstance(action, (AcceptAction, RejectAction)):
                raise IllegalActionError("voting turns require accept or reject")
            self._require_public_protocol_recipients(action)
            outstanding = self._require_outstanding_offer(state, action.offer_id)
            vote = OfferVote(
                offer_id=outstanding.offer_id,
                voter_id=action.actor_id,
                approve=isinstance(action, AcceptAction),
            )
            votes = state.votes + (vote,)
            approvals = tuple(item.voter_id for item in votes if item.approve)
            required = set(self.configuration.required_participants)
            if (
                len(approvals) >= self.configuration.required_approval_count
                and required.issubset(set(approvals))
            ):
                agreement = Agreement(
                    agreement_id=AgreementId(f"agreement-{len(state.event_sequence) + 1}"),
                    offer=outstanding,
                    accepted_by=approvals,
                    round_number=state.round_number,
                )
                return {
                    "current_turn": None,
                    "offer_acceptances": approvals,
                    "votes": votes,
                    "phase": ProtocolPhase.AGREED,
                    "outcome": TerminalOutcome(
                        status=OutcomeStatus.AGREEMENT,
                        final_round=state.round_number,
                        agreement=agreement,
                    ),
                }
            remaining = tuple(
                item for item in self.configuration.eligible_voters
                if item not in {vote.voter_id for vote in votes}
            )
            required_rejected = any(
                not item.approve and item.voter_id in required for item in votes
            )
            possible_approvals = len(approvals) + len(remaining)
            if required_rejected or possible_approvals < self.configuration.required_approval_count or not remaining:
                reason = (
                    "A required participant rejected the proposal."
                    if required_rejected
                    else "The configured acceptance rule was not satisfied."
                )
                return {
                    "current_turn": None,
                    "offer_acceptances": approvals,
                    "votes": votes,
                    "phase": ProtocolPhase.FAILED,
                    "outcome": TerminalOutcome(
                        status=OutcomeStatus.NO_AGREEMENT,
                        final_round=state.round_number,
                        reason=reason,
                    ),
                }
            return {"current_turn": remaining[0], "offer_acceptances": approvals, "votes": votes}
        raise IllegalActionError(f"unsupported multiparty stage '{stage.value}'")

    def _enter_stage(
        self,
        state: NegotiationSession,
        stage: ProtocolStage,
        current_turn: ParticipantId,
        **updates: Any,
    ) -> dict[str, Any]:
        next_round = state.round_number + 1
        if next_round > state.maximum_rounds:
            return {
                "current_turn": None,
                "phase": ProtocolPhase.EXPIRED,
                "outcome": TerminalOutcome(
                    status=OutcomeStatus.NO_AGREEMENT,
                    final_round=state.round_number,
                    reason="Maximum negotiation rounds reached.",
                ),
                **updates,
            }
        return {
            "phase": ProtocolPhase.ACTIVE,
            "protocol_stage": stage,
            "current_turn": current_turn,
            "round_number": next_round,
            **updates,
        }

    def _validate_coalition_message(self, action: MessageAction) -> None:
        public_recipients = set(self.participants) - {action.actor_id}
        if set(action.recipients) == public_recipients:
            return
        coalition = frozenset((action.actor_id,) + action.recipients)
        if not self.configuration.coalition_restrictions.permits(coalition):
            raise IllegalActionError("private coalition communication is not permitted")

    def _require_public_protocol_recipients(self, action: NegotiationAction) -> None:
        expected = set(self.participants) - {action.actor_id}
        if set(action.recipients) != expected:
            raise IllegalActionError(
                "proposals, revisions, and votes must address every other principal"
            )

    @staticmethod
    def _next_in(actor_id: ParticipantId, order: Tuple[ParticipantId, ...]) -> ParticipantId:
        return order[(order.index(actor_id) + 1) % len(order)]

    def replay(self, events: Tuple[Any, ...]) -> NegotiationSession:
        state = self._created_session()
        for recorded in events:
            if isinstance(recorded, SystemTerminatedEvent):
                if self.feasible_region is not FeasibleRegionStatus.IMPOSSIBLE:
                    raise ReplayError("impossible-region event does not match protocol configuration")
                result = self._terminate_impossible(state)
            elif isinstance(recorded, ActionAppliedEvent):
                result = self.transition(state, recorded.action)
            else:
                raise ReplayError("unsupported event in multiparty replay")
            if result.events[0] != recorded:
                raise ReplayError(
                    f"event {recorded.sequence_number} does not match deterministic transition output"
                )
            state = result.state
        return state


class MultipartyProtocolFactory:
    supports_multiparty = True

    def __init__(self, configuration: MultipartyProtocolConfiguration) -> None:
        self.configuration = configuration

    def create(
        self,
        *,
        scenario: NegotiationScenario,
        maximum_rounds: int,
        initial_turn: ParticipantId,
        feasible_region: FeasibleRegionStatus,
        mediator_ids: Tuple[ParticipantId, ...],
    ) -> MultipartyNegotiationProtocol:
        del initial_turn
        return MultipartyNegotiationProtocol(
            scenario,
            maximum_rounds,
            self.configuration,
            feasible_region=feasible_region,
            mediator_ids=mediator_ids,
        )


@dataclass(frozen=True)
class DeliberativePolicy:
    """Stage-aware wrapper adding public critique and voting to any offer policy."""

    offer_policy: NegotiationPolicy
    critique: str = "I am evaluating the public proposals against the stated issues."
    name: str = "deliberative"

    def choose_action(self, memory: MemorySnapshot) -> NegotiationAction:
        stage = memory.working.protocol_stage
        recipients = tuple(
            item.participant_id for item in memory.long_term.public_participants
            if item.participant_id != memory.participant_id
        )
        if stage is ProtocolStage.DELIBERATION:
            return MessageAction(
                actor_id=memory.participant_id,
                recipients=recipients,
                round_number=memory.working.round_number,
                content=self.critique,
            )
        if stage is ProtocolStage.VOTING:
            outstanding = memory.working.outstanding_offer
            if outstanding is None:
                raise PolicyError("voting requires an outstanding offer")
            from .domain import calculate_utility

            utility = calculate_utility(
                memory.scenario, outstanding, memory.long_term.own_preferences
            )
            action_type = AcceptAction if utility + 1e-9 >= (
                memory.long_term.own_preferences.reservation.reservation_utility
            ) else RejectAction
            return action_type(
                actor_id=memory.participant_id,
                recipients=recipients,
                round_number=memory.working.round_number,
                offer_id=outstanding.offer_id,
            )
        if stage is ProtocolStage.REVISION:
            outstanding = memory.working.outstanding_offer
            if outstanding is None:
                raise PolicyError("revision requires an outstanding offer")
            target_method = getattr(self.offer_policy, "target_utility", None)
            if target_method is None:
                raise PolicyError("wrapped offer policy cannot produce a utility target")
            digest = hashlib.sha256(
                (
                    f"{memory.working.negotiation_id}|{memory.participant_id}|revision|"
                    f"{memory.working.public_event_count}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            revised = generate_offer_for_utility(
                memory.scenario,
                memory.long_term.own_preferences,
                target_method(memory),
                OfferId(f"offer-{digest}"),
            )
            return CounterAction(
                actor_id=memory.participant_id,
                recipients=recipients,
                round_number=memory.working.round_number,
                offer=revised,
                responds_to=outstanding.offer_id,
            )
        return self.offer_policy.choose_action(memory)


class TeamMemberConfiguration(DomainModel):
    """Research assignment; model settings remain separate from the private AgentProfile."""

    profile: AgentProfile
    strategy: str = Field(min_length=1, max_length=100)
    model_configuration: Optional[ModelConfiguration] = None


def three_principal_scenario() -> tuple[
    NegotiationScenario, Mapping[ParticipantId, AgentProfile]
]:
    """Public two-issue resource allocation with heterogeneous private utilities."""

    participants = (
        Participant(participant_id=ParticipantId("operations"), display_name="Operations", role=ParticipantRole.NEGOTIATOR),
        Participant(participant_id=ParticipantId("research"), display_name="Research", role=ParticipantRole.NEGOTIATOR),
        Participant(participant_id=ParticipantId("community"), display_name="Community", role=ParticipantRole.NEGOTIATOR),
    )
    budget = NumericIssue(issue_id="budget_share", name="Research budget share", minimum=0.0, maximum=100.0, unit="percent")
    schedule = CategoricalIssue(issue_id="launch", name="Launch schedule", choices=("early", "balanced", "late"))
    scenario = NegotiationScenario(
        scenario_id="three-principal-resource-allocation",
        title="Three-principal resource allocation",
        participants=participants,
        issues=(budget, schedule),
    )
    specifications = (
        (participants[0], "Pragmatic delivery steward", 0.65, PreferenceDirection.MINIMIZE, (0.9, 1.0, 0.2), 0.35),
        (participants[1], "Evidence-focused research advocate", 0.75, PreferenceDirection.MAXIMIZE, (0.3, 0.8, 1.0), 0.25),
        (participants[2], "Consensus-seeking community representative", 0.45, PreferenceDirection.MINIMIZE, (0.5, 1.0, 0.6), 0.45),
    )
    profiles = {}
    for participant, persona, weight, direction, categories, reservation in specifications:
        preferences = ParticipantPreferences(
            participant_id=participant.participant_id,
            issue_preferences=(
                NumericPreference(issue_id=budget.issue_id, weight=weight, direction=direction),
                CategoricalPreference(
                    issue_id=schedule.issue_id,
                    weight=1.0 - weight,
                    category_utilities=tuple(
                        CategoryUtility(category=choice, utility=value)
                        for choice, value in zip(schedule.choices, categories)
                    ),
                ),
            ),
            reservation=ReservationPolicy(
                reservation_utility=reservation,
                batna_utility=reservation,
                batna_description="Retain the status quo allocation.",
            ),
        )
        profiles[participant.participant_id] = AgentProfile(
            identity=PublicAgentIdentity(participant=participant, persona=persona),
            preferences=preferences,
        )
    return scenario, profiles


def heterogeneous_three_principal_team(
    model_configurations: Optional[Mapping[ParticipantId, ModelConfiguration]] = None,
) -> Mapping[ParticipantId, TeamMemberConfiguration]:
    """Return heterogeneous research assignments without putting model settings in profiles."""

    _scenario, profiles = three_principal_scenario()
    strategies = {
        ParticipantId("operations"): "boulware",
        ParticipantId("research"): "linear",
        ParticipantId("community"): "conceder",
    }
    configured_models = dict(model_configurations or {})
    unknown = set(configured_models) - set(profiles)
    if unknown:
        raise ValueError("model configurations contain an unknown team participant")
    return {
        participant_id: TeamMemberConfiguration(
            profile=profile,
            strategy=strategies[participant_id],
            model_configuration=configured_models.get(participant_id),
        )
        for participant_id, profile in profiles.items()
    }
