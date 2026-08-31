from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

from .domain import CounterAction, NegotiationScenario, Offer, ParticipantId, ProposeAction
from .memory import AgentMemory, AgentObservation, MemorySnapshot
from .policies import AgentProfile, NegotiationPolicy
from .protocol import NegotiationProtocol, NegotiationSession, TERMINAL_PHASES
from .verification import (
    CorrectionProvider,
    VerificationCoordinator,
    VerificationLogEntry,
    VerifiedProtocolExecutor,
)


@dataclass(frozen=True)
class BenchmarkResult:
    session: NegotiationSession
    offer_trajectory: Tuple[Offer, ...]
    memory_snapshots: Mapping[ParticipantId, MemorySnapshot] = field(default_factory=dict)
    verification_log: Tuple[VerificationLogEntry, ...] = ()


def run_policy_session(
    scenario: NegotiationScenario,
    profiles: Mapping[ParticipantId, AgentProfile],
    policies: Mapping[ParticipantId, NegotiationPolicy],
    maximum_rounds: int,
    verification_coordinator: Optional[VerificationCoordinator] = None,
    correction_providers: Optional[Mapping[ParticipantId, CorrectionProvider]] = None,
) -> BenchmarkResult:
    """Run a deterministic policy comparison through the authoritative protocol."""

    participant_ids = tuple(participant.participant_id for participant in scenario.participants)
    expected = set(participant_ids)
    if set(profiles) != expected or set(policies) != expected:
        raise ValueError("profiles and policies must cover every scenario participant exactly once")
    for participant_id, profile in profiles.items():
        if profile.identity.participant.participant_id != participant_id:
            raise ValueError("profile mapping key must match its participant identity")

    protocol = NegotiationProtocol(
        scenario=scenario,
        maximum_rounds=maximum_rounds,
        initial_turn=participant_ids[0],
    )
    coordinator = verification_coordinator or VerificationCoordinator()
    verified_protocol = VerifiedProtocolExecutor(protocol, coordinator)
    correctors = correction_providers or {}
    state = protocol.create_session()
    memories = {
        participant_id: AgentMemory(
            profiles[participant_id],
            scenario,
            strategy_guidance=(f"Use the {policies[participant_id].name} policy.",),
        )
        for participant_id in participant_ids
    }
    offers = []
    while state.phase not in TERMINAL_PHASES:
        participant_id = state.current_turn
        observation = AgentObservation(
            scenario=scenario,
            session=state,
            own_profile=profiles[participant_id],
        )
        memory = memories[participant_id].update(
            observation,
            legal_actions=protocol.legal_action_types(state, participant_id),
            active_plan=(f"Choose the next {policies[participant_id].name} policy action.",),
        )
        action = policies[participant_id].choose_action(memory)
        for belief in getattr(policies[participant_id], "last_belief_states", ()):
            memories[participant_id].record_belief_snapshot(belief)
        verified = verified_protocol.submit(
            state,
            profiles[participant_id],
            action,
            declared_strategy=(policies[participant_id].name,),
            correction_provider=correctors.get(participant_id),
        )
        selected_action = verified.verification.action
        result = verified.transition
        state = result.state
        if isinstance(selected_action, (ProposeAction, CounterAction)):
            offers.append(selected_action.offer)
    return BenchmarkResult(
        session=state,
        offer_trajectory=tuple(offers),
        memory_snapshots={
            key: value.snapshot()
            for key, value in memories.items()
            if value.working is not None
        },
        verification_log=coordinator.audit_log,
    )
