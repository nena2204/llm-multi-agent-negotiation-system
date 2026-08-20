from dataclasses import dataclass
from typing import Mapping, Tuple

from .domain import CounterAction, NegotiationScenario, Offer, ParticipantId, ProposeAction
from .policies import AgentObservation, AgentProfile, NegotiationPolicy
from .protocol import NegotiationProtocol, NegotiationSession, TERMINAL_PHASES


@dataclass(frozen=True)
class BenchmarkResult:
    session: NegotiationSession
    offer_trajectory: Tuple[Offer, ...]


def run_policy_session(
    scenario: NegotiationScenario,
    profiles: Mapping[ParticipantId, AgentProfile],
    policies: Mapping[ParticipantId, NegotiationPolicy],
    maximum_rounds: int,
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
    state = protocol.create_session()
    offers = []
    while state.phase not in TERMINAL_PHASES:
        participant_id = state.current_turn
        observation = AgentObservation(
            scenario=scenario,
            session=state,
            own_profile=profiles[participant_id],
        )
        action = policies[participant_id].choose_action(observation)
        result = protocol.transition(state, action)
        state = result.state
        if isinstance(action, (ProposeAction, CounterAction)):
            offers.append(action.offer)
    return BenchmarkResult(session=state, offer_trajectory=tuple(offers))
