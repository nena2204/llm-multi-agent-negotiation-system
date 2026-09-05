from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional, Tuple

from .communication import MessageBus
from .domain import CounterAction, NegotiationScenario, Offer, ParticipantId, ProposeAction
from .memory import AgentMemory, AgentProfile, MemorySnapshot
from .orchestration import (
    AUDIT_READER_ID,
    InMemoryMemoryStore,
    NegotiationOrchestrator,
    OrchestratorConfiguration,
)
from .policies import NegotiationPolicy
from .protocol import NegotiationSession
from .verification import CorrectionProvider, VerificationCoordinator, VerificationLogEntry


class _BenchmarkClock:
    def now(self) -> datetime:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    def monotonic(self) -> float:
        return 0.0


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
    """Run the historical benchmark API through the shared orchestration lifecycle."""

    participant_ids = tuple(item.participant_id for item in scenario.participants)
    memories = {
        participant_id: AgentMemory(
            profiles[participant_id],
            scenario,
            strategy_guidance=(f"Use the {policies[participant_id].name} policy.",),
        )
        for participant_id in participant_ids
    }
    store = InMemoryMemoryStore(memories)
    coordinator = verification_coordinator or VerificationCoordinator()
    service = NegotiationOrchestrator(
        scenario=scenario,
        profiles=profiles,
        policies=policies,
        message_bus=MessageBus(
            scenario.scenario_id,
            participants=participant_ids,
            mediator_ids=(),
            audit_reader_ids=(AUDIT_READER_ID,),
        ),
        verifier=coordinator,
        mediator=None,
        model_gateway=None,
        memory_store=store,
        clock=_BenchmarkClock(),
        random_seed=0,
        configuration=OrchestratorConfiguration(maximum_rounds=maximum_rounds),
        correction_providers=correction_providers,
    )
    episode = service.run()
    offers = tuple(
        event.action.offer
        for event in episode.audit_events
        if getattr(event, "action", None) is not None
        and isinstance(event.action, (ProposeAction, CounterAction))
    )
    return BenchmarkResult(
        session=episode.final_session,
        offer_trajectory=offers,
        memory_snapshots={key: memories[key].snapshot() for key in participant_ids},
        verification_log=coordinator.audit_log,
    )
