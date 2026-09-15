from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .agents import BuyerAgent, SellerAgent, _strategy_step
from .domain import (
    AcceptAction,
    CounterAction,
    MediationAccessMode,
    NumericIssueValue,
    Offer,
    OfferId,
    ParticipantId,
    ProposeAction,
    price_only_from_legacy,
)
from .domain.legacy import BUYER_ID, PRICE_ISSUE_ID, SELLER_ID
from .judge import Judge
from .learning import LegacyRewardScorer
from .mediation import DeadlockConfiguration, MediatorConfiguration
from .memory import AgentProfile, MemorySnapshot, PublicAgentIdentity
from .orchestration import NegotiationOrchestrator, OrchestratorConfiguration
from .protocol import ActionAppliedEvent, FeasibleRegionStatus, MediatorInterventionEvent


@dataclass(frozen=True)
class _LegacyPricePolicy:
    """Typed compatibility policy preserving the historical price-step formula."""

    actor_id: ParticipantId
    initial_price: float
    reservation_price: float
    strategy: str
    is_buyer: bool

    @property
    def name(self) -> str:
        return f"legacy-{self.strategy}"

    def choose_action(self, memory: MemorySnapshot):
        own_offers = [
            item.offer.value_for(PRICE_ISSUE_ID).value
            for item in memory.working.recent_offers
            if item.actor_id == self.actor_id
        ]
        current = own_offers[-1] if own_offers else self.initial_price
        recipients = tuple(
            item.participant_id
            for item in memory.long_term.public_participants
            if item.participant_id != self.actor_id
        )
        outstanding = memory.working.outstanding_offer
        if outstanding is None:
            return ProposeAction(
                actor_id=self.actor_id,
                recipients=recipients,
                round_number=memory.working.round_number,
                offer=self._offer(memory, current),
            )
        incoming = outstanding.value_for(PRICE_ISSUE_ID).value
        acceptable = current >= incoming if self.is_buyer else incoming >= current
        if acceptable:
            return AcceptAction(
                actor_id=self.actor_id,
                recipients=recipients,
                round_number=memory.working.round_number,
                offer_id=outstanding.offer_id,
            )
        target = (
            min(self.reservation_price, incoming)
            if self.is_buyer
            else max(self.reservation_price, incoming)
        )
        price = round(_strategy_step(current, target, self.strategy), 2)
        return CounterAction(
            actor_id=self.actor_id,
            recipients=recipients,
            round_number=memory.working.round_number,
            responds_to=outstanding.offer_id,
            offer=self._offer(memory, price),
        )

    def _offer(self, memory: MemorySnapshot, price: float) -> Offer:
        return Offer(
            offer_id=OfferId(
                f"legacy-{self.actor_id}-{memory.working.public_event_count + 1}"
            ),
            values=(NumericIssueValue(issue_id=PRICE_ISSUE_ID, value=price),),
        )


class NegotiationManager:
    """Legacy CLI adapter; execution is delegated entirely to NegotiationOrchestrator."""

    def __init__(self, product_name: str, buyer: BuyerAgent, seller: SellerAgent, rounds: int = 6):
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
            raise ValueError("rounds must be a positive integer")
        self.product_name = product_name
        self.buyer = buyer
        self.seller = seller
        self.rounds = rounds
        self.history: List[str] = []
        self.deal_reached = False
        self.final_price: Optional[float] = None
        self.mediator_suggestion: Optional[float] = None
        self.protocol_state = None
        self.episode_result = None

    def _build_orchestrator(self) -> NegotiationOrchestrator:
        migration = price_only_from_legacy(
            self.product_name, self.buyer, self.seller, self.rounds
        )
        profiles = {
            BUYER_ID: AgentProfile(
                identity=PublicAgentIdentity(
                    participant=migration.scenario.participant(BUYER_ID),
                    persona="Legacy buyer",
                ),
                preferences=migration.buyer_preferences,
            ),
            SELLER_ID: AgentProfile(
                identity=PublicAgentIdentity(
                    participant=migration.scenario.participant(SELLER_ID),
                    persona="Legacy seller",
                ),
                preferences=migration.seller_preferences,
            ),
        }
        policies = {
            BUYER_ID: _LegacyPricePolicy(
                BUYER_ID, self.buyer.initial_offer, self.buyer.max_price,
                self.buyer.strategy, True,
            ),
            SELLER_ID: _LegacyPricePolicy(
                SELLER_ID, self.seller.initial_price, self.seller.min_acceptable,
                self.seller.strategy, False,
            ),
        }
        feasible = (
            FeasibleRegionStatus.FEASIBLE
            if self.buyer.max_price >= self.seller.min_acceptable
            else FeasibleRegionStatus.IMPOSSIBLE
        )
        mediator_configuration = MediatorConfiguration(
            access_mode=MediationAccessMode.CONFIDENTIAL_SUMMARY,
            deadlock=DeadlockConfiguration(enabled=False, deadline_rounds_remaining=0),
        )
        return NegotiationOrchestrator.with_defaults(
            scenario=migration.scenario,
            profiles=profiles,
            policies=policies,
            random_seed=0,
            configuration=OrchestratorConfiguration(
                maximum_rounds=self.rounds + 1,
                mediation_enabled=True,
                pause_after_mediator_intervention=True,
            ),
            feasible_region=feasible,
            mediator_configuration=mediator_configuration,
        )

    def _reset_legacy_state(self) -> None:
        self.buyer.current_offer = self.buyer.initial_offer
        self.seller.current_offer = self.seller.initial_price
        self.buyer.history.clear()
        self.seller.history.clear()
        self.history = []
        self.deal_reached = False
        self.final_price = None
        self.mediator_suggestion = None
        self.protocol_state = None
        self.episode_result = None

    def _render_history(self) -> None:
        state = self.protocol_state
        if state.phase.value == "failed":
            self.history = [f"System (protocol): {state.outcome.reason}"]
            return
        self.history = [
            f"{self.seller.name} ({self.seller.role}): I ask for ${self.seller.initial_price:.2f}.",
            f"{self.buyer.name} ({self.buyer.role}): I offer ${self.buyer.initial_offer:.2f}.",
        ]
        offer_count = 0
        for event in state.event_sequence:
            if isinstance(event, ActionAppliedEvent):
                action = event.action
                if isinstance(action, ProposeAction):
                    offer_count += 1
                    continue
                if isinstance(action, CounterAction):
                    offer_count += 1
                    price = action.offer.value_for(PRICE_ISSUE_ID).value
                    if action.actor_id == SELLER_ID:
                        self.seller.current_offer = price
                        self.history.append(
                            f"{self.seller.name} ({self.seller.role}): I can lower to ${price:.2f}."
                        )
                    else:
                        self.buyer.current_offer = price
                        self.history.append(
                            f"{self.buyer.name} ({self.buyer.role}): I can go to ${price:.2f}."
                        )
                elif isinstance(action, AcceptAction):
                    price = state.latest_valid_offer.value_for(PRICE_ISSUE_ID).value
                    speaker = (
                        self.buyer
                        if action.actor_id == BUYER_ID or offer_count == 1
                        else self.seller
                    )
                    self.history.append(
                        f"{speaker.name} ({speaker.role}): Deal accepted at ${price:.2f}."
                    )
            elif isinstance(event, MediatorInterventionEvent):
                if event.intervention.offer is not None:
                    price = event.intervention.offer.value_for(PRICE_ISSUE_ID).value
                    self.mediator_suggestion = round(price, 2)
                    self.history.append(
                        f"Mediator (mediator): I suggest a compromise price of ${price:.2f}."
                    )

    def run(self) -> Dict[str, Any]:
        self._reset_legacy_state()
        orchestrator = self._build_orchestrator()
        self.episode_result = orchestrator.run()
        self.protocol_state = self.episode_result.final_session
        agreement = self.protocol_state.outcome.agreement if self.protocol_state.outcome else None
        if agreement is not None:
            self.deal_reached = True
            self.final_price = round(
                agreement.offer.value_for(PRICE_ISSUE_ID).value, 2
            )
        self._render_history()
        action_events = tuple(
            item for item in self.protocol_state.event_sequence
            if isinstance(item, ActionAppliedEvent)
        )
        rounds_used = 0 if not action_events else min(self.rounds, self.protocol_state.round_number)
        evaluation = Judge().evaluate(
            history=self.history,
            deal_reached=self.deal_reached,
            final_price=self.final_price,
            buyer=self.buyer,
            seller=self.seller,
            rounds_used=rounds_used,
        )
        reward = LegacyRewardScorer().score_result(evaluation)
        return {
            "product": self.product_name,
            "history": self.history,
            "deal_reached": self.deal_reached,
            "final_price": self.final_price,
            "mediator_suggestion": self.mediator_suggestion,
            "evaluation": evaluation,
            "reward": reward,
        }

    def run_many(self, simulations: int = 10) -> Dict[str, Any]:
        if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        strategy_rewards: Dict[str, float] = {}
        details: List[Dict[str, Any]] = []
        for _ in range(simulations):
            result = self.run()
            details.append(result)
            key = f"buyer={self.buyer.strategy}|seller={self.seller.strategy}"
            strategy_rewards[key] = strategy_rewards.get(key, 0.0) + result["reward"]
        best = max(strategy_rewards.items(), key=lambda item: item[1])
        return {"aggregate_rewards": strategy_rewards, "best_strategy": best, "details": details}
