from typing import Any, Dict, List, Optional
from .agents import BuyerAgent, SellerAgent, MediatorAgent
from .domain import (
    AcceptAction,
    CounterAction,
    NumericIssueValue,
    NegotiationAction,
    Offer,
    OfferId,
    ProposeAction,
    ParticipantId,
    RequestMediationAction,
    MediationAccessMode,
    MediationTrigger,
    price_only_from_legacy,
)
from .domain.legacy import BUYER_ID, PRICE_ISSUE_ID, SELLER_ID
from .judge import Judge
from .learning import LearningAgent
from .protocol import FeasibleRegionStatus, NegotiationProtocol, NegotiationSession, ProtocolPhase
from .mediation import DeterministicMediator, MediatorConfiguration, MediatorContext


class NegotiationManager:
    def __init__(self, product_name: str, buyer: BuyerAgent, seller: SellerAgent, rounds: int = 6):
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
            raise ValueError("rounds must be a positive integer")
        self.product_name = product_name
        self.buyer = buyer
        self.seller = seller
        self.mediator = MediatorAgent(name="Mediator", role="mediator")
        self.rounds = rounds
        self.history: List[str] = []
        self.deal_reached: bool = False
        self.final_price: Optional[float] = None
        self.mediator_suggestion: Optional[float] = None
        self.protocol_state: Optional[NegotiationSession] = None
        self._protocol: Optional[NegotiationProtocol] = None
        self._offer_sequence = 0
        self._migration = None

    def _record(self, message: str):
        self.history.append(message)

    def _reset(self) -> None:
        self.buyer.current_offer = self.buyer.initial_offer
        self.seller.current_offer = self.seller.initial_price
        self.buyer.history.clear()
        self.seller.history.clear()
        self.mediator.history.clear()
        self.history = []
        self.deal_reached = False
        self.final_price = None
        self.mediator_suggestion = None
        self._offer_sequence = 0

        migration = price_only_from_legacy(self.product_name, self.buyer, self.seller, self.rounds)
        self._migration = migration
        self._protocol = NegotiationProtocol(
            scenario=migration.scenario,
            maximum_rounds=self.rounds + 1,
            initial_turn=BUYER_ID,
            feasible_region=(
                FeasibleRegionStatus.FEASIBLE
                if self.buyer.max_price >= self.seller.min_acceptable
                else FeasibleRegionStatus.IMPOSSIBLE
            ),
        )
        self.protocol_state = self._protocol.create_session()
        if self.protocol_state.phase is ProtocolPhase.FAILED:
            return
        self._apply_protocol_action(
            ProposeAction(
                actor_id=BUYER_ID,
                recipients=(SELLER_ID,),
                round_number=self.protocol_state.round_number,
                offer=migration.buyer_initial_offer,
            )
        )

    def _next_price_offer(self, actor: str, price: float) -> Offer:
        self._offer_sequence += 1
        return Offer(
            offer_id=OfferId(f"legacy-{actor}-{self._offer_sequence}"),
            values=(NumericIssueValue(issue_id=PRICE_ISSUE_ID, value=price),),
        )

    def _apply_protocol_action(self, action: NegotiationAction) -> None:
        if self.protocol_state is None or self._protocol is None:
            raise RuntimeError("protocol session has not been initialized")
        self.protocol_state = self._protocol.transition(self.protocol_state, action).state

    def _accept_outstanding_offer(self, actor_id: ParticipantId) -> None:
        if self.protocol_state is None or self.protocol_state.latest_valid_offer is None:
            raise RuntimeError("cannot accept without an outstanding protocol offer")
        self._apply_protocol_action(
            AcceptAction(
                actor_id=actor_id,
                round_number=self.protocol_state.round_number,
                offer_id=self.protocol_state.latest_valid_offer.offer_id,
            )
        )
        agreement = self.protocol_state.outcome.agreement if self.protocol_state.outcome else None
        if agreement is None:
            raise RuntimeError("legacy bilateral acceptance did not produce an agreement")
        price_value = agreement.offer.value_for(PRICE_ISSUE_ID)
        if not isinstance(price_value, NumericIssueValue):
            raise RuntimeError("legacy price agreement does not contain a numeric price")
        self.deal_reached = True
        self.final_price = round(price_value.value, 2)

    def _run_price_negotiation(self) -> int:
        # initial proposals
        self._record(self.seller.propose_initial())
        self._record(self.buyer.propose_initial())

        rounds_used = 0
        for round_number in range(1, self.rounds + 1):
            rounds_used = round_number
            remaining = self.rounds - round_number + 1
            # Seller gives a price -> Buyer counters
            seller_accepts = self.buyer.current_offer >= self.seller.current_offer
            seller_msg = self.seller.counter_offer(self.buyer.current_offer, remaining)
            self._record(seller_msg)
            if seller_accepts:
                self._accept_outstanding_offer(SELLER_ID)
                self._record(self.buyer.speak(f"Deal accepted at ${self.final_price:.2f}."))
                break

            seller_offer = self._next_price_offer("seller", self.seller.current_offer)
            self._apply_protocol_action(
                CounterAction(
                    actor_id=SELLER_ID,
                    recipients=(BUYER_ID,),
                    round_number=self.protocol_state.round_number,
                    offer=seller_offer,
                    responds_to=self.protocol_state.latest_valid_offer.offer_id,
                )
            )
            # check acceptance
            if self.buyer.current_offer >= self.seller.current_offer:
                self._accept_outstanding_offer(BUYER_ID)
                self._record(self.buyer.speak(f"Deal accepted at ${self.final_price:.2f}."))
                break

            buyer_msg = self.buyer.counter_offer(self.seller.current_offer, remaining)
            self._record(buyer_msg)
            buyer_offer = self._next_price_offer("buyer", self.buyer.current_offer)
            self._apply_protocol_action(
                CounterAction(
                    actor_id=BUYER_ID,
                    recipients=(SELLER_ID,),
                    round_number=self.protocol_state.round_number,
                    offer=buyer_offer,
                    responds_to=self.protocol_state.latest_valid_offer.offer_id,
                )
            )
            # check acceptance
            if self.buyer.current_offer >= self.seller.current_offer:
                self._accept_outstanding_offer(SELLER_ID)
                self._record(self.seller.speak(f"Deal accepted at ${self.final_price:.2f}."))
                break

        if not self.deal_reached:
            self._apply_protocol_action(
                RequestMediationAction(
                    actor_id=self.protocol_state.current_turn,
                    round_number=self.protocol_state.round_number,
                    reason="Legacy price negotiation exhausted its configured rounds.",
                )
            )
            mediator = DeterministicMediator(
                MediatorConfiguration(
                    access_mode=MediationAccessMode.SIMULATION_ORACLE,
                    allow_simulation_oracle=True,
                )
            )
            intervention = mediator.intervene(
                MediatorContext(
                    scenario=self._migration.scenario,
                    session=self.protocol_state,
                    access_mode=MediationAccessMode.SIMULATION_ORACLE,
                    simulation_ground_truth=(
                        self._migration.buyer_preferences,
                        self._migration.seller_preferences,
                    ),
                ),
                MediationTrigger.EXPLICIT_REQUEST,
            )
            self.protocol_state = self._protocol.apply_mediator_intervention(
                self.protocol_state, intervention
            ).state
            price = intervention.offer.value_for(PRICE_ISSUE_ID)
            if not isinstance(price, NumericIssueValue):
                raise RuntimeError("legacy mediator did not produce a numeric price")
            self.mediator_suggestion = round(price.value, 2)
            self._record(self.mediator.suggest_compromise(self.buyer.max_price, self.seller.min_acceptable))
        return rounds_used

    def run(self) -> Dict[str, Any]:
        self._reset()
        if self.protocol_state.phase is ProtocolPhase.FAILED:
            rounds_used = 0
            self._record(f"System (protocol): {self.protocol_state.outcome.reason}")
        else:
            rounds_used = self._run_price_negotiation()

        # judge evaluation
        judge = Judge()
        evaluation = judge.evaluate(
            history=self.history,
            deal_reached=self.deal_reached,
            final_price=self.final_price,
            buyer=self.buyer,
            seller=self.seller,
            rounds_used=rounds_used,
        )

        # learning agent (optional) - we can return score per strategies
        learner = LearningAgent()
        reward = learner.score_result(evaluation)

        result = {
            "product": self.product_name,
            "history": self.history,
            "deal_reached": self.deal_reached,
            "final_price": self.final_price,
            "mediator_suggestion": self.mediator_suggestion,
            "evaluation": evaluation,
            "reward": reward,
        }
        return result

    def run_many(self, simulations: int = 10) -> Dict[str, Any]:
        if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
            raise ValueError("simulations must be a positive integer")

        # run multiple simulations to compare strategies
        strategy_rewards: Dict[str, float] = {}
        details: List[Dict] = []
        for _ in range(simulations):
            res = self.run()
            details.append(res)
            # aggregate reward per strategy pair
            key = f"buyer={self.buyer.strategy}|seller={self.seller.strategy}"
            strategy_rewards[key] = strategy_rewards.get(key, 0) + res["reward"]

        # pick best-performing strategy key
        best_strategy = max(strategy_rewards.items(), key=lambda x: x[1]) if strategy_rewards else (None, 0)
        return {
            "aggregate_rewards": strategy_rewards,
            "best_strategy": best_strategy,
            "details": details,
        }

