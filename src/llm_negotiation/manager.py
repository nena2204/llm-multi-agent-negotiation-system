from typing import Dict, Any, List, Optional, Tuple
from .agents import BuyerAgent, SellerAgent, MediatorAgent
from .judge import Judge
from .learning import LearningAgent


class NegotiationManager:
    def __init__(self, product_name: str, buyer: BuyerAgent, seller: SellerAgent, rounds: int = 6):
        self.product_name = product_name
        self.buyer = buyer
        self.seller = seller
        self.mediator = MediatorAgent(name="Mediator", role="mediator")
        self.rounds = rounds
        self.history: List[str] = []
        self.deal_reached: bool = False
        self.final_price: Optional[float] = None
        self.mediator_suggestion: Optional[float] = None

    def _record(self, message: str):
        self.history.append(message)

    def run(self) -> Dict[str, Any]:
        # initial proposals
        self._record(self.seller.propose_initial())
        self._record(self.buyer.propose_initial())

        for r in range(1, self.rounds + 1):
            remaining = self.rounds - r + 1
            # Seller gives a price -> Buyer counters
            seller_msg = self.seller.counter_offer(self.buyer.current_offer, remaining)
            self._record(seller_msg)
            # check acceptance
            if self.buyer.current_offer >= self.seller.current_offer:
                self.deal_reached = True
                self.final_price = round(self.seller.current_offer, 2)
                self._record(self.buyer.speak(f"Deal accepted at ${self.final_price:.2f}."))
                break

            buyer_msg = self.buyer.counter_offer(self.seller.current_offer, remaining)
            self._record(buyer_msg)
            # check acceptance
            if self.buyer.current_offer >= self.seller.current_offer:
                self.deal_reached = True
                self.final_price = round(self.buyer.current_offer, 2)
                self._record(self.seller.speak(f"Deal accepted at ${self.final_price:.2f}."))
                break

        if not self.deal_reached:
            # mediator suggests compromise
            compromise = round((self.buyer.max_price + self.seller.min_acceptable) / 2.0, 2)
            self.mediator_suggestion = compromise
            self._record(self.mediator.suggest_compromise(self.buyer.max_price, self.seller.min_acceptable))

        # judge evaluation
        judge = Judge()
        evaluation = judge.evaluate(
            history=self.history,
            deal_reached=self.deal_reached,
            final_price=self.final_price,
            buyer=self.buyer,
            seller=self.seller,
            rounds_used=min(self.rounds, len(self.history)),
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
        # run multiple simulations (resetting states) to compare strategies
        strategy_rewards: Dict[str, float] = {}
        details: List[Dict] = []
        for i in range(simulations):
            # clone simple state by copying values
            # Reset agents to their initial starting offers
            self.seller.current_offer = float(self.seller.initial_price)
            self.buyer.current_offer = float(self.buyer.current_offer)
            self.history = []
            self.deal_reached = False
            self.final_price = None
            self.mediator_suggestion = None

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

