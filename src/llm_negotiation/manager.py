from typing import Any, Dict, List, Optional
from .agents import BuyerAgent, SellerAgent, MediatorAgent
from .judge import Judge
from .learning import LearningAgent


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

    def run(self) -> Dict[str, Any]:
        self._reset()
        # initial proposals
        self._record(self.seller.propose_initial())
        self._record(self.buyer.propose_initial())

        rounds_used = 0
        for round_number in range(1, self.rounds + 1):
            rounds_used = round_number
            remaining = self.rounds - round_number + 1
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

