import math
from dataclasses import dataclass, field
from typing import List


def _validate_price(name: str, value: float) -> float:
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(price) or price < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return price


def _strategy_step(current: float, target: float, strategy: str, remaining_rounds: int = 1) -> float:
    """Compute next offer moving current toward target based on strategy.

    - aggressive: small steps (10% of gap)
    - neutral: medium steps (25% of gap)
    - cooperative: large steps (50% of gap)
    """
    if remaining_rounds <= 0:
        remaining_rounds = 1
    gap = target - current
    if strategy == "aggressive":
        step = gap * 0.10
    elif strategy == "neutral":
        step = gap * 0.25
    elif strategy == "cooperative":
        step = gap * 0.5
    else:
        # default to neutral
        step = gap * 0.25
    # avoid overshoot
    next_val = current + step
    if gap > 0:
        return min(next_val, target)
    else:
        return max(next_val, target)


@dataclass
class BaseAgent:
    name: str
    role: str
    strategy: str = "neutral"
    history: List[str] = field(default_factory=list)

    def speak(self, message: str) -> str:
        self.history.append(message)
        return f"{self.name} ({self.role}): {message}"


@dataclass
class BuyerAgent(BaseAgent):
    max_price: float = 100.0
    current_offer: float = 0.0
    initial_offer: float = field(init=False)

    def __post_init__(self) -> None:
        self.max_price = _validate_price("buyer maximum price", self.max_price)
        self.current_offer = _validate_price("buyer initial offer", self.current_offer)
        if self.current_offer > self.max_price:
            raise ValueError("buyer initial offer must not exceed buyer maximum price")
        self.initial_offer = self.current_offer

    def propose_initial(self) -> str:
        self.current_offer = float(self.current_offer)
        return self.speak(f"I offer ${self.current_offer:.2f}.")

    def counter_offer(self, seller_offer: float, remaining_rounds: int) -> str:
        # buyer wants lower price -> move toward max_price? buyer max_price is highest they will pay
        # If seller_offer is lower than current_offer, accept
        if self.current_offer >= seller_offer:
            return self.speak(f"I accept the seller's price of ${seller_offer:.2f}.")
        # move buyer offer up toward buyer max_price (they can increase their offer)
        next_offer = _strategy_step(self.current_offer, min(self.max_price, seller_offer), self.strategy, remaining_rounds)
        self.current_offer = round(next_offer, 2)
        return self.speak(f"I can go to ${self.current_offer:.2f}.")


@dataclass
class SellerAgent(BaseAgent):
    initial_price: float = 150.0
    min_acceptable: float = 50.0
    current_offer: float = 0.0

    def __post_init__(self) -> None:
        self.initial_price = _validate_price("seller initial price", self.initial_price)
        self.min_acceptable = _validate_price("seller minimum acceptable price", self.min_acceptable)
        if self.min_acceptable > self.initial_price:
            raise ValueError("seller minimum acceptable price must not exceed initial ask")
        self.current_offer = _validate_price("seller current offer", self.current_offer)

    def propose_initial(self) -> str:
        self.current_offer = float(self.initial_price)
        return self.speak(f"I ask for ${self.current_offer:.2f}.")

    def counter_offer(self, buyer_offer: float, remaining_rounds: int) -> str:
        # seller wants higher price -> move toward min_acceptable (they can reduce their ask)
        if buyer_offer >= self.current_offer:
            return self.speak(f"I accept the buyer's price of ${buyer_offer:.2f}.")
        next_offer = _strategy_step(self.current_offer, max(self.min_acceptable, buyer_offer), self.strategy, remaining_rounds)
        self.current_offer = round(next_offer, 2)
        return self.speak(f"I can lower to ${self.current_offer:.2f}.")


@dataclass
class MediatorAgent(BaseAgent):
    def suggest_compromise(self, buyer_max: float, seller_min: float) -> str:
        # fair compromise: midpoint weighted toward buyer and seller equally
        compromise = round((buyer_max + seller_min) / 2.0, 2)
        return self.speak(f"I suggest a compromise price of ${compromise:.2f}.")

