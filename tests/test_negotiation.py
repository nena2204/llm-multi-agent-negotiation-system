import math

import pytest

from llm_negotiation.agents import BuyerAgent, SellerAgent
from llm_negotiation.manager import NegotiationManager
from llm_negotiation.protocol import ProtocolPhase


def make_manager(
    *,
    buyer_initial=100.0,
    buyer_max=120.0,
    seller_initial=100.0,
    seller_min=80.0,
    rounds=5,
    strategy="cooperative",
):
    buyer = BuyerAgent(
        name="Buyer",
        role="buyer",
        strategy=strategy,
        max_price=buyer_max,
        current_offer=buyer_initial,
    )
    seller = SellerAgent(
        name="Seller",
        role="seller",
        strategy=strategy,
        initial_price=seller_initial,
        min_acceptable=seller_min,
    )
    return NegotiationManager(product_name="Test", buyer=buyer, seller=seller, rounds=rounds)


def test_deal_is_reached_and_rounds_are_counted():
    manager = make_manager()
    result = manager.run()

    assert result["deal_reached"] is True
    assert result["final_price"] == 100.0
    assert result["mediator_suggestion"] is None
    assert result["evaluation"]["rounds_used"] == 1
    assert manager.protocol_state.phase is ProtocolPhase.AGREED
    assert manager.protocol_state.outcome.agreement is not None


def test_failed_negotiation_invokes_mediator():
    manager = make_manager(
        buyer_initial=10,
        buyer_max=100,
        seller_initial=100,
        seller_min=80,
        rounds=1,
        strategy="aggressive",
    )
    result = manager.run()

    assert result["deal_reached"] is False
    assert result["final_price"] is None
    assert result["mediator_suggestion"] == 90.0
    assert result["evaluation"]["rounds_used"] == 1
    assert result["history"][-1] == "Mediator (mediator): I suggest a compromise price of $90.00."
    assert manager.protocol_state.phase is ProtocolPhase.MEDIATION
    assert manager.protocol_state.outcome is None


def test_impossible_price_region_fails_before_bargaining():
    manager = make_manager(
        buyer_initial=10,
        buyer_max=40,
        seller_initial=100,
        seller_min=80,
        rounds=2,
        strategy="aggressive",
    )

    result = manager.run()

    assert result["deal_reached"] is False
    assert result["mediator_suggestion"] is None
    assert result["evaluation"]["rounds_used"] == 0
    assert result["history"] == ["System (protocol): The feasible region is empty."]
    assert manager.protocol_state.phase is ProtocolPhase.FAILED
    assert manager.protocol_state.outcome is not None


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: make_manager(rounds=0), "rounds must be a positive integer"),
        (lambda: make_manager(buyer_initial=121), "buyer initial offer must not exceed"),
        (lambda: make_manager(seller_initial=50, seller_min=51), "seller minimum acceptable price must not exceed"),
        (lambda: make_manager(buyer_initial=-1), "finite non-negative"),
        (lambda: make_manager(buyer_max=math.inf), "finite non-negative"),
        (lambda: make_manager(seller_initial=math.nan), "finite non-negative"),
    ],
)
def test_invalid_configuration_is_rejected(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_repeated_runs_are_deterministic_and_reset_state():
    manager = make_manager()

    first = manager.run()
    second = manager.run()
    many = manager.run_many(simulations=3)

    assert second == first
    assert all(result == first for result in many["details"])
    assert many["aggregate_rewards"] == {"buyer=cooperative|seller=cooperative": first["reward"] * 3}

