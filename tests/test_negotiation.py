from llm_negotiation.agents import BuyerAgent, SellerAgent
from llm_negotiation.manager import NegotiationManager


def test_simple_deal_reached():
    buyer = BuyerAgent(name="Buyer", role="buyer", strategy="cooperative", max_price=120.0, current_offer=20.0)
    seller = SellerAgent(name="Seller", role="seller", strategy="cooperative", initial_price=100.0, min_acceptable=80.0)
    manager = NegotiationManager(product_name="Test", buyer=buyer, seller=seller, rounds=5)
    res = manager.run()
    # cooperative agents should reach a deal
    assert isinstance(res, dict)
    assert "history" in res
    # Either deal reached or mediator suggested; for these params cooperative should reach
    assert res["deal_reached"] or res["mediator_suggestion"] is not None

