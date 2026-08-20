import argparse
from llm_negotiation.agents import BuyerAgent, SellerAgent
from llm_negotiation.domain import price_only_from_legacy, price_only_to_legacy
from llm_negotiation.manager import NegotiationManager
from llm_negotiation.utils import format_history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a negotiation simulation")
    p.add_argument("--product", default="Widget", help="Product name")
    p.add_argument("--seller_initial", type=float, default=150.0, help="Seller initial price")
    p.add_argument("--seller_min", type=float, default=50.0, help="Seller minimum acceptable price")
    p.add_argument("--buyer_initial", type=float, default=20.0, help="Buyer initial offer")
    p.add_argument("--buyer_max", type=float, default=120.0, help="Buyer maximum price")
    p.add_argument("--rounds", type=int, default=6, help="Number of negotiation rounds")
    p.add_argument("--buyer_strategy", choices=["aggressive","neutral","cooperative"], default="neutral")
    p.add_argument("--seller_strategy", choices=["aggressive","neutral","cooperative"], default="neutral")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    buyer = BuyerAgent(name="Buyer", role="buyer", strategy=args.buyer_strategy, max_price=args.buyer_max, current_offer=args.buyer_initial)
    seller = SellerAgent(name="Seller", role="seller", strategy=args.seller_strategy, initial_price=args.seller_initial, min_acceptable=args.seller_min)
    price_domain = price_only_from_legacy(args.product, buyer, seller, args.rounds)
    buyer, seller, rounds = price_only_to_legacy(price_domain)

    manager = NegotiationManager(product_name=args.product, buyer=buyer, seller=seller, rounds=rounds)
    result = manager.run()

    print("--- Negotiation History ---")
    print(format_history(result["history"]))
    print("\n--- Outcome ---")
    print(f"Deal reached: {result['deal_reached']}")
    print(f"Final price: {result['final_price']}")
    print(f"Mediator suggestion: {result['mediator_suggestion']}")
    print("\n--- Judge Evaluation ---")
    for k, v in result["evaluation"].items():
        print(f"{k}: {v}")
    print(f"\nReward points: {result['reward']}")


if __name__ == '__main__':
    main()

