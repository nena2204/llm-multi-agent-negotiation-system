import streamlit as st
from llm_negotiation.agents import BuyerAgent, SellerAgent
from llm_negotiation.manager import NegotiationManager
from llm_negotiation.utils import format_history


def app():
    st.title("LLM-Based Multi-Agent Negotiation System (Demo)")

    st.sidebar.header("Negotiation Parameters")
    product = st.sidebar.text_input("Product name", "Widget")
    seller_initial = st.sidebar.number_input("Seller initial price", value=150.0)
    seller_min = st.sidebar.number_input("Seller minimum acceptable price", value=50.0)
    buyer_initial = st.sidebar.number_input("Buyer initial offer", value=20.0)
    buyer_max = st.sidebar.number_input("Buyer maximum price", value=120.0)
    rounds = st.sidebar.slider("Rounds", 1, 20, 6)
    buyer_strategy = st.sidebar.selectbox("Buyer strategy", ["aggressive","neutral","cooperative"], index=1)
    seller_strategy = st.sidebar.selectbox("Seller strategy", ["aggressive","neutral","cooperative"], index=1)

    if st.button("Run negotiation"):
        buyer = BuyerAgent(name="Buyer", role="buyer", strategy=buyer_strategy, max_price=buyer_max, current_offer=buyer_initial)
        seller = SellerAgent(name="Seller", role="seller", strategy=seller_strategy, initial_price=seller_initial, min_acceptable=seller_min)
        manager = NegotiationManager(product_name=product, buyer=buyer, seller=seller, rounds=rounds)
        result = manager.run()

        st.subheader("Negotiation History")
        st.text(format_history(result["history"]))

        st.subheader("Outcome")
        st.write("Deal reached:", result["deal_reached"])
        st.write("Final price:", result["final_price"])
        st.write("Mediator suggestion:", result["mediator_suggestion"])

        st.subheader("Judge Evaluation")
        st.json(result["evaluation"])

        st.subheader("Reward")
        st.write(result["reward"])


if __name__ == '__main__':
    app()

