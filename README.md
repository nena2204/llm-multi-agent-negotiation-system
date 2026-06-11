# LLM-Based Multi-Agent Negotiation System

This project simulates a negotiation between autonomous agents: BuyerAgent, SellerAgent, MediatorAgent, JudgeAgent, and an optional LearningAgent.

Features:
- Rule-based agents with simple negotiation strategies: aggressive, neutral, cooperative
- Negotiation manager to run multi-round negotiations
- Mediator suggestion when agents fail to reach agreement
- Judge evaluates fairness and produces a fairness score and explanation
- LearningAgent awards rewards to strategies across multiple simulations
- CLI entrypoint and optional Streamlit UI

See `src/llm_negotiation/cli.py` for usage and `src/llm_negotiation/ui_streamlit.py` for the optional UI.

To run tests:

    pytest -q

To run the CLI example:

    python -m src.llm_negotiation.cli

To launch the Streamlit UI (optional):

    streamlit run src/llm_negotiation/ui_streamlit.py

