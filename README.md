# LLM-Based Multi-Agent Negotiation System

This project is a deterministic, rule-based simulation of negotiation between buyer, seller,
mediator, judge, and learning roles. It does not currently call an LLM or an external API.

## Features

- Buyer and seller strategies: aggressive, neutral, and cooperative
- Multi-round negotiation with a mediator fallback
- Outcome validity and fairness evaluation
- Deterministic rewards and repeated simulations
- Command-line interface and optional Streamlit UI

## Domain model

The `llm_negotiation.domain` package defines validated participants, issues, offers, private utility
profiles, actions, agreements, and terminal outcomes. Public scenarios contain participants and
issue definitions but never private preferences. Numeric utilities are linearly normalized, while
categorical utilities use participant-specific scores in `[0, 1]`. Issue weights must sum to `1.0`
within an absolute tolerance of `1e-6`.

The current deterministic price negotiation remains available through a compatibility adapter while
the orchestration layer is migrated incrementally.

The `llm_negotiation.protocol` module defines the authoritative negotiation lifecycle. It validates
typed actions against the current phase, participant turn, outstanding offer, and round deadline;
every successful transition emits a replayable event. Natural-language messages never change
protocol meaning.

Protocol phases are `created`, `active`, `mediation`, `agreed`, `failed`, `withdrawn`, and `expired`.
A round is one complete rotation through the configured participant order, anchored to the initial
participant. Mediation is entered by an explicit `request_mediation` action; no mediator policy is
implemented yet. A trusted coordinator may mark the feasible region `impossible` during session
creation, allowing deterministic failure without adding private preference data to public state.

## Setup

Python 3.9 or newer is required. From the repository root, create a virtual environment and install
the package in editable mode with its test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run the CLI

Use the installed console command:

```powershell
llm-negotiation
```

Or run the package module directly:

```powershell
python -m llm_negotiation.cli
```

For all options, run `llm-negotiation --help`.

## Verify

```powershell
python -c "import llm_negotiation; print(llm_negotiation.__file__)"
pytest -q
llm-negotiation --rounds 2
python -m llm_negotiation.cli --rounds 2
```

## Optional Streamlit UI

Streamlit is deliberately separate from the core runtime and test dependencies:

```powershell
python -m pip install -e ".[ui]"
streamlit run src/llm_negotiation/ui_streamlit.py
```

