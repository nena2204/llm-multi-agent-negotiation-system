# Repository Guide

## Layout

- `src/llm_negotiation/`: the single Python package implementation.
- `src/llm_negotiation/domain/`: typed public negotiation models, private preferences, and adapters.
- `src/llm_negotiation/protocol.py`: deterministic protocol state machine and replayable events.
- `src/llm_negotiation/policies.py`: deterministic non-LLM policies and agent-visible observations.
- `src/llm_negotiation/benchmark.py`: protocol-backed deterministic policy comparison runner.
- `src/llm_negotiation/communication.py`: typed message routing, audit replay, and formatting.
- `src/llm_negotiation/memory.py`: participant-owned long-term, working, and episodic memory.
- `src/llm_negotiation/llm/`: provider-independent LLM contracts, offline fake, and optional adapters.
- `src/llm_negotiation/llm_policy.py`: staged LLM negotiation policy, grounding, and telemetry.
- `src/llm_negotiation/beliefs.py`: typed uncertain opponent-belief and calibration models.
- `src/llm_negotiation/opponent.py`: observable evidence, heuristic/LLM modellers, and evaluation.
- `src/llm_negotiation/verification.py`: deterministic/LLM action checks and bounded correction.
- `src/llm_negotiation/mediation.py`: mediation triggers, privacy modes, compromise search, and services.
- `src/llm_negotiation/llm_prompts/`: versioned structured-output prompt templates.
- `tests/`: pytest unit and CLI smoke tests.
- `pyproject.toml`: packaging, dependencies, console script, and pytest configuration.

## Setup, Run, and Test

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
llm-negotiation
python -m llm_negotiation.cli
```

Install the optional Streamlit UI with `python -m pip install -e ".[ui]"` and run it with
`streamlit run src/llm_negotiation/ui_streamlit.py`.

## Engineering Constraints

- Keep all user-facing imports under `llm_negotiation`; do not recreate a root compatibility package.
- Preserve deterministic negotiation behavior unless a change explicitly requires otherwise.
- Keep private participant preferences out of public scenarios, offers, actions, and presentation output.
- Keep protocol transitions deterministic, immutable, and independent of agent implementations.
- Pass bounded participant-specific memory snapshots to policies; keep raw audit trails external.
- Keep model/provider execution configuration separate from agent strategy, persona, and preferences.
- Keep provider SDK imports behind the common LLM interface and default tests network-disabled.
- Never log prompts, credentials, private memory, or provider exception bodies by default.
- Keep LLM prompts versioned and request concise rationale/evidence, never hidden chain-of-thought.
- Ground every LLM-selected action through domain and participant-visible protocol constraints.
- Infer opponent beliefs only from participant-visible evidence; simulation ground truth is evaluation-only.
- Keep deterministic verification authoritative; qualitative verifiers may reject but never override it.
- Keep mediator suggestions non-binding, privacy-mode scoped, and separate from participant actions.
- Validate public inputs and add focused tests for behavioral changes.
- Keep runtime dependencies minimal and separate optional/development dependencies.
- Do not add LLM calls, databases, or web APIs unless a task explicitly requests them.
- Avoid unrelated formatting changes and never commit or push unless explicitly asked.

## Definition of Done

A future task is complete when its requested behavior is implemented, relevant tests are added or
updated, `pytest -q` passes from an editable install, documented CLI commands still work, and the
final diff contains only intentional changes.
