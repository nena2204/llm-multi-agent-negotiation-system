# Repository Guide

## Layout

- `src/llm_negotiation/`: the single Python package implementation.
- `src/llm_negotiation/domain/`: typed public negotiation models, private preferences, and adapters.
- `src/llm_negotiation/protocol.py`: deterministic protocol state machine and replayable events.
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
- Validate public inputs and add focused tests for behavioral changes.
- Keep runtime dependencies minimal and separate optional/development dependencies.
- Do not add LLM calls, databases, or web APIs unless a task explicitly requests them.
- Avoid unrelated formatting changes and never commit or push unless explicitly asked.

## Definition of Done

A future task is complete when its requested behavior is implemented, relevant tests are added or
updated, `pytest -q` passes from an editable install, documented CLI commands still work, and the
final diff contains only intentional changes.
