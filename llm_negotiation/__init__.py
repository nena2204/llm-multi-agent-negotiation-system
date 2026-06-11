"""Compatibility wrapper that re-exports the implementation under src.llm_negotiation

This allows top-level imports like `import llm_negotiation` while keeping sources under `src/`.
"""
from importlib import import_module

_impl = import_module("src.llm_negotiation")

# Re-export public names
for _name in getattr(_impl, "__all__", [n for n in dir(_impl) if not n.startswith("_")]):
    try:
        globals()[_name] = getattr(_impl, _name)
    except AttributeError:
        pass

# Also keep a reference to the implementation module
__implementation__ = _impl

