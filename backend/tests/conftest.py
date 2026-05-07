"""Shared fixtures and early mocking for backend tests.

Heavy ML dependencies (transformers, torch, bitsandbytes, etc.) are not
available on every dev machine.  We stub them at the *sys.modules* level
so that the rest of the app code can be imported without error.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy ML packages before any app code is imported
# ---------------------------------------------------------------------------

_STUBS = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "chromadb",
]

for _name in _STUBS:
    if _name not in sys.modules:
        mod = ModuleType(_name)
        # Make attribute access return MagicMock so things like
        # ``from transformers import AutoProcessor`` don't blow up.
        mod.__dict__.setdefault("__getattr__", lambda *_a, **_k: MagicMock())
        sys.modules[_name] = mod  # type: ignore[assignment]
