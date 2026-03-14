"""
browser_ai.compat
~~~~~~~~~~~~~~~~~
Client-specific compatibility patches.

Each module in this package provides a CompatHandler that can be registered
with the server at startup via --compat <name>.  When no --compat flag is
given, no patches are applied and the server behaves as a plain
OpenAI-compatible endpoint.

Available compat modules
------------------------
continue    First-class support for the Continue VS Code extension.
"""

from __future__ import annotations

from typing import Optional

from browser_ai.compat.continue_compat import ContinueCompat

REGISTRY: dict[str, type] = {
    "continue": ContinueCompat,
}


def get_compat(name: Optional[str]):
    """
    Return an instantiated CompatHandler for *name*, or None if name is None.
    Raises ValueError for unrecognised names.
    """
    if name is None:
        return None
    try:
        return REGISTRY[name.lower()]()
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown compat module '{name}'. Available: {available}"
        )