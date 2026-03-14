from __future__ import annotations

from typing import Type

from browser_ai.backends.base import BrowserBackend
from browser_ai.backends.gemini import GeminiBackend
from browser_ai.backends.chatgpt import ChatGPTBackend
from browser_ai.backends.perplexity import PerplexityBackend

REGISTRY: dict[str, Type[BrowserBackend]] = {
    "gemini": GeminiBackend,
    "chatgpt": ChatGPTBackend,
    "perplexity": PerplexityBackend,
}


def get_backend_class(name: str) -> Type[BrowserBackend]:
    try:
        return REGISTRY[name.lower()]
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {available}"
        )