"""
browser_ai.backends.base
~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class that every browser backend must implement.

The interface is intentionally minimal: start, ask, close, reset.
The session manager and route handlers depend only on this interface,
never on a concrete backend class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BrowserBackend(ABC):
    """
    Abstract browser backend.

    Implementations manage a persistent browser session pointed at a specific
    LLM web UI.  Each instance corresponds to one browser page / conversation.
    """

    @abstractmethod
    async def start(self) -> None:
        """
        Launch the browser and navigate to the LLM UI.

        Must be idempotent — safe to call multiple times.  Subsequent calls
        after the first successful start should be no-ops.
        """

    @abstractmethod
    async def ask(self, prompt: str) -> str:
        """
        Submit *prompt* to the LLM and return the complete response text.

        Blocks until the response is stable (no longer being generated).
        Must be safe to call concurrently — implementations should serialise
        access with an asyncio.Lock.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Tear down the browser session and release all resources.

        Must be idempotent.
        """

    async def reset(self) -> None:
        """
        Close and reopen the browser session from scratch.

        The default implementation calls close() then start().
        Backends may override this to use a lighter-weight reset mechanism
        (e.g. clicking a "New chat" button) when available.
        """
        await self.close()
        await self.start()
