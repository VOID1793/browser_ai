"""
browser_ai.backends.perplexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend stub for Perplexity AI (https://www.perplexity.ai).

Status: NOT IMPLEMENTED
-------
This stub exists so the backend can be referenced by name in the CLI
and registry without import errors.

Implementation notes for contributors
--------------------------------------
1. TARGET_URL = "https://www.perplexity.ai"
2. Login: Perplexity supports anonymous use but rate-limits aggressively.
   A logged-in session (Google or email) is strongly recommended. Use
   --visible on first run to authenticate.
3. Input selector: a textarea or contenteditable div in the search/ask bar.
   The element ID or placeholder text ("Ask anything...") can be used as
   a reliable selector anchor.
4. Submit: Enter key or a submit button (magnifying glass / arrow icon).
5. Response container: Perplexity streams responses into a div with a class
   like "prose" or "answer". Wait for the copy button to appear as a signal
   that generation is complete.
6. Note: Perplexity's responses include citations/sources. The extraction
   logic should consider stripping or preserving these depending on use case.
"""

from __future__ import annotations

from browser_ai.backends.base import BrowserBackend

_NOT_IMPLEMENTED_MSG = (
    "The Perplexity backend is not yet implemented. "
    "See browser_ai/backends/perplexity.py for implementation notes."
)


class PerplexityBackend(BrowserBackend):
    """Browser backend stub for Perplexity AI."""

    label: str = "perplexity"

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    async def start(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def ask(self, prompt: str) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def close(self) -> None:
        pass
