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
1. URL = "https://www.perplexity.ai"
2. Login: Perplexity supports anonymous use but rate-limits aggressively.
   A logged-in session (Google or email) is recommended.  Use --no-headless
   on first run to authenticate.
3. Input selector: a textarea or contenteditable div in the ask bar.
   The placeholder text ("Ask anything...") is a reliable selector anchor.
4. Submit: Enter key or a submit button (arrow icon).
5. Response container: Perplexity streams into a div with a class like
   "prose" or "answer". The copy button appearing is a reliable signal
   that generation is complete.
6. Perplexity responses include citation links.  Consider stripping them
   in a custom extract_clean_response_text override.
"""

from __future__ import annotations
from typing import Optional

from browser_ai.backends.base import BrowserBackend

_NOT_IMPLEMENTED_MSG = (
    "The Perplexity backend is not yet implemented. "
    "See browser_ai/backends/perplexity.py for implementation notes."
)


class PerplexityBackend(BrowserBackend):
    """Browser backend stub for Perplexity AI."""

    label: str = "perplexity"
    DEFAULT_HEADLESS: bool = True

    def __init__(self, headless: Optional[bool] = None) -> None:
        # headless argument accepted for API compatibility; ignored since
        # this backend raises NotImplementedError before ever launching.
        self.headless = headless if headless is not None else self.DEFAULT_HEADLESS

    async def start(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def ask(self, prompt: str) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def close(self) -> None:
        pass