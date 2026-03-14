"""
browser_ai.backends.grok
~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend stub for xAI Grok (https://grok.com).

Status: NOT IMPLEMENTED
-------
This stub exists so the backend can be referenced by name in the CLI
and registry without import errors.

Implementation notes for contributors
--------------------------------------
1. TARGET_URL = "https://grok.com"
2. Login: Grok requires an X (Twitter) account login. Use --visible on first
   run to authenticate; the session persists in the browser context.
3. Input selector: typically a contenteditable div or textarea; inspect the
   page to confirm current selectors as the UI changes frequently.
4. Submit: look for a send button or Enter key submission.
5. Response container: the last assistant message bubble; inspect for a
   stable CSS class or data attribute to target.
6. Generating indicator: a spinner or "Stop" button that disappears when
   generation is complete.
"""

from __future__ import annotations

from browser_ai.backends.base import BrowserBackend

_NOT_IMPLEMENTED_MSG = (
    "The Grok backend is not yet implemented. "
    "See browser_ai/backends/grok.py for implementation notes."
)


class GrokBackend(BrowserBackend):
    """Browser backend stub for xAI Grok."""

    label: str = "grok"

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    async def start(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def ask(self, prompt: str) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def close(self) -> None:
        pass
