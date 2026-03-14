"""
browser_ai.backends.chatgpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend stub for OpenAI ChatGPT (https://chat.openai.com).

Status: NOT IMPLEMENTED
-------
This stub exists so the backend can be referenced by name in the CLI
and registry without import errors.  Attempting to use it will raise
NotImplementedError with guidance on what needs to be implemented.

Implementation notes for contributors
--------------------------------------
1. TARGET_URL = "https://chat.openai.com"
2. Login detection: ChatGPT requires an authenticated session.  On first
   run with --visible the user should log in; the session is persisted in
   the Playwright browser context directory.
3. Input selector: the prompt textarea is typically:
       textarea[data-id="root"]
   or  div[contenteditable="true"][id="prompt-textarea"]
4. Submit: a send button with aria-label "Send prompt" or pressing Enter.
5. Response container: div.markdown.prose, or more specifically the last
   article[data-testid^="conversation-turn-"] > div.prose
6. Generating indicator: button[aria-label="Stop generating"]
7. Clipboard paste: same execCommand approach as GeminiBackend works fine.
"""

from __future__ import annotations

from browser_ai.backends.base import BrowserBackend

_NOT_IMPLEMENTED_MSG = (
    "The ChatGPT backend is not yet implemented. "
    "See browser_ai/backends/chatgpt.py for implementation notes."
)


class ChatGPTBackend(BrowserBackend):
    """Browser backend stub for OpenAI ChatGPT."""

    label: str = "chatgpt"

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    async def start(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def ask(self, prompt: str) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def close(self) -> None:
        pass  # Nothing to tear down in a stub
