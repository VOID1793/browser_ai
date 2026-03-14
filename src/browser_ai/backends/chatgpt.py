"""
browser_ai.backends.chatgpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend for OpenAI ChatGPT (https://chat.openai.com).

ChatGPT's free tier (GPT-4o mini) is accessible without login.
Navigating to https://chatgpt.com/ starts an anonymous session.

Status: selectors defined, needs live validation.
-------
The selectors below were derived by inspecting the ChatGPT DOM.  Run with
--visible to validate them against the live UI before relying on this backend.

To debug selector issues:
    browser-ai serve --backend chatgpt --visible
"""

from __future__ import annotations

from browser_ai.backends.base import BaseBrowserBackend


class ChatGPTBackend(BaseBrowserBackend):
    """Playwright-backed browser session for OpenAI ChatGPT."""

    label = "chatgpt"
    URL = "https://chatgpt.com/"

    # ChatGPT may show a cookie consent banner in some regions.
    CONSENT_SELECTORS = [
        'button:has-text("Accept all")',
        'button:has-text("Accept cookies")',
        'button:has-text("Agree")',
    ]

    # The prompt textarea has a stable data-id and placeholder.
    INPUT_SELECTORS = [
        'div[id="prompt-textarea"][contenteditable="true"]',
        'div[contenteditable="true"][data-id="root"]',
        'textarea[data-id="root"]',
        'div[contenteditable="true"][placeholder*="Message" i]',
        'div[contenteditable="true"]',
        'textarea',
    ]

    # The send button has a stable data-testid.
    SEND_SELECTORS = [
        'button[data-testid="send-button"]',
        'button[aria-label*="Send" i]',
        'button[aria-label*="send" i]',
    ]

    # Assistant responses are wrapped in article elements with a data-testid.
    RESPONSE_SELECTORS = [
        'article[data-testid^="conversation-turn-"] div.prose',
        'div[data-message-author-role="assistant"]',
        '.prose.dark\\:prose-invert',
        '.markdown',
    ]

    # ChatGPT shows a stop button while streaming.
    GENERATING_SELECTORS = [
        'button[data-testid="stop-button"]',
        'button[aria-label*="Stop" i]',
        'button[aria-label="Stop streaming"]',
    ]