"""
browser_ai.backends.chatgpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend for OpenAI ChatGPT (https://chatgpt.com).

ChatGPT's free tier is accessible without login.

Selector status: VALIDATED against live DOM (2025-03-14)
---------------------------------------------------------
Input: div#prompt-textarea[contenteditable="true"].ProseMirror
  - There is also a hidden textarea[name="prompt-textarea"] (display:none)
    which is what 'textarea' matches. Clipboard paste works to either.
  - Keyboard typing must target the visible contenteditable div.

Response: article[data-turn="assistant"] div.markdown
  - Each turn is an <article data-testid="conversation-turn-N">
  - data-turn="assistant" distinguishes model replies from user messages
  - Content lives in div.markdown (also has class prose)

Send button: button[data-testid="send-button"]
  - Confirmed present in DOM. Becomes enabled after content is entered.

Generating indicator:
  - Not visible in completed-response HTML.
  - Assumed to be button[data-testid="stop-button"] during streaming.
"""

from __future__ import annotations

from browser_ai.backends.base import BaseBrowserBackend


class ChatGPTBackend(BaseBrowserBackend):
    """Playwright-backed browser session for OpenAI ChatGPT."""

    label = "chatgpt"
    URL = "https://chatgpt.com/"

    CONSENT_SELECTORS = [
        'button:has-text("Accept all")',
        'button:has-text("Accept cookies")',
        'button:has-text("Agree")',
    ]

    # Validated: the visible ProseMirror editor has id="prompt-textarea".
    # The hidden textarea[name="prompt-textarea"] is display:none and used
    # as a fallback for clipboard paste only.
    INPUT_SELECTORS = [
        'div#prompt-textarea[contenteditable="true"]',
        'div[contenteditable="true"].ProseMirror',
        'div[id="prompt-textarea"][contenteditable="true"]',
        'textarea',  # hidden backing element — works for clipboard paste
    ]

    # Validated: send button has data-testid="send-button".
    # It becomes enabled only after content is typed/pasted.
    SEND_SELECTORS = [
        'button[data-testid="send-button"]',
        'button[aria-label="Send prompt"]',
        'button[aria-label*="Send" i]',
    ]

    # Validated: assistant turns are article[data-turn="assistant"].
    # Content is inside div.markdown (which also has class prose).
    RESPONSE_SELECTORS = [
        'article[data-turn="assistant"] div.markdown',
        'article[data-turn="assistant"] .prose',
        'div[data-message-author-role="assistant"] div.markdown',
        'div[data-message-author-role="assistant"]',
    ]

    # Best-effort: stop button appears during streaming.
    GENERATING_SELECTORS = [
        'button[data-testid="stop-button"]',
        'button[aria-label="Stop streaming"]',
        'button[aria-label*="Stop" i]',
    ]

    # Selectors for mid-session popups that block the UI.
    # ChatGPT occasionally shows a "log in or stay logged out" interstitial.
    INTERSTITIAL_SELECTORS = [
        # "Stay logged out" button on login nudge
        'button:has-text("Stay logged out")',
        'button:has-text("stay logged out")',
        # Generic close/dismiss on modal dialogs
        'button[aria-label="Close"]',
        # "Maybe later" on upsell nudges
        'button:has-text("Maybe later")',
        'button:has-text("No thanks")',
    ]

    async def _post_deliver_wait(self, pg) -> None:
        """
        ChatGPT's send button is disabled until content is entered.
        Wait for the send button to become enabled, up to 3 seconds.
        Also dismiss any interstitial that may have appeared.
        """
        # Poll until send button is enabled (it starts disabled)
        for _ in range(15):
            try:
                btn = pg.locator('button[data-testid="send-button"]').first
                if await btn.is_enabled(timeout=200):
                    break
            except Exception:
                pass
            await pg.wait_for_timeout(200)
        await self._dismiss_interstitials(pg)

    async def _dismiss_interstitials(self, pg) -> None:
        """
        Dismiss any mid-session modal or interstitial overlays.
        Called before sending and during response polling.
        ChatGPT sometimes shows a login nudge with a "Stay logged out" button.
        """
        for selector in self.INTERSTITIAL_SELECTORS:
            try:
                btn = pg.locator(selector).first
                if await btn.is_visible(timeout=500):
                    print(
                        f"[{self.label}] Dismissing interstitial: {selector}",
                        flush=True,
                    )
                    await btn.click()
                    await pg.wait_for_timeout(400)
                    return
            except Exception:
                pass