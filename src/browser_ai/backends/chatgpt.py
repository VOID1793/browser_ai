"""
browser_ai.backends.chatgpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hardened ChatGPT backend with aggressive interstitial dismissal.
"""

from __future__ import annotations
from browser_ai.backends.base import BaseBrowserBackend

class ChatGPTBackend(BaseBrowserBackend):
    """Playwright-backed browser session for OpenAI ChatGPT."""

    label = "chatgpt"
    URL = "https://chatgpt.com/"

    # Selectors for closing common ChatGPT popups/nudges
    INTERSTITIAL_SELECTORS = [
        'button:has-text("Stay logged out")',
        'button:has-text("No thanks")',
        'div[role="dialog"] button:has-text("Close")',
        'div[role="dialog"] button:has-text("Dismiss")',
        'button:has-text("Maybe later")',
        '.bg-token-main-surface-primary button', 
    ]

    # Validated input selectors
    INPUT_SELECTORS = [
        'div#prompt-textarea',
        'textarea[name="prompt-textarea"]',
        'textarea',
    ]

    SEND_SELECTORS = [
        'button[data-testid="send-button"]',
        'button[aria-label="Send prompt"]',
    ]

    # Explicitly target the assistant turn while ignoring nudges
    RESPONSE_SELECTORS = [
        'article[data-turn="assistant"] div.markdown',
        'div[data-message-author-role="assistant"] div.markdown',
        'article:has(div.markdown):not(:has(div[data-turn="user"])) div.markdown',
    ]

    # The stop button and streaming class are the best signals
    GENERATING_SELECTORS = [
        'button[data-testid="stop-button"]',
        '.result-streaming',
    ]

    async def _post_deliver_wait(self, pg) -> None:
        """
        Wait for the UI to stabilize and ensure the send button is truly enabled.
        """
        # 1. Clear any login nudges that might have popped up during prompt injection
        for _ in range(5):
            if await self._dismiss_interstitials(pg):
                await pg.wait_for_timeout(400)
            else:
                break

        # 2. Wait for the send button to become enabled
        for _ in range(15):
            try:
                # Ensure a new nudge hasn't appeared
                await self._dismiss_interstitials(pg)
                
                btn = pg.locator(self.SEND_SELECTORS[0]).first
                if await btn.is_enabled(timeout=200):
                    break
            except Exception:
                pass
            await pg.wait_for_timeout(200)

    async def _dismiss_interstitials(self, pg) -> bool:
        """
        Aggressively dismiss blocking UI elements using forced clicks.
        """
        clicked = False
        for selector in self.INTERSTITIAL_SELECTORS:
            try:
                btn = pg.locator(selector).first
                if await btn.is_visible(timeout=100):
                    # Use force=True to bypass hit-testing issues with overlays
                    await btn.click(force=True, timeout=500)
                    clicked = True
                    await pg.wait_for_timeout(300)
            except Exception:
                continue
        return clicked