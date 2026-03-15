"""
browser_ai.backends.chatgpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend for OpenAI ChatGPT (https://chatgpt.com).

Why visible + off-screen?
--------------------------
ChatGPT's web UI reliably detects headless Chromium and responds by degrading
the session — either serving a stripped DOM or silently rate-limiting.

The solution is a real (non-headless) Chromium window positioned off-screen at
launch via --window-position=-2000,-2000. Chromium applies this before drawing
anything, so the window never appears on screen or steals focus. No CDP calls,
no subprocess tools, no race conditions. Works on Linux, WSL2, and anywhere
Chromium respects window-position.

ProseMirror input
-----------------
INPUT_SELECTORS lists only the visible contenteditable div. The hidden backing
textarea (display:none) is excluded — Playwright's is_visible() matches it
inconsistently, causing clipboard paste to silently fail.
"""

from __future__ import annotations

from browser_ai.backends.base import BaseBrowserBackend


class ChatGPTBackend(BaseBrowserBackend):
    """Playwright-backed browser session for OpenAI ChatGPT."""

    label = "chatgpt"
    URL = "https://chatgpt.com/"

    # Run visible but positioned off-screen to avoid headless detection.
    # --window-position is applied by Chromium before the window is drawn,
    # so it never appears on screen or steals focus.
    DEFAULT_HEADLESS = False
    MINIMIZE_WINDOW = True  # used for banner/log labelling only

    EXTRA_LAUNCH_ARGS = [
        "--window-position=-2000,-2000",
        "--window-size=1280,900",
    ]

    # ── Selectors ─────────────────────────────────────────────────────────────

    INPUT_SELECTORS = [
        'div#prompt-textarea[contenteditable="true"]',
        'div#prompt-textarea',
    ]

    SEND_SELECTORS = [
        'button[data-testid="send-button"]',
        'button[aria-label="Send prompt"]',
    ]

    RESPONSE_SELECTORS = [
        'article[data-turn="assistant"] div.markdown',
        'article[data-turn="assistant"]',
        'div[data-message-author-role="assistant"] div.markdown',
        'div[data-message-author-role="assistant"]',
    ]

    GENERATING_SELECTORS = [
        'button[data-testid="stop-button"]',
        '[data-testid="stop-button"]',
        '.result-streaming',
    ]

    INTERSTITIAL_SELECTORS = [
        'button:has-text("Stay logged out")',
        'button:has-text("No thanks")',
        'button:has-text("Maybe later")',
        'div[role="dialog"] button:has-text("Close")',
        'div[role="dialog"] button:has-text("Dismiss")',
        'div[role="dialog"] button[aria-label="Close"]',
    ]

    # ── Hooks ─────────────────────────────────────────────────────────────────

    async def _post_deliver_wait(self, pg) -> None:
        """Dismiss any nudges that appeared during paste, wait for send button."""
        for _ in range(5):
            if await self._dismiss_interstitials(pg):
                await pg.wait_for_timeout(400)
            else:
                break

        for _ in range(20):
            try:
                await self._dismiss_interstitials(pg)
                btn = pg.locator(self.SEND_SELECTORS[0]).first
                if await btn.is_enabled(timeout=200):
                    break
            except Exception:
                pass
            await pg.wait_for_timeout(200)

    async def _dismiss_interstitials(self, pg) -> bool:
        """Dismiss blocking modals/nudges. Returns True if anything was clicked."""
        clicked = False
        for selector in self.INTERSTITIAL_SELECTORS:
            try:
                btn = pg.locator(selector).first
                if await btn.is_visible(timeout=100):
                    await btn.click(force=True, timeout=500)
                    clicked = True
                    await pg.wait_for_timeout(300)
            except Exception:
                continue
        return clicked