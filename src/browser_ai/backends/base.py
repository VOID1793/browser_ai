"""
browser_ai.backends.base
~~~~~~~~~~~~~~~~~~~~~~~~
Two-layer base class system for browser backends.

Layer 1 — BrowserBackend (Abstract Interface)
    The minimal public contract that the session manager and route handlers
    depend on.  Contains only start(), ask(), close(), and reset().
    Nothing in the rest of the codebase imports a concrete backend class —
    only this interface.

Layer 2 — BaseBrowserBackend (Shared Machinery)
    A concrete (but not instantiable) implementation of BrowserBackend that
    provides all the logic shared across every backend:
      - Chromium launch and teardown
      - Clipboard-aware prompt delivery
      - Stable-read response polling
      - The ask() lock/deadlock-avoidance pattern

    Concrete backends subclass BaseBrowserBackend and only need to define:
      - label                str          name used in logs and the model card
      - URL                  str          page to navigate to
      - INPUT_SELECTORS      list[str]    ordered selectors for the input box
      - SEND_SELECTORS       list[str]    ordered selectors for the send button
      - RESPONSE_SELECTORS   list[str]    selectors matching response bubbles
      - GENERATING_SELECTORS list[str]    selectors visible while generating
      - CONSENT_SELECTORS    list[str]    selectors for consent dialogs (optional)

    Optional class-level overrides:
      - DEFAULT_HEADLESS     bool         per-backend headless default (True)
      - MINIMIZE_WINDOW      bool         minimize the window after launch (False)
                                          Only meaningful when headless=False.
      - EXTRA_LAUNCH_ARGS    list[str]    additional Chromium launch args

Adding a new backend
--------------------
1. Create browser_ai/backends/<n>.py
2. Subclass BaseBrowserBackend
3. Set the class-level selector attributes above
4. Register in browser_ai/backends/__init__.py
5. Add to --backend choices in cli.py
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from playwright.async_api import async_playwright

from browser_ai.cleaning import clean_response_text, extract_clean_response_text
from browser_ai.config import (
    CLIPBOARD_PASTE_THRESHOLD,
    LAUNCH_TIMEOUT_MS,
    POLL_INTERVAL_S,
    RESPONSE_TIMEOUT_MS,
    STABLE_READS,
)


# =============================================================================
# Layer 1 — Public interface
# =============================================================================

class BrowserBackend(ABC):
    """
    Minimal public interface for a browser-backed LLM session.

    The session manager and all route handlers program against this interface
    exclusively.  They never import a concrete backend class.
    """

    #: Short name used in log messages, the model card, and the CLI.
    label: str = "base"

    #: Per-backend headless default.  Overridden by the CLI --headless /
    #: --no-headless flags at runtime.  Set to False for backends (e.g.
    #: ChatGPT) that don't work reliably in headless Chromium.
    DEFAULT_HEADLESS: bool = True

    @abstractmethod
    async def start(self) -> None:
        """
        Launch the browser and navigate to the LLM UI.

        Idempotent — safe to call multiple times.  Subsequent calls after a
        successful start are no-ops.
        """

    @abstractmethod
    async def ask(self, prompt: str) -> str:
        """
        Submit *prompt* to the LLM and return the complete response text.

        Blocks until the response is stable.  Implementations must serialise
        concurrent calls with an asyncio.Lock so a single page is never used
        by two coroutines simultaneously.
        """

    @abstractmethod
    async def close(self) -> None:
        """
        Tear down the browser session and release all resources.

        Idempotent.
        """

    async def reset(self) -> None:
        """
        Close and reopen the browser session from scratch.

        Backends may override this with a lighter approach (e.g. clicking a
        'New chat' button) when available.
        """
        await self.close()
        await self.start()


# =============================================================================
# Layer 2 — Shared machinery
# =============================================================================

class BaseBrowserBackend(BrowserBackend):
    """
    Concrete base class providing all shared browser automation machinery.

    Subclasses declare selector lists and a URL; this class handles everything
    else.  See module docstring for the full list of attributes to define.
    """

    # ── Subclasses must define these ──────────────────────────────────────────

    URL: str = ""
    INPUT_SELECTORS: List[str] = []
    SEND_SELECTORS: List[str] = []
    RESPONSE_SELECTORS: List[str] = []
    GENERATING_SELECTORS: List[str] = []
    CONSENT_SELECTORS: List[str] = []

    # ── Subclasses may override these ─────────────────────────────────────────

    DEFAULT_HEADLESS: bool = True

    # When True the browser window is launched visible but immediately
    # minimized.  This side-steps headless detection while keeping the window
    # out of the user's way.  Only takes effect when headless=False.
    MINIMIZE_WINDOW: bool = False

    # Additional Chromium launch args appended to the shared baseline set.
    # Backends that need special flags (e.g. --window-position) add them here.
    EXTRA_LAUNCH_ARGS: List[str] = []

    # ── Internal state ────────────────────────────────────────────────────────

    def __init__(self, headless: Optional[bool] = None) -> None:
        # None → use the backend's own DEFAULT_HEADLESS preference.
        # Explicit True/False → CLI override wins.
        if headless is None:
            self.headless = self.DEFAULT_HEADLESS
        else:
            self.headless = headless
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._lock = asyncio.Lock()
        self._started = False

    @property
    def page(self):
        return self._page

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch Chromium and navigate to the backend URL. Idempotent."""
        if self._started and self._page is not None:
            return

        async with self._lock:
            if self._started and self._page is not None:
                return

            mode = "headless" if self.headless else (
                "minimized" if self.MINIMIZE_WINDOW else "visible"
            )
            print(f"[{self.label}] Launching Chromium ({mode})...", flush=True)

            self._playwright = await async_playwright().start()

            launch_args = [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-dev-shm-usage",
            ] + self.EXTRA_LAUNCH_ARGS

            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=launch_args,
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="en-US",
            )
            self._page = await self._context.new_page()
            await self._goto_and_prepare()
            self._started = True
            print(f"[{self.label}] Ready.", flush=True)

    async def close(self) -> None:
        """Tear down the browser session. Idempotent."""
        async with self._lock:
            for attr, method in [
                ("_context", "close"),
                ("_browser", "close"),
                ("_playwright", "stop"),
            ]:
                obj = getattr(self, attr, None)
                if obj is not None:
                    try:
                        await getattr(obj, method)()
                    except Exception:
                        pass
            self._playwright = None
            self._browser = None
            self._context = None
            self._page = None
            self._started = False

    # ── Navigation ────────────────────────────────────────────────────────────

    async def _goto_and_prepare(self) -> None:
        """
        Navigate to self.URL, dismiss consent dialogs, and wait for input.

        Backends with unusual loading behaviour may override this method.
        """
        assert self._page is not None

        if not self.URL:
            raise RuntimeError(
                f"[{self.label}] URL is not set. "
                "Did you forget to define URL on your backend class?"
            )

        print(f"[{self.label}] Navigating to {self.URL}...", flush=True)
        await self._page.goto(
            self.URL,
            wait_until="domcontentloaded",
            timeout=LAUNCH_TIMEOUT_MS,
        )

        # Dismiss consent / cookie dialogs
        for selector in self.CONSENT_SELECTORS:
            try:
                btn = self._page.locator(selector).first
                if await btn.is_visible(timeout=3_000):
                    print(
                        f"[{self.label}] Dismissing consent dialog: {selector}",
                        flush=True,
                    )
                    await btn.click()
                    await self._page.wait_for_timeout(800)
                    break
            except Exception:
                pass

        if not self.INPUT_SELECTORS:
            raise RuntimeError(
                f"[{self.label}] INPUT_SELECTORS is empty — "
                "cannot determine when the page is ready."
            )

        combined_input = ", ".join(self.INPUT_SELECTORS)
        print(f"[{self.label}] Waiting for chat input...", flush=True)
        await self._page.wait_for_selector(combined_input, timeout=LAUNCH_TIMEOUT_MS)
        print(f"[{self.label}] Chat input found.", flush=True)

    # ── Prompt delivery ───────────────────────────────────────────────────────

    async def _deliver_prompt(self, pg, input_box, prompt: str) -> None:
        """
        Type or paste *prompt* into *input_box*.

        Short prompts use keyboard.type() with a per-character delay.
        Long prompts use clipboard paste to avoid multi-minute delays for
        large file-context payloads from editors like Continue.

        In headless mode the async Clipboard API is always denied by Chromium,
        so we go straight to the execCommand textarea trick.  In visible mode
        we try the modern API first and fall back if denied.

        The clipboard is cleared after pasting so prompt text does not persist
        on the host system's clipboard.
        """
        char_count = len(prompt)

        if char_count < CLIPBOARD_PASTE_THRESHOLD:
            print(
                f"[{self.label}] Typing prompt ({char_count} chars, keyboard mode)...",
                flush=True,
            )
            await pg.keyboard.type(prompt, delay=18)
            return

        print(
            f"[{self.label}] Pasting prompt ({char_count} chars, clipboard mode)...",
            flush=True,
        )

        async def _write_clipboard(text: str) -> None:
            await pg.evaluate(
                """(text) => {
                    const el = document.createElement('textarea');
                    el.value = text;
                    el.style.position = 'fixed';
                    el.style.opacity = '0';
                    document.body.appendChild(el);
                    el.focus();
                    el.select();
                    document.execCommand('copy');
                    document.body.removeChild(el);
                }""",
                text,
            )

        if not self.headless:
            try:
                await pg.evaluate(
                    "(text) => navigator.clipboard.writeText(text)", prompt
                )
            except Exception as exc:
                print(
                    f"[{self.label}] clipboard.writeText failed ({exc}); "
                    "using execCommand fallback.",
                    flush=True,
                )
                await _write_clipboard(prompt)
        else:
            await _write_clipboard(prompt)

        await input_box.click()
        await pg.keyboard.press("Control+V")
        await pg.wait_for_timeout(400)

        try:
            await _write_clipboard("")
        except Exception:
            pass

    # ── Response polling ──────────────────────────────────────────────────────

    async def _wait_for_response(self, prev_response_count: int) -> str:
        """
        Poll the page until a new, stable response appears.

        Waits for the response count to exceed prev_response_count, then
        requires STABLE_READS consecutive identical reads with no active
        generating indicator before returning the cleaned response text.
        """
        assert self._page is not None

        response_sel = ", ".join(self.RESPONSE_SELECTORS) if self.RESPONSE_SELECTORS else ""
        generating_sel = ", ".join(self.GENERATING_SELECTORS) if self.GENERATING_SELECTORS else ""

        deadline = time.time() + RESPONSE_TIMEOUT_MS / 1000
        last_text = ""
        stable_count = 0

        print(
            f"[{self.label}] Waiting for response "
            f"(prev_count={prev_response_count})...",
            flush=True,
        )

        while time.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)

            is_generating = False
            if generating_sel:
                try:
                    is_generating = (
                        await self._page.locator(generating_sel).count() > 0
                    )
                except Exception:
                    pass

            responses = []
            if response_sel:
                try:
                    responses = await self._page.locator(response_sel).all()
                except Exception:
                    pass

            # Per-backend hook: dismiss popups that may have appeared
            # mid-response (e.g. ChatGPT login nudge).
            await self._dismiss_interstitials(self._page)

            if len(responses) <= prev_response_count:
                continue

            try:
                current_text = await extract_clean_response_text(responses[-1])
            except Exception:
                current_text = ""

            if current_text and not is_generating:
                if current_text == last_text:
                    stable_count += 1
                    if stable_count >= STABLE_READS:
                        # Generation is stable. Give the backend a chance to
                        # reveal hidden source text (e.g. click Code-view
                        # toggles in ChatGPT) before the final extraction.
                        try:
                            await self._prepare_response_element(responses[-1])
                            # Re-extract after DOM manipulation.
                            final_text = await extract_clean_response_text(responses[-1])
                        except Exception:
                            final_text = current_text
                        print(
                            f"[{self.label}] Response stable after "
                            f"{stable_count} reads.",
                            flush=True,
                        )
                        return final_text
                else:
                    last_text = current_text
                    stable_count = 1
            else:
                last_text = current_text
                stable_count = 0

        print(f"[{self.label}] Response timed out.", flush=True)
        return (
            clean_response_text(last_text)
            if last_text
            else f"[browser-ai: {self.label} response timed out]"
        )

    async def _post_deliver_wait(self, pg) -> None:
        """
        Optional hook called after prompt delivery and before send-button
        search.  Default is a no-op.  Override in backends whose UI reveals
        the send button asynchronously after input (e.g. Grok).
        """
        pass

    async def _dismiss_interstitials(self, pg) -> None:
        """
        Optional hook to dismiss mid-session modals/popups.
        Called once per polling iteration in _wait_for_response.
        Default is a no-op.  Override in backends that show interstitials
        (e.g. ChatGPT login nudge).
        """
        pass

    async def _prepare_response_element(self, element) -> None:
        """
        Optional hook called once on the final response element, immediately
        before text is extracted, after generation is confirmed stable.

        Use this to perform any DOM manipulation that reveals hidden source
        text before the JS walker runs.  The canonical use-case is clicking
        "Code" view toggles in ChatGPT's mermaid/code-block renderer, which
        renders diagrams as SVG images and hides the raw source unless the
        Code tab is explicitly selected.

        Default is a no-op.  Backends override as needed.  Failures are
        silently swallowed so that a broken toggle never aborts extraction.
        """
        pass

    # ── Public interface ──────────────────────────────────────────────────────

    async def ask(self, prompt: str) -> str:
        """
        Submit *prompt* to the LLM and return the scraped response.

        start() is called before the lock is acquired to avoid a self-deadlock
        (start() acquires self._lock internally).
        """
        await self.start()

        async with self._lock:
            assert self._page is not None
            pg = self._page

            if not pg.url or pg.url == "about:blank":
                await self._goto_and_prepare()

            response_sel = ", ".join(self.RESPONSE_SELECTORS) if self.RESPONSE_SELECTORS else ""
            existing_responses = 0
            if response_sel:
                try:
                    existing_responses = await pg.locator(response_sel).count()
                except Exception:
                    pass
            print(f"[{self.label}] Existing response count: {existing_responses}", flush=True)

            # Locate the chat input
            input_box = None
            for sel in self.INPUT_SELECTORS:
                try:
                    candidate = pg.locator(sel).first
                    if await candidate.is_visible(timeout=2_000):
                        input_box = candidate
                        print(
                            f"[{self.label}] Input found via selector: {sel}",
                            flush=True,
                        )
                        break
                except Exception:
                    continue

            if input_box is None:
                raise RuntimeError(
                    f"[{self.label}] Could not locate chat input box. "
                    "Check INPUT_SELECTORS for this backend."
                )

            await input_box.click()
            await pg.wait_for_timeout(300)

            # Clear any stale text
            try:
                await input_box.fill("")
            except Exception:
                try:
                    await pg.keyboard.press("Control+A")
                    await pg.keyboard.press("Backspace")
                except Exception:
                    pass

            await self._deliver_prompt(pg, input_box, prompt)
            await pg.wait_for_timeout(300)

            # Optional per-backend hook: extra wait after delivery so UIs
            # that reveal their send button asynchronously have time to do so.
            await self._post_deliver_wait(pg)

            # Find and click the send button, or fall back to Enter
            sent = False
            for btn_sel in self.SEND_SELECTORS:
                try:
                    btn = pg.locator(btn_sel).first
                    if (
                        await btn.is_visible(timeout=1_500)
                        and await btn.is_enabled(timeout=1_500)
                    ):
                        await btn.click()
                        sent = True
                        print(
                            f"[{self.label}] Sent via button: {btn_sel}",
                            flush=True,
                        )
                        break
                except Exception:
                    continue

            if not sent:
                print(
                    f"[{self.label}] Send button not found; submitting via Enter.",
                    flush=True,
                )
                await pg.keyboard.press("Enter")

            return await self._wait_for_response(existing_responses)