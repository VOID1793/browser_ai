"""
browser_ai.backends.gemini
~~~~~~~~~~~~~~~~~~~~~~~~~~
Browser backend for Google Gemini (https://gemini.google.com/app).

This is the reference implementation and the only fully working backend.
All automation logic was developed and tested against the Gemini web UI.
"""

from __future__ import annotations

import asyncio
import time

from playwright.async_api import async_playwright

from browser_ai.backends.base import BrowserBackend
from browser_ai.cleaning import extract_clean_response_text, clean_response_text
from browser_ai.config import (
    CLIPBOARD_PASTE_THRESHOLD,
    LAUNCH_TIMEOUT_MS,
    POLL_INTERVAL_S,
    RESPONSE_TIMEOUT_MS,
    STABLE_READS,
)

GEMINI_URL = "https://gemini.google.com/app"


class GeminiBackend(BrowserBackend):
    """Playwright-backed browser session for Google Gemini."""

    #: Human-readable name used in log messages and the model card.
    label: str = "gemini"

    def __init__(self, headless: bool = True) -> None:
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
        """Launch Chromium and navigate to Gemini. Idempotent."""
        if self._started and self._page is not None:
            return

        async with self._lock:
            if self._started and self._page is not None:
                return

            print("[gemini] Launching Chromium...", flush=True)
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                ],
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
            print("[gemini] Ready.", flush=True)

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
        assert self._page is not None
        print(f"[gemini] Navigating to {GEMINI_URL}...", flush=True)
        await self._page.goto(
            GEMINI_URL,
            wait_until="domcontentloaded",
            timeout=LAUNCH_TIMEOUT_MS,
        )

        # Dismiss consent / cookie dialogs
        for selector in [
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Agree")',
            '[aria-label="Accept all"]',
        ]:
            try:
                btn = self._page.locator(selector).first
                if await btn.is_visible(timeout=3_000):
                    print(f"[gemini] Dismissing consent dialog: {selector}", flush=True)
                    await btn.click()
                    await self._page.wait_for_timeout(800)
                    break
            except Exception:
                pass

        input_selector = (
            'rich-textarea, textarea, '
            '[contenteditable="true"][aria-label*="message"], '
            '[contenteditable="true"][placeholder], '
            'div[role="textbox"]'
        )
        print("[gemini] Waiting for chat input...", flush=True)
        await self._page.wait_for_selector(input_selector, timeout=LAUNCH_TIMEOUT_MS)
        print("[gemini] Chat input found.", flush=True)

    # ── Prompt delivery ───────────────────────────────────────────────────────

    async def _deliver_prompt(self, pg, input_box, prompt: str) -> None:
        """
        Type or paste *prompt* into *input_box*.

        Short prompts use keyboard.type() with a small per-character delay.
        Long prompts use clipboard paste (execCommand fallback in headless mode)
        to avoid multi-minute delays for large file-context payloads.
        """
        char_count = len(prompt)

        if char_count < CLIPBOARD_PASTE_THRESHOLD:
            print(f"[gemini] Typing prompt ({char_count} chars, keyboard mode)...", flush=True)
            await pg.keyboard.type(prompt, delay=18)
            return

        print(f"[gemini] Pasting prompt ({char_count} chars, clipboard mode)...", flush=True)

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
                await pg.evaluate("(text) => navigator.clipboard.writeText(text)", prompt)
            except Exception as exc:
                print(f"[gemini] clipboard.writeText failed ({exc}); using execCommand fallback.", flush=True)
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
        assert self._page is not None

        deadline = time.time() + RESPONSE_TIMEOUT_MS / 1000
        last_text = ""
        stable_count = 0

        print(f"[gemini] Waiting for response (prev_count={prev_response_count})...", flush=True)

        while time.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)

            try:
                is_generating = await self._page.locator(
                    '[data-test-id="stop-generating-button"], '
                    'button[aria-label*="Stop"], '
                    'mat-spinner'
                ).count() > 0
            except Exception:
                is_generating = False

            try:
                responses = await self._page.locator(
                    "model-response, "
                    "[data-response-index], "
                    ".model-response-text, "
                    "message-content.model-response-text"
                ).all()
            except Exception:
                responses = []

            if not responses:
                try:
                    responses = await self._page.locator(".response-content, .markdown").all()
                except Exception:
                    responses = []

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
                        print(f"[gemini] Response stable after {stable_count} reads.", flush=True)
                        return current_text
                else:
                    last_text = current_text
                    stable_count = 1
            else:
                last_text = current_text
                stable_count = 0

        print("[gemini] Response timed out.", flush=True)
        return clean_response_text(last_text) if last_text else "[browser-ai: response timed out]"

    # ── Public interface ──────────────────────────────────────────────────────

    async def ask(self, prompt: str) -> str:
        """
        Submit *prompt* to Gemini and return the scraped response.

        Starts the browser if not already running.  Serialises concurrent
        calls with self._lock so a single page is never used by two coroutines
        simultaneously.

        NOTE: start() acquires self._lock internally.  We call start() BEFORE
        acquiring the lock here to avoid a self-deadlock.
        """
        await self.start()

        async with self._lock:
            assert self._page is not None
            pg = self._page

            if pg.url == "about:blank":
                await self._goto_and_prepare()

            existing_responses = await pg.locator(
                "model-response, [data-response-index], "
                ".model-response-text, message-content.model-response-text"
            ).count()
            print(f"[gemini] Existing response count: {existing_responses}", flush=True)

            # Locate the chat input
            input_box = None
            for sel in [
                'rich-textarea div[contenteditable="true"]',
                'div[contenteditable="true"][aria-label*="message"]',
                'div[contenteditable="true"][aria-label*="prompt"]',
                'div[contenteditable="true"][aria-label*="Enter"]',
                'div[contenteditable="true"]',
                "textarea",
            ]:
                try:
                    candidate = pg.locator(sel).first
                    if await candidate.is_visible(timeout=2_000):
                        input_box = candidate
                        print(f"[gemini] Input found via selector: {sel}", flush=True)
                        break
                except Exception:
                    continue

            if input_box is None:
                raise RuntimeError("[gemini] Could not locate chat input box")

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

            # Submit
            sent = False
            for btn_sel in [
                'button[aria-label*="Send"]',
                'button[aria-label*="send"]',
                'button[data-test-id="send-button"]',
                'button.send-button',
                'button[jsname="V67aGc"]',
            ]:
                try:
                    btn = pg.locator(btn_sel).first
                    if await btn.is_visible(timeout=1_500) and await btn.is_enabled(timeout=1_500):
                        await btn.click()
                        sent = True
                        print(f"[gemini] Sent via button: {btn_sel}", flush=True)
                        break
                except Exception:
                    continue

            if not sent:
                print("[gemini] Send button not found; submitting via Enter.", flush=True)
                await pg.keyboard.press("Enter")

            return await self._wait_for_response(existing_responses)
