#!/usr/bin/env python3
"""
gemini_openai_server.py

Async Playwright-backed Gemini browser automation with an OpenAI-compatible API.

Endpoints:
    GET  /healthz
    GET  /v1/models
    POST /v1/chat/completions

Examples:
    # Start server
    python gemini_openai_server.py serve --host 127.0.0.1 --port 8000

    # Visible browser for debugging
    python gemini_openai_server.py serve --visible

    # Single prompt from CLI
    python gemini_openai_server.py chat "Explain monads simply"

    # Interactive session
    python gemini_openai_server.py chat --session

Requirements:
    pip install fastapi uvicorn pydantic playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

# ── Tunables ──────────────────────────────────────────────────────────────────
GEMINI_URL = "https://gemini.google.com/app"

LAUNCH_TIMEOUT_MS = 30_000
RESPONSE_TIMEOUT_MS = 180_000   # 3 min — file-context answers can be slow
POLL_INTERVAL_S = 0.8
STABLE_READS = 3
HEADLESS = True

# Prompts longer than this threshold are delivered via clipboard paste
# instead of keyboard.type(), avoiding multi-minute input delays for
# large file-context payloads from editors like Continue.
CLIPBOARD_PASTE_THRESHOLD = 200  # characters

SESSION_IDLE_TTL_S = 30 * 60       # close idle browser sessions after 30 min
SESSION_CLEANUP_INTERVAL_S = 60    # check every minute

MODEL_ID = "gemini-browser"
SYSTEM_FINGERPRINT = "gemini-browser-playwright-v1"

# UI junk lines to strip from scraped responses
_JUNK_LINES: frozenset[str] = frozenset({
    "export to sheets",
    "copy",
    "retry",
    "share",
    "thumbs up",
    "thumbs down",
    "more options",
    "report a problem",
    "show drafts",
    "hide drafts",
})
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Utility helpers
# =============================================================================

def now_ts() -> int:
    return int(time.time())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    # Rough heuristic only; NOT true provider token counts.
    # Good enough for OpenAI-compatible usage metadata fields.
    if not text:
        return 0
    return max(1, len(text) // 4)


def canonical_messages(messages: List[Dict[str, Any]]) -> str:
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def extract_text_content(content: Any) -> str:
    """
    Normalize OpenAI-style content:
      - plain string
      - array of {"type": "text", "text": "..."} parts
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()

    return str(content).strip()


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = extract_text_content(m.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def build_transcript_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Fallback strategy when we can't incrementally continue the browser session.
    We serialize the conversation into a single prompt.
    """
    system_parts: List[str] = []
    convo_parts: List[str] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
        elif role == "developer":
            system_parts.append(content)
        elif role == "user":
            convo_parts.append(f"User:\n{content}")
        elif role == "assistant":
            convo_parts.append(f"Assistant:\n{content}")
        elif role == "tool":
            convo_parts.append(f"Tool:\n{content}")
        else:
            convo_parts.append(f"{role.capitalize()}:\n{content}")

    system_block = ""
    if system_parts:
        system_block = (
            "System instructions:\n"
            + "\n\n".join(system_parts).strip()
            + "\n\n"
        )

    convo_block = "\n\n".join(convo_parts).strip()

    return (
        f"{system_block}"
        "Continue the following conversation faithfully. "
        "Respond as the assistant to the most recent user request.\n\n"
        f"{convo_block}"
    ).strip()


def build_incremental_prompt(
    previous_messages: Optional[List[Dict[str, str]]],
    current_messages: List[Dict[str, str]],
) -> Tuple[str, bool]:
    """
    Returns: (prompt, reset_required)

    Strategy:
      - If no previous history exists, send a shaped prompt containing system+latest user.
      - If current history cleanly extends previous history, send only the newest user text.
      - Otherwise, request session reset and use transcript fallback.
    """
    if not current_messages:
        return "Hello.", False

    # First request: fold system/developer into first prompt if present
    if not previous_messages:
        system_parts = [
            m["content"] for m in current_messages
            if m["role"] in {"system", "developer"} and m["content"].strip()
        ]
        latest_user = ""
        for m in reversed(current_messages):
            if m["role"] == "user" and m["content"].strip():
                latest_user = m["content"].strip()
                break

        if not latest_user:
            return build_transcript_prompt(current_messages), False

        if system_parts:
            prompt = (
                "Please follow these instructions for this conversation:\n"
                + "\n\n".join(system_parts).strip()
                + "\n\n"
                + latest_user
            )
            return prompt, False

        return latest_user, False

    # If history is identical, nothing new; just resend last user as a fallback
    if previous_messages == current_messages:
        for m in reversed(current_messages):
            if m["role"] == "user" and m["content"].strip():
                return m["content"].strip(), False
        return build_transcript_prompt(current_messages), False

    # Prefix extension case
    if (
        len(current_messages) >= len(previous_messages)
        and current_messages[: len(previous_messages)] == previous_messages
    ):
        delta = current_messages[len(previous_messages):]

        # Safe incremental path only if delta contains a new user turn and no tool/system/developer
        bad_roles = {"system", "developer", "tool"}
        if any(m["role"] in bad_roles for m in delta):
            return build_transcript_prompt(current_messages), True

        for m in reversed(delta):
            if m["role"] == "user" and m["content"].strip():
                return m["content"].strip(), False

        # No new user content found -> fallback
        return build_transcript_prompt(current_messages), True

    # History diverged; safest approach is reset + transcript prompt
    return build_transcript_prompt(current_messages), True


# =============================================================================
# Response text cleaning
# =============================================================================

def clean_response_text(raw: str) -> str:
    """
    Clean a plain-text string scraped from Gemini's response container.

    Steps:
      1. Two-pass HTML entity unescaping.
         Gemini renders some characters (especially inside code blocks) as
         literal HTML entity text on screen — e.g. the browser displays the
         five characters '&gt;' rather than '>'. inner_text() faithfully
         returns those five characters. A single html.unescape() pass fixes
         normal entities; a second pass catches double-escaped cases like
         '&amp;gt;' → '&gt;' → '>'.
      2. Remove known UI junk lines (case-insensitive exact match).
      3. Collapse runs of 3+ consecutive blank lines to 2.
    """
    if not raw:
        return raw

    # 1. Two-pass unescape to handle both singly- and doubly-escaped entities
    text = html.unescape(html.unescape(raw))

    # 2. Remove junk lines
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in _JUNK_LINES:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 3. Collapse runs of more than 2 consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# JavaScript injected into the page to walk the DOM and reconstruct
# markdown-style output, preserving code blocks with fences.
# This runs in the browser context so we get structural information
# that inner_text() loses (e.g. which text was inside <pre><code>).
_EXTRACT_JS = """
(element) => {
    // Collect all <pre> elements upfront so we can suppress their preceding
    // language-label siblings when we encounter them as text nodes.
    const allPres = Array.from(element.querySelectorAll('pre'));

    // Build a Set of DOM nodes we should skip because we've already consumed
    // them as part of a code block's language label.
    const suppressedNodes = new Set();

    for (const pre of allPres) {
        // Walk backwards through preceding siblings looking for a language label.
        // Gemini typically emits the label as a plain text node or a span/div
        // immediately before the <pre> inside the same parent wrapper div.
        let sib = pre.previousSibling;
        while (sib) {
            const sibText = sib.textContent ? sib.textContent.trim() : '';
            if (sibText) {
                // Mark this sibling for suppression — we'll consume it in walk()
                suppressedNodes.add(sib);
                break;
            }
            sib = sib.previousSibling;
        }

        // Also suppress dedicated label elements inside the parent
        const parent = pre.parentElement;
        if (parent) {
            const labelEl = parent.querySelector(
                '.code-block-decoration, .language-label, [class*="lang"]'
            );
            if (labelEl) suppressedNodes.add(labelEl);
        }
    }

    function getLangForPre(pre) {
        // 1. Check for a data attribute on the <pre> or its parent
        if (pre.dataset && pre.dataset.lang) return pre.dataset.lang.toLowerCase();
        const parent = pre.parentElement;
        if (parent && parent.dataset && parent.dataset.lang) {
            return parent.dataset.lang.toLowerCase();
        }

        // 2. Check for a class like 'language-python' on the inner <code>
        const codeEl = pre.querySelector('code');
        if (codeEl) {
            for (const cls of codeEl.classList) {
                if (cls.startsWith('language-')) return cls.slice(9).toLowerCase();
            }
        }

        // 3. Use the text of the suppressed preceding sibling
        let sib = pre.previousSibling;
        while (sib) {
            const sibText = sib.textContent ? sib.textContent.trim() : '';
            if (sibText) return sibText.toLowerCase();
            sib = sib.previousSibling;
        }

        // 4. Check a dedicated label element inside the parent
        if (parent) {
            const labelEl = parent.querySelector(
                '.code-block-decoration, .language-label, [class*="lang"]'
            );
            if (labelEl) return labelEl.textContent.trim().toLowerCase();
        }

        return '';
    }

    function walk(node) {
        // Skip nodes we've already accounted for as language labels
        if (suppressedNodes.has(node)) return '';

        // Text node: return its text content directly
        if (node.nodeType === Node.TEXT_NODE) {
            return node.textContent;
        }

        if (node.nodeType !== Node.ELEMENT_NODE) {
            return '';
        }

        const tag = node.tagName.toLowerCase();

        // Skip known UI chrome elements entirely
        const junkTags = ['button', 'mat-icon', 'svg', 'tool-use'];
        if (junkTags.includes(tag)) return '';

        // Code block: wrap in markdown fences
        if (tag === 'pre') {
            const codeEl = node.querySelector('code');
            const codeText = codeEl ? codeEl.innerText : node.innerText;
            const lang = getLangForPre(node);
            return '\\n```' + lang + '\\n' + codeText.trimEnd() + '\\n```\\n';
        }

        // Inline code (but not when it's inside a <pre> — handled above)
        if (tag === 'code') {
            return '`' + node.innerText + '`';
        }

        // Headings
        if (/^h[1-6]$/.test(tag)) {
            const level = parseInt(tag[1]);
            const prefix = '#'.repeat(level);
            return '\\n' + prefix + ' ' + node.innerText.trim() + '\\n';
        }

        // Block elements
        const blockTags = ['p', 'div', 'li', 'tr', 'blockquote', 'br'];
        const isBlock = blockTags.includes(tag);

        // Recurse into children
        let result = '';
        for (const child of node.childNodes) {
            result += walk(child);
        }

        if (tag === 'br') return '\\n';
        if (tag === 'li') return '\\n- ' + result.trim();
        if (tag === 'tr') return result + '\\n';
        if (isBlock && result.trim()) return '\\n' + result.trim() + '\\n';

        return result;
    }

    return walk(element);
}
"""


async def extract_clean_response_text(element) -> str:
    """
    Extract clean markdown-friendly text from a Gemini response DOM element.

    Strategy (in order):
      1. Run _EXTRACT_JS in the browser to walk the DOM and reconstruct
         code fences, headings, etc. from structural tags.
      2. If JS extraction yields nothing useful, fall back to inner_text()
         on a tighter child container if available.
      3. Last resort: inner_text() on the element itself.

    All paths run through clean_response_text() for entity unescaping
    and junk-line removal.
    """
    # 1. Try JS-based structural extraction first
    try:
        raw = await element.evaluate(_EXTRACT_JS)
        if raw and raw.strip():
            return clean_response_text(raw)
    except Exception as exc:
        print(f"[browser] JS extraction failed: {exc}", flush=True)

    # 2. Try a tighter child container via inner_text()
    for child_sel in [
        ".response-content",
        ".markdown",
        "model-response-text",
        ".model-response-text__content",
    ]:
        try:
            child = element.locator(child_sel).first
            if await child.count() > 0:
                raw = (await child.inner_text()).strip()
                if raw:
                    return clean_response_text(raw)
        except Exception:
            pass

    # 3. Fallback: full element inner_text()
    try:
        raw = (await element.inner_text()).strip()
        return clean_response_text(raw)
    except Exception:
        return ""


# =============================================================================
# OpenAI-compatible request/response models
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


# =============================================================================
# Gemini browser session (async Playwright)
# =============================================================================

class GeminiBrowserSession:
    def __init__(self, headless: bool = True):
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

    async def start(self) -> None:
        """
        Ensure the browser is launched and Gemini is loaded.
        Safe to call multiple times; is a no-op if already started.
        Uses a dedicated flag to avoid re-entering start() from ask().
        """
        if self._started and self._page is not None:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self._started and self._page is not None:
                return

            print("[browser] Launching Chromium...", flush=True)
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
            print("[browser] Ready.", flush=True)

    async def _goto_and_prepare(self) -> None:
        assert self._page is not None
        print(f"[browser] Navigating to {GEMINI_URL}...", flush=True)
        await self._page.goto(
            GEMINI_URL,
            wait_until="domcontentloaded",
            timeout=LAUNCH_TIMEOUT_MS,
        )

        # Dismiss common consent/cookie dialogs
        for selector in [
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Agree")',
            '[aria-label="Accept all"]',
        ]:
            try:
                btn = self._page.locator(selector).first
                if await btn.is_visible(timeout=3_000):
                    print(f"[browser] Dismissing consent dialog: {selector}", flush=True)
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
        print("[browser] Waiting for chat input...", flush=True)
        await self._page.wait_for_selector(input_selector, timeout=LAUNCH_TIMEOUT_MS)
        print("[browser] Chat input found.", flush=True)

    async def close(self) -> None:
        async with self._lock:
            try:
                if self._context is not None:
                    await self._context.close()
            except Exception:
                pass
            try:
                if self._browser is not None:
                    await self._browser.close()
            except Exception:
                pass
            try:
                if self._playwright is not None:
                    await self._playwright.stop()
            except Exception:
                pass
            finally:
                self._playwright = None
                self._browser = None
                self._context = None
                self._page = None
                self._started = False

    async def reset(self) -> None:
        print("[browser] Resetting session...", flush=True)
        await self.close()
        await self.start()

    async def _wait_for_response(self, prev_response_count: int) -> str:
        assert self._page is not None

        deadline = time.time() + RESPONSE_TIMEOUT_MS / 1000
        last_text = ""
        stable_count = 0

        print(f"[browser] Waiting for response (prev_count={prev_response_count})...", flush=True)

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
                # Use improved extraction that tries tighter child nodes first
                current_text = await extract_clean_response_text(responses[-1])
            except Exception:
                current_text = ""

            if current_text and not is_generating:
                if current_text == last_text:
                    stable_count += 1
                    if stable_count >= STABLE_READS:
                        print(f"[browser] Response stable after {stable_count} reads.", flush=True)
                        return current_text
                else:
                    last_text = current_text
                    stable_count = 1
            else:
                last_text = current_text
                stable_count = 0

        print("[browser] Response timed out.", flush=True)
        return clean_response_text(last_text) if last_text else "[gemini_openai_server: response timed out]"

    async def _deliver_prompt(self, pg, input_box, prompt: str) -> None:
        """
        Deliver a prompt string into the Gemini input box.

        Strategy:
          - Short prompts (< CLIPBOARD_PASTE_THRESHOLD chars): use keyboard.type()
            with a small per-character delay. This is the most reliable path for
            short text and avoids clipboard race conditions.
          - Long prompts (>= CLIPBOARD_PASTE_THRESHOLD chars): write the text to
            the system clipboard via JavaScript and paste with Ctrl+V. This is
            near-instantaneous regardless of prompt length, making file-context
            payloads from editors like Continue viable.

        Both paths clear the input box first. The clipboard path restores the
        clipboard to an empty string after pasting so we don't leave potentially
        sensitive prompt text sitting on the user's clipboard.
        """
        char_count = len(prompt)

        if char_count < CLIPBOARD_PASTE_THRESHOLD:
            print(f"[browser] Typing prompt ({char_count} chars, keyboard mode)...", flush=True)
            await pg.keyboard.type(prompt, delay=18)
            return

        print(f"[browser] Pasting prompt ({char_count} chars, clipboard mode)...", flush=True)

        # Write prompt to clipboard via browser JS, then paste with Ctrl+V.
        #
        # navigator.clipboard.writeText() (async Clipboard API) requires a
        # user-gesture context and clipboard-write permission. Headless Chromium
        # denies this unconditionally, so we skip straight to the execCommand
        # textarea trick in headless mode. In visible mode we try the modern API
        # first and fall back if it is denied.
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
            # Visible mode: try the modern Clipboard API first (cleaner, no DOM side-effects)
            try:
                await pg.evaluate("(text) => navigator.clipboard.writeText(text)", prompt)
            except Exception as exc:
                print(f"[browser] clipboard.writeText failed ({exc}); using execCommand fallback.", flush=True)
                await _write_clipboard(prompt)
        else:
            # Headless mode: go straight to execCommand — modern API is always denied here
            await _write_clipboard(prompt)

        # Focus the input and paste
        await input_box.click()
        await pg.keyboard.press("Control+V")

        # Brief pause to let the paste event settle before we check/submit
        await pg.wait_for_timeout(400)

        # Clear clipboard so sensitive prompt text doesn't persist
        try:
            await _write_clipboard("")
        except Exception:
            pass

    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to Gemini and return the scraped response text.

        Starts the browser if not already running.
        Uses self._lock to serialize concurrent asks on the same page.

        NOTE: start() must NOT be called while self._lock is held, because
        start() also acquires self._lock internally. We call start() first
        (it is safe to call multiple times), then acquire the lock for the
        actual page interaction.
        """
        # Start browser BEFORE taking the interaction lock (avoids self-deadlock)
        await self.start()

        async with self._lock:
            assert self._page is not None
            pg = self._page

            if pg.url == "about:blank":
                await self._goto_and_prepare()

            # Count existing responses before we submit
            existing_responses = await pg.locator(
                "model-response, [data-response-index], "
                ".model-response-text, message-content.model-response-text"
            ).count()
            print(f"[browser] Existing response count: {existing_responses}", flush=True)

            # Locate input box
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
                        print(f"[browser] Input found via selector: {sel}", flush=True)
                        break
                except Exception:
                    continue

            if input_box is None:
                raise RuntimeError("Could not locate Gemini chat input box")

            await input_box.click()
            await pg.wait_for_timeout(300)

            # Clear any existing text
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

            # Try to click Send button
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
                        print(f"[browser] Sent via button: {btn_sel}", flush=True)
                        break
                except Exception:
                    continue

            if not sent:
                print("[browser] Send button not found; submitting via Enter.", flush=True)
                await pg.keyboard.press("Enter")

            return await self._wait_for_response(existing_responses)


# =============================================================================
# Session manager
# =============================================================================

@dataclass
class SessionState:
    session_id: str
    browser: GeminiBrowserSession
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    # Stored for future extension (e.g., per-session auth routing).
    # We do NOT validate this token; it's an opaque passthrough.
    auth_header: Optional[str] = None
    previous_messages: Optional[List[Dict[str, str]]] = None


class SessionManager:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start(self) -> None:
        print("[session_manager] Starting cleanup loop.", flush=True)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        print("[session_manager] Shutting down...", flush=True)
        self._closed = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except Exception:
                pass

        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for state in sessions:
            await state.browser.close()

        print("[session_manager] Shutdown complete.", flush=True)

    async def get_or_create(self, session_key: str, auth_header: Optional[str]) -> SessionState:
        async with self._lock:
            state = self._sessions.get(session_key)
            if state is None:
                print(f"[session_manager] Creating new session: {session_key[:12]}...", flush=True)
                browser = GeminiBrowserSession(headless=self.headless)
                state = SessionState(
                    session_id=session_key,
                    browser=browser,
                    auth_header=auth_header,
                )
                self._sessions[session_key] = state
            else:
                print(f"[session_manager] Reusing session: {session_key[:12]}...", flush=True)

            state.last_used_at = time.time()
            return state

    async def reset(self, session_key: str) -> SessionState:
        print(f"[session_manager] Resetting session: {session_key[:12]}...", flush=True)
        async with self._lock:
            old = self._sessions.pop(session_key, None)

        if old:
            await old.browser.close()

        browser = GeminiBrowserSession(headless=self.headless)
        state = SessionState(session_id=session_key, browser=browser)
        async with self._lock:
            self._sessions[session_key] = state
        return state

    async def _cleanup_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL_S)
            cutoff = time.time() - SESSION_IDLE_TTL_S

            stale: List[SessionState] = []
            async with self._lock:
                stale_keys = [
                    key for key, state in self._sessions.items()
                    if state.last_used_at < cutoff
                ]
                for key in stale_keys:
                    print(f"[session_manager] Expiring idle session: {key[:12]}...", flush=True)
                    stale.append(self._sessions.pop(key))

            for state in stale:
                await state.browser.close()


# =============================================================================
# FastAPI app
# =============================================================================

session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_manager
    print("[server] Starting up...", flush=True)
    session_manager = SessionManager(headless=HEADLESS)
    await session_manager.start()
    yield
    if session_manager:
        await session_manager.shutdown()
    print("[server] Shutdown complete.", flush=True)


app = FastAPI(title="Gemini Browser OpenAI-Compatible Server", lifespan=lifespan)


@app.exception_handler(PlaywrightTimeoutError)
async def playwright_timeout_handler(_: Request, exc: PlaywrightTimeoutError):
    return JSONResponse(
        status_code=504,
        content={
            "error": {
                "message": f"Browser automation timed out: {str(exc)}",
                "type": "timeout_error",
                "code": "playwright_timeout",
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_error",
            }
        },
    )


def make_session_key(
    authorization: Optional[str],
    x_session_id: Optional[str],
    openai_user: Optional[str],
) -> str:
    """
    Derive an opaque session key from available identity signals.

    Auth header passthrough behavior:
      - We do NOT validate the Authorization value.
      - We accept it as-is and use it purely for session namespace isolation.
      - The auth_header is stored on SessionState for future extension.
    """
    auth_part = authorization or "anon"
    explicit_session = x_session_id or openai_user or "default"
    raw = f"{auth_part}::{explicit_session}"
    return sha256_text(raw)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": MODEL_ID}


@app.get("/v1/models")
async def list_models():
    # locally synthesized model list — not fetched from upstream Gemini API
    return {
        "object": "list",
        "data": [
            ModelCard(
                id=MODEL_ID,
                created=now_ts(),
            ).model_dump()
        ],
    }


def _sse_chunk(chunk_id: str, created: int, delta_content: str) -> str:
    """
    Format a single SSE data line containing an OpenAI-style stream chunk.
    delta_content=None signals the final [DONE] chunk.
    """
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done_chunk(chunk_id: str, created: int) -> str:
    """Final SSE chunk signalling end of stream."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


async def _stream_response(
    response_text: str,
    chunk_id: str,
    created: int,
    x_session_id: Optional[str],
) -> Any:
    """
    Async generator that yields the full response as a sequence of SSE chunks.

    We already have the complete response from the browser before we start
    streaming, so this is a simulated stream — we chunk the text and yield
    it word-by-word with no artificial delay. Continue and other OpenAI-
    compatible clients will render it as a normal streaming response.
    """
    # Split on word boundaries, preserving whitespace tokens so the
    # reassembled text is identical to the original.
    tokens = re.split(r"(\s+)", response_text)
    for token in tokens:
        if token:
            yield _sse_chunk(chunk_id, created, token)
            # Yield control back to the event loop between chunks so the
            # response actually streams rather than buffering entirely.
            await asyncio.sleep(0)
    yield _sse_done_chunk(chunk_id, created)


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None),
    x_reset_session: Optional[str] = Header(default=None),
):
    global session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")

    normalized = normalize_messages([m.model_dump() for m in body.messages])
    if not normalized:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    session_key = make_session_key(
        authorization=authorization,
        x_session_id=x_session_id,
        openai_user=body.user,
    )

    force_reset = (x_reset_session or "").strip().lower() in {"1", "true", "yes"}
    if force_reset:
        print(f"[server] Force-resetting session {session_key[:12]}...", flush=True)
        state = await session_manager.reset(session_key)
        state.auth_header = authorization
    else:
        state = await session_manager.get_or_create(session_key, authorization)

    prompt, reset_required = build_incremental_prompt(
        previous_messages=state.previous_messages,
        current_messages=normalized,
    )

    # If the history diverged from what the browser session contains,
    # reset the session and replay via a transcript-style prompt
    if reset_required:
        print("[server] History diverged; resetting browser session.", flush=True)
        state = await session_manager.reset(session_key)
        state.auth_header = authorization

    state.last_used_at = time.time()

    try:
        print(f"[server] Calling browser.ask() for session {session_key[:12]}...", flush=True)
        response_text = await asyncio.wait_for(
            state.browser.ask(prompt),
            timeout=RESPONSE_TIMEOUT_MS / 1000 + 15,
        )
        state.previous_messages = normalized
        state.last_used_at = time.time()
        print(f"[server] Got response ({len(response_text)} chars).", flush=True)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out waiting for browser response.",
        )
    except PlaywrightTimeoutError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Estimate tokens heuristically (NOT true provider token counts)
    prompt_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in normalized if m["content"].strip()
    )
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(response_text)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = now_ts()

    resp_headers = {}
    if x_session_id:
        resp_headers["X-Session-Id"] = x_session_id

    if body.stream:
        # Simulated streaming: we have the full response already; chunk it
        # into SSE events so Continue and other streaming clients are happy.
        print(f"[server] Streaming response as SSE ({len(response_text)} chars).", flush=True)
        return StreamingResponse(
            _stream_response(response_text, completion_id, created, x_session_id),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    # Non-streaming response
    # Locally synthesized OpenAI-compatible wrapper fields:
    # - id, object, created, model, system_fingerprint, usage are all local metadata
    # - choices[0].message.content is the true browser-scraped Gemini response
    response_payload = {
        "id": completion_id,                    # locally generated
        "object": "chat.completion",            # locally synthesized
        "created": created,                     # locally generated timestamp
        "model": MODEL_ID,                      # local alias, not upstream model name
        "system_fingerprint": SYSTEM_FINGERPRINT,  # local constant
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",        # assumed; not read from upstream
                "message": {
                    "role": "assistant",
                    # ↓ TRUE browser-scraped content from Gemini
                    "content": response_text,
                },
            }
        ],
        "usage": {
            # Heuristic estimates only (len/4); NOT upstream token counts
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    return JSONResponse(content=response_payload, headers=resp_headers)


# =============================================================================
# CLI helpers
# =============================================================================

async def ask_once(prompt: str, *, visible: bool = False) -> str:
    browser = GeminiBrowserSession(headless=not visible)
    try:
        await browser.start()
        return await browser.ask(prompt)
    finally:
        await browser.close()


async def interactive_chat(*, visible: bool = False) -> None:
    browser = GeminiBrowserSession(headless=not visible)
    try:
        await browser.start()
        print("[gemini_openai_server] Starting interactive session (Ctrl+C to quit)", file=sys.stderr)
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not prompt:
                continue

            try:
                resp = await browser.ask(prompt)
                print(f"\n{resp}\n", flush=True)
            except Exception as exc:
                print(f"[error] {exc}", file=sys.stderr)
    finally:
        await browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Gemini browser automation with OpenAI-compatible API server."
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    serve_p = sub.add_parser("serve", help="Start the OpenAI-compatible API server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--visible", action="store_true", help="Show browser window (disable headless)")

    chat_p = sub.add_parser("chat", help="Single prompt or interactive browser chat")
    chat_p.add_argument("prompt", nargs="?", default=None)
    chat_p.add_argument("--visible", action="store_true", help="Show browser window (disable headless)")
    chat_p.add_argument("--session", action="store_true", help="Interactive session mode")

    # Backward-ish compatibility if user runs without subcommand
    parser.add_argument("--visible", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    cmd = args.cmd or "chat"

    global HEADLESS

    if cmd == "serve":
        HEADLESS = not args.visible
        import uvicorn
        # Use the app object directly so renaming this file won't break it.
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return

    if cmd == "chat":
        visible = getattr(args, "visible", False)
        HEADLESS = not visible

        if getattr(args, "session", False) or args.prompt is None:
            asyncio.run(interactive_chat(visible=visible))
        else:
            result = asyncio.run(ask_once(args.prompt, visible=visible))
            print(result)
        return


if __name__ == "__main__":
    main()