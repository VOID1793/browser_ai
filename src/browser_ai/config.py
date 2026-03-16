"""
browser_ai.config
~~~~~~~~~~~~~~~~~
Global tunables.  All values can be overridden at startup via CLI flags or
environment variables.  Nothing in this file imports from the rest of the
package so it is safe to import from anywhere.
"""

from __future__ import annotations

# ── Browser ───────────────────────────────────────────────────────────────────
LAUNCH_TIMEOUT_MS: int = 30_000
RESPONSE_TIMEOUT_MS: int = 180_000     # 3 min — large file-context responses can be slow
POLL_INTERVAL_S: float = 0.8
STABLE_READS: int = 3

# Default headless mode.  Individual backends may override this via their
# DEFAULT_HEADLESS class attribute (e.g. ChatGPT forces visible+minimized).
# The CLI --headless / --no-headless flags override both.
HEADLESS: bool = True

# Prompts longer than this are delivered via clipboard paste (Ctrl+V) instead
# of character-by-character keyboard.type(), avoiding multi-minute delays for
# large payloads from editors like Continue.
CLIPBOARD_PASTE_THRESHOLD: int = 200

# ── Session management ────────────────────────────────────────────────────────
SESSION_IDLE_TTL_S: int = 30 * 60      # expire idle browser sessions after 30 min
SESSION_CLEANUP_INTERVAL_S: int = 60   # check for idle sessions every minute

# ── OpenAI compatibility metadata ─────────────────────────────────────────────
# These are locally synthesised fields returned in API responses.
# They are NOT upstream provider metadata.
MODEL_ID: str = "browser-ai"
SYSTEM_FINGERPRINT: str = "browser-ai-playwright-v1"

# ── Response cleaning ─────────────────────────────────────────────────────────
# UI junk lines stripped from scraped response text (case-insensitive exact match).
JUNK_LINES: frozenset[str] = frozenset({
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
    # Gemini code block UI labels
    "code snippet",
    "run in google colab",
    "use code with caution",
})

# ── Tool calling ──────────────────────────────────────────────────────────────
# Tool names that write content to a file.  Used by the argument-rescue logic
# to patch missing `contents` arguments from the last assistant message.
FILE_WRITE_TOOLS: frozenset[str] = frozenset({
    "create_new_file",
    "write_file",
    "create_file",
    "write_to_file",
})

# Argument names that carry file content, in priority order.
CONTENT_ARG_NAMES: tuple[str, ...] = ("contents", "content", "text", "body", "data")