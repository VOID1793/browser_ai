"""
browser_ai.compat.continue_compat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compatibility patches for the Continue VS Code extension.

All Continue-specific behaviour that was previously scattered through the
server code now lives here.  When --compat continue is NOT passed, none of
this code is executed.

Patches applied
---------------
1. Chat-titling stub
   Continue fires a background /v1/chat/completions request after every
   response to auto-generate a sidebar title.  These requests have a single
   user message containing "reply with a title".  We intercept them, return
   a cheap stub response, and leave the browser session completely untouched.

2. Code-context reordering
   Continue injects file contents as a leading fenced code block before the
   user's actual instruction.  For large files (40k+ chars) the LLM ignores
   the instruction entirely.  We move the instruction to the front.

3. Soft prefix matching
   When the user switches from chat mode to agent mode, Continue replaces the
   system message, shifting the message history enough to trigger a false
   divergence detection and unnecessary browser reset.  We strip system
   messages from both sides before comparison.

   NOTE: soft prefix matching is already built into build_incremental_prompt()
   in prompt.py because it is broadly useful.  It is listed here for
   documentation clarity but does not require extra code in this module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Throwaway request patterns ────────────────────────────────────────────────
# Extend this list if Continue's wording changes or if other common one-shot
# meta-requests need to be intercepted.
THROWAWAY_PATTERNS: List[str] = [
    "reply with a title",
]

STUB_TITLE = "Browser AI Chat"


class ContinueCompat:
    """
    Compatibility handler for the Continue VS Code extension.

    The server calls the methods below at appropriate points in the request
    lifecycle.  If the method returns a non-None value, the server returns
    that value directly without further processing.
    """

    # ── Throwaway detection ───────────────────────────────────────────────────

    def is_throwaway_request(self, normalized_messages: List[Dict[str, str]]) -> bool:
        """
        Return True if this request is a background meta-request (e.g. title
        generation) that should be answered with a stub rather than forwarded
        to the browser.
        """
        if len(normalized_messages) != 1:
            return False
        if normalized_messages[0]["role"] != "user":
            return False
        content_lower = normalized_messages[0]["content"].lower()
        return any(pat in content_lower for pat in THROWAWAY_PATTERNS)

    def stub_title_response(self) -> str:
        """Return the stub title string for throwaway title requests."""
        return STUB_TITLE

    # ── Prompt post-processing ────────────────────────────────────────────────

    def process_user_message(self, user_content: str) -> str:
        """
        Apply Continue-specific prompt transformations.

        reorder_user_message() is now applied inside build_incremental_prompt()
        on the raw user message before system-prefix assembly.  This method
        is kept as an extension point for any future Continue-specific
        transformations that cannot be handled generically.
        """
        return user_content