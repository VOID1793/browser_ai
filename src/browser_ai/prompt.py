"""
browser_ai.prompt
~~~~~~~~~~~~~~~~~
Prompt construction, conversation history tracking, and message normalisation.

None of these functions perform I/O — they are pure transformations of text
and message lists.

Reorder contract
----------------
reorder_user_message() must only ever be called on a RAW user message string,
never on an assembled prompt that already has a system-instructions prefix.
build_incremental_prompt() applies it internally to the latest_user content
before assembling the final prompt, so callers (the server) must NOT call it
again after the fact.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from browser_ai.config import MODEL_ID  # noqa: F401  (re-exported for server.py)


# ── Message normalisation ─────────────────────────────────────────────────────

def extract_text_content(content: Any) -> str:
    """
    Normalise OpenAI-style message content to a plain string.

    Handles:
      - plain str
      - list of {"type": "text", "text": "..."} content parts
      - None
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
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Basic normaliser — converts every message to {role, content} strings."""
    normalized: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = extract_text_content(m.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_messages_with_tools(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Tool-aware normaliser.

    Flattens tool_calls on assistant messages and tool result messages into
    plain text representations so that build_incremental_prompt's prefix-
    comparison logic continues to work without triggering spurious resets.
    """
    normalized: List[Dict[str, str]] = []

    for m in messages:
        role = str(m.get("role", "user")).strip().lower()

        if role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                parts = []
                raw_content = extract_text_content(m.get("content", ""))
                if raw_content:
                    parts.append(raw_content)
                for tc in tool_calls:
                    try:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        tc_name = fn.get("name", "unknown")
                        tc_args = fn.get("arguments", "{}")
                        tc_json = json.dumps({"name": tc_name, "arguments": tc_args})
                        parts.append(f"TOOL_CALL: {tc_json}")
                    except Exception:
                        pass
                normalized.append({"role": "assistant", "content": "\n".join(parts)})
                continue

        if role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            content = extract_text_content(m.get("content", ""))
            label = f"[tool_result id={tool_call_id}]" if tool_call_id else "[tool_result]"
            normalized.append({"role": "tool", "content": f"{label}\n{content}"})
            continue

        content = extract_text_content(m.get("content", ""))
        normalized.append({"role": role, "content": content})

    return normalized


# ── Prompt builders ───────────────────────────────────────────────────────────


def reorder_user_message(user_content: str) -> str:
    """
    Reorder a user message that embeds a file block before the instruction.

    Continue and similar IDE extensions produce messages shaped like:

        ```lang path/to/file (lines N-M)
        <file contents — may contain inner fenced blocks>
        ```
        <actual user instruction>

    Because the file block dominates, the LLM often ignores the instruction.
    This function moves the instruction to the front:

        <actual instruction>

        Here is the relevant file context:
        ```lang path/to/file (lines N-M)
        <file contents>
        ```

    Algorithm:
      - The FIRST line that starts with ``` is the outer opener.
      - The outer closer is the LAST line that is exactly ``` (nothing after).
        This works because inner closing fences are always followed by more
        content, while the outer closer is the final ``` in the message.
      - Everything after the outer closer is the instruction.
      - Requires at least 3 words of instruction to avoid spurious reordering.

    This function MUST only be called on the raw user message string, not on
    an assembled prompt that already has system-instruction text prepended.
    """
    content = user_content.strip()
    lines = content.splitlines()

    # Find the first line that opens a fenced block (starts with ```)
    opener_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            opener_idx = i
            break

    if opener_idx is None:
        return content

    # Find the LAST line that is a bare closing fence (exactly ```)
    closer_idx = None
    for i in range(len(lines) - 1, opener_idx, -1):
        if lines[i].strip() == "```":
            closer_idx = i
            break

    if closer_idx is None or closer_idx <= opener_idx:
        return content

    # Everything after the closer is the instruction
    instruction = "\n".join(lines[closer_idx + 1:]).strip()

    # Require at least 3 words of meaningful instruction
    if len(instruction.split()) < 3:
        return content

    code_block = "\n".join(lines[opener_idx : closer_idx + 1])
    return f"{instruction}\n\nHere is the relevant file context:\n\n{code_block}"


def build_transcript_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Serialise a full conversation history into a single prompt string.

    Used as a fallback when the browser session must be reset and we need
    to replay context in a single turn.
    """
    system_parts: List[str] = []
    convo_parts: List[str] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if not content:
            continue
        if role in {"system", "developer"}:
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
    Derive the next prompt to send to the browser and whether a session reset
    is required.

    Returns: (prompt_string, reset_required)

    reorder_user_message() is applied to the raw latest_user content BEFORE
    the system-instructions prefix is prepended.  Callers must NOT call it
    again on the assembled result.

    Strategy:
      - No previous history → first turn; fold system messages + latest user.
      - Identical history  → resend latest user (defensive fallback).
      - Exact prefix match → incremental; send only the new user turn.
      - Soft prefix match  → system message swapped (e.g. chat→agent mode in
                             Continue); strip system messages and re-compare.
      - Diverged           → reset + transcript fallback.
    """
    if not current_messages:
        return "Hello.", False

    # ── First turn ────────────────────────────────────────────────────────────
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

        # Apply reorder to the raw user message BEFORE prepending system text.
        latest_user = reorder_user_message(latest_user)

        if system_parts:
            prompt = (
                "Please follow these instructions for this conversation:\n"
                + "\n\n".join(system_parts).strip()
                + "\n\n"
                + latest_user
            )
            return prompt, False

        return latest_user, False

    # ── Identical history ─────────────────────────────────────────────────────
    if previous_messages == current_messages:
        for m in reversed(current_messages):
            if m["role"] == "user" and m["content"].strip():
                return reorder_user_message(m["content"].strip()), False
        return build_transcript_prompt(current_messages), False

    # ── Exact prefix extension ────────────────────────────────────────────────
    if (
        len(current_messages) >= len(previous_messages)
        and current_messages[: len(previous_messages)] == previous_messages
    ):
        delta = current_messages[len(previous_messages):]
        bad_roles = {"system", "developer", "tool"}
        if any(m["role"] in bad_roles for m in delta):
            return build_transcript_prompt(current_messages), True
        for m in reversed(delta):
            if m["role"] == "user" and m["content"].strip():
                return reorder_user_message(m["content"].strip()), False
        return build_transcript_prompt(current_messages), True

    # ── Soft prefix match (system message swap) ───────────────────────────────
    def _strip_system(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return [m for m in msgs if m["role"] not in {"system", "developer"}]

    cur_stripped = _strip_system(current_messages)
    prev_stripped = _strip_system(previous_messages)

    if (
        len(cur_stripped) >= len(prev_stripped)
        and cur_stripped[: len(prev_stripped)] == prev_stripped
    ):
        delta = cur_stripped[len(prev_stripped):]
        if any(m["role"] == "tool" for m in delta):
            return build_transcript_prompt(current_messages), True
        for m in reversed(delta):
            if m["role"] == "user" and m["content"].strip():
                print("[prompt] Soft prefix match — sending delta only.", flush=True)
                return reorder_user_message(m["content"].strip()), False
        return build_transcript_prompt(current_messages), True

    # ── Diverged ──────────────────────────────────────────────────────────────
    return build_transcript_prompt(current_messages), True