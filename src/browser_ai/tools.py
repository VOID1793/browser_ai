"""
browser_ai.tools
~~~~~~~~~~~~~~~~
Prompt-engineering layer for OpenAI-style tool / function calling.

Because the browser backends are plain chat UIs with no native tool-call
support, tool calling is implemented via:
  1. Serialising the tools array into a structured natural-language preamble
     appended to the prompt (serialize_tools_to_prompt).
  2. Parsing the LLM's plain-text response for a TOOL_CALL: {...} line
     (extract_tool_call).
  3. Rescuing missing large-string arguments (e.g. file contents) from the
     most recent assistant message in history (rescue_tool_arguments).

All functions are pure (no I/O) and never raise — failures return None or
the input unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from browser_ai.config import CONTENT_ARG_NAMES, FILE_WRITE_TOOLS


# ── Prompt serialisation ──────────────────────────────────────────────────────

def serialize_tools_to_prompt(tools: List[Any]) -> str:
    """
    Convert an OpenAI-style tools array into a plain-text block to be
    appended to the end of the prompt.

    Appending (rather than prepending) ensures the format instruction is the
    last thing the LLM reads before generating its response, which significantly
    improves compliance when the prompt contains large code context.

    The format the LLM is asked to use:

        TOOL_CALL: {"name": "<tool_name>", "arguments": {<args as JSON>}}

    This string is easy to extract with a regex and is unlikely to appear in
    normal prose responses.
    """
    if not tools:
        return ""

    lines = [
        "=" * 60,
        "IMPORTANT INSTRUCTIONS — READ THIS LAST, BEFORE RESPONDING",
        "=" * 60,
        "",
        "You have access to tools. To use a tool you MUST respond with",
        "EXACTLY the following on its own line as your ENTIRE response",
        "(no prose before or after the TOOL_CALL line):",
        "",
        '    TOOL_CALL: {"name": "<tool_name>", "arguments": {<args as JSON>}}',
        "",
        "If no tool is needed, respond normally in plain text.",
        "If you use a tool, output ONLY the TOOL_CALL line — nothing else.",
        "",
        "Available tools:",
        "",
    ]

    for tool in tools:
        try:
            if hasattr(tool, "function"):
                fn = tool.function
                name = fn.name
                desc = fn.description or ""
                params = fn.parameters or {}
            else:
                fn = tool.get("function", {})
                name = fn.get("name", "unknown")
                desc = fn.get("description", "")
                params = fn.get("parameters", {})

            lines.append(f"- {name}: {desc}")

            props = params.get("properties", {}) if isinstance(params, dict) else {}
            required = params.get("required", []) if isinstance(params, dict) else []
            for pname, pschema in props.items():
                req_marker = " (required)" if pname in required else ""
                pdesc = pschema.get("description", "") if isinstance(pschema, dict) else ""
                ptype = pschema.get("type", "") if isinstance(pschema, dict) else ""
                lines.append(f"    - {pname} [{ptype}]{req_marker}: {pdesc}")

        except Exception:
            # Malformed tool definition — skip silently
            continue

    return "\n".join(lines) + "\n"


# ── Response parsing ──────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\{.*\})", re.DOTALL)


def extract_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract a structured tool call from the LLM's response.

    Returns a dict with keys:
        name       str   — function name
        arguments  dict  — parsed argument dict

    Returns None if no valid TOOL_CALL line is found.  Never raises.
    """
    match = _TOOL_CALL_RE.search(response_text)
    if not match:
        return None

    json_str = match.group(1).strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Salvage attempt: trim to the last closing brace
        try:
            json_str = json_str[: json_str.rfind("}") + 1]
            parsed = json.loads(json_str)
        except Exception:
            return None

    name = parsed.get("name")
    arguments = parsed.get("arguments", {})

    if not name or not isinstance(name, str):
        return None
    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": name, "arguments": arguments}


# ── Argument rescue ───────────────────────────────────────────────────────────

def rescue_tool_arguments(
    tool_call: Dict[str, Any],
    normalized_messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Patch missing or empty content arguments on file-write tool calls.

    LLMs reliably produce the tool name but frequently omit large string
    arguments (e.g. `contents` for create_new_file) because accurately
    JSON-encoding kilobytes of markdown inside a single string is beyond what
    prompt-engineering reliably achieves.

    Strategy: if the detected tool is a file-write call and its primary
    content argument is absent or empty, walk the message history backwards
    and use the last substantial assistant message as the content.

    This is semantically safe when the user said "write what you just gave me
    to a file" — the most recent assistant turn IS the intended content.

    Returns the tool_call dict (mutated in-place) with the rescued argument,
    or the original dict unchanged if rescue was not needed or not possible.
    """
    if tool_call.get("name") not in FILE_WRITE_TOOLS:
        return tool_call

    args = tool_call.get("arguments", {})
    content_key = next((k for k in CONTENT_ARG_NAMES if k in args), None)
    content_val = args.get(content_key, "") if content_key else ""

    if content_val and str(content_val).strip():
        # Content is already present — nothing to rescue
        return tool_call

    # Walk history backwards for the last substantial assistant message
    rescued = None
    for msg in reversed(normalized_messages):
        if msg["role"] == "assistant" and len(msg["content"].strip()) > 100:
            rescued = msg["content"].strip()
            break

    if rescued:
        target_key = content_key or "contents"
        tool_call["arguments"][target_key] = rescued
        print(
            f"[tools] Rescued empty '{target_key}' for {tool_call['name']} "
            f"from last assistant message ({len(rescued)} chars).",
            flush=True,
        )
    else:
        print(
            f"[tools] WARNING: {tool_call['name']} has empty content argument "
            f"and no prior assistant message to rescue from.",
            flush=True,
        )

    return tool_call
