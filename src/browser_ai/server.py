"""
browser_ai.server
~~~~~~~~~~~~~~~~~
FastAPI application, route handlers, and SSE streaming helpers.

The server exposes an OpenAI-compatible HTTP API:

    GET  /healthz
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions          (legacy, used by Continue edit mode)

The app object is created by build_app() which accepts runtime configuration
(backend class, compat handler, headless flag, quiet flag) so that the CLI can
configure it without mutating module-level globals.  Multiple independent
instances can be created pointing at different backends and bound to different
ports — each has its own SessionManager and browser pool.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Type

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from browser_ai.backends.base import BrowserBackend
from browser_ai.config import MODEL_ID, RESPONSE_TIMEOUT_MS, SYSTEM_FINGERPRINT
from browser_ai.models import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
)
from browser_ai.prompt import (
    build_incremental_prompt,
    normalize_messages_with_tools,
    reorder_user_message,
)
from browser_ai.cleaning import strip_edit_response
from browser_ai.session import SessionManager, make_session_key
from browser_ai.tools import extract_tool_call, rescue_tool_arguments, serialize_tools_to_prompt


# ── Utility ───────────────────────────────────────────────────────────────────

def now_ts() -> int:
    return int(time.time())


def estimate_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token.  NOT upstream provider counts."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_chunk(chunk_id: str, created: int, delta_content: str) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [
            {"index": 0, "delta": {"content": delta_content}, "finish_reason": None}
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_done_chunk(chunk_id: str, created: int) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


async def _stream_response(response_text: str, chunk_id: str, created: int):
    """
    Yield the full response as a sequence of SSE chunks, one word at a time.

    We have the complete response before streaming starts (the browser blocks
    until generation is stable).  Splitting on word boundaries and yielding
    with asyncio.sleep(0) between chunks lets uvicorn flush each chunk so
    clients see a streaming effect rather than a single large burst.
    """
    import re
    tokens = re.split(r"(\s+)", response_text)
    for token in tokens:
        if token:
            yield _sse_chunk(chunk_id, created, token)
            await asyncio.sleep(0)
    yield _sse_done_chunk(chunk_id, created)


def _sse_completion_chunk(chunk_id: str, created: int, text: str) -> str:
    payload = {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "text": text, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_completion_done(chunk_id: str, created: int) -> str:
    payload = {
        "id": chunk_id,
        "object": "text_completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


async def _stream_completion(response_text: str, chunk_id: str, created: int):
    import re
    tokens = re.split(r"(\s+)", response_text)
    for token in tokens:
        if token:
            yield _sse_completion_chunk(chunk_id, created, token)
            await asyncio.sleep(0)
    yield _sse_completion_done(chunk_id, created)


# ── Tool-call SSE helpers ─────────────────────────────────────────────────────

async def _stream_tool_call(
    tool_call: Dict[str, Any],
    chunk_id: str,
    created: int,
    tool_call_id: str,
):
    """Yield an SSE stream that carries a single tool_call delta."""
    role_chunk = {
        "id": chunk_id, "object": "chat.completion.chunk",
        "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": None, "delta": {
            "role": "assistant",
            "tool_calls": [{"index": 0, "id": tool_call_id, "type": "function",
                            "function": {"name": tool_call["name"], "arguments": ""}}],
        }}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    args_chunk = {
        "id": chunk_id, "object": "chat.completion.chunk",
        "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": None, "delta": {
            "tool_calls": [{"index": 0, "function": {
                "arguments": json.dumps(tool_call["arguments"])
            }}],
        }}],
    }
    yield f"data: {json.dumps(args_chunk)}\n\n"

    done_chunk = {
        "id": chunk_id, "object": "chat.completion.chunk",
        "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n"


# ── Edit-mode instruction injection ──────────────────────────────────────────
# Injected into every /v1/completions prompt (non-FIM) so the LLM returns
# ONLY replacement code.
#
# PLACEMENT STRATEGY:
#   Continue's edit-mode prompt for ChatGPT ends with TWO ChatML tokens:
#     <|im_end|>  — closes the user turn
#     <|im_start|>assistant [prefill text] — OPENS the assistant turn with a
#     pre-written sentence ("Sure! Here's the entire code block...").
#
#   This assistant prefill is ChatGPT-specific and it is the ROOT CAUSE of
#   ChatGPT edit mode failure.  ChatGPT continues from that prefill and
#   reproduces the ENTIRE file, completely ignoring any instruction in the
#   user turn.  Our injection before <|im_end|> was therefore also ignored.
#
#   FIX:
#   1. Inject our instruction immediately before <|im_end|> (last user-turn
#      position — highest authority for the user turn).
#   2. Strip the prefill text that follows <|im_start|>assistant, leaving only
#      the bare token.  ChatGPT then generates fresh from zero context,
#      sees our instruction, and returns only the fenced code.
#
#   For Gemini (no ChatML tokens), we append the instruction after the full
#   prompt.  Either way the instruction is the last thing the model reads.
#
# WORDING: kept deliberately terse.  Verbose preambles cause some models to
# repeat them in the response, which breaks the fence extractor.
_EDIT_MODE_INSTRUCTION = (
    "CRITICAL: Respond with ONLY the replacement fenced code block, "
    "preserving the opening fence line (e.g. ```mermaid or ```python) "
    "and closing ``` exactly as they appear in the original. "
    "No prose before or after — just the complete fence."
)

# ChatML boundary tokens injected by Continue into edit-mode prompts.
_CHATML_USER_END = "<|im_end|>"
_CHATML_ASSISTANT_START = "<|im_start|>assistant"


def _strip_assistant_prefill(prompt: str) -> str:
    """
    Remove the pre-filled assistant text that Continue appends to ChatGPT prompts.

    Continue appends ``<|im_start|>assistant [prefill sentence]`` to prime
    ChatGPT's response.  That prefill ("Sure! Here's the entire code block,
    including the rewritten portion:") instructs ChatGPT to reproduce the full
    file, overriding every instruction in the user turn.

    This function keeps the ``<|im_start|>assistant`` token (so ChatGPT knows
    it is generating as the assistant) but strips the priming sentence, letting
    ChatGPT generate from a clean slate.

    No-op for prompts that do not contain the ChatML assistant-start token
    (e.g. Gemini prompts).
    """
    if _CHATML_ASSISTANT_START not in prompt:
        return prompt
    head, tail = prompt.split(_CHATML_ASSISTANT_START, 1)
    # tail is the prefill sentence (one line) — strip it entirely.
    # If there is content after the first newline (unusual) preserve it.
    newline_idx = tail.find('\n')
    if newline_idx == -1:
        return head + _CHATML_ASSISTANT_START
    after = tail[newline_idx + 1:]
    return head + _CHATML_ASSISTANT_START + ("\n" + after if after.strip() else "")


# ── App factory ───────────────────────────────────────────────────────────────

def build_app(
    backend_class: Type[BrowserBackend],
    headless=None,
    compat=None,
    quiet: bool = False,
) -> FastAPI:
    """
    Build and return a configured FastAPI application.

    Parameters
    ----------
    backend_class : Type[BrowserBackend]
        The browser backend to use for all sessions.
    headless : bool | None
        True = force headless, False = force visible, None = use backend default.
    compat : CompatHandler | None
        An optional compat handler instance (e.g. ContinueCompat).
        When None, no client-specific patches are applied.
    quiet : bool
        Suppress internal browser-ai log lines when True.  HTTP access logs
        are controlled separately by uvicorn's log_level.

    Multiple independent instances may be created (e.g. one per backend) and
    bound to different ports via separate uvicorn.run() calls in separate
    processes.  Each instance owns its own SessionManager and browser pool.
    """
    # _log() replaces bare print() throughout this module so that quiet mode
    # suppresses our internal lines without touching uvicorn's access logs.
    def _log(*args, **kwargs) -> None:
        if not quiet:
            print(*args, **kwargs)

    session_manager: Optional[SessionManager] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal session_manager
        effective = headless if headless is not None else backend_class.DEFAULT_HEADLESS
        _log(
            f"[server] Starting up (backend={backend_class.label}, "
            f"headless={effective})...",
            flush=True,
        )
        session_manager = SessionManager(backend_class=backend_class, headless=headless)
        await session_manager.start()
        try:
            yield
        finally:
            # finally runs on both normal shutdown and Ctrl+C (SIGINT).
            # CancelledError (BaseException, not Exception) must be caught here;
            # if it escapes, Starlette prints a misleading ERROR traceback.
            try:
                await session_manager.shutdown()
            except (asyncio.CancelledError, Exception) as exc:
                _log(f"[server] Shutdown warning (non-fatal): {exc}", flush=True)
            _log("[server] Shutdown complete.", flush=True)

    app = FastAPI(
        title="browser-ai OpenAI-Compatible Server",
        description="OpenAI-compatible API backed by browser automation.",
        version="0.2.0",
        lifespan=lifespan,
    )

    # ── Exception handlers ────────────────────────────────────────────────────

    @app.exception_handler(PlaywrightTimeoutError)
    async def playwright_timeout_handler(_: Request, exc: PlaywrightTimeoutError):
        return JSONResponse(
            status_code=504,
            content={"error": {
                "message": f"Browser automation timed out: {exc}",
                "type": "timeout_error",
                "code": "playwright_timeout",
            }},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(_: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_error",
            }},
        )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/healthz")
    async def healthz():
        from browser_ai import __version__
        effective_headless = headless if headless is not None else backend_class.DEFAULT_HEADLESS
        mode = "headless" if effective_headless else (
            "visible+minimized" if getattr(backend_class, "MINIMIZE_WINDOW", False)
            else "visible"
        )
        return {
            "ok": True,
            "version": __version__,
            "backend": backend_class.label,
            "model": MODEL_ID,
            "mode": mode,
            "sessions": session_manager.session_count() if session_manager else 0,
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [ModelCard(id=MODEL_ID, created=now_ts()).model_dump()],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_session_id: Optional[str] = Header(default=None),
        x_reset_session: Optional[str] = Header(default=None),
    ):
        if session_manager is None:
            raise HTTPException(status_code=503, detail="Session manager not ready")

        normalized = normalize_messages_with_tools([m.model_dump() for m in body.messages])
        if not normalized:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        # ── Compat: throwaway request interception ────────────────────────────
        if compat is not None and compat.is_throwaway_request(normalized):
            _log("[server] Intercepted throwaway request via compat handler.", flush=True)
            stub_id = f"chatcmpl-{uuid.uuid4().hex}"
            stub_title = compat.stub_title_response()
            if body.stream:
                async def _title_stream():
                    yield _sse_chunk(stub_id, now_ts(), stub_title)
                    yield _sse_done_chunk(stub_id, now_ts())
                return StreamingResponse(_title_stream(), media_type="text/event-stream")
            return JSONResponse(content={
                "id": stub_id, "object": "chat.completion", "created": now_ts(),
                "model": MODEL_ID, "system_fingerprint": SYSTEM_FINGERPRINT,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": stub_title}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        # ─────────────────────────────────────────────────────────────────────

        session_key = make_session_key(
            authorization=authorization,
            x_session_id=x_session_id,
            openai_user=body.user,
        )

        force_reset = (x_reset_session or "").strip().lower() in {"1", "true", "yes"}
        if force_reset:
            _log(f"[server] Force-resetting session {session_key[:12]}...", flush=True)
            state = await session_manager.reset(session_key)
            state.auth_header = authorization
        else:
            state = await session_manager.get_or_create(session_key, authorization)

        prompt, reset_required = build_incremental_prompt(
            previous_messages=state.previous_messages,
            current_messages=normalized,
        )

        if reset_required:
            _log("[server] History diverged; resetting browser session.", flush=True)
            state = await session_manager.reset(session_key)
            state.auth_header = authorization

        state.last_used_at = time.time()

        # Compat handler may apply additional prompt transformations.
        # Note: reorder_user_message is now applied inside build_incremental_prompt
        # on the raw user content, before system-prefix assembly. Do NOT call it
        # again here.
        if compat is not None:
            prompt = compat.process_user_message(prompt)

        # ── Tool preamble injection ───────────────────────────────────────────
        has_tools = bool(body.tools)
        if has_tools:
            state.last_tools = body.tools
        elif state.last_tools:
            has_tools = True

        if has_tools:
            active_tools = body.tools if body.tools else state.last_tools or []
            prompt = prompt + "\n\n" + serialize_tools_to_prompt(active_tools)
            _log(f"[server] Tool preamble appended ({len(active_tools)} tools).", flush=True)
        # ─────────────────────────────────────────────────────────────────────

        try:
            _log(f"[server] Calling browser.ask() for session {session_key[:12]}...", flush=True)
            response_text = await asyncio.wait_for(
                state.browser.ask(prompt),
                timeout=RESPONSE_TIMEOUT_MS / 1000 + 15,
            )
            state.previous_messages = normalized + [
                {"role": "assistant", "content": response_text}
            ]
            state.last_used_at = time.time()
            _log(f"[server] Got response ({len(response_text)} chars).", flush=True)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out.")
        except PlaywrightTimeoutError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in normalized
            if isinstance(m.get("content"), str) and m["content"].strip()
        )
        prompt_tokens = estimate_tokens(prompt_text)
        completion_tokens = estimate_tokens(response_text)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = now_ts()

        resp_headers: Dict[str, str] = {}
        if x_session_id:
            resp_headers["X-Session-Id"] = x_session_id

        # ── Tool-call response path ───────────────────────────────────────────
        if has_tools:
            tool_call = extract_tool_call(response_text)
            if tool_call:
                tool_call = rescue_tool_arguments(tool_call, normalized)
                _log(
                    f"[server] Tool call: {tool_call['name']}"
                    f"({list(tool_call['arguments'].keys())})",
                    flush=True,
                )
                tool_call_id = f"call_{uuid.uuid4().hex[:16]}"
                tc_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id, "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        },
                    }],
                }
                tc_payload = {
                    "id": completion_id, "object": "chat.completion",
                    "created": created, "model": MODEL_ID,
                    "system_fingerprint": SYSTEM_FINGERPRINT,
                    "choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": tc_message}],
                    "usage": {"prompt_tokens": prompt_tokens,
                              "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens},
                }
                if body.stream:
                    return StreamingResponse(
                        _stream_tool_call(tool_call, completion_id, created, tool_call_id),
                        media_type="text/event-stream",
                        headers=resp_headers,
                    )
                return JSONResponse(content=tc_payload, headers=resp_headers)
            else:
                _log("[server] Tools present but no TOOL_CALL detected — returning plain text.", flush=True)
        # ─────────────────────────────────────────────────────────────────────

        if body.stream:
            _log(f"[server] Streaming response ({len(response_text)} chars).", flush=True)
            return StreamingResponse(
                _stream_response(response_text, completion_id, created),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        return JSONResponse(content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": response_text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }, headers=resp_headers)

    @app.post("/v1/completions")
    async def completions(
        body: CompletionRequest,
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_session_id: Optional[str] = Header(default=None),
        x_reset_session: Optional[str] = Header(default=None),
    ):
        """
        Legacy text completions endpoint.

        Continue's edit mode sends the selected text + instruction here as a
        plain string rather than a messages array.
        """
        _log(f"[server] /v1/completions REQUEST received (stream={body.stream})", flush=True)
        if session_manager is None:
            raise HTTPException(status_code=503, detail="Session manager not ready")

        raw_prompt = (
            "\n".join(str(p) for p in body.prompt if p)
            if isinstance(body.prompt, list)
            else str(body.prompt or "").strip()
        )
        if not raw_prompt:
            raise HTTPException(status_code=400, detail="prompt cannot be empty")

        # NOTE: reorder_user_message is intentionally NOT called here.
        # Continue's edit-mode prompt has its own carefully structured layout
        # (file context, <START/STOP EDITING HERE> markers, template text, and
        # ChatML tokens).  reorder_user_message would detect the leading ```
        # fence and relocate the ChatML tokens to the front of the prompt,
        # destroying the token sequence that ChatGPT relies on.
        # The edit prompt is already correctly structured — leave it alone.
        prompt = raw_prompt
        if compat is not None:
            prompt = compat.process_user_message(prompt)

        # Step 2: inject the edit-mode instruction and strip the ChatGPT prefill.
        # Only skipped for FIM payloads (non-empty `suffix` field).
        is_fim = bool(body.suffix and body.suffix.strip())
        if not is_fim:
            if _CHATML_USER_END in prompt:
                # ChatML (ChatGPT via Continue): inject before the user-turn closer
                # so our instruction is the LAST thing in the user turn.
                prompt = prompt.replace(
                    _CHATML_USER_END,
                    "\n\n" + _EDIT_MODE_INSTRUCTION + "\n" + _CHATML_USER_END,
                    1,
                )
                # Step 2b: strip the pre-filled assistant text that follows
                # <|im_start|>assistant.  See _strip_assistant_prefill() docstring.
                # This is the root-cause fix for ChatGPT ignoring edit instructions.
                original_len = len(prompt)
                prompt = _strip_assistant_prefill(prompt)
                if len(prompt) != original_len:
                    _log("[server] Stripped ChatGPT assistant prefill for edit mode.", flush=True)
                _log("[server] Injected edit instruction at ChatML user-turn boundary.", flush=True)
            else:
                # Non-ChatML (Gemini, others): append so instruction is last.
                prompt = prompt + "\n\n" + _EDIT_MODE_INSTRUCTION

        session_key = make_session_key(
            authorization=authorization,
            x_session_id=x_session_id,
            openai_user=body.user,
        )

        force_reset = (x_reset_session or "").strip().lower() in {"1", "true", "yes"}
        if force_reset:
            state = await session_manager.reset(session_key)
            state.auth_header = authorization
        else:
            state = await session_manager.get_or_create(session_key, authorization)

        state.last_used_at = time.time()
        _log(
            f"[server] /v1/completions for session {session_key[:12]} "
            f"({len(prompt)} chars)...",
            flush=True,
        )

        try:
            response_text = await asyncio.wait_for(
                state.browser.ask(prompt),
                timeout=RESPONSE_TIMEOUT_MS / 1000 + 15,
            )
            state.last_used_at = time.time()
            raw_len = len(response_text)
            # Edit mode: extract the code content from any surrounding prose/fences
            response_text = strip_edit_response(response_text)
            _log(
                f"[server] /v1/completions response: raw={raw_len} chars, "
                f"after strip={len(response_text)} chars. "
                f"Preview: {repr(response_text[:80])}",
                flush=True,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out.")
        except PlaywrightTimeoutError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        chunk_id = f"cmpl-{uuid.uuid4().hex}"
        created = now_ts()
        prompt_tokens = estimate_tokens(raw_prompt)
        completion_tokens = estimate_tokens(response_text)

        resp_headers: Dict[str, str] = {}
        if x_session_id:
            resp_headers["X-Session-Id"] = x_session_id

        if body.stream:
            _log(f"[server] Streaming /v1/completions ({len(response_text)} chars).", flush=True)
            try:
                return StreamingResponse(
                    _stream_completion(response_text, chunk_id, created),
                    media_type="text/event-stream",
                    headers=resp_headers,
                )
            except Exception as exc:
                _log(f"[server] Streaming error: {exc}", flush=True)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={
            "id": chunk_id,
            "object": "text_completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "text": response_text,
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }, headers=resp_headers)

    return app