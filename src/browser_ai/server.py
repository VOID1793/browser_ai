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
(backend class, compat handler, headless flag) so that the CLI can configure
it without mutating module-level globals.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type

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


# ── App factory ───────────────────────────────────────────────────────────────

def build_app(
    backend_class: Type[BrowserBackend],
    headless: bool = True,
    compat=None,
) -> FastAPI:
    """
    Build and return a configured FastAPI application.

    Parameters
    ----------
    backend_class : Type[BrowserBackend]
        The browser backend to use for all sessions.
    headless : bool
        Whether to run the browser without a visible window.
    compat : CompatHandler | None
        An optional compat handler instance (e.g. ContinueCompat).
        When None, no client-specific patches are applied.
    """
    session_manager: Optional[SessionManager] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal session_manager
        print(f"[server] Starting up (backend={backend_class.label}, headless={headless})...", flush=True)
        session_manager = SessionManager(backend_class=backend_class, headless=headless)
        await session_manager.start()
        yield
        if session_manager:
            await session_manager.shutdown()
        print("[server] Shutdown complete.", flush=True)

    app = FastAPI(
        title="browser-ai OpenAI-Compatible Server",
        description="OpenAI-compatible API backed by browser automation.",
        version="0.1.0",
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
        return {"ok": True, "backend": backend_class.label, "model": MODEL_ID}

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
            print("[server] Intercepted throwaway request via compat handler.", flush=True)
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
            print(f"[server] Force-resetting session {session_key[:12]}...", flush=True)
            state = await session_manager.reset(session_key)
            state.auth_header = authorization
        else:
            state = await session_manager.get_or_create(session_key, authorization)

        prompt, reset_required = build_incremental_prompt(
            previous_messages=state.previous_messages,
            current_messages=normalized,
        )

        if reset_required:
            print("[server] History diverged; resetting browser session.", flush=True)
            state = await session_manager.reset(session_key)
            state.auth_header = authorization

        state.last_used_at = time.time()

        # ── Compat: prompt post-processing ────────────────────────────────────
        # Apply client-specific prompt transformations (e.g. code reordering).
        # This fires on the first turn (when the full user message is the prompt)
        # and is a no-op on incremental turns that send only a short delta.
        if compat is not None:
            prompt = compat.process_user_message(prompt)
        # ─────────────────────────────────────────────────────────────────────

        # ── Tool preamble injection ───────────────────────────────────────────
        has_tools = bool(body.tools)
        if has_tools:
            state.last_tools = body.tools
        elif state.last_tools:
            has_tools = True

        if has_tools:
            active_tools = body.tools if body.tools else state.last_tools or []
            prompt = prompt + "\n\n" + serialize_tools_to_prompt(active_tools)
            print(f"[server] Tool preamble appended ({len(active_tools)} tools).", flush=True)
        # ─────────────────────────────────────────────────────────────────────

        try:
            print(f"[server] Calling browser.ask() for session {session_key[:12]}...", flush=True)
            response_text = await asyncio.wait_for(
                state.browser.ask(prompt),
                timeout=RESPONSE_TIMEOUT_MS / 1000 + 15,
            )
            state.previous_messages = normalized + [
                {"role": "assistant", "content": response_text}
            ]
            state.last_used_at = time.time()
            print(f"[server] Got response ({len(response_text)} chars).", flush=True)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out.")
        except PlaywrightTimeoutError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in normalized if m["content"].strip()
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
                print(
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
                print("[server] Tools present but no TOOL_CALL detected — returning plain text.", flush=True)
        # ─────────────────────────────────────────────────────────────────────

        if body.stream:
            print(f"[server] Streaming response ({len(response_text)} chars).", flush=True)
            return StreamingResponse(
                _stream_response(response_text, completion_id, created),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        return JSONResponse(content={
            "id": completion_id,                    # locally generated
            "object": "chat.completion",            # locally synthesised
            "created": created,                     # locally generated timestamp
            "model": MODEL_ID,                      # local alias, not upstream name
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": response_text}}],  # TRUE scraped content
            "usage": {
                # Heuristic estimates only — NOT upstream token counts
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
        if session_manager is None:
            raise HTTPException(status_code=503, detail="Session manager not ready")

        raw_prompt = (
            "\n".join(str(p) for p in body.prompt if p)
            if isinstance(body.prompt, list)
            else str(body.prompt or "").strip()
        )
        if not raw_prompt:
            raise HTTPException(status_code=400, detail="prompt cannot be empty")

        # Apply compat prompt processing if active, otherwise fall back to
        # the general reorder_user_message which is always safe.
        if compat is not None:
            prompt = compat.process_user_message(raw_prompt)
        else:
            prompt = reorder_user_message(raw_prompt)

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
        print(f"[server] /v1/completions for session {session_key[:12]} ({len(prompt)} chars)...", flush=True)

        try:
            response_text = await asyncio.wait_for(
                state.browser.ask(prompt),
                timeout=RESPONSE_TIMEOUT_MS / 1000 + 15,
            )
            # Edit-mode is stateless — don't update previous_messages
            state.last_used_at = time.time()
            print(f"[server] /v1/completions response ({len(response_text)} chars).", flush=True)
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
            print(f"[server] Streaming /v1/completions ({len(response_text)} chars).", flush=True)
            return StreamingResponse(
                _stream_completion(response_text, chunk_id, created),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        return JSONResponse(content={
            "id": chunk_id,
            "object": "text_completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "text": response_text,  # TRUE scraped content
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        }, headers=resp_headers)

    return app
