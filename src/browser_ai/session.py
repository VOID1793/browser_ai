"""
browser_ai.session
~~~~~~~~~~~~~~~~~~
Session state and lifecycle management.

Each session corresponds to one persistent browser page.  Sessions are keyed
by a hash of the caller's identity signals (Authorization header, X-Session-Id
header, OpenAI user field) so that different callers get isolated browser
contexts automatically.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from browser_ai.backends.base import BrowserBackend
from browser_ai.config import SESSION_CLEANUP_INTERVAL_S, SESSION_IDLE_TTL_S


# ── Session key derivation ────────────────────────────────────────────────────

def make_session_key(
    authorization: Optional[str],
    x_session_id: Optional[str],
    openai_user: Optional[str],
) -> str:
    """
    Derive an opaque session key from available caller-identity signals.

    Auth header passthrough:
      - The Authorization value is NOT validated.
      - It is accepted as-is and used purely for session namespace isolation.
      - Different auth tokens → different browser sessions automatically.
    """
    auth_part = authorization or "anon"
    explicit_session = x_session_id or openai_user or "default"
    raw = f"{auth_part}::{explicit_session}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    session_id: str
    browser: BrowserBackend
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    # Stored for future extension (e.g. per-session auth routing).
    # We do NOT validate this token — it is an opaque passthrough.
    auth_header: Optional[str] = None

    # Normalised message history for the current conversation.
    # Includes both user messages and the assistant replies we sent back,
    # so that prefix-matching in build_incremental_prompt works correctly
    # when clients resend full history on every request.
    previous_messages: Optional[List[Dict[str, str]]] = None

    # Last tools array seen for this session.  Persisted so that follow-up
    # turns that omit the tools array (common in agentic loops) still get
    # the tool preamble injected into the prompt.
    last_tools: Optional[List[Any]] = None


# ── Session manager ───────────────────────────────────────────────────────────

class SessionManager:
    """
    Manages a pool of browser sessions, one per unique session key.

    Responsibilities:
      - Create new sessions on demand.
      - Return existing sessions for repeat callers.
      - Expire and close sessions that have been idle longer than SESSION_IDLE_TTL_S.
      - Clean shutdown of all sessions on server stop.
    """

    def __init__(
        self,
        backend_class: Type[BrowserBackend],
        headless: bool = True,
    ) -> None:
        self.backend_class = backend_class
        self.headless = headless
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start(self) -> None:
        """Start the background idle-session cleanup loop."""
        print("[session_manager] Starting cleanup loop.", flush=True)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def shutdown(self) -> None:
        """Cancel cleanup, close all sessions, and release resources."""
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

    async def get_or_create(
        self,
        session_key: str,
        auth_header: Optional[str] = None,
    ) -> SessionState:
        """Return an existing session or create a new one for *session_key*."""
        async with self._lock:
            state = self._sessions.get(session_key)
            if state is None:
                print(
                    f"[session_manager] Creating new session: {session_key[:12]}...",
                    flush=True,
                )
                browser = self.backend_class(headless=self.headless)
                state = SessionState(
                    session_id=session_key,
                    browser=browser,
                    auth_header=auth_header,
                )
                self._sessions[session_key] = state
            else:
                print(
                    f"[session_manager] Reusing session: {session_key[:12]}...",
                    flush=True,
                )

            state.last_used_at = time.time()
            return state

    async def reset(self, session_key: str) -> SessionState:
        """Close and remove an existing session, then create a fresh one."""
        print(
            f"[session_manager] Resetting session: {session_key[:12]}...",
            flush=True,
        )
        async with self._lock:
            old = self._sessions.pop(session_key, None)

        if old:
            await old.browser.close()

        browser = self.backend_class(headless=self.headless)
        state = SessionState(session_id=session_key, browser=browser)

        async with self._lock:
            self._sessions[session_key] = state

        return state

    async def _cleanup_loop(self) -> None:
        """Periodically close sessions that have been idle too long."""
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
                    print(
                        f"[session_manager] Expiring idle session: {key[:12]}...",
                        flush=True,
                    )
                    stale.append(self._sessions.pop(key))

            for state in stale:
                await state.browser.close()
