# Agent Guidelines for browser-ai

This document provides guidelines for agents working on this codebase. Any LLM should be able to understand the project goals and interact with it by reading this file + README.md.

## Project Goals

**Mission:** Bridge free GUI LLM applications (ChatGPT, Gemini) with a local OpenAI-compatible API server using Playwright browser automation.

**Key Outcomes:**
1. Headless, API-key-free access to web LLMs
2. Strict OpenAI schema compatibility for client transparency
3. Zero regression across backends
4. Rapid adaptation to UI changes

### Tier 1 Backends (Gold Standard)
- **Gemini** (primary)
- **ChatGPT** (co-primary)
- Baseline: Feature parity for tool calling, streaming, and context handling

### Tier 1 Client Integrations
1. Continue Extension (Chat, Inline Edits, Codebase Context)
2. Open-WebUI (conversational flow + history)
3. OpenCode

---

## Build, Test, and Lint Commands

**Note:** This project uses an existing `.venv` in the repository root. Activate it with `source .venv/bin/activate` from `/home/tsell1/PROJECTS/browser_ai`.

### Installation
```bash
source .venv/bin/activate
cd src && pip install -e .
playwright install chromium
```

### Running
```bash
browser-ai serve --backend gemini --compat continue  # API server
browser-ai chat --backend gemini "prompt"            # One-shot
browser-ai chat --backend gemini --session          # Interactive
browser-ai backends                                 # List backends
```

### Smoke Test
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -H "X-Session-Id: smoke-test" \
  -d '{"model":"browser-ai","messages":[{"role":"user","content":"What is 2+2?"}],"stream":false}'
```

### Visible Debugging
```bash
browser-ai serve --backend gemini --no-headless
```

---

## HARD CONSTRAINTS (Enforced)

### 1. Cross-Backend Regression Guard

**RULE:** When modifying shared modules (`cleaning.py`, `session.py`, `base.py`, `prompt.py`), you MUST explicitly state:
```
"This change does not impact [OTHER_BACKEND] because..."
```

**Preference:** Backend-specific overrides over global regex/DOM changes.

**Trigger questions before touching shared code:**
- Does this work universally or only for one backend?
- Can the fix be a backend override instead?
- Have I checked the impact on all Tier 1 backends?

### 2. Two-Strike Anti-Looping Rule

**RULE:** After two failed attempts to fix an issue:
1. STOP immediately
2. State the fundamental architecture problem
3. Propose a novel approach
4. Request diagnostic artifacts (see below)

### 3. Diagnostic Artifacts on Request

**When intervention is needed, provide exact commands:**

```bash
# Page HTML for DOM inspection
await page.content()

# Network logs
page.on("console", lambda msg: print(f"CONSOLE: {msg.text}"))

# Playwright trace (upload to trace.playwright.dev)
await context.tracing.start(screenshots=True, snapshots=True)
# ... reproduce issue ...
await context.tracing.stop(path="trace.zip")
```

### 4. Selector Robustness Preference

**Prefer in order:**
1. ARIA labels (`[aria-label="..."]`)
2. Text-based (`button:has-text("Send")`)
3. Data attributes (`[data-test-id="..."]`)
4. Structural selectors (`preceding-sibling`, `nth-child`)
5. CSS classes (LAST resort — brittle)

### 5. Frugal Architect Protocol

**Token budget is real.** Every output token costs money.

- **Output ratio:** 3:1 prose:code minimum
- **Artifacts-first:** Output code/config/errors, not explanations
- **Zero fluff:** No "Certainly!", "Here's the fix:", "In conclusion"
- **Dense comments:** `# key insight` inline, not paragraphs

---

## Code Style Guidelines

### General
- Python 3.9+
- `from __future__ import annotations` everywhere
- Absolute imports: `from browser_ai.config import ...`
- `async/await` only — no blocking I/O in async functions

### Module Structure
```python
"""
browser_ai.<module>
~~~~~~~~~~~~~~~~~~
Purpose. Contracts/notes.
"""

from __future__ import annotations

# stdlib (alphabetical)
import asyncio
from typing import Any, Dict, List, Optional

# third-party
from pydantic import BaseModel

# local
from browser_ai.config import X
```

### Naming
| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case.py` | `session_manager.py` |
| Classes | `PascalCase` | `SessionManager` |
| Functions | `snake_case` | `make_session_key()` |
| Constants | `UPPER_SNAKE_CASE` | `MODEL_ID` |
| Private | `_prefix` | `_sessions` |

### Type Annotations
- `Optional[X]` (not `X | None`)
- `List`, `Dict` from `typing`
- Return type `-> None` for procedures
- Pydantic: `field(default=...)` or `field(default_factory=...)`

### Error Handling
- Specific exceptions over `except Exception:`
- Re-raise with context: `raise HTTPException(...) from exc`
- `try/finally` for cleanup
- Handle `asyncio.CancelledError` explicitly (BaseException)

```python
try:
    await session_manager.shutdown()
except (asyncio.CancelledError, Exception) as exc:
    _log(f"Shutdown warning: {exc}", flush=True)
```

### Logging
- `print(..., flush=True)` for user output
- Format: `[module] Message`
- Suppressible via `quiet` param

---

## Browser Backend Pattern

### Adding a Backend
1. Create `browser_ai/backends/<name>.py`
2. Subclass `BaseBrowserBackend`
3. Define selectors: `INPUT_`, `SEND_`, `RESPONSE_`, `GENERATING_`, `CONSENT_`
4. Optional: `DEFAULT_HEADLESS`, `MINIMIZE_WINDOW`, `EXTRA_LAUNCH_ARGS`
5. Hooks: `_goto_and_prepare()`, `_dismiss_interstitials()`, `_prepare_response_element()`
6. Register in `__init__.py` + CLI in `cli.py`

### Selector Priority
```python
INPUT_SELECTORS = [
    'div[contenteditable="true"][aria-label*="prompt"]',  # ARIA-first
    'div[contenteditable="true"]',                       # Text-friendly
    'textarea',                                          # Fallback
]
```

---

## File Organization

```
browser_ai/
├── __init__.py           # Version
├── config.py             # Tunables (import-safe: no package deps)
├── models.py             # Pydantic request/response
├── cleaning.py           # Text cleaning + DOM extraction
├── prompt.py             # Prompt building + history
├── tools.py              # Tool serialization
├── session.py            # SessionManager + SessionState
├── server.py             # FastAPI routes + SSE
├── cli.py                # Entry point
├── backends/
│   ├── __init__.py      # REGISTRY, get_backend_class()
│   ├── base.py          # BrowserBackend ABC + BaseBrowserBackend
│   ├── gemini.py        # Gemini selectors
│   ├── chatgpt.py       # ChatGPT selectors
│   └── perplexity.py    # Stub
└── compat/
    ├── __init__.py      # get_compat()
    └── continue_compat.py  # Continue patches
```

---

## API Contract

### Endpoints
- `GET /healthz` → status
- `GET /v1/models` → model list
- `POST /v1/chat/completions` → streaming/non-streaming
- `POST /v1/completions` → legacy (Continue edit mode)

### Required Response Fields
```python
{
    "id": f"chatcmpl-{uuid}",
    "object": "chat.completion",
    "created": now_ts(),
    "model": "browser-ai",
    "system_fingerprint": "browser-ai-playwright-v1",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {...}}],
    "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
}
```

---

## Performance

- Clipboard paste for prompts > 200 chars
- SSE streaming: `await asyncio.sleep(0)` between chunks
- Session cleanup: every 60s, 30min idle expiry
- Per-page lock: prevents concurrent requests
