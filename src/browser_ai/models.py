"""
browser_ai.models
~~~~~~~~~~~~~~~~~
Pydantic models for the OpenAI-compatible API wire format.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from browser_ai.config import MODEL_ID


# ── Inbound ───────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any                        # str | list[ContentPart] | None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    # "auto" | "none" | {"type": "function", "function": {"name": "..."}}
    # Accepted and stored but not enforced — the backend decides whether to call tools.
    tool_choice: Optional[Any] = None


class CompletionRequest(BaseModel):
    """Legacy /v1/completions request (used by Continue edit mode)."""
    model: str
    prompt: Any                         # str | list[str]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    suffix: Optional[str] = None        # present in some edit-mode payloads; ignored


# ── Outbound ──────────────────────────────────────────────────────────────────

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
