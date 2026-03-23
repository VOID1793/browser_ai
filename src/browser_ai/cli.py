"""
browser_ai.cli
~~~~~~~~~~~~~~
Command-line entry point for the browser-ai package.

Installed as the `browser-ai` console script via pyproject.toml.

Commands
--------
browser-ai serve     Start the OpenAI-compatible API server.
browser-ai chat      Send a single prompt or start an interactive session.
browser-ai backends  List available backends and their implementation status.

Examples
--------
# Gemini backend, Continue compat, default port
browser-ai serve --backend gemini --compat continue

# ChatGPT backend (runs in a visible browser window automatically)
browser-ai serve --backend chatgpt --compat continue

# Multiple instances on different ports (run in separate terminals)
browser-ai serve --backend gemini --port 8000
browser-ai serve --backend chatgpt --port 8001

# Force a visible window (useful for first-time setup or debugging)
browser-ai serve --backend chatgpt --no-headless

# Force headless (advanced — may not work for all backends)
browser-ai serve --backend gemini --headless

# Quiet mode — suppress browser-ai log lines, keep uvicorn access logs
browser-ai serve --backend gemini --quiet

# One-shot prompt
browser-ai chat --backend gemini "Explain async/await in Python"

# Interactive session
browser-ai chat --backend gemini --session
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="browser-ai",
        description="OpenAI-compatible API server backed by browser automation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=False)

    # ── serve ─────────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the OpenAI-compatible API server.")
    serve_p.add_argument(
        "--backend", default="gemini",
        choices=["gemini", "chatgpt", "perplexity"],
        help="Browser backend to use (default: gemini).",
    )
    serve_p.add_argument(
        "--compat", default=None,
        choices=["continue"],
        help="Enable client-specific compatibility patches (default: none).",
    )
    serve_p.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host (default: 127.0.0.1). Use 0.0.0.0 to accept external connections.",
    )
    serve_p.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000). Run separate instances on different ports for multi-model setups.",
    )
    serve_p.add_argument(
        "--quiet", action="store_true",
        help="Reduce log verbosity (suppress browser-ai internal logs; keep HTTP access logs).",
    )

    headless_group = serve_p.add_mutually_exclusive_group()
    headless_group.add_argument(
        "--headless", dest="headless", action="store_true", default=None,
        help="Force headless mode regardless of backend default.",
    )
    headless_group.add_argument(
        "--no-headless", dest="headless", action="store_false",
        help="Force a visible browser window. Useful for first-time login or debugging.",
    )

    # ── chat ──────────────────────────────────────────────────────────────────
    chat_p = sub.add_parser("chat", help="Send a prompt or start an interactive session.")
    chat_p.add_argument(
        "--backend", default="gemini",
        choices=["gemini", "chatgpt", "perplexity"],
        help="Browser backend to use (default: gemini).",
    )
    chat_p.add_argument(
        "prompt", nargs="?", default=None,
        help="Prompt to send. If omitted, starts an interactive session.",
    )
    chat_p.add_argument(
        "--session", action="store_true",
        help="Force interactive session mode even if a prompt argument is given.",
    )
    headless_group_chat = chat_p.add_mutually_exclusive_group()
    headless_group_chat.add_argument(
        "--headless", dest="headless", action="store_true", default=None,
        help="Force headless mode.",
    )
    headless_group_chat.add_argument(
        "--no-headless", dest="headless", action="store_false",
        help="Force visible browser window.",
    )

    # ── backends ──────────────────────────────────────────────────────────────
    sub.add_parser("backends", help="List available backends and their status.")

    return parser


# ── Sub-command implementations ───────────────────────────────────────────────

def cmd_backends(_args) -> None:
    from browser_ai.backends import REGISTRY
    print("\nAvailable backends:\n")
    statuses = {
        "gemini":     ("✓ working",    "headless",          "https://gemini.google.com/app"),
        "chatgpt":    ("✓ working",    "visible",           "https://chatgpt.com"),
        "perplexity": ("○ stub only",  "headless",          "https://www.perplexity.ai"),
    }
    print(f"  {'NAME':<14} {'STATUS':<16} {'DEFAULT MODE':<16} URL")
    print(f"  {'-'*13} {'-'*15} {'-'*15} {'-'*32}")
    for name in sorted(REGISTRY):
        status, mode, url = statuses.get(name, ("?", "?", ""))
        print(f"  {name:<14} {status:<16} {mode:<16} {url}")
    print()


def _resolve_headless(args) -> Optional[bool]:
    """
    Resolve the effective headless flag.

    Priority (highest first):
      1. Explicit CLI flag (--headless or --no-headless)
      2. None — let the backend constructor use its own DEFAULT_HEADLESS
    """
    return getattr(args, "headless", None)


def cmd_serve(args) -> None:
    import uvicorn
    from browser_ai.backends import get_backend_class
    from browser_ai.compat import get_compat
    from browser_ai.server import build_app

    try:
        backend_class = get_backend_class(args.backend)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        compat = get_compat(args.compat)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    headless = _resolve_headless(args)
    quiet = getattr(args, "quiet", False)

    effective_headless = headless if headless is not None else backend_class.DEFAULT_HEADLESS
    if not effective_headless and getattr(backend_class, "MINIMIZE_WINDOW", False):
        mode_str = "visible+minimized"
    elif effective_headless:
        mode_str = "headless"
    else:
        mode_str = "visible"

    compat_label = args.compat or "none"

    if not quiet:
        print(
            f"[browser-ai] backend={args.backend}  compat={compat_label}  "
            f"mode={mode_str}  http://{args.host}:{args.port}",
            flush=True,
        )

    app = build_app(
        backend_class=backend_class,
        headless=headless,
        compat=compat,
        quiet=quiet,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning" if quiet else "info",
    )


async def _ask_once(backend_name: str, prompt: str, headless: Optional[bool]) -> str:
    from browser_ai.backends import get_backend_class
    backend_class = get_backend_class(backend_name)
    backend = backend_class(headless=headless)
    try:
        await backend.start()
        return await backend.ask(prompt)
    finally:
        await backend.close()


async def _interactive(backend_name: str, headless: Optional[bool]) -> None:
    from browser_ai.backends import get_backend_class
    backend_class = get_backend_class(backend_name)
    backend = backend_class(headless=headless)
    try:
        await backend.start()
        print("[browser-ai] Interactive session started (Ctrl+C to quit)", file=sys.stderr)
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            try:
                resp = await backend.ask(prompt)
                print(f"\n{resp}\n", flush=True)
            except Exception as exc:
                print(f"[error] {exc}", file=sys.stderr)
    finally:
        await backend.close()


def cmd_chat(args) -> None:
    headless = _resolve_headless(args)
    interactive = getattr(args, "session", False) or args.prompt is None

    if interactive:
        asyncio.run(_interactive(args.backend, headless))
    else:
        result = asyncio.run(_ask_once(args.backend, args.prompt, headless))
        print(result)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cmd = args.cmd or "serve"

    if cmd == "serve":
        cmd_serve(args)
    elif cmd == "chat":
        cmd_chat(args)
    elif cmd == "backends":
        cmd_backends(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()