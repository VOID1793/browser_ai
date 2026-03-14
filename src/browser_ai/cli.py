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

# ChatGPT backend, visible browser for first-time login
browser-ai serve --backend chatgpt --visible

# One-shot prompt
browser-ai chat --backend gemini "Explain async/await in Python"

# Interactive session with visible browser
browser-ai chat --backend gemini --session --visible
"""

from __future__ import annotations

import argparse
import asyncio
import sys


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
        choices=["gemini", "chatgpt", "grok", "perplexity"],
        help="Browser backend to use (default: gemini).",
    )
    serve_p.add_argument(
        "--compat", default=None,
        choices=["continue"],
        help="Enable client-specific compatibility patches (default: none).",
    )
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    serve_p.add_argument(
        "--visible", action="store_true",
        help="Show the browser window (disables headless mode). "
             "Useful for first-time login or debugging.",
    )

    # ── chat ──────────────────────────────────────────────────────────────────
    chat_p = sub.add_parser("chat", help="Send a prompt or start an interactive session.")
    chat_p.add_argument(
        "--backend", default="gemini",
        choices=["gemini", "chatgpt", "grok", "perplexity"],
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
    chat_p.add_argument(
        "--visible", action="store_true",
        help="Show the browser window.",
    )

    # ── backends ──────────────────────────────────────────────────────────────
    sub.add_parser("backends", help="List available backends and their status.")

    return parser


# ── Sub-command implementations ───────────────────────────────────────────────

def cmd_backends(_args) -> None:
    from browser_ai.backends import REGISTRY
    print("\nAvailable backends:\n")
    statuses = {
        "gemini":     ("✓ implemented", "https://gemini.google.com/app"),
        "chatgpt":    ("○ stub only",   "https://chat.openai.com"),
        "grok":       ("○ stub only",   "https://grok.com"),
        "perplexity": ("○ stub only",   "https://www.perplexity.ai"),
    }
    for name in sorted(REGISTRY):
        status, url = statuses.get(name, ("?", ""))
        print(f"  {name:<14} {status:<20} {url}")
    print()


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

    headless = not args.visible
    app = build_app(backend_class=backend_class, headless=headless, compat=compat)

    compat_label = args.compat or "none"
    print(
        f"[browser-ai] backend={args.backend}  compat={compat_label}  "
        f"headless={headless}  http://{args.host}:{args.port}",
        flush=True,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


async def _ask_once(backend_name: str, prompt: str, visible: bool) -> str:
    from browser_ai.backends import get_backend_class
    backend_class = get_backend_class(backend_name)
    backend = backend_class(headless=not visible)
    try:
        await backend.start()
        return await backend.ask(prompt)
    finally:
        await backend.close()


async def _interactive(backend_name: str, visible: bool) -> None:
    from browser_ai.backends import get_backend_class
    backend_class = get_backend_class(backend_name)
    backend = backend_class(headless=not visible)
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
    visible = getattr(args, "visible", False)
    interactive = getattr(args, "session", False) or args.prompt is None

    if interactive:
        asyncio.run(_interactive(args.backend, visible))
    else:
        result = asyncio.run(_ask_once(args.backend, args.prompt, visible))
        print(result)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cmd = args.cmd or "serve"   # default to serve if no subcommand given

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
