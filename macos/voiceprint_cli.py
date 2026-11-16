#!/usr/bin/env python3
"""macOS CLI helper: grabs clipboard -> rewrites in active VoicePrint DNA -> copies back."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.rewriter import rewrite_text  # noqa: E402


_UTF8_ENV = {**os.environ, "LANG": "en_US.UTF-8", "LC_CTYPE": "UTF-8"}


def read_clipboard() -> str:
    """Read text from the macOS clipboard using pbpaste."""
    proc = subprocess.run(
        ["pbpaste"], capture_output=True, text=True, encoding="utf-8", env=_UTF8_ENV, check=False
    )
    return proc.stdout or ""


def write_clipboard(text: str) -> None:
    """Write text to the macOS clipboard using pbcopy."""
    subprocess.run(
        ["pbcopy"], input=text, text=True, encoding="utf-8", env=_UTF8_ENV, check=True
    )


def notify_macos(title: str, subtitle: str, message: str) -> None:
    """Display a native macOS notification banner."""
    safe_title = title.replace('"', '\\"')
    safe_sub = subtitle.replace('"', '\\"')
    safe_msg = message.replace('"', '\\"')
    script = f'display notification "{safe_msg}" with title "{safe_title}" subtitle "{safe_sub}"'
    subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)


def rewrite_via_server_or_local(
    text: str,
    context_mode: str | None = None,
    intensity: str | None = None,
) -> dict:
    """Use running VoicePrint daemon if reachable, otherwise invoke the local engine directly."""
    port = os.environ.get("PORT", "8765")
    url = f"http://127.0.0.1:{port}/api/rewrite"
    payload = {"text": text, "auto_copy": False}
    if context_mode:
        payload["context_mode"] = context_mode
    if intensity:
        payload["intensity"] = intensity

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return rewrite_text(text, context_mode=context_mode, intensity=intensity, save_history=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="VoicePrint clipboard rewriter")
    parser.add_argument("--mode", choices=["casual_chat", "work_email", "long_form", "social_post"], default=None)
    parser.add_argument("--intensity", choices=["subtle", "natural", "deep"], default=None)
    parser.add_argument("--stdin", action="store_true", help="Read from stdin instead of clipboard")
    parser.add_argument("--dry-run", action="store_true", help="Print result instead of writing to clipboard")
    parser.add_argument("--no-notify", action="store_true", help="Suppress macOS banner notification")
    args = parser.parse_args()

    source = sys.stdin.read() if args.stdin else read_clipboard()
    if not source.strip():
        if not args.no_notify:
            notify_macos("VoicePrint", "Clipboard is empty", "Copy some text first, then run VoicePrint.")
        print("VoicePrint: clipboard is empty.", file=sys.stderr)
        return 1

    result = rewrite_via_server_or_local(source, context_mode=args.mode, intensity=args.intensity)
    rewritten = result["rewritten"]
    after = result["after"]
    mode = result["context_mode"]

    if not args.dry_run:
        write_clipboard(rewritten)

    if not args.no_notify and not args.dry_run:
        notify_macos(
            "VoicePrint — Clipboard Rewritten",
            f"Mode: {mode} | Provider: {result['provider_used']}",
            f"Human Score: {after['human_score']}% | Burstiness: {after['burstiness_score']}% | Match: {after['style_match_score']}%",
        )

    print(rewritten)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
