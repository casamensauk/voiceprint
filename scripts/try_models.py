#!/usr/bin/env python3
"""Rewrite one draft with several models and (optionally) scan each result with Pangram.

Dry run by default: no network, nothing is sent. Run it yourself:
  --run   makes one paid OpenRouter call per model
  --scan  also makes one paid Pangram scan per rewrite (implies --run)
The draft text is never printed. Rewrites are saved to data/experiments/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.db import DEFAULT_DB_PATH, get_connection, get_settings, init_db, persist_pangram_result  # noqa: E402
from engine.pangram_client import (  # noqa: E402
    PANGRAM_PASS_MAX_AI,
    PANGRAM_PASS_MAX_HUMANIZER,
    PangramError,
    estimate_pangram_cost,
    passes_pangram,
    resolve_pangram_key,
    scan_text_pangram,
)
from engine.rewriter import rewrite_with_model  # noqa: E402

APP_SUPPORT_DB = Path.home() / "Library" / "Application Support" / "VoicePrint" / "voiceprint.db"
DEFAULT_MODELS = "anthropic/claude-haiku-5.5,openai/gpt-6-luna,mistralai/mistral-large-4-0"
MIN_DRAFT_CHARS = 50


def _load_draft(args: argparse.Namespace, db_path: Path) -> tuple[str, str]:
    """Return (draft_text, source description). Raises ValueError with a clear message."""
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            raise ValueError(f"No such file: {path}")
        return path.read_text(encoding="utf-8"), f"file {path.name}"
    with get_connection(db_path) as conn:
        if args.history_id is not None:
            row = conn.execute("SELECT id, input_text FROM rewrite_history WHERE id = ?", (args.history_id,)).fetchone()
            if row is None:
                raise ValueError(f"No rewrite_history row with id {args.history_id}")
        else:
            row = conn.execute("SELECT id, input_text FROM rewrite_history ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                raise ValueError("No draft: rewrite_history is empty. Use --file PATH.")
    return row["input_text"] or "", f"rewrite_history id {row['id']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="SQLite db (settings, voice corpus, history)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--file", type=Path, default=None, help="UTF-8 text file holding the draft")
    src.add_argument("--history-id", type=int, default=None, help="Use input_text of this rewrite_history row")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated OpenRouter model slugs")
    ap.add_argument("--mode", default=None, help="Context mode (default: from settings)")
    ap.add_argument("--intensity", default=None, help="Intensity (default: from settings, else deep)")
    ap.add_argument("--run", action="store_true", help="Make the paid OpenRouter rewrite calls")
    ap.add_argument("--scan", action="store_true", help="Also scan each rewrite with Pangram (paid; implies --run)")
    ap.add_argument("--show", action="store_true", help="Print the rewrite texts to the terminal")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write the JSON (default data/experiments)")
    args = ap.parse_args(argv)
    run = args.run or args.scan

    db_path = args.db or (APP_SUPPORT_DB if APP_SUPPORT_DB.exists() else DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1
    init_db(db_path)

    try:
        draft, source = _load_draft(args, db_path)
    except (ValueError, OSError) as err:
        print(str(err))
        return 1
    draft = draft.strip()
    if len(draft) < MIN_DRAFT_CHARS:
        print(f"Draft is too short ({len(draft)} characters, need at least {MIN_DRAFT_CHARS}).")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models given.")
        return 1

    settings = get_settings(db_path)
    openrouter_key = (settings.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    pangram_key = resolve_pangram_key(settings)
    est = estimate_pangram_cost(draft)

    if not run:
        per = est["estimated_cost_usd"]
        print("DRY RUN (no network). Add --run to rewrite, --scan to also scan with Pangram.")
        print(f"Draft: {est['word_count']} words ({source})")
        print(f"Models: {', '.join(models)}")
        print(f"Model calls: {len(models)}")
        print(f"Estimated Pangram cost: ${per:.4f} per scan, ${per * len(models):.4f} for {len(models)} scans")
        print(f"OpenRouter key configured: {'yes' if openrouter_key else 'no'}")
        print(f"Pangram key configured: {'yes' if pangram_key else 'no'}")
        return 0

    if not openrouter_key:
        print("No OpenRouter key configured; set one in Settings or OPENROUTER_API_KEY. Nothing was called.")
        return 1
    if args.scan and not pangram_key:
        print("No Pangram key configured; set one in Settings or PANGRAM_API_KEY. Nothing was called.")
        return 1

    pg_model = settings.get("pangram_model") or "pangram-4"
    pg_public = settings.get("pangram_public_link", "false").lower() == "true"

    rows: list[dict] = []
    for model in models:
        try:
            res = rewrite_with_model(draft, model, context_mode=args.mode, intensity=args.intensity, db_path=db_path)
        except Exception as err:  # one model failing must not stop the rest
            res = {"model": model, "rewritten": "", "error": str(err)[:200]}
        row: dict = {**res, "model": model}
        text = row.get("rewritten") or ""
        if text and args.scan:
            try:
                scan = scan_text_pangram(text, api_key=pangram_key, model=pg_model, public_dashboard_link=pg_public)
                scan_id = persist_pangram_result(scan, f"experiment: {model}", db_path=db_path)
                row["pangram"] = {
                    "scan_id": scan_id,
                    "fraction_ai": scan.get("fraction_ai"),
                    "fraction_human": scan.get("fraction_human"),
                    "max_humanizer_score": scan.get("max_humanizer_score"),
                    "headline": scan.get("headline"),
                    "passed": passes_pangram(scan),
                }
            except PangramError as err:
                row["pangram_error"] = err.message
            except Exception as err:
                row["pangram_error"] = str(err)[:200]
        rows.append(row)

    header = f"{'model':<34} {'words':>5} {'in-house':>8} {'hidden':>6}"
    if args.scan:
        header += f" {'AI%':>5} {'humanizer':>9} {'result':>6}"
    print(header)
    passed = 0
    for row in rows:
        text = row.get("rewritten") or ""
        if row.get("error") or not text:
            print(f"{row['model']:<34} ERROR: {row.get('error') or 'empty response'}")
            continue
        hidden = int((row.get("hidden_characters") or {}).get("removed") or 0)
        line = f"{row['model']:<34} {len(text.split()):>5} {row.get('inhouse_human_score', 0):>8} {hidden:>6}"
        if args.scan:
            pg = row.get("pangram")
            if pg:
                ok = bool(pg["passed"])
                passed += ok
                ai_pct = f"{float(pg['fraction_ai']) * 100:.0f}%"
                line += f" {ai_pct:>5} {float(pg['max_humanizer_score']):>9.2f} {'PASS' if ok else 'FAIL':>6}"
            else:
                line += f" Pangram error: {row.get('pangram_error', 'unknown')}"
        print(line)

    print()
    if args.scan:
        print(
            f"{passed} of {len(rows)} rewrites passed Pangram "
            f"(AI <= {PANGRAM_PASS_MAX_AI * 100:.0f}%, humanizer < {PANGRAM_PASS_MAX_HUMANIZER * 100:.0f}%)"
        )
    else:
        print("No Pangram scan was run (add --scan).")

    out_dir = args.out_dir or (ROOT / "data" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = out_dir / f"{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "draft_source": source,
                "draft_words": est["word_count"],
                "mode": args.mode,
                "intensity": args.intensity,
                "scanned": bool(args.scan),
                "results": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {out_path}")

    if args.show:
        for row in rows:
            print(f"\n--- {row['model']} ---")
            print(row.get("rewritten") or f"(none: {row.get('error', 'empty')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
