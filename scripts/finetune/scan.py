#!/usr/bin/env python3
"""Scan the adapter's test answers with Pangram. Owner-run: this is the only paid step.

Dry run by default: counts, estimated cost and whether a Pangram key is configured; no network.
--run makes one paid Pangram scan per text, stores each result in the db and writes the scores
back into the outputs file. Texts are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

from engine.db import get_settings, init_db, persist_pangram_result  # noqa: E402
from engine.pangram_client import (  # noqa: E402
    PANGRAM_PASS_MAX_AI,
    PANGRAM_PASS_MAX_HUMANIZER,
    PangramError,
    estimate_pangram_cost,
    passes_pangram,
    resolve_pangram_key,
    scan_text_pangram,
)


def newest_outputs(layout: common.Layout) -> Path | None:
    files = sorted(layout.outputs.glob("*.json")) if layout.outputs.exists() else []
    return files[-1] if files else None


def collect_items(data: dict, include_baseline: bool) -> list[tuple[dict, str, dict]]:
    """(draft entry, label, variant dict) for every scannable text."""
    items = []
    for entry in data.get("drafts", []):
        for label in ("adapter", "baseline") if include_baseline else ("adapter",):
            v = entry.get(label)
            if v and v.get("text"):
                items.append((entry, label, v))
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="SQLite db (Pangram key setting, stored scans)")
    ap.add_argument("--work-dir", type=Path, default=None, help="Base directory (default data/finetune)")
    ap.add_argument("--file", type=Path, default=None, help="Outputs file to scan (default: newest in outputs/)")
    ap.add_argument("--include-baseline", action="store_true", help="Also scan the base-model outputs")
    ap.add_argument("--run", action="store_true", help="Make the paid Pangram scans")
    args = ap.parse_args(argv)

    layout = common.Layout(args.work_dir)
    path = args.file or newest_outputs(layout)
    if path is None or not Path(path).is_file():
        print(f"No outputs file found in {layout.outputs}. Run generate.py --run first.")
        return 1
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = collect_items(data, args.include_baseline)
    if not items:
        print("Nothing to scan: no output texts in that file.")
        return 1

    db_path = common.resolve_db(args.db)
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1
    init_db(db_path)
    settings = get_settings(db_path)
    key = resolve_pangram_key(settings)
    ests = [estimate_pangram_cost(v["text"]) for _, _, v in items]
    total_words = sum(e["word_count"] for e in ests)
    total_cost = sum(e["estimated_cost_usd"] for e in ests)

    if not args.run:
        print("DRY RUN (no network). Add --run to scan; this is the paid step.")
        print(f"File: {Path(path).name}")
        print(f"Texts: {len(items)} ({total_words} words)")
        print(f"Estimated Pangram cost: ${total_cost / len(items):.4f} per scan on average, ${total_cost:.4f} in total")
        print(f"Pangram key configured: {'yes' if key else 'no'}")
        return 0
    if not key:
        print("No Pangram key configured; set one in Settings or PANGRAM_API_KEY. Nothing was called.")
        return 1

    model = settings.get("pangram_model") or "pangram-4"
    public = settings.get("pangram_public_link", "false").lower() == "true"
    print(f"{'draft':<26} {'kind':<9} {'words':>5} {'AI%':>5} {'humanizer':>9} {'result':>6}")
    passed = 0
    for entry, label, v in items:
        words = common.word_count(v["text"])
        try:
            scan = scan_text_pangram(v["text"], api_key=key, model=model, public_dashboard_link=public)
            scan_id = persist_pangram_result(scan, f"finetune: {entry['source']}", db_path=db_path)
            ok = bool(passes_pangram(scan))
            v["pangram"] = {
                "scan_id": scan_id,
                "fraction_ai": scan.get("fraction_ai"),
                "fraction_human": scan.get("fraction_human"),
                "max_humanizer_score": scan.get("max_humanizer_score"),
                "headline": scan.get("headline"),
                "passed": ok,
            }
            passed += ok
            ai = f"{float(scan.get('fraction_ai') or 0) * 100:.0f}%"
            hum = float(scan.get("max_humanizer_score") or 0)
            print(f"{entry['source']:<26} {label:<9} {words:>5} {ai:>5} {hum:>9.2f} {'PASS' if ok else 'FAIL':>6}")
        except PangramError as err:
            v["pangram_error"] = err.message
            print(f"{entry['source']:<26} {label:<9} {words:>5} Pangram error: {err.message}")
        except Exception as err:  # one failed scan must not stop the rest
            v["pangram_error"] = str(err)[:200]
            print(f"{entry['source']:<26} {label:<9} {words:>5} Pangram error: {str(err)[:200]}")

    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(
        f"{passed} of {len(items)} passed Pangram "
        f"(AI <= {PANGRAM_PASS_MAX_AI * 100:.0f}%, humanizer < {PANGRAM_PASS_MAX_HUMANIZER * 100:.0f}%)"
    )
    print(f"Scores written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
