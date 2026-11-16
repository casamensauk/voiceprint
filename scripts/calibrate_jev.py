#!/usr/bin/env python3
"""Compare Jev's detection answers with your stored Pangram results. Dry run by default.

Run with --run yourself to make the paid calls. Texts are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.ai_detector.agreement import MIN_CHARS, PANGRAM_SCAN_TYPES  # noqa: E402
from engine.db import DEFAULT_DB_PATH, get_connection, get_jev_estimate, get_settings, init_db, save_jev_estimate  # noqa: E402
from engine.jev_client import DEFAULT_JEV_MODEL, JEV_INPUT_USD_PER_MTOK, JevError  # noqa: E402
from engine.jev_judge import judge_detection  # noqa: E402

TOKENS_PER_WORD = 1.4
TOKENS_PER_REQUEST = 150
APP_SUPPORT_DB = Path.home() / "Library" / "Application Support" / "VoicePrint" / "voiceprint.db"


def _load_rows(db_path: Path) -> list[dict]:
    marks = ",".join("?" for _ in PANGRAM_SCAN_TYPES)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, input_text, fraction_ai, max_humanizer_score FROM pangram_scans "
            f"WHERE scan_type IN ({marks}) ORDER BY id",
            PANGRAM_SCAN_TYPES,
        ).fetchall()
    return [dict(r) for r in rows if len((r["input_text"] or "").strip()) >= MIN_CHARS]


def _mean(vals: list[float]) -> str:
    return f"{sum(vals) / len(vals):.2f}" if vals else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=None, help="SQLite db holding pangram_scans")
    ap.add_argument("--run", action="store_true", help="Actually call Jev (costs money). Default is a dry run.")
    ap.add_argument("--force", action="store_true", help="Re-query texts already in jev_estimates")
    args = ap.parse_args(argv)

    db_path = args.db or (APP_SUPPORT_DB if APP_SUPPORT_DB.exists() else DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1
    rows = _load_rows(db_path)

    if not args.run:
        words = sum(len(r["input_text"].split()) for r in rows)
        tokens = int(words * TOKENS_PER_WORD + TOKENS_PER_REQUEST * len(rows))
        cost = tokens * JEV_INPUT_USD_PER_MTOK / 1_000_000
        print("DRY RUN (no network). Add --run to send.")
        print(f"Texts that would be sent: {len(rows)}")
        print(f"Total words: {words}")
        print(f"Estimated input tokens: {tokens}")
        print(f"Estimated cost: ${cost:.4f}")
        print("These texts would be sent to api.typesafe.ai.")
        return 0

    init_db(db_path)
    settings = get_settings(db_path)
    results = []
    tokens_total = 0
    cost_total = 0.0
    try:
        for r in rows:
            text = r["input_text"]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            est = None if args.force else get_jev_estimate(digest, db_path)
            if est is None:
                j = judge_detection(text, settings=settings)
                in_tok = int(j["usage"].get("input_tokens") or 0)
                save_jev_estimate(digest, j["model"], j["model_written"], j["paraphrased"], in_tok, j["cost_usd"], db_path)
                tokens_total += in_tok
                cost_total += j["cost_usd"]
                est = {"model_written": j["model_written"], "paraphrased": j["paraphrased"]}
            results.append((r, len(text.split()), est))
    except JevError as err:
        print(f"Jev error ({err.status_code}): {err.message}")
        return 1

    print(f"{'scan':>5} {'words':>6} {'pg_ai':>6} {'pg_hum':>6} {'jev_ai':>7} {'jev_para':>8}")
    for r, words, est in results:
        print(
            f"{r['id']:>5} {words:>6} {float(r['fraction_ai']):>6.2f} {float(r['max_humanizer_score']):>6.2f} "
            f"{est['model_written']:>7.2f} {est['paraphrased']:>8.2f}"
        )

    n = len(results)
    ai = [e["model_written"] for r, _, e in results if float(r["fraction_ai"]) >= 0.5]
    hu = [e["model_written"] for r, _, e in results if float(r["fraction_ai"]) < 0.5]
    agree = sum((e["model_written"] >= 0.5) == (float(r["fraction_ai"]) >= 0.5) for r, _, e in results)
    hz = [e["paraphrased"] for r, _, e in results if float(r["max_humanizer_score"]) >= 0.5]
    nh = [e["paraphrased"] for r, _, e in results if float(r["max_humanizer_score"]) < 0.5]
    agree_p = sum((e["paraphrased"] >= 0.5) == (float(r["max_humanizer_score"]) >= 0.5) for r, _, e in results)
    print()
    print(f"model_written agrees with Pangram AI verdict: {agree} of {n}")
    print(f"mean model_written, Pangram-AI group: {_mean(ai)} ({len(ai)}); Pangram-human group: {_mean(hu)} ({len(hu)})")
    print(f"paraphrased agrees with Pangram humanizer verdict: {agree_p} of {n}")
    print(f"mean paraphrased, humanizer group: {_mean(hz)} ({len(hz)}); other group: {_mean(nh)} ({len(nh)})")
    print(f"Input tokens this run: {tokens_total}; actual cost: ${cost_total:.4f} (model {DEFAULT_JEV_MODEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
