"""Local agreement between stored Pangram API scans and the in-house estimate (offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..analyzer import analyze_text
from ..db import get_connection, init_db

PANGRAM_SCAN_TYPES = ("text", "file")  # in-house rows use "inhouse"/"inhouse_file"
MIN_CHARS = 50

# Re-scoring every stored scan costs ~0.25s, and the state payload is loaded often,
# so reuse the result until a scan is added or removed.
_cache: dict[tuple[str, int, int], dict[str, Any]] = {}


def pangram_agreement(db_path: Path | None = None) -> dict[str, Any]:
    """Compare Pangram's AI verdict (fraction_ai >= 0.5) with in-house (human_score < 50). No text returned."""
    init_db(db_path)
    marks = ",".join("?" for _ in PANGRAM_SCAN_TYPES)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, input_text, fraction_ai FROM pangram_scans WHERE scan_type IN ({marks}) ORDER BY id DESC",
            PANGRAM_SCAN_TYPES,
        ).fetchall()

    cache_key = (str(db_path), len(rows), int(rows[0]["id"]) if rows else 0)
    if cache_key in _cache:
        return _cache[cache_key]

    scans = agree = 0
    disagreements: list[dict[str, Any]] = []
    for row in rows:
        text = row["input_text"] or ""
        if len(text.strip()) < MIN_CHARS:
            continue
        human = int(analyze_text(text)["human_score"])
        frac = float(row["fraction_ai"])
        scans += 1
        if (frac >= 0.5) == (human < 50):
            agree += 1
        else:
            disagreements.append(
                {"id": int(row["id"]), "pangram_fraction_ai": frac, "inhouse_human_score": human}
            )
    _cache.clear()
    _cache[cache_key] = result = {
        "scans": scans,
        "agree": agree,
        "rate": (agree / scans) if scans else None,
        "disagreements": disagreements,
    }
    return result
