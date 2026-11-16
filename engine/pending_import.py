"""In-memory pending batch for Google imports: review first, write to the samples table only on confirm."""

from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any

from engine.db import add_samples_bulk, record_imported_ids

COUNT_KEYS = (
    "listed",
    "already_imported",
    "not_user",
    "automatic",
    "empty",
    "too_short",
    "duplicates",
    "failed",
    "kept",
)
EXCERPT_CHARS = 300
EXCERPT_COUNT = 5

_lock = threading.Lock()
_batch: dict[str, Any] | None = None


def new_counts() -> dict[str, int]:
    return {k: 0 for k in COUNT_KEYS}


class ScanProgress:
    """Thread-safe progress of the background scan."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self.reset()

    def reset(self, **fields: Any) -> None:
        with self._lock:
            self._data = {"running": False, "stage": "idle", "done": 0, "total": 0, "error": ""}
            self._data.update(fields)

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._data.update(fields)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


PROGRESS = ScanProgress()


def _words(text: str) -> int:
    return len(text.split())


def set_pending(
    source_type: str,
    context_mode: str,
    texts: list[str],
    ids: list[tuple[str, str]],
    counts: dict[str, int],
) -> None:
    global _batch
    counts = {**new_counts(), **counts, "kept": len(texts)}
    with _lock:
        _batch = {
            "source_type": source_type,
            "context_mode": context_mode,
            "texts": list(texts),
            "ids": list(ids),
            "counts": counts,
            "total_words": sum(_words(t) for t in texts),
        }


def has_pending() -> bool:
    with _lock:
        return _batch is not None


def get_summary() -> dict[str, Any]:
    """Counts, total words and up to 5 random excerpts. Never the full batch."""
    with _lock:
        b = _batch
        if b is None:
            return {"has_pending": False}
        texts = list(b["texts"])
        out = {
            "has_pending": True,
            "source_type": b["source_type"],
            "context_mode": b["context_mode"],
            "counts": dict(b["counts"]),
            "total_words": b["total_words"],
        }
    picks = random.sample(texts, min(EXCERPT_COUNT, len(texts)))
    out["samples"] = [t if len(t) <= EXCERPT_CHARS else t[: EXCERPT_CHARS - 1].rstrip() + "…" for t in picks]
    return out


def commit_pending(db_path: Path | None = None) -> dict[str, Any]:
    """Write the kept texts as ordinary samples and remember their message ids."""
    global _batch
    with _lock:
        b = _batch
        _batch = None
    if b is None:
        return {"inserted": 0, "committed": False}
    inserted = add_samples_bulk(
        b["texts"], context_mode=b["context_mode"], source_type=b["source_type"], weight=1.0, db_path=db_path
    )
    by_source: dict[str, list[str]] = {}
    for source, ext_id in b["ids"]:
        by_source.setdefault(source, []).append(ext_id)
    for source, ids in by_source.items():
        record_imported_ids(source, ids, db_path)
    return {"inserted": inserted, "committed": True}


def discard_pending() -> bool:
    global _batch
    with _lock:
        had = _batch is not None
        _batch = None
    if not PROGRESS.snapshot()["running"]:
        PROGRESS.reset()
    return had
