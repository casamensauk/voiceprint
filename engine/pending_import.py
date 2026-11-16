"""In-memory pending batch for Google imports: review first, write to the samples table only on confirm."""

from __future__ import annotations

import random
import threading
from pathlib import Path
from typing import Any

from engine.db import commit_import_batch

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

_lock = threading.RLock()
_commit_lock = threading.Lock()
_batch: dict[str, Any] | None = None
_generation = 0
_active = ""  # "scan" or "upload" while one is running


def new_counts() -> dict[str, int]:
    return {k: 0 for k in COUNT_KEYS}


class ScanProgress:
    """Thread-safe progress of the background scan, with a cancel flag."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._cancel = threading.Event()
        self.reset()

    def reset(self, **fields: Any) -> None:
        with self._lock:
            self._data = {"running": False, "stage": "idle", "done": 0, "total": 0, "error": ""}
            self._data.update(fields)
        self._cancel.clear()

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._data.update(fields)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def request_cancel(self) -> bool:
        """Ask the running scan to stop. False when no scan is running."""
        if not self.snapshot()["running"]:
            return False
        self._cancel.set()
        return True

    def cancelled(self) -> bool:
        return self._cancel.is_set()


PROGRESS = ScanProgress()


def _words(text: str) -> int:
    return len(text.split())


def current_generation() -> int:
    with _lock:
        return _generation


def active() -> str:
    """What is running right now: "scan", "upload" or "" (nothing)."""
    with _lock:
        if _active:
            return _active
    return "scan" if PROGRESS.snapshot()["running"] else ""


def claim(kind: str) -> str:
    """Start a scan or an upload. Returns "" on success, otherwise a plain-language reason it cannot start."""
    global _active
    with _lock:
        if _active == "scan" or (not _active and PROGRESS.snapshot()["running"]):
            return "A scan is already running."
        if _active == "upload":
            return "A file is already being read."
        if _batch is not None:
            return "Import or discard the list you are reviewing first."
        _active = kind
        return ""


def release(kind: str) -> None:
    global _active
    with _lock:
        if _active == kind:
            _active = ""


def set_pending(
    source_type: str,
    context_mode: str,
    texts: list[str],
    ids: list[tuple[str, ...]],
    counts: dict[str, int],
    expected_generation: int | None = None,
    partial: bool = False,
    partial_reason: str = "",
) -> bool:
    """Park a batch for review. False (and nothing parked) if a discard or a newer batch got in first."""
    global _batch, _generation
    counts = {**new_counts(), **counts, "kept": len(texts)}
    items = [(i[0], i[1], i[2] if len(i) > 2 else "kept") for i in ids]
    with _lock:
        if expected_generation is not None and expected_generation != _generation:
            return False
        _generation += 1
        _batch = {
            "source_type": source_type,
            "context_mode": context_mode,
            "texts": list(texts),
            "ids": items,
            "counts": counts,
            "total_words": sum(_words(t) for t in texts),
            "partial": bool(partial),
            "partial_reason": partial_reason if partial else "",
            "generation": _generation,
        }
        return True


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
            "partial": b["partial"],
            "partial_reason": b["partial_reason"],
        }
    picks = random.sample(texts, min(EXCERPT_COUNT, len(texts)))
    out["samples"] = [t if len(t) <= EXCERPT_CHARS else t[: EXCERPT_CHARS - 1].rstrip() + "…" for t in picks]
    return out


def commit_pending(db_path: Path | None = None) -> dict[str, Any]:
    """Write the kept texts as ordinary samples and remember every processed message id, in one transaction.

    The batch is cleared only after the transaction commits; on failure it is kept so the user can retry.
    """
    global _batch, _generation
    with _commit_lock:
        with _lock:
            b = _batch
        if b is None:
            return {"inserted": 0, "committed": False}
        try:
            inserted = commit_import_batch(
                b["texts"],
                b["ids"],
                context_mode=b["context_mode"],
                source_type=b["source_type"],
                weight=1.0,
                db_path=db_path,
            )
        except Exception:
            return {
                "inserted": 0,
                "committed": False,
                "error": "Your samples could not be saved, so nothing was imported. Your list is still here; try again.",
            }
        with _lock:
            if _batch is b:
                _batch = None
                _generation += 1
        return {"inserted": inserted, "committed": True}


def discard_pending(db_path: Path | None = None) -> bool:
    """Drop the batch. Anything still being read is dropped too when it finishes."""
    global _batch, _generation
    with _commit_lock:
        with _lock:
            b = _batch
            busy = bool(_active) or PROGRESS.snapshot()["running"]
            if b is not None or busy:
                _generation += 1
            _batch = None
    if b is not None and not b["texts"] and not b["partial"] and b["ids"]:
        # Nothing was worth importing: still remember the mail that was read, so it is not fetched again.
        try:
            commit_import_batch([], b["ids"], b["context_mode"], b["source_type"], db_path=db_path)
        except Exception:
            pass
    if not PROGRESS.snapshot()["running"]:
        PROGRESS.reset()
    return b is not None
