"""Read a Google Chat Takeout export (messages.json, or a Takeout .zip) and keep only the user's own messages."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from engine import pending_import
from engine.db import get_imported_ids, get_sample_contents
from engine.own_text import is_substantive, mask_private_details

MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_MESSAGES_JSON_BYTES = 60 * 1024 * 1024
MAX_SAMPLE_WORDS = 400
SOURCE = "google_chat"
_MEMBER_RE = re.compile(r"(?:^|/)Google Chat/Groups/[^/]+/messages\.json$", re.I)


class ChatTakeoutError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _unsafe_name(name: str) -> bool:
    norm = name.replace("\\", "/")
    return norm.startswith("/") or bool(re.match(r"^[A-Za-z]:", norm)) or ".." in PurePosixPath(norm).parts


def read_message_files(data: bytes) -> list[dict[str, Any]]:
    """Return the parsed messages.json documents from raw bytes (a .json file or a Takeout zip)."""
    if len(data) > MAX_UNCOMPRESSED_BYTES:
        raise ChatTakeoutError("That file is larger than 200 MB.", 413)
    if not zipfile.is_zipfile(io.BytesIO(data)):
        if len(data) > MAX_MESSAGES_JSON_BYTES:
            raise ChatTakeoutError("That messages.json file is larger than 60 MB.", 413)
        return [_load_json(data)]
    docs: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            for info in infos:
                if _unsafe_name(info.filename):
                    raise ChatTakeoutError("The archive contains an unsafe file path, so it was not opened.")
            if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
                raise ChatTakeoutError("The archive is larger than 200 MB once unpacked.", 413)
            budget = MAX_UNCOMPRESSED_BYTES
            for info in infos:
                if info.is_dir() or not _MEMBER_RE.search(info.filename.replace("\\", "/")):
                    continue
                limit = min(budget, MAX_MESSAGES_JSON_BYTES)
                if info.file_size > limit:
                    raise ChatTakeoutError("A messages.json file in the archive is larger than 60 MB.", 413)
                with zf.open(info) as fh:
                    raw = fh.read(limit + 1)
                if len(raw) > limit:
                    raise ChatTakeoutError("The archive is larger than 200 MB once unpacked.", 413)
                budget -= len(raw)
                docs.append(_load_json(raw))
    except zipfile.BadZipFile:
        raise ChatTakeoutError("That zip file could not be read.") from None
    if not docs:
        raise ChatTakeoutError("No Google Chat messages were found in that archive.")
    return docs


def _load_json(raw: bytes) -> dict[str, Any]:
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError):
        raise ChatTakeoutError("That file is not a Google Chat messages.json export.") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("messages"), list):
        raise ChatTakeoutError("That file is not a Google Chat messages.json export.")
    return doc


def _creator_email(msg: dict[str, Any]) -> str:
    creator = msg.get("creator")
    if isinstance(creator, dict):
        return str(creator.get("email") or "").strip().lower()
    return ""


def _key(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def _chunks(text: str, max_words: int) -> list[str]:
    """Split text into pieces of at most max_words words, preferring line boundaries."""
    if len(text.split()) <= max_words:
        return [text]
    out: list[str] = []
    cur: list[str] = []
    count = 0

    def flush_cur() -> None:
        nonlocal cur, count
        if cur:
            out.append("\n".join(cur))
            cur, count = [], 0

    for line in text.split("\n"):
        words = line.split()
        if len(words) > max_words:
            flush_cur()
            for k in range(0, len(words), max_words):
                out.append(" ".join(words[k : k + max_words]))
            continue
        if count + len(words) > max_words:
            flush_cur()
        cur.append(line)
        count += len(words)
    flush_cur()
    return out


def build_batch(
    docs: list[dict[str, Any]],
    addresses: set[str],
    db_path: Path | None = None,
    expected_generation: int | None = None,
) -> bool:
    """Filter, merge, mask and park the result as the pending batch (nothing is saved yet)."""
    addresses = {a.strip().lower() for a in addresses if a.strip()}
    if not addresses:
        raise ChatTakeoutError("Connect Google or add your address in Settings first, so your own messages can be found.")
    counts = pending_import.new_counts()
    kept: list[str] = []
    items: list[tuple[str, str, str]] = []
    seen: set[str] = {_key(t) for t in get_sample_contents(db_path)}
    done_ids = get_imported_ids(SOURCE, db_path)

    def flush(run: list[str], run_ids: list[str]) -> None:
        if not run:
            return
        text = mask_private_details("\n".join(run))
        any_kept = False
        for piece in _chunks(text, MAX_SAMPLE_WORDS):
            key = _key(piece)
            if not is_substantive(piece):
                counts["too_short"] += 1
            elif key in seen:
                counts["duplicates"] += 1
            else:
                seen.add(key)
                kept.append(piece)
                any_kept = True
        outcome = "kept" if any_kept else "skipped"
        items.extend((SOURCE, mid, outcome) for mid in run_ids)

    for doc in docs:
        run: list[str] = []
        run_ids: list[str] = []
        run_words = 0
        for msg in doc.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            counts["listed"] += 1
            if _creator_email(msg) not in addresses:
                counts["not_user"] += 1
                flush(run, run_ids)
                run, run_ids, run_words = [], [], 0
                continue
            raw_id = msg.get("message_id")
            mid = raw_id.strip() if isinstance(raw_id, str) else ""
            if mid and mid in done_ids:
                counts["already_imported"] += 1
                continue
            text = msg.get("text")
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                counts["empty"] += 1
                if mid:
                    items.append((SOURCE, mid, "skipped"))
                continue
            if run and run_words + len(text.split()) > MAX_SAMPLE_WORDS:
                flush(run, run_ids)
                run, run_ids, run_words = [], [], 0
            run.append(text)
            run_words += len(text.split())
            if mid:
                run_ids.append(mid)
        flush(run, run_ids)
    return pending_import.set_pending(
        source_type=SOURCE,
        context_mode="casual_chat",
        texts=kept,
        ids=items,
        counts=counts,
        expected_generation=expected_generation,
    )


def import_export(
    data: bytes,
    addresses: set[str],
    db_path: Path | None = None,
    expected_generation: int | None = None,
) -> bool:
    return build_batch(read_message_files(data), addresses, db_path, expected_generation)
