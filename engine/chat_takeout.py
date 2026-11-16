"""Read a Google Chat Takeout export (messages.json, or a Takeout .zip) and keep only the user's own messages."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import PurePosixPath
from typing import Any

from engine import pending_import
from engine.own_text import is_substantive, mask_private_details

MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
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
                with zf.open(info) as fh:
                    raw = fh.read(budget + 1)
                budget -= len(raw)
                if budget < 0:
                    raise ChatTakeoutError("The archive is larger than 200 MB once unpacked.", 413)
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


def build_batch(docs: list[dict[str, Any]], addresses: set[str]) -> None:
    """Filter, merge, mask and park the result as the pending batch (nothing is saved yet)."""
    addresses = {a.strip().lower() for a in addresses if a.strip()}
    if not addresses:
        raise ChatTakeoutError("Connect Google or add your address in Settings first, so your own messages can be found.")
    counts = pending_import.new_counts()
    kept: list[str] = []
    seen: set[str] = set()

    def flush(run: list[str]) -> None:
        if not run:
            return
        text = mask_private_details("\n".join(run))
        key = hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
        if not is_substantive(text):
            counts["too_short"] += 1
        elif key in seen:
            counts["duplicates"] += 1
        else:
            seen.add(key)
            kept.append(text)

    for doc in docs:
        run: list[str] = []
        for msg in doc.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            counts["listed"] += 1
            if _creator_email(msg) not in addresses:
                counts["not_user"] += 1
                flush(run)
                run = []
                continue
            text = msg.get("text")
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                counts["empty"] += 1
                continue
            run.append(text)
        flush(run)
    pending_import.set_pending(
        source_type="google_chat", context_mode="casual_chat", texts=kept, ids=[], counts=counts
    )


def import_export(data: bytes, addresses: set[str]) -> None:
    build_batch(read_message_files(data), addresses)
