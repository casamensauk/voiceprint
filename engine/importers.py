"""Bulk parsers for WhatsApp, Discord/Slack, Emails, CSV/iMessage, and plain docs."""

from __future__ import annotations

import csv
import io
import json
import re

# WhatsApp formats:
# [12/05/2025, 14:22:10] Name: message
# 12/05/2025, 14:22 - Name: message
_WHATSAPP_LINE_RE = re.compile(
    r"^(?:\[?\d{1,4}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?\]?(?:\s*-\s*|\s+))"
    r"([^:\n]{1,60}):\s*(.*)$"
)
_DISCORD_HEADER_RE = re.compile(
    r"^([^\n]{1,50}?)\s*(?:—|-)\s*(?:Today|Yesterday|\d{1,2}/\d{1,2}/\d{2,4})[^\n]*$"
)
_SLACK_HEADER_RE = re.compile(r"^([^\n]{1,40}?)\s+\[\d{1,2}:\d{2}(?:\s*[AP]M)?\]\s*$")
_EMAIL_HEADER_RE = re.compile(r"^(?:From|To|Cc|Bcc|Date|Subject|Sent|Reply-To):\s*.*$", re.IGNORECASE)
_EMAIL_REPLY_RE = re.compile(r"^On\s+.+\s+wrote:\s*$", re.IGNORECASE)
_SYSTEM_MSG_SKIP = (
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "messages and calls are end-to-end encrypted",
    "this message was deleted",
    "you deleted this message",
)


def _matches_sender(author: str, sender_name: str) -> bool:
    if not sender_name.strip():
        return True
    return sender_name.strip().lower() in author.strip().lower()


def _parse_whatsapp(raw_text: str, sender_name: str = "") -> list[str]:
    messages: list[tuple[str, str]] = []
    current_author = ""
    current_lines: list[str] = []

    for line in raw_text.splitlines():
        m = _WHATSAPP_LINE_RE.match(line.strip())
        if m:
            if current_lines:
                messages.append((current_author, "\n".join(current_lines).strip()))
            current_author = m.group(1).strip()
            current_lines = [m.group(2).strip()]
        elif current_lines:
            current_lines.append(line.strip())
    if current_lines:
        messages.append((current_author, "\n".join(current_lines).strip()))

    results: list[str] = []
    for author, body in messages:
        if not body or any(skip in body.lower() for skip in _SYSTEM_MSG_SKIP):
            continue
        if _matches_sender(author, sender_name) and len(body.split()) >= 3:
            results.append(body)
    return results


def _parse_discord_slack(raw_text: str, sender_name: str = "") -> list[str]:
    stripped = raw_text.strip()
    # Try JSON export first
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
            items = data.get("messages", []) if isinstance(data, dict) else data
            if isinstance(items, list):
                results: list[str] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    content = str(item.get("content") or item.get("text") or "").strip()
                    author_obj = item.get("author") or item.get("user") or item.get("sender") or ""
                    if isinstance(author_obj, dict):
                        author = str(author_obj.get("name") or author_obj.get("username") or "")
                    else:
                        author = str(author_obj)
                    if content and _matches_sender(author, sender_name) and len(content.split()) >= 3:
                        results.append(content)
                if results:
                    return results
        except json.JSONDecodeError:
            pass

    # Copy-pasted Discord / Slack log format
    results = []
    current_author = ""
    buf: list[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        m_disc = _DISCORD_HEADER_RE.match(s)
        m_slack = _SLACK_HEADER_RE.match(s)
        if m_disc or m_slack:
            if buf and _matches_sender(current_author, sender_name):
                joined = "\n".join(buf).strip()
                if len(joined.split()) >= 3:
                    results.append(joined)
            current_author = (m_disc or m_slack).group(1).strip()
            buf = []
        elif s:
            buf.append(s)
    if buf and _matches_sender(current_author, sender_name):
        joined = "\n".join(buf).strip()
        if len(joined.split()) >= 3:
            results.append(joined)
    if results:
        return results
    return _parse_plain_text(raw_text)


def _parse_email(raw_text: str) -> list[str]:
    # Split multiple emails by common separator if present, otherwise clean single/thread email
    chunks = re.split(r"\n-{4,}\n|\n={4,}\n", raw_text)
    results: list[str] = []
    for chunk in chunks:
        cleaned_lines: list[str] = []
        for line in chunk.splitlines():
            s = line.strip()
            if _EMAIL_HEADER_RE.match(s):
                continue
            if _EMAIL_REPLY_RE.match(s):
                break
            if s.startswith(">"):
                continue
            if s in ("--", "---"):
                break
            cleaned_lines.append(line.rstrip())
        body = "\n".join(cleaned_lines).strip()
        if len(body.split()) >= 4:
            results.append(body)
    return results


def _parse_csv(raw_text: str, sender_name: str = "") -> list[str]:
    reader = csv.DictReader(io.StringIO(raw_text))
    if not reader.fieldnames:
        return _parse_plain_text(raw_text)
    fields_lower = {f.lower(): f for f in reader.fieldnames}
    text_col = next(
        (fields_lower[k] for k in ("text", "message", "body", "content") if k in fields_lower),
        None,
    )
    sender_col = next(
        (fields_lower[k] for k in ("sender", "author", "from", "is_from_me") if k in fields_lower),
        None,
    )
    if not text_col:
        return _parse_plain_text(raw_text)
    results: list[str] = []
    for row in reader:
        msg = (row.get(text_col) or "").strip()
        if not msg or len(msg.split()) < 3:
            continue
        if sender_col and sender_name.strip():
            if not _matches_sender(str(row.get(sender_col) or ""), sender_name):
                continue
        results.append(msg)
    return results


def _parse_plain_text(raw_text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n+", raw_text.strip())
    if len(blocks) == 1 and "\n" in blocks[0]:
        lines = [ln.strip() for ln in blocks[0].splitlines() if len(ln.strip().split()) >= 3]
        if len(lines) > 1:
            return lines
    return [b.strip() for b in blocks if len(b.strip().split()) >= 3]


def detect_format(raw_text: str) -> str:
    """Auto-detect the input log/document format."""
    sample = raw_text[:4000]
    lines = [ln.strip() for ln in sample.splitlines() if ln.strip()]
    if any(_WHATSAPP_LINE_RE.match(ln) for ln in lines[:12]):
        return "whatsapp"
    stripped = sample.strip()
    if stripped.startswith(("{", "[")) or any(
        _DISCORD_HEADER_RE.match(ln) or _SLACK_HEADER_RE.match(ln) for ln in lines[:12]
    ):
        return "discord_slack"
    if any(_EMAIL_HEADER_RE.match(ln) or _EMAIL_REPLY_RE.match(ln) for ln in lines[:12]):
        return "email"
    if len(lines) >= 2 and "," in lines[0] and any(k in lines[0].lower() for k in ("message", "text", "sender", "body")):
        return "csv"
    return "document"


def parse_imported_content(
    raw_text: str,
    source_format: str = "auto",
    sender_name: str = "",
) -> dict[str, object]:
    """Parse bulk text/exports into cleaned sample strings."""
    if not raw_text or not raw_text.strip():
        return {"detected_format": "document", "samples": []}

    fmt = detect_format(raw_text) if source_format in ("", "auto") else source_format
    if fmt == "whatsapp":
        samples = _parse_whatsapp(raw_text, sender_name=sender_name)
    elif fmt == "discord_slack":
        samples = _parse_discord_slack(raw_text, sender_name=sender_name)
    elif fmt == "email":
        samples = _parse_email(raw_text)
    elif fmt == "csv":
        samples = _parse_csv(raw_text, sender_name=sender_name)
    else:
        samples = _parse_plain_text(raw_text)

    return {"detected_format": fmt, "samples": samples}
