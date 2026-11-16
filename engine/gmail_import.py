"""Read the user's own sent mail from the Gmail REST API (read-only) and turn it into voice samples."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parseaddr
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from engine import google_auth, pending_import
from engine.db import get_imported_ids, get_samples, get_settings
from engine.own_text import extract_own_text, is_substantive, mask_private_details

API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
SOURCE = "gmail"
_RETRY_STATUSES = (429, 500, 502, 503, 504)
_sleep = time.sleep


class GmailError(Exception):
    """Gmail API failure. Never carries tokens or message text."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# --------------------------------------------------------------------------- http


def _http_get(url: str, token: str, timeout: float = 30.0) -> dict[str, Any]:
    """The single HTTP call (patched in tests). Raises GmailError carrying the status code."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        raise GmailError(f"Gmail request failed ({int(err.code)}).", int(err.code)) from None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        raise GmailError("Could not reach Gmail.", 502) from None
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        raise GmailError("Unexpected Gmail response.", 502) from None
    if not isinstance(data, dict):
        raise GmailError("Unexpected Gmail response.", 502)
    return data


def _api_get(path: str, params: dict[str, Any] | None = None, db_path: Path | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {}, doseq=True)
    url = f"{API_BASE}/{path}" + (f"?{query}" if query else "")
    token = google_auth.get_access_token(db_path)
    try:
        return _http_get(url, token)
    except GmailError as err:
        if err.status_code not in _RETRY_STATUSES:
            raise
    _sleep(1.5)
    return _http_get(url, google_auth.get_access_token(db_path))


def get_profile(db_path: Path | None = None) -> str:
    addr = str(_api_get("profile", db_path=db_path).get("emailAddress") or "").strip()
    if not addr:
        raise GmailError("Gmail did not return an address.", 502)
    return addr


def list_sent_ids(db_path: Path | None = None) -> list[str]:
    ids: list[str] = []
    page = ""
    while True:
        params: dict[str, Any] = {"labelIds": "SENT", "maxResults": 500}
        if page:
            params["pageToken"] = page
        data = _api_get("messages", params, db_path)
        for item in data.get("messages") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        page = str(data.get("nextPageToken") or "")
        if not page:
            return ids


def get_message(message_id: str, db_path: Path | None = None) -> dict[str, Any]:
    return _api_get(f"messages/{urllib.parse.quote(message_id, safe='')}", {"format": "full"}, db_path)


# --------------------------------------------------------------------------- message filtering


def own_addresses(settings: dict[str, str], profile_email: str = "") -> set[str]:
    found = {profile_email.strip().lower(), (settings.get("google_account_email") or "").strip().lower()}
    for part in re.split(r"[,;\s]+", settings.get("google_extra_addresses") or ""):
        found.add(part.strip().lower())
    return {a for a in found if a}


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for h in payload.get("headers") or []:
        if isinstance(h, dict) and h.get("name"):
            out.setdefault(str(h["name"]).lower(), str(h.get("value") or ""))
    return out


def _iter_parts(payload: dict[str, Any]):
    yield payload
    for child in payload.get("parts") or []:
        if isinstance(child, dict):
            yield from _iter_parts(child)


def _decode_part(part: dict[str, Any]) -> str:
    data = ((part.get("body") or {}).get("data")) or ""
    if not data:
        return ""
    raw = base64.urlsafe_b64decode(str(data) + "=" * (-len(str(data)) % 4))
    charset = "utf-8"
    for h in part.get("headers") or []:
        if isinstance(h, dict) and str(h.get("name", "")).lower() == "content-type":
            m = re.search(r"charset=\"?([\w.-]+)", str(h.get("value") or ""), re.I)
            if m:
                charset = m.group(1)
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


class _HtmlText(HTMLParser):
    """Strip tags; drop quoted replies (blockquote, div.gmail_quote) and everything from the signature on."""

    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "ul", "ol", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_tag = ""
        self._skip_depth = 0
        self._ignore = 0
        self.stopped = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.stopped:
            return
        classes = (dict(attrs).get("class") or "").split()
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag == "div" and "gmail_signature" in classes:
            self.stopped = True
            return
        if tag == "blockquote" or (tag == "div" and "gmail_quote" in classes):
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag in ("script", "style"):
            self._ignore += 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.stopped:
            return
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = ""
            return
        if tag in ("script", "style") and self._ignore:
            self._ignore -= 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not (self.stopped or self._skip_tag or self._ignore):
            self.parts.append(data)


def html_to_text(html_text: str) -> str:
    parser = _HtmlText()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        pass
    text = "".join(parser.parts).replace("\xa0", " ")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_body(payload: dict[str, Any]) -> str:
    """Prefer text/plain; otherwise strip tags from text/html."""
    plain = html_part = None
    for part in _iter_parts(payload):
        if part.get("filename"):
            continue
        mime = str(part.get("mimeType") or "").lower()
        if mime == "text/plain" and plain is None and (part.get("body") or {}).get("data"):
            plain = part
        elif mime == "text/html" and html_part is None and (part.get("body") or {}).get("data"):
            html_part = part
    if plain is not None:
        return _decode_part(plain)
    if html_part is not None:
        return html_to_text(_decode_part(html_part))
    return ""


def classify_message(msg: dict[str, Any], addresses: set[str]) -> tuple[str, str]:
    """Return (status, body): status is ok | not_user | automatic."""
    labels = {str(x) for x in (msg.get("labelIds") or [])}
    payload = msg.get("payload") or {}
    if "SENT" not in labels or "DRAFT" in labels:
        return "not_user", ""
    headers = _headers(payload)
    sender = parseaddr(headers.get("from", ""))[1].strip().lower()
    if not sender or sender not in addresses:
        return "not_user", ""
    auto = headers.get("auto-submitted", "").strip().lower()
    if auto and auto != "no":
        return "automatic", ""
    if any(str(p.get("mimeType") or "").lower() == "text/calendar" for p in _iter_parts(payload)):
        return "automatic", ""
    return "ok", extract_body(payload)


# --------------------------------------------------------------------------- scan

_scan_lock = threading.Lock()
_scan_running = False


def _text_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def scan_running() -> bool:
    return pending_import.PROGRESS.snapshot()["running"]


def start_scan(db_path: Path | None = None) -> bool:
    """Start the background scan. Returns False if one is already running."""
    global _scan_running
    with _scan_lock:
        if _scan_running:
            return False
        _scan_running = True
    pending_import.PROGRESS.reset(running=True, stage="listing")

    def _run() -> None:
        global _scan_running
        try:
            scan_sent_mail(db_path, pending_import.PROGRESS)
        finally:
            with _scan_lock:
                _scan_running = False

    threading.Thread(target=_run, daemon=True).start()
    return True


def scan_sent_mail(db_path: Path | None, progress: pending_import.ScanProgress) -> None:
    counts = pending_import.new_counts()
    progress.reset(running=True, stage="listing")
    try:
        settings = get_settings(db_path)
        addresses = own_addresses(settings, get_profile(db_path=db_path))
        ids = list_sent_ids(db_path)
        counts["listed"] = len(ids)
        done_ids = get_imported_ids(SOURCE, db_path)
        todo = [i for i in ids if i not in done_ids]
        counts["already_imported"] = len(ids) - len(todo)
        progress.update(stage="reading", done=0, total=len(todo))
        kept: list[str] = []
        kept_ids: list[str] = []
        seen: set[str] = {_text_key(s["content"]) for s in get_samples(limit=1_000_000, db_path=db_path)}
        for n, mid in enumerate(todo, start=1):
            try:
                msg = get_message(mid, db_path)
            except google_auth.GoogleAuthError:
                raise
            except GmailError:
                counts["failed"] += 1
                progress.update(done=n)
                continue
            status, body = classify_message(msg, addresses)
            if status == "not_user":
                counts["not_user"] += 1
            elif status == "automatic":
                counts["automatic"] += 1
            else:
                text = extract_own_text(body)
                if not text:
                    counts["empty"] += 1
                else:
                    text = mask_private_details(text)
                    key = _text_key(text)
                    if not is_substantive(text):
                        counts["too_short"] += 1
                    elif key in seen:
                        counts["duplicates"] += 1
                    else:
                        seen.add(key)
                        kept.append(text)
                        kept_ids.append(mid)
            progress.update(done=n)
        pending_import.set_pending(
            source_type="gmail", context_mode="work_email", texts=kept, ids=[(SOURCE, i) for i in kept_ids], counts=counts
        )
        progress.update(running=False, stage="review")
    except (google_auth.GoogleAuthError, GmailError) as err:
        progress.update(running=False, stage="error", error=err.message)
    except Exception:
        progress.update(running=False, stage="error", error="The scan stopped unexpectedly.")
