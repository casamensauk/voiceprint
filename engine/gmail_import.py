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
from engine.db import get_imported_ids, get_sample_contents, get_settings
from engine.own_text import extract_own_text, is_substantive, mask_private_details

API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
SOURCE = "gmail"
_RETRY_STATUSES = (429, 500, 502, 503, 504)
_RATE_LIMIT_REASONS = ("ratelimitexceeded", "userratelimitexceeded")
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
_MAX_RETRY_AFTER = 30.0
_MAX_LIST_PAGES = 2000
_MAX_CONSECUTIVE_FAILURES = 25
_SIGN_IN_EXPIRED = "Your Google sign-in expired. Connect again to keep going."
_sleep = time.sleep


class GmailError(Exception):
    """Gmail API failure. Never carries tokens or message text."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        retry_after: float | None = None,
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after
        self.reason = reason


# --------------------------------------------------------------------------- http


def _error_reason(raw: bytes) -> str:
    """The machine reason Google gives for an error (e.g. rateLimitExceeded). Never message text."""
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            errors = err.get("errors")
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                return str(errors[0].get("reason") or "")
            return str(err.get("status") or "")
    except (ValueError, AttributeError):
        pass
    return ""


def _http_get(url: str, token: str, timeout: float = 30.0) -> dict[str, Any]:
    """The single HTTP call (patched in tests). Raises GmailError carrying the status code."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        retry_after: float | None = None
        try:
            retry_after = float(err.headers.get("Retry-After")) if err.headers.get("Retry-After") else None
        except (TypeError, ValueError):
            retry_after = None
        reason = ""
        try:
            reason = _error_reason(err.read(8192))
        except Exception:
            reason = ""
        raise GmailError(f"Gmail request failed ({int(err.code)}).", int(err.code), retry_after, reason) from None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        raise GmailError("Could not reach Gmail.", 502) from None
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        raise GmailError("Unexpected Gmail response.", 502) from None
    if not isinstance(data, dict):
        raise GmailError("Unexpected Gmail response.", 502)
    return data


def _retryable(err: GmailError) -> bool:
    if err.status_code in _RETRY_STATUSES:
        return True
    return err.status_code == 403 and err.reason.strip().lower() in _RATE_LIMIT_REASONS


def _api_get(path: str, params: dict[str, Any] | None = None, db_path: Path | None = None) -> dict[str, Any]:
    """GET with up to 4 attempts (1, 2, 4 s backoff, honouring Retry-After) and one token refresh on a 401."""
    query = urllib.parse.urlencode(params or {}, doseq=True)
    url = f"{API_BASE}/{path}" + (f"?{query}" if query else "")
    attempt = 0
    refreshed = False
    while True:
        token = google_auth.get_access_token(db_path)
        try:
            return _http_get(url, token)
        except GmailError as err:
            if err.status_code == 401:
                if refreshed:
                    raise google_auth.GoogleAuthError(_SIGN_IN_EXPIRED, 401) from None
                refreshed = True
                google_auth.invalidate_access_token()
                continue
            attempt += 1
            if not _retryable(err) or attempt >= _MAX_ATTEMPTS:
                raise
            delay = _BACKOFF_SECONDS[attempt - 1]
            if err.retry_after is not None:
                delay = max(delay, min(float(err.retry_after), _MAX_RETRY_AFTER))
            _sleep(delay)


def get_profile(db_path: Path | None = None) -> str:
    addr = str(_api_get("profile", db_path=db_path).get("emailAddress") or "").strip()
    if not addr:
        raise GmailError("Gmail did not return an address.", 502)
    return addr


def list_sent_ids(db_path: Path | None = None) -> list[str]:
    ids: list[str] = []
    seen_pages: set[str] = set()
    page = ""
    for _ in range(_MAX_LIST_PAGES):
        params: dict[str, Any] = {"labelIds": "SENT", "maxResults": 500}
        if page:
            params["pageToken"] = page
        data = _api_get("messages", params, db_path)
        for item in data.get("messages") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        page = str(data.get("nextPageToken") or "")
        if not page or page in seen_pages:
            break
        seen_pages.add(page)
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


def own_text_from_payload(payload: dict[str, Any]) -> str:
    """The user's own words. With both parts present, clean each and keep the shorter non-empty result.

    The HTML part has real quote and signature markup removed, so it catches quotes in any language;
    the plain part catches what only text rules can. The shorter result is the one with less of other people's words.
    """
    plain = html_part = None
    for part in _iter_parts(payload):
        if part.get("filename"):
            continue
        mime = str(part.get("mimeType") or "").lower()
        if mime == "text/plain" and plain is None and (part.get("body") or {}).get("data"):
            plain = part
        elif mime == "text/html" and html_part is None and (part.get("body") or {}).get("data"):
            html_part = part
    candidates: list[str] = []
    if plain is not None:
        candidates.append(extract_own_text(_decode_part(plain)))
    if html_part is not None:
        candidates.append(extract_own_text(html_to_text(_decode_part(html_part))))
    candidates = [c for c in candidates if c]
    return min(candidates, key=len) if candidates else ""


# --------------------------------------------------------------------------- scan

_FAILURE_REASONS = {
    "auth": "your Google sign-in expired",
    "network": "Google stopped responding",
    "cancel": "you cancelled it",
}


def _text_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def scan_running() -> bool:
    return pending_import.active() == "scan"


def start_scan(db_path: Path | None = None) -> bool:
    """Start the background scan. Returns False if a scan or upload is running, or a list is waiting for review."""
    if pending_import.claim("scan"):
        return False
    pending_import.PROGRESS.reset(running=True, stage="listing")

    def _run() -> None:
        try:
            scan_sent_mail(db_path, pending_import.PROGRESS)
        finally:
            if pending_import.PROGRESS.snapshot()["running"]:
                pending_import.PROGRESS.update(running=False)
            pending_import.release("scan")

    threading.Thread(target=_run, daemon=True).start()
    return True


def _plain_error(err: Exception) -> str:
    if isinstance(err, google_auth.GoogleAuthError):
        if err.status_code == 401:
            return _SIGN_IN_EXPIRED
        return err.message
    if isinstance(err, GmailError):
        return "Google could not be reached. Check your connection and try again."
    return "The scan stopped unexpectedly. Try again."


def scan_sent_mail(db_path: Path | None, progress: pending_import.ScanProgress) -> None:
    generation = pending_import.current_generation()
    counts = pending_import.new_counts()
    kept: list[str] = []
    items: list[tuple[str, str, str]] = []  # (source, message id, kept|skipped) for every message read successfully
    partial_reason = ""
    fatal: Exception | None = None
    progress.update(running=True, stage="listing", done=0, total=0, error="")
    try:
        settings = get_settings(db_path)
        addresses = own_addresses(settings, get_profile(db_path=db_path))
        ids = list_sent_ids(db_path)
        counts["listed"] = len(ids)
        done_ids = get_imported_ids(SOURCE, db_path)
        todo = [i for i in ids if i not in done_ids]
        counts["already_imported"] = len(ids) - len(todo)
        progress.update(stage="reading", done=0, total=len(todo))
        seen: set[str] = {_text_key(c) for c in get_sample_contents(db_path)}
        consecutive_failures = 0
        for n, mid in enumerate(todo, start=1):
            if progress.cancelled():
                partial_reason = _FAILURE_REASONS["cancel"]
                break
            try:
                msg = get_message(mid, db_path)
                status, _body = classify_message(msg, addresses)
                outcome = "skipped"
                if status == "not_user":
                    counts["not_user"] += 1
                elif status == "automatic":
                    counts["automatic"] += 1
                else:
                    text = own_text_from_payload(msg.get("payload") or {})
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
                            outcome = "kept"
                items.append((SOURCE, mid, outcome))
                consecutive_failures = 0
            except google_auth.GoogleAuthError as err:
                fatal = err
                partial_reason = _FAILURE_REASONS["auth"] if err.status_code == 401 else "Google sign-in could not be refreshed"
                break
            except Exception as err:
                counts["failed"] += 1
                consecutive_failures += 1
                if isinstance(err, GmailError) and consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    fatal = err
                    partial_reason = _FAILURE_REASONS["network"]
                    progress.update(done=n)
                    break
            progress.update(done=n)
    except (google_auth.GoogleAuthError, GmailError) as err:
        fatal = err
    except Exception as err:
        fatal = err
    if fatal is not None and not partial_reason:
        # Failed before anything could be read (sign-in, address or listing): nothing to park.
        progress.update(running=False, stage="error", error=_plain_error(fatal))
        return
    if partial_reason and not kept:
        if fatal is not None:
            progress.update(running=False, stage="error", error=_plain_error(fatal))
        else:
            progress.update(running=False, stage="idle", error="")
        return
    parked = pending_import.set_pending(
        source_type="gmail",
        context_mode="work_email",
        texts=kept,
        ids=items,
        counts=counts,
        expected_generation=generation,
        partial=bool(partial_reason),
        partial_reason=partial_reason,
    )
    if parked:
        progress.update(running=False, stage="review")
    else:
        progress.update(running=False, stage="idle")
