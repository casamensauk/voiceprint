"""Google OAuth 2.0 for a Desktop-app client: loopback redirect + PKCE. Stdlib only.

Tokens: the access token and its expiry live in memory only. The refresh token goes to the
macOS Keychain (via /usr/bin/security). Nothing here ever returns, logs or puts a code, token
or client secret into an exception message.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
CALLBACK_PATH = "/api/google/callback"

KEYCHAIN_SERVICE = "VoicePrint"
KEYCHAIN_ACCOUNT = "google_refresh_token"
SECURITY_BIN = "/usr/bin/security"

FLOW_TTL_SECONDS = 600
_EXPIRY_MARGIN = 60
_MAX_PENDING = 10


class GoogleAuthError(Exception):
    """Sign-in problem. The message is always safe to show (it never contains secrets)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_lock = threading.Lock()
_pending: dict[str, dict[str, Any]] = {}
_access: dict[str, Any] = {"token": "", "expires_at": 0.0}
_memory_refresh: dict[str, str] = {"token": ""}  # only used if the Keychain write fails
_state: dict[str, str] = {"note": ""}


# --------------------------------------------------------------------------- network


def _post_form(url: str, data: dict[str, str], timeout: float = 20.0) -> tuple[int, dict[str, Any]]:
    """The single HTTP call (patched in tests). Returns (status, parsed JSON object)."""
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        status = int(getattr(err, "code", 502) or 502)
        try:
            raw = err.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        raise GoogleAuthError("Could not reach Google. Check your connection and try again.", 502) from None
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except ValueError:
        parsed = {}
    return status, parsed if isinstance(parsed, dict) else {}


# --------------------------------------------------------------------------- keychain


def _keychain_write(token: str) -> bool:
    """Store the refresh token. Returns False on any failure (patched in tests)."""
    try:
        proc = subprocess.run(
            [SECURITY_BIN, "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w", token],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _keychain_read() -> str:
    """Return the stored refresh token, or '' (patched in tests)."""
    try:
        proc = subprocess.run(
            [SECURITY_BIN, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _keychain_delete() -> bool:
    """Remove the stored refresh token (patched in tests)."""
    try:
        proc = subprocess.run(
            [SECURITY_BIN, "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------- helpers


def _settings(db_path: Path | None = None) -> dict[str, str]:
    from engine.db import get_settings

    return get_settings(db_path)


def _client_credentials(db_path: Path | None = None) -> tuple[str, str]:
    s = _settings(db_path)
    return (s.get("google_client_id") or "").strip(), (s.get("google_client_secret") or "").strip()


def redirect_uri(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}{CALLBACK_PATH}"


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _purge_expired(now: float) -> None:
    for key in [k for k, v in _pending.items() if v["expires_at"] <= now]:
        _pending.pop(key, None)


def has_pending_flow() -> bool:
    with _lock:
        _purge_expired(time.time())
        return bool(_pending)


def _stored_refresh_token() -> str:
    with _lock:
        mem = _memory_refresh["token"]
    return mem or _keychain_read()


def _clear_tokens() -> None:
    with _lock:
        _access["token"] = ""
        _access["expires_at"] = 0.0
        _memory_refresh["token"] = ""
    _keychain_delete()


# --------------------------------------------------------------------------- flow


def start_flow(port: int, db_path: Path | None = None) -> str:
    """Begin sign-in and return the Google consent URL."""
    client_id, client_secret = _client_credentials(db_path)
    if not client_id or not client_secret:
        raise GoogleAuthError("Add your Google client ID and secret in Settings first.")
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    now = time.time()
    with _lock:
        _purge_expired(now)
        while len(_pending) >= _MAX_PENDING:
            _pending.pop(next(iter(_pending)))
        _pending[state] = {
            "verifier": verifier,
            "redirect_uri": redirect_uri(port),
            "expires_at": now + FLOW_TTL_SECONDS,
        }
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(port),
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
        "code_challenge": _challenge(verifier),
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def finish_flow(code: str, state: str, port: int, db_path: Path | None = None) -> dict[str, Any]:
    """Complete sign-in from the loopback callback. Single-use, time-limited state."""
    with _lock:
        _purge_expired(time.time())
        entry = _pending.pop(state, None) if state else None
    if not entry or not code:
        raise GoogleAuthError("This sign-in link is unknown or has expired. Start again from VoicePrint.")
    client_id, client_secret = _client_credentials(db_path)
    status, data = _post_form(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": entry["redirect_uri"],
            "code_verifier": entry["verifier"],
        },
    )
    access = str(data.get("access_token") or "")
    if status != 200 or not access:
        raise GoogleAuthError("Google did not accept the sign-in. Check the client ID and secret, then try again.")
    refresh = str(data.get("refresh_token") or "")
    try:
        expires_in = float(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0
    note = ""
    with _lock:
        _access["token"] = access
        _access["expires_at"] = time.time() + expires_in
        _memory_refresh["token"] = ""
    if refresh:
        if not _keychain_write(refresh):
            with _lock:
                _memory_refresh["token"] = refresh
            note = "Could not save the sign-in to the Keychain, so you are connected for this session only."
    else:
        note = "Google did not issue a long-lived sign-in, so you are connected for this session only."
    with _lock:
        _state["note"] = note
    try:
        from engine import gmail_import
        from engine.db import update_settings

        email_addr = gmail_import.get_profile(db_path=db_path)
        update_settings({"google_account_email": email_addr}, db_path=db_path)
    except Exception:
        # Connected, but the address could not be read; the status note says so.
        with _lock:
            _state["note"] = (note + " " if note else "") + "Connected, but your address could not be read yet."
    return status_dict(db_path)


def get_access_token(db_path: Path | None = None) -> str:
    """Return a valid access token, refreshing it from the stored refresh token if needed."""
    with _lock:
        if _access["token"] and _access["expires_at"] - _EXPIRY_MARGIN > time.time():
            return str(_access["token"])
    refresh = _stored_refresh_token()
    if not refresh:
        raise GoogleAuthError("Not connected to Google.", 401)
    client_id, client_secret = _client_credentials(db_path)
    status, data = _post_form(
        TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    if data.get("error") == "invalid_grant":
        _clear_tokens()
        with _lock:
            _state["note"] = "sign-in expired"
        raise GoogleAuthError("sign-in expired", 401)
    access = str(data.get("access_token") or "")
    if status != 200 or not access:
        raise GoogleAuthError("Could not refresh the Google sign-in.", 502)
    try:
        expires_in = float(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0
    with _lock:
        _access["token"] = access
        _access["expires_at"] = time.time() + expires_in
    return access


def invalidate_access_token() -> None:
    """Forget the cached access token so the next call refreshes it from the stored refresh token."""
    with _lock:
        _access["token"] = ""
        _access["expires_at"] = 0.0


def disconnect(db_path: Path | None = None) -> dict[str, Any]:
    """Revoke the grant at Google (best effort), delete the stored token, forget the account."""
    with _lock:
        token = _memory_refresh["token"] or ""
        access = str(_access["token"] or "")
    token = token or _keychain_read() or access
    if token:
        try:
            _post_form(REVOKE_URL, {"token": token})
        except GoogleAuthError:
            pass
    _clear_tokens()
    with _lock:
        _state["note"] = ""
    from engine.db import update_settings

    update_settings({"google_account_email": ""}, db_path=db_path)
    return status_dict(db_path)


def status_dict(db_path: Path | None = None) -> dict[str, Any]:
    client_id, client_secret = _client_credentials(db_path)
    with _lock:
        has_access = bool(_access["token"]) and _access["expires_at"] > time.time()
        note = _state["note"]
        has_mem_refresh = bool(_memory_refresh["token"])
    connected = has_access or has_mem_refresh or bool(_keychain_read())
    email_addr = (_settings(db_path).get("google_account_email") or "").strip() if connected else ""
    return {
        "configured": bool(client_id and client_secret),
        "connected": connected,
        "email": email_addr,
        "note": note,
    }


def status(db_path: Path | None = None) -> dict[str, Any]:
    """{"configured", "connected", "email", "note"}; never any token."""
    return status_dict(db_path)


def _reset_for_tests() -> None:
    with _lock:
        _pending.clear()
        _access["token"] = ""
        _access["expires_at"] = 0.0
        _memory_refresh["token"] = ""
        _state["note"] = ""
