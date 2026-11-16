"""TypeSafe AI Jev (System One) client: typed questions in, calibrated probabilities out. Stdlib only."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-1.14.0"
JEV_INPUT_USD_PER_MTOK = 0.042
_OFF_VALUES = ("0", "false", "off")
_RETRY_STATUSES = (429, 529)


class JevError(Exception):
    """Structured error for Jev failures. Never carries the submitted text."""

    def __init__(self, message: str, status_code: int = 400, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after


def resolve_typesafe_key(settings: dict[str, str] | None = None, override_key: str = "") -> str:
    """Resolve the key: override, then setting, then TYPESAFE_API_KEY, then JEV_API_KEY."""
    if override_key and override_key.strip():
        return override_key.strip()
    if settings and (settings.get("typesafe_api_key") or "").strip():
        return settings["typesafe_api_key"].strip()
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


def _switched_off(settings: dict[str, str] | None) -> bool:
    return str((settings or {}).get("jev_enabled", "1")).strip().lower() in _OFF_VALUES


def is_jev_enabled(settings: dict[str, str] | None = None) -> bool:
    return not _switched_off(settings) and bool(resolve_typesafe_key(settings))


def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {"true": true or "", "false": false or ""}
    return q


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    if not 1 <= len(levels) <= 10:
        raise JevError("A score question needs 1-10 levels.", 400)
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def choice(instructions: str, options: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


_STATUS_NAMES = {
    400: "bad request",
    401: "key rejected",
    413: "request too large",
    415: "unsupported media type",
    422: "invalid request",
    429: "rate limited",
    529: "service overloaded",
}


def _post_json(url: str, api_key: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """The single HTTP call (patched in tests). Maps HTTP failures to JevError."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as err:
        code = int(getattr(err, "code", 502) or 502)
        retry_after = None
        try:
            retry_after = float(err.headers.get("retry-after"))
        except (TypeError, ValueError, AttributeError):
            pass
        detail = ""
        if code == 422:
            try:
                detail = str(json.loads(err.read().decode("utf-8", errors="replace")).get("detail", ""))[:160]
            except Exception:
                detail = ""
        msg = f"Jev request failed ({code}): {_STATUS_NAMES.get(code, 'error')}" + (f" - {detail}" if detail else "")
        raise JevError(msg, code, retry_after) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise JevError(f"Jev network error: {type(err).__name__}", 502) from None
    if not isinstance(data, dict):
        raise JevError("Unexpected Jev response format.", 502)
    return data


def evaluate(
    state: Any,
    questions: dict[str, dict[str, Any]],
    settings: dict[str, str] | None = None,
    override_key: str = "",
    timeout: float = 12.0,
) -> dict[str, Any]:
    model = ((settings or {}).get("jev_model") or "").strip() or DEFAULT_JEV_MODEL
    if "latest" in model.lower() or "preview" in model.lower():
        raise JevError("Jev model must be a pinned version, not an alias.", 400)
    if _switched_off(settings):
        raise JevError("Jev is turned off in settings.", 400)
    key = resolve_typesafe_key(settings, override_key)
    if not key:
        raise JevError("Missing TypeSafe API key - add it in Settings.", 400)

    body = {"model": model, "state": state, "questions": questions}
    try:
        data = _post_json(JEV_URL, key, body, timeout)
    except JevError as err:
        if err.status_code not in _RETRY_STATUSES:
            raise
        time.sleep(min(err.retry_after, 5.0) if err.retry_after else 1.0)
        data = _post_json(JEV_URL, key, body, timeout)

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    in_tok = int(usage.get("input_tokens") or 0)
    return {
        "model": str(data.get("model") or model),
        "answers": data.get("answers") if isinstance(data.get("answers"), dict) else {},
        "usage": usage,
        "cost_usd": in_tok * JEV_INPUT_USD_PER_MTOK / 1_000_000,
    }
