"""Pangram Labs Pay-As-You-Go REST client (pangram-4 AI/humanizer, file upload, plagiarism)."""

from __future__ import annotations

import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

PANGRAM_TEXT_BASE = "https://text.external-api.pangram.com"
PANGRAM_FILE_BASE = "https://file-external.api.pangram.com"
PANGRAM_PLAGIARISM_BASE = "https://plagiarism.api.pangram.com"

COST_PER_100_WORDS_USD = 0.05

PANGRAM_PASS_MAX_AI = 0.20
PANGRAM_PASS_MAX_HUMANIZER = 0.50


def passes_pangram(result: dict[str, Any]) -> bool:
    """True when a Pangram result reads as human: AI fraction <= 20% and humanizer score < 50%."""
    return (
        float(result.get("fraction_ai") or 0.0) <= PANGRAM_PASS_MAX_AI
        and float(result.get("max_humanizer_score") or 0.0) < PANGRAM_PASS_MAX_HUMANIZER
    )


class PangramError(Exception):
    """Structured error raised for Pangram API failures or validation errors."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def resolve_pangram_key(settings: dict[str, str] | None = None, override_key: str = "") -> str:
    """Resolve Pangram API key from explicit override, DB settings, or PANGRAM_API_KEY env."""
    if override_key and override_key.strip():
        return override_key.strip()
    if settings and (settings.get("pangram_api_key") or "").strip():
        return settings["pangram_api_key"].strip()
    return (os.environ.get("PANGRAM_API_KEY") or "").strip()


def estimate_pangram_cost(text: str, is_bulk: bool = False) -> dict[str, Any]:
    """Estimate Pangram Pay-As-You-Go developer cost ($0.05 / 100 words, 20% off bulk)."""
    stripped = text.strip()
    char_count = len(stripped)
    word_count = len(stripped.split()) if stripped else 0
    raw_cost = (word_count / 100.0) * COST_PER_100_WORDS_USD
    if is_bulk:
        raw_cost *= 0.8
    return {
        "char_count": char_count,
        "word_count": word_count,
        "meets_min_chars": char_count >= 50,
        "rate_per_100_words_usd": COST_PER_100_WORDS_USD,
        "estimated_cost_usd": round(raw_cost, 4),
    }


def _map_http_error(err: urllib.error.HTTPError, context: str) -> PangramError:
    try:
        detail = err.read().decode("utf-8", errors="replace").strip()
    except Exception:
        detail = str(err)
    code = int(getattr(err, "code", 502) or 502)
    if code in (401, 403):
        return PangramError(
            f"Pangram authentication/entitlement rejected ({code}) during {context}: {detail or 'Check API key and model entitlement.'}",
            code,
        )
    if code == 404:
        return PangramError(f"Pangram resource not found or expired (404) during {context}: {detail}", 404)
    if code == 413:
        return PangramError("Submission exceeds Pangram payload body limit (413).", 413)
    return PangramError(f"Pangram {context} failed ({code}): {detail}", code)


def _request_json(
    url: str,
    api_key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_s: float = 20.0,
) -> Any:
    headers = {"x-api-key": api_key}
    body_bytes: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as err:
        raise _map_http_error(err, f"{method} {url}") from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise PangramError(f"Pangram network/JSON error calling {url}: {err}", 502) from err


def list_pangram_models(api_key: str) -> list[str]:
    """List enabled Pangram models via GET /models."""
    if not api_key.strip():
        raise PangramError("Missing Pangram API key — configure it in AI Engine Settings.", 400)
    data = _request_json(f"{PANGRAM_TEXT_BASE}/models", api_key=api_key.strip(), method="GET")
    if isinstance(data, list):
        return [str(m) for m in data]
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [str(m) for m in data["models"]]
    return ["default", "pangram-4"]


def _normalize_pangram_result(data: dict[str, Any], submitted_text: str, model: str) -> dict[str, Any]:
    """Normalize completed Pangram prediction payload with derived humanizer & cost summary."""
    raw_windows = data.get("windows") or []
    windows: list[dict[str, Any]] = []
    max_humanizer = 0.0
    has_humanized = False

    for w in raw_windows:
        if not isinstance(w, dict):
            continue
        is_hum = bool(w.get("is_humanized", False))
        hum_score = float(w.get("humanizer_score") or 0.0)
        if is_hum or hum_score >= 0.5:
            has_humanized = True
        if hum_score > max_humanizer:
            max_humanizer = hum_score
        windows.append(
            {
                "text": str(w.get("text") or ""),
                "label": str(w.get("label") or "Unknown"),
                "ai_assistance_score": round(float(w.get("ai_assistance_score") or 0.0), 4),
                "confidence": str(w.get("confidence") or ""),
                "is_humanized": is_hum,
                "humanizer_score": round(hum_score, 4),
                "start_index": int(w.get("start_index") or 0),
                "end_index": int(w.get("end_index") or 0),
                "word_count": int(w.get("word_count") or len(str(w.get("text") or "").split())),
                "token_length": int(w.get("token_length") or 0),
            }
        )

    effective_text = str(data.get("text") or submitted_text)
    estimate = estimate_pangram_cost(effective_text)
    return {
        "scan_type": "text",
        "task_id": str(data.get("task_id") or ""),
        "stage": str(data.get("stage") or "STAGE_SUCCESS"),
        "version": str(data.get("version") or ("4.1" if model == "pangram-4" else "default")),
        "model": model,
        "text": effective_text,
        "headline": str(data.get("headline") or data.get("prediction_short") or "Completed"),
        "prediction": str(data.get("prediction") or ""),
        "prediction_short": str(data.get("prediction_short") or data.get("headline") or "Completed"),
        "fraction_ai": round(float(data.get("fraction_ai") or 0.0), 4),
        "fraction_ai_assisted": round(float(data.get("fraction_ai_assisted") or 0.0), 4),
        "fraction_human": round(float(data.get("fraction_human") or 0.0), 4),
        "num_ai_segments": int(data.get("num_ai_segments") or 0),
        "num_ai_assisted_segments": int(data.get("num_ai_assisted_segments") or 0),
        "num_human_segments": int(data.get("num_human_segments") or 0),
        "dashboard_link": str(data.get("dashboard_link") or ""),
        "has_humanized_segments": has_humanized,
        "max_humanizer_score": round(max_humanizer, 4),
        "windows": windows,
        "estimate": estimate,
    }


def scan_text_pangram(
    text: str,
    api_key: str,
    model: str = "pangram-4",
    public_dashboard_link: bool = False,
    timeout_s: float = 30.0,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    """Run Pangram's async POST /task -> GET /task/{task_id} AI & humanizer scan."""
    key = api_key.strip()
    if not key:
        raise PangramError("Missing Pangram API key — configure it in AI Engine Settings.", 400)
    stripped = text.strip()
    if len(stripped) < 50:
        raise PangramError("Pangram requires at least 50 characters of text to run a scan.", 400)

    selected_model = (model or "pangram-4").strip()
    submit_payload = {
        "text": text,
        "model": selected_model,
        "public_dashboard_link": bool(public_dashboard_link),
    }
    submit_res = _request_json(
        f"{PANGRAM_TEXT_BASE}/task",
        api_key=key,
        method="POST",
        payload=submit_payload,
    )
    task_id = str(submit_res.get("task_id") or "").strip()
    if not task_id:
        raise PangramError("Pangram POST /task did not return a task_id.", 502)

    deadline = time.monotonic() + max(2.0, timeout_s)
    sleep_interval = max(0.1, poll_s)

    while time.monotonic() < deadline:
        time.sleep(sleep_interval)
        poll_data = _request_json(
            f"{PANGRAM_TEXT_BASE}/task/{task_id}",
            api_key=key,
            method="GET",
        )
        stage = str(poll_data.get("stage") or "")
        if stage == "STAGE_SUCCESS":
            poll_data.setdefault("task_id", task_id)
            return _normalize_pangram_result(poll_data, text, selected_model)
        if stage == "STAGE_FAILED":
            headline = str(poll_data.get("headline") or poll_data.get("prediction") or "Unknown error")
            raise PangramError(f"Pangram scan failed: {headline}", 422)

    raise PangramError(f"Pangram task {task_id} timed out after {timeout_s:.0f}s.", 504)


def extract_docx_text(file_bytes: bytes) -> str:
    """Extract paragraph text from a .docx archive using stdlib zipfile + XML parsing."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise PangramError("Invalid .docx file: missing word/document.xml.", 400)
            xml_bytes = zf.read("word/document.xml")
    except zipfile.BadZipFile as err:
        raise PangramError("Uploaded .docx file is corrupted or not a valid ZIP archive.", 400) from err

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("}p") or elem.tag == "p":
            runs = [t.text for t in elem.iter() if (t.tag.endswith("}t") or t.tag == "t") and t.text]
            para = "".join(runs).strip()
            if para:
                paragraphs.append(para)
    return "\n\n".join(paragraphs)


def _encode_multipart(filename: str, file_bytes: bytes, public_dashboard_link: bool) -> tuple[bytes, str]:
    boundary = f"----VoicePrintBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    safe_name = Path(filename).name.replace('"', "")

    parts: list[bytes] = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="public_dashboard_link"\r\n\r\n'.encode("utf-8"),
        f"{'true' if public_dashboard_link else 'false'}\r\n".encode("utf-8"),
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="files"; filename="{safe_name}"\r\n'.encode("utf-8"),
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def scan_file_pangram(
    filename: str,
    file_bytes: bytes,
    api_key: str,
    prefer_pangram_4: bool = True,
    public_dashboard_link: bool = False,
) -> dict[str, Any]:
    """Scan a .docx, .pdf, .rtf, .txt, or .md file with Pangram.

    Because Pangram's multipart file endpoint currently uses the `default` model,
    .docx and plain-text files extract text locally and route through `pangram-4`
    when `prefer_pangram_4=True` so humanizer scores are available.
    """
    key = api_key.strip()
    if not key:
        raise PangramError("Missing Pangram API key — configure it in AI Engine Settings.", 400)
    if not file_bytes:
        raise PangramError("Uploaded file is empty.", 400)

    ext = Path(filename).suffix.lower()
    if prefer_pangram_4 and ext == ".docx":
        extracted = extract_docx_text(file_bytes)
        res = scan_text_pangram(
            extracted,
            api_key=key,
            model="pangram-4",
            public_dashboard_link=public_dashboard_link,
        )
        res["scan_type"] = "file"
        res["filename"] = Path(filename).name
        return res

    if ext in (".txt", ".md"):
        decoded = file_bytes.decode("utf-8", errors="replace")
        res = scan_text_pangram(
            decoded,
            api_key=key,
            model="pangram-4" if prefer_pangram_4 else "default",
            public_dashboard_link=public_dashboard_link,
        )
        res["scan_type"] = "file"
        res["filename"] = Path(filename).name
        return res

    body, content_type = _encode_multipart(filename, file_bytes, public_dashboard_link)
    req = urllib.request.Request(
        f"{PANGRAM_FILE_BASE}/",
        data=body,
        headers={"x-api-key": key, "Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise _map_http_error(err, "POST file upload") from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise PangramError(f"Pangram file upload failed: {err}", 502) from err

    item = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(item, dict):
        raise PangramError("Unexpected response format from Pangram file upload endpoint.", 502)
    extracted_text = str(item.get("text") or "")
    normalized = _normalize_pangram_result(item, extracted_text, "default")
    normalized["scan_type"] = "file"
    normalized["filename"] = str(item.get("filename") or Path(filename).name)
    return normalized


def check_web_plagiarism_pangram(text: str, api_key: str) -> dict[str, Any]:
    """Run synchronous web plagiarism check via https://plagiarism.api.pangram.com/."""
    key = api_key.strip()
    if not key:
        raise PangramError("Missing Pangram API key — configure it in AI Engine Settings.", 400)
    if len(text.strip()) < 50:
        raise PangramError("Pangram requires at least 50 characters for a plagiarism check.", 400)

    data = _request_json(
        f"{PANGRAM_PLAGIARISM_BASE}/",
        api_key=key,
        method="POST",
        payload={"text": text},
        timeout_s=30.0,
    )
    if not isinstance(data, dict):
        data = {"raw": data}

    matches = data.get("matches") or data.get("sources") or data.get("results") or []
    return {
        "scan_type": "plagiarism",
        "text": text,
        "plagiarism_score": round(float(data.get("plagiarism_score") or data.get("similarity") or 0.0), 4),
        "plagiarism_detected": bool(data.get("plagiarism_detected", bool(matches))),
        "matches": matches if isinstance(matches, list) else [],
        "estimate": estimate_pangram_cost(text),
        "raw": data,
    }
