"""Main orchestrator for the 3-Stage In-House AI Authorship & Humanizer Detector."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from engine.db import get_cached_ai_detection, get_settings, set_cached_ai_detection

from .arbiter import DEFAULT_ARBITER_MODEL, run_arbiter_synthesis
from .calibrator import calibrate_report
from .dna_regeneration import DEFAULT_OPENROUTER_URL, DEFAULT_REGEN_MODELS, run_dna_regeneration
from .preprocessor import extract_document_text, segment_document
from .stylometrics import score_window_stylometrics


class InHouseDetectorError(Exception):
    """Structured error raised for invalid detector inputs."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def resolve_openrouter_config(
    settings: dict[str, str] | None = None,
    override_api_key: str = "",
) -> dict[str, str]:
    """Resolve OpenRouter API key, base URL, and model selection from settings or env."""
    cfg = settings or {}
    api_key = (
        override_api_key.strip()
        or (cfg.get("openrouter_api_key") or "").strip()
        or (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    )
    base_url = (
        (cfg.get("openrouter_base_url") or "").strip()
        or (os.environ.get("OPENROUTER_BASE_URL") or "").strip()
        or DEFAULT_OPENROUTER_URL
    )
    regen_models = (
        (cfg.get("ai_detect_regen_models") or "").strip()
        or (os.environ.get("AI_DETECT_REGEN_MODELS") or "").strip()
        or DEFAULT_REGEN_MODELS
    )
    arbiter_model = (
        (cfg.get("ai_detect_arbiter_model") or "").strip()
        or (os.environ.get("AI_DETECT_ARBITER_MODEL") or "").strip()
        or DEFAULT_ARBITER_MODEL
    )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "regen_models": regen_models,
        "arbiter_model": arbiter_model,
    }


def _build_cache_key(text: str, cfg: dict[str, str], use_openrouter: bool) -> str:
    mode_tag = "ensemble" if (use_openrouter and cfg["api_key"]) else "offline_v1"
    raw = f"{mode_tag}|{cfg['regen_models']}|{cfg['arbiter_model']}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def analyze_text_for_ai(
    text: str,
    use_openrouter: bool = True,
    override_api_key: str = "",
    bypass_cache: bool = False,
    scan_type: str = "inhouse",
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Run the 3-Stage Multi-Signal AI Writing Detector with SHA-256 caching."""
    if not text or len(text.strip()) < 50:
        raise InHouseDetectorError("Provide at least 50 characters to run AI authorship detection.", 400)

    settings = get_settings(db_path)
    cfg = resolve_openrouter_config(settings, override_api_key=override_api_key)
    cache_key = _build_cache_key(text, cfg, use_openrouter=use_openrouter)

    if not bypass_cache:
        cached = get_cached_ai_detection(cache_key, db_path=db_path)
        if cached:
            cached["cache_hit"] = True
            return cached

    segments = segment_document(text)
    stylometrics_by_idx = {
        idx: score_window_stylometrics(seg.text)
        for idx, seg in enumerate(segments)
        if not seg.is_excluded_reference
    }

    dna_by_idx: dict[int, dict[str, Any]] = {}
    arbiter_res: dict[str, Any] = {}
    total_tokens = 0
    total_cost = 0.0

    if use_openrouter and cfg["api_key"]:
        dna_by_idx = run_dna_regeneration(
            segments=segments,
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            regen_models_csv=cfg["regen_models"],
        )
        for d in dna_by_idx.values():
            total_tokens += int(d.get("tokens") or 0)
            total_cost += float(d.get("cost_usd") or 0.0)

        arbiter_res = run_arbiter_synthesis(
            segments=segments,
            stylometrics_by_idx=stylometrics_by_idx,
            dna_by_idx=dna_by_idx,
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            arbiter_model=cfg["arbiter_model"],
        )
        total_tokens += int(arbiter_res.get("tokens") or 0)
        total_cost += float(arbiter_res.get("cost_usd") or 0.0)

    model_desc = (
        f"inhouse-3stage ({cfg['arbiter_model']})"
        if (dna_by_idx or arbiter_res)
        else "inhouse-forensic-stage1"
    )
    report = calibrate_report(
        text=text,
        segments=segments,
        stylometrics_by_idx=stylometrics_by_idx,
        dna_by_idx=dna_by_idx,
        arbiter_res=arbiter_res,
        model_name=model_desc,
        scan_type=scan_type,
        total_api_tokens=total_tokens,
        total_api_cost_usd=total_cost,
    )
    payload = report.to_dict()
    set_cached_ai_detection(cache_key, payload, db_path=db_path)
    return payload


def scan_file_inhouse(
    filename: str,
    file_bytes: bytes,
    use_openrouter: bool = True,
    override_api_key: str = "",
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Extract text from .docx/.rtf/.txt/.md and run the 3-Stage AI detector."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        raise InHouseDetectorError(
            "Local text extraction supports .docx, .rtf, .txt, and .md. For PDFs, switch engine to Pangram Labs API or paste the PDF text.",
            400,
        )
    if not file_bytes:
        raise InHouseDetectorError("Uploaded file is empty.", 400)
    extracted = extract_document_text(filename, file_bytes)
    res = analyze_text_for_ai(
        extracted,
        use_openrouter=use_openrouter,
        override_api_key=override_api_key,
        scan_type="inhouse_file",
        db_path=db_path,
    )
    res["filename"] = Path(filename).name
    return res
