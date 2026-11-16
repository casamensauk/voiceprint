"""Main orchestrator for the 3-Stage In-House AI Authorship & Humanizer Detector."""

from __future__ import annotations

import hashlib
import os
import urllib.request
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


def _is_ollama_up(ollama_url: str) -> bool:
    try:
        req = urllib.request.Request(f"{ollama_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.35) as resp:
            return resp.status == 200
    except Exception:
        return False


def resolve_openrouter_config(
    settings: dict[str, str] | None = None,
    override_api_key: str = "",
) -> dict[str, str]:
    """Resolve active LLM config: OpenRouter first, falling back to configured Gemini/OpenAI/Anthropic/Ollama."""
    cfg = settings or {}
    openrouter_key = (
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

    if openrouter_key:
        return {
            "provider": "openrouter",
            "api_key": openrouter_key,
            "base_url": base_url,
            "regen_models": regen_models,
            "arbiter_model": arbiter_model,
        }

    # Automatic multi-provider fallback when user already saved Gemini / OpenAI / Anthropic / Ollama
    if (cfg.get("gemini_api_key") or "").strip():
        gm = (cfg.get("gemini_model") or "gemini-2.5-flash").strip()
        return {
            "provider": "gemini",
            "api_key": cfg["gemini_api_key"].strip(),
            "base_url": "",
            "regen_models": gm,
            "arbiter_model": gm,
        }
    if (cfg.get("openai_api_key") or "").strip():
        om = (cfg.get("openai_model") or "gpt-4o-mini").strip()
        return {
            "provider": "openai",
            "api_key": cfg["openai_api_key"].strip(),
            "base_url": "",
            "regen_models": om,
            "arbiter_model": om,
        }
    if (cfg.get("anthropic_api_key") or "").strip():
        am = (cfg.get("anthropic_model") or "claude-3-5-haiku-latest").strip()
        return {
            "provider": "anthropic",
            "api_key": cfg["anthropic_api_key"].strip(),
            "base_url": "",
            "regen_models": am,
            "arbiter_model": am,
        }
    ollama_url = (cfg.get("ollama_url") or "http://localhost:11434").strip()
    if (cfg.get("provider") == "ollama") and _is_ollama_up(ollama_url):
        ol_model = (cfg.get("ollama_model") or "llama3.2").strip()
        return {
            "provider": "ollama",
            "api_key": "ollama",
            "base_url": ollama_url,
            "regen_models": ol_model,
            "arbiter_model": ol_model,
        }

    return {
        "provider": "openrouter",
        "api_key": "",
        "base_url": base_url,
        "regen_models": regen_models,
        "arbiter_model": arbiter_model,
    }


def _build_cache_key(text: str, cfg: dict[str, str], use_openrouter: bool) -> str:
    mode_tag = (
        f"ensemble_{cfg['provider']}"
        if (use_openrouter and cfg["api_key"])
        else "offline_v2"
    )
    raw = f"v2_3|{mode_tag}|{cfg['regen_models']}|{cfg['arbiter_model']}|{text}"
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
            regen_models_csv=cfg["regen_models"],
            provider_cfg=cfg,
        )
        for d in dna_by_idx.values():
            total_tokens += int(d.get("tokens") or 0)
            total_cost += float(d.get("cost_usd") or 0.0)

        arbiter_res = run_arbiter_synthesis(
            segments=segments,
            stylometrics_by_idx=stylometrics_by_idx,
            dna_by_idx=dna_by_idx,
            arbiter_model=cfg["arbiter_model"],
            provider_cfg=cfg,
        )
        total_tokens += int(arbiter_res.get("tokens") or 0)
        total_cost += float(arbiter_res.get("cost_usd") or 0.0)

    model_desc = (
        f"inhouse-3stage ({cfg['provider']}: {cfg['arbiter_model']})"
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
