"""Stage 3: OpenRouter structured JSON synthesis & AI-humanizer detector."""

from __future__ import annotations

import json
from typing import Any

from .dna_regeneration import DEFAULT_OPENROUTER_URL, OpenRouterError, openrouter_chat_completion
from .preprocessor import PreprocessedSegment

DEFAULT_ARBITER_MODEL = "google/gemini-2.5-flash"

_ARBITER_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ai_detection_arbiter",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "overall_summary": {"type": "string"},
                "windows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "window_index": {"type": "integer"},
                            "ai_score": {"type": "number"},
                            "is_humanized": {"type": "boolean"},
                            "humanizer_score": {"type": "number"},
                            "reasons": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "window_index",
                            "ai_score",
                            "is_humanized",
                            "humanizer_score",
                            "reasons",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["overall_summary", "windows"],
            "additionalProperties": False,
        },
    },
}


def run_arbiter_synthesis(
    segments: list[PreprocessedSegment],
    stylometrics_by_idx: dict[int, dict[str, Any]],
    dna_by_idx: dict[int, dict[str, Any]],
    api_key: str,
    base_url: str = DEFAULT_OPENROUTER_URL,
    arbiter_model: str = DEFAULT_ARBITER_MODEL,
) -> dict[str, Any]:
    """Synthesize forensic stylometrics + DNA-GPT evidence via structured JSON arbitration."""
    if not api_key.strip():
        return {}

    evidence_blocks: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if seg.is_excluded_reference:
            continue
        sty = stylometrics_by_idx.get(idx, {})
        dna = dna_by_idx.get(idx, {})
        evidence_blocks.append(
            {
                "window_index": idx,
                "word_count": seg.word_count,
                "text": seg.text[:900],
                "stage1_stylometrics": {
                    "stylometric_ai_score": sty.get("stylometric_ai_score"),
                    "stylometric_humanizer_score": sty.get("stylometric_humanizer_score"),
                    "burstiness_cv": sty.get("burstiness_cv"),
                    "contraction_rate": sty.get("contraction_rate"),
                    "mattr": sty.get("mattr"),
                    "ai_lexicon_hits": sty.get("ai_lexicon_hits"),
                    "unicode_anomaly_count": sty.get("unicode_anomaly_count"),
                    "homoglyph_count": sty.get("homoglyph_count"),
                },
                "stage2_dna_gpt": {
                    "dna_bscore": dna.get("dna_bscore"),
                    "dna_entropy": dna.get("dna_entropy"),
                },
            }
        )

    if not evidence_blocks:
        return {}

    system_prompt = (
        "You are an academic forensics arbiter synthesizing multi-signal evidence for AI authorship and AI-humanizer scrubbing.\n"
        "CRITICAL GUIDELINES:\n"
        "1. NEVER rely on formality alone to flag text as AI. Formal academic or technical prose with low DNA-GPT BScore (< 0.35) "
        "and zero homoglyph/watermark tampering is very often HUMAN.\n"
        "2. Weigh Stage 2 DNA-GPT BScore heavily: high BScore (>= 0.55) indicates strong LLM continuation alignment.\n"
        "3. Detect AI-Humanizer evasion (is_humanized=true, humanizer_score 0.0-1.0) when you observe awkward thesaurus swaps, "
        "mixed-script homoglyphs, invisible Unicode watermarks, or uniform AI structure masked by forced contractions.\n"
        "4. Return strict JSON matching the schema with ai_score in [0.0, 1.0], humanizer_score in [0.0, 1.0], and 1-3 concise forensic reasons per window."
    )
    user_prompt = json.dumps({"windows_evidence": evidence_blocks}, ensure_ascii=False)

    model = (arbiter_model or DEFAULT_ARBITER_MODEL).strip()
    resp: dict[str, Any] = {}
    try:
        resp = openrouter_chat_completion(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=1200,
            response_format=_ARBITER_JSON_SCHEMA,
            timeout_s=28.0,
        )
    except OpenRouterError:
        try:
            resp = openrouter_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + "\nReturn a JSON object."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.15,
                max_tokens=1200,
                response_format={"type": "json_object"},
                timeout_s=28.0,
            )
        except OpenRouterError:
            return {}

    choices = resp.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    raw_content = str((choices[0].get("message") or {}).get("content") or "").strip()
    if not raw_content:
        return {}
    if raw_content.startswith("```"):
        raw_content = raw_content.strip("`")
        if raw_content.startswith("json"):
            raw_content = raw_content[4:].strip()
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return {}

    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    tokens = int(usage.get("total_tokens") or 0)
    cost = float(usage["cost"]) if usage.get("cost") is not None else (tokens / 1_000_000.0) * 0.15

    by_window: dict[int, dict[str, Any]] = {}
    for item in parsed.get("windows") or []:
        if not isinstance(item, dict):
            continue
        widx = int(item.get("window_index", -1))
        if widx >= 0:
            by_window[widx] = {
                "arbiter_ai_score": min(1.0, max(0.0, float(item.get("ai_score") or 0.0))),
                "is_humanized": bool(item.get("is_humanized", False)),
                "arbiter_humanizer_score": min(1.0, max(0.0, float(item.get("humanizer_score") or 0.0))),
                "reasons": [str(r) for r in (item.get("reasons") or []) if str(r).strip()],
            }

    return {
        "overall_summary": str(parsed.get("overall_summary") or ""),
        "by_window": by_window,
        "tokens": tokens,
        "cost_usd": cost,
    }
