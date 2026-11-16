"""Stage 4: Weighted ensemble calibration, short-text clamping, and 1%-19%* flagging."""

from __future__ import annotations

from typing import Any

from .preprocessor import PreprocessedSegment
from .schemas import DetectorReport, DetectorWindow, WindowSignals


def _confidence_label(score: float, total_words: int, is_full_ensemble: bool) -> str:
    dist = abs(score - 0.5)
    if total_words < 300:
        # Short-text confidence clamping (< 300 words)
        if dist >= 0.30 and is_full_ensemble:
            return "Medium"
        return "Low"
    if dist >= 0.30:
        return "High"
    if dist >= 0.14:
        return "Medium"
    return "Low"


def calibrate_report(
    text: str,
    segments: list[PreprocessedSegment],
    stylometrics_by_idx: dict[int, dict[str, Any]],
    dna_by_idx: dict[int, dict[str, Any]],
    arbiter_res: dict[str, Any],
    model_name: str,
    scan_type: str = "inhouse",
    total_api_tokens: int = 0,
    total_api_cost_usd: float = 0.0,
) -> DetectorReport:
    """Combine Stage 1-3 signals into a Pangram v4.1-compatible calibrated DetectorReport."""
    total_words = len(text.split())
    low_word_count_warning = total_words < 300
    arbiter_windows = arbiter_res.get("by_window") or {}
    has_dna = any((d.get("dna_bscore") is not None) for d in dna_by_idx.values())
    has_arbiter = bool(arbiter_windows)
    mode = "ensemble_3_stage" if (has_dna or has_arbiter) else "forensic_offline"

    windows: list[DetectorWindow] = []
    body_word_total = 0
    weighted_ai_words = 0.0
    weighted_assisted_words = 0.0
    weighted_human_words = 0.0

    num_ai = 0
    num_assisted = 0
    num_human = 0
    has_humanized_any = False
    max_humanizer_score = 0.0

    sty_scores_all: list[float] = []
    dna_scores_all: list[float] = []
    arb_scores_all: list[float] = []

    for idx, seg in enumerate(segments):
        if seg.is_excluded_reference:
            windows.append(
                DetectorWindow(
                    text=seg.text,
                    label="Human",
                    ai_assistance_score=0.0,
                    confidence="High",
                    is_humanized=False,
                    humanizer_score=0.0,
                    start_index=seg.start_index,
                    end_index=seg.end_index,
                    word_count=seg.word_count,
                    token_length=int(round(seg.word_count * 1.3)),
                    is_excluded_reference=True,
                    reasons=["Excluded reference/bibliography section (protected from false positives)"],
                    signals=WindowSignals(),
                )
            )
            continue

        sty = stylometrics_by_idx.get(idx, {})
        dna = dna_by_idx.get(idx, {})
        arb = arbiter_windows.get(idx, {})

        s_ai = float(sty.get("stylometric_ai_score") or 0.0)
        s_hum = float(sty.get("stylometric_humanizer_score") or 0.0)
        d_bscore = dna.get("dna_bscore")
        d_entropy = dna.get("dna_entropy")
        a_ai = arb.get("arbiter_ai_score")
        a_hum = arb.get("arbiter_humanizer_score")

        sty_scores_all.append(s_ai)
        if d_bscore is not None:
            dna_scores_all.append(float(d_bscore))
        if a_ai is not None:
            arb_scores_all.append(float(a_ai))

        # Weighted ensemble combination
        w_sum = 0.25
        score_sum = 0.25 * s_ai
        if d_bscore is not None:
            w_sum += 0.40
            score_sum += 0.40 * float(d_bscore)
        if a_ai is not None:
            w_sum += 0.35
            score_sum += 0.35 * float(a_ai)
        raw_ai = score_sum / w_sum

        # Combine humanizer score
        hum_score = max(s_hum, float(a_hum)) if a_hum is not None else s_hum
        # Hard guard: mixed-script homoglyphs or invisible watermarks indicate deliberate obfuscation
        if int(sty.get("homoglyph_count") or 0) > 0 or int(sty.get("unicode_anomaly_count") or 0) > 0:
            hum_score = max(hum_score, 0.72)
            raw_ai = max(raw_ai, 0.72)

        # Short-text conservative clamping (< 300 words): pull borderline predictions slightly toward 0.35
        if low_word_count_warning and 0.38 <= raw_ai <= 0.68 and int(sty.get("homoglyph_count") or 0) == 0:
            raw_ai = 0.75 * raw_ai + 0.25 * 0.35

        raw_ai = round(min(1.0, max(0.0, raw_ai)), 4)
        hum_score = round(min(1.0, max(0.0, hum_score)), 4)
        is_humanized = bool(arb.get("is_humanized", False)) or hum_score >= 0.55

        if is_humanized:
            has_humanized_any = True
        if hum_score > max_humanizer_score:
            max_humanizer_score = hum_score

        if raw_ai >= 0.72:
            label = "AI"
            num_ai += 1
            weighted_ai_words += seg.word_count
        elif raw_ai >= 0.42:
            label = "AI-Assisted"
            num_assisted += 1
            weighted_assisted_words += seg.word_count
        else:
            label = "Human"
            num_human += 1
            weighted_human_words += seg.word_count

        body_word_total += seg.word_count

        merged_reasons: list[str] = []
        for r in (arb.get("reasons") or []) + (sty.get("reasons") or []):
            if r not in merged_reasons:
                merged_reasons.append(r)
        if d_bscore is not None:
            merged_reasons.insert(0, f"DNA-GPT regeneration BScore: {float(d_bscore):.2f}")

        windows.append(
            DetectorWindow(
                text=seg.text,
                label=label,
                ai_assistance_score=raw_ai,
                confidence=_confidence_label(raw_ai, total_words, mode == "ensemble_3_stage"),
                is_humanized=is_humanized,
                humanizer_score=hum_score,
                start_index=seg.start_index,
                end_index=seg.end_index,
                word_count=seg.word_count,
                token_length=int(round(seg.word_count * 1.3)),
                is_excluded_reference=False,
                reasons=merged_reasons[:4],
                signals=WindowSignals(
                    stylometric_ai_score=s_ai,
                    stylometric_humanizer_score=s_hum,
                    dna_bscore=d_bscore,
                    dna_entropy=d_entropy,
                    arbiter_ai_score=a_ai,
                    arbiter_humanizer_score=a_hum,
                    burstiness_cv=float(sty.get("burstiness_cv") or 0.0),
                    contraction_rate=float(sty.get("contraction_rate") or 0.5),
                    mattr=float(sty.get("mattr") or 0.0),
                    ai_lexicon_hits=int(sty.get("ai_lexicon_hits") or 0),
                    unicode_anomaly_count=int(sty.get("unicode_anomaly_count") or 0),
                    homoglyph_count=int(sty.get("homoglyph_count") or 0),
                ),
            )
        )

    denom = max(1, body_word_total)
    frac_ai = round(weighted_ai_words / denom, 4)
    frac_assisted = round(weighted_assisted_words / denom, 4)
    frac_human = round(max(0.0, 1.0 - frac_ai - frac_assisted), 4)

    ai_combined = frac_ai + frac_assisted
    low_confidence_asterisk = 0.01 <= ai_combined <= 0.19

    if frac_ai >= 0.50:
        headline = "Likely AI-Generated"
        pred_short = "AI"
    elif ai_combined >= 0.20:
        headline = "AI-Assisted / Mixed"
        pred_short = "AI-Assisted"
    elif low_confidence_asterisk:
        headline = f"Low-Signal Review ({int(round(ai_combined * 100))}%*)"
        pred_short = "Low-Confidence"
    else:
        headline = "Likely Human"
        pred_short = "Human"

    notes: list[str] = []
    if arbiter_res.get("overall_summary"):
        notes.append(str(arbiter_res["overall_summary"]))
    else:
        notes.append(
            f"Analyzed {body_word_total} body words across {num_ai + num_assisted + num_human} window(s) "
            f"using {'3-Stage Ensemble (Stylometrics + DNA-GPT + Arbiter)' if mode == 'ensemble_3_stage' else 'Offline Stage 1 Deterministic Forensics'}."
        )
    if low_word_count_warning:
        notes.append(
            f"Short-text advisory ({total_words} words < 300-word threshold): confidence clamped conservatively."
        )
    if low_confidence_asterisk:
        notes.append(
            "Turnitin-style 1%–19%* low-confidence band: small fraction detected — treat as inconclusive without corroboration."
        )
    if has_humanized_any:
        notes.append(
            f"AI-humanizer or obfuscation cues detected (peak humanizer score {int(round(max_humanizer_score * 100))}%)."
        )

    cost_per_1k = (total_api_cost_usd / max(1, total_words)) * 1000.0
    ensemble_breakdown = {
        "mode": mode,
        "stage1_avg_ai": round(sum(sty_scores_all) / max(1, len(sty_scores_all)), 3),
        "stage2_avg_bscore": round(sum(dna_scores_all) / len(dna_scores_all), 3) if dna_scores_all else None,
        "stage3_avg_arbiter": round(sum(arb_scores_all) / len(arb_scores_all), 3) if arb_scores_all else None,
        "api_tokens": total_api_tokens,
        "api_cost_usd": round(total_api_cost_usd, 6),
        "cost_per_1k_words_usd": round(cost_per_1k, 6),
    }

    return DetectorReport(
        scan_type=scan_type,
        engine="inhouse",
        mode=mode,
        version="inhouse-3stage-1.0",
        model=model_name,
        text=text,
        headline=headline,
        prediction=" ".join(notes),
        prediction_short=pred_short,
        fraction_ai=frac_ai,
        fraction_ai_assisted=frac_assisted,
        fraction_human=frac_human,
        num_ai_segments=num_ai,
        num_ai_assisted_segments=num_assisted,
        num_human_segments=num_human,
        has_humanized_segments=has_humanized_any,
        max_humanizer_score=round(max_humanizer_score, 4),
        low_word_count_warning=low_word_count_warning,
        low_confidence_asterisk=low_confidence_asterisk,
        cache_hit=False,
        ensemble_breakdown=ensemble_breakdown,
        windows=windows,
        estimate={
            "char_count": len(text.strip()),
            "word_count": total_words,
            "meets_min_chars": len(text.strip()) >= 50,
            "estimated_cost_usd": round(total_api_cost_usd, 6),
        },
        dashboard_link="",
    )
