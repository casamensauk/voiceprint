"""Stage 1: Pure deterministic stylometrics, Unicode/homoglyph & humanizer forensics."""

from __future__ import annotations

import re
import statistics
from typing import Any

from engine.analyzer import (
    HIGH_RISK_AI_WORDS,
    ROBOTIC_TRANSITIONS,
    compute_contraction_stats,
    split_sentences,
)

# Zero-width & invisible watermark characters commonly inserted by obfuscators/LLMs
_ZERO_WIDTH_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM / zero-width no-break space
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
    "\u180e",  # Mongolian vowel separator
}
_EXOTIC_SPACES = {"\u00a0", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007", "\u2008", "\u2009", "\u200a", "\u202f", "\u205f"}

# Cyrillic / Greek lookalike codepoints used in homoglyph spoofing attacks
_HOMOGLYPH_CHARS = set(
    "аеорсухіАЕОРСУХІКМНТВ"  # Cyrillic lookalikes
    "αβγεζηικμνορστυχωΑΒΕΖΗΙΚΜΝΟΡΤΧ"  # Greek lookalikes
)

_TOKEN_ALPHA_RE = re.compile(r"\S+")
_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")
_NOT_ONLY_RE = re.compile(r"\bnot only\b.{2,120}\bbut also\b", re.IGNORECASE | re.DOTALL)

# Awkward thesaurus/humanizer swaps (stiff lexical artifacts often left by paraphrasers)
_HUMANIZER_ARTIFACTS = (
    "peruse",
    "aforementioned",
    "inquisitive",
    "thereby",
    "whereby",
    "in essence",
    "to put it simply",
    "by the same token",
    "it goes without saying",
    "one might contend",
    "in light of the aforementioned",
    "commence to",
    "exhibits a propensity",
    "endeavor to",
)


def compute_mattr(words: list[str], window_size: int = 40) -> float:
    """Compute Moving-Average Type-Token Ratio (lexical diversity independent of length)."""
    if not words:
        return 0.0
    low = [w.lower() for w in words]
    if len(low) <= window_size:
        return len(set(low)) / float(len(low))
    ratios = []
    for i in range(len(low) - window_size + 1):
        chunk = low[i : i + window_size]
        ratios.append(len(set(chunk)) / float(window_size))
    return sum(ratios) / float(len(ratios))


def detect_unicode_and_homoglyphs(text: str) -> dict[str, Any]:
    """Scan text for invisible watermark chars, exotic whitespace, and mixed-script homoglyphs."""
    zero_width_count = sum(1 for ch in text if ch in _ZERO_WIDTH_CHARS)
    exotic_space_count = sum(1 for ch in text if ch in _EXOTIC_SPACES)
    homoglyph_tokens: list[str] = []

    for tok in _TOKEN_ALPHA_RE.findall(text):
        has_latin = any("a" <= c.lower() <= "z" for c in tok)
        has_homoglyph = any(c in _HOMOGLYPH_CHARS for c in tok)
        if has_latin and has_homoglyph:
            homoglyph_tokens.append(tok)

    return {
        "zero_width_count": zero_width_count,
        "exotic_space_count": exotic_space_count,
        "unicode_anomaly_count": zero_width_count + exotic_space_count,
        "homoglyph_count": len(homoglyph_tokens),
        "homoglyph_tokens": homoglyph_tokens[:5],
    }


def score_window_stylometrics(text: str) -> dict[str, Any]:
    """Run deterministic Stage 1 forensics on a single window, returning scores and reasons."""
    cleaned = text.strip()
    words = _WORD_RE.findall(cleaned)
    word_count = max(1, len(words))
    sentences = split_sentences(cleaned)
    sent_lengths = [len(s.split()) for s in sentences if s.split()]

    # 1. Burstiness CV
    if len(sent_lengths) >= 2:
        mean_l = statistics.mean(sent_lengths)
        std_l = statistics.pstdev(sent_lengths)
        cv = (std_l / mean_l) if mean_l > 0 else 0.0
        adjacent_deltas = [abs(sent_lengths[i] - sent_lengths[i - 1]) for i in range(1, len(sent_lengths))]
        avg_delta = statistics.mean(adjacent_deltas) if adjacent_deltas else 0.0
    else:
        cv = 0.35
        avg_delta = 5.0

    # 2. Contraction ratio
    c_cnt, e_cnt, contraction_rate = compute_contraction_stats(cleaned)
    has_contraction_ops = (c_cnt + e_cnt) > 0

    # 3. Lexical diversity (MATTR)
    mattr = compute_mattr(words, window_size=40)

    # 4. AI vocabulary & robotic transitions
    vocab_hits: list[str] = []
    for word in HIGH_RISK_AI_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", cleaned, re.IGNORECASE):
            vocab_hits.append(word)

    transition_hits: list[str] = []
    for phrase in ROBOTIC_TRANSITIONS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", cleaned, re.IGNORECASE):
            transition_hits.append(phrase)

    em_dash_count = cleaned.count("—")
    has_not_only = bool(_NOT_ONLY_RE.search(cleaned))

    # 5. Unicode / watermark / homoglyph checks
    uni = detect_unicode_and_homoglyphs(cleaned)

    # 6. Humanizer artifact heuristics
    artifact_hits = [
        art for art in _HUMANIZER_ARTIFACTS if re.search(r"\b" + re.escape(art) + r"\b", cleaned, re.IGNORECASE)
    ]

    # Combine deterministic AI signals into [0.0, 1.0]
    ai_score = 0.12
    reasons: list[str] = []

    if vocab_hits:
        ai_score += min(0.42, len(vocab_hits) * 0.14)
        reasons.append(f"AI lexicon markers ({len(vocab_hits)}): {', '.join(vocab_hits[:4])}")
    if transition_hits:
        ai_score += min(0.30, len(transition_hits) * 0.15)
        reasons.append(f"Formulaic transitions: {', '.join(transition_hits[:3])}")
    if len(sent_lengths) >= 3 and cv < 0.24:
        ai_score += 0.18
        reasons.append(f"Uniform sentence cadence (burstiness CV={cv:.2f})")
    elif len(sent_lengths) >= 3 and avg_delta <= 2.5:
        ai_score += 0.10
        reasons.append(f"Low inter-sentence length delta ({avg_delta:.1f}w)")

    if em_dash_count >= 2 or (em_dash_count >= 1 and word_count <= 90):
        ai_score += 0.12
        reasons.append(f"Elevated em-dash density ({em_dash_count})")
    if has_not_only:
        ai_score += 0.10
        reasons.append("Correlative 'Not only ... but also ...' template")
    if has_contraction_ops and contraction_rate == 0.0 and word_count >= 35:
        ai_score += 0.08
        reasons.append("Zero contractions despite formal expansion opportunities")
    if mattr < 0.62 and word_count >= 45:
        ai_score += 0.08
        reasons.append(f"Low moving-average lexical diversity (MATTR={mattr:.2f})")

    # Humanizer / spoofing signal
    humanizer_score = 0.05
    if uni["homoglyph_count"] > 0:
        humanizer_score += min(0.85, 0.55 + uni["homoglyph_count"] * 0.15)
        ai_score = max(ai_score, 0.78)
        reasons.append(
            f"Mixed-script homoglyph spoofing detected ({uni['homoglyph_count']} token(s): {', '.join(uni['homoglyph_tokens'])})"
        )
    if uni["unicode_anomaly_count"] > 0:
        humanizer_score += min(0.65, 0.35 + uni["unicode_anomaly_count"] * 0.10)
        ai_score = max(ai_score, 0.68)
        reasons.append(f"Invisible/exotic Unicode characters detected ({uni['unicode_anomaly_count']})")
    if artifact_hits:
        humanizer_score += min(0.45, len(artifact_hits) * 0.20)
        reasons.append(f"Thesaurus/paraphraser phrasing cues: {', '.join(artifact_hits[:3])}")
    if len(sent_lengths) >= 3 and cv < 0.22 and not vocab_hits and contraction_rate > 0.7:
        # Classic humanizer signature: forced contractions inside unnaturally uniform cadence
        humanizer_score += 0.28
        reasons.append("Uniform sentence length masked with forced contractions")

    if not reasons:
        reasons.append("Natural lexical diversity and human sentence cadence")

    return {
        "stylometric_ai_score": round(min(1.0, max(0.0, ai_score)), 4),
        "stylometric_humanizer_score": round(min(1.0, max(0.0, humanizer_score)), 4),
        "burstiness_cv": round(cv, 3),
        "contraction_rate": round(contraction_rate, 3),
        "mattr": round(mattr, 3),
        "ai_lexicon_hits": len(vocab_hits) + len(transition_hits),
        "unicode_anomaly_count": uni["unicode_anomaly_count"],
        "homoglyph_count": uni["homoglyph_count"],
        "reasons": reasons,
    }
