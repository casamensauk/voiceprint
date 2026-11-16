"""Stage 1: Pure deterministic stylometrics, coursework taxonomy, RLHF prose, & Unicode forensics."""

from __future__ import annotations

import re
import statistics
from typing import Any

from engine.analyzer import (
    HIGH_RISK_AI_WORDS,
    ROBOTIC_TRANSITIONS,
    compute_contraction_stats,
)

from .preprocessor import split_sentences_recover_collapsed

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
_EXOTIC_SPACES = {
    "\u00a0", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
    "\u2007", "\u2008", "\u2009", "\u200a", "\u202f", "\u205f",
}

# Cyrillic / Greek lookalike codepoints used in homoglyph spoofing attacks
_HOMOGLYPH_CHARS = set(
    "аеорсухіАЕОРСУХІКМНТВ"
    "αβγεζηικμνορστυχωΑΒΕΖΗΙΚΜΝΟΡΤΧ"
)

_TOKEN_ALPHA_RE = re.compile(r"\S+")
_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")
_NOT_ONLY_RE = re.compile(r"\bnot only\b.{2,120}\bbut also\b", re.IGNORECASE | re.DOTALL)

# Modern GPT-4o / Claude / Gemini RLHF discourse framing constructions
_RLHF_FRAMING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Instead/Rather than -ing framing", re.compile(r"\b(?:instead|rather)\s+of\s+[a-z]+ing\b", re.IGNORECASE)),
    ("Participial 'By ...ing, ...' opener", re.compile(r"(?:^|[.!?]\s+)by\s+[a-z]+ing\b[^.!?\n]{4,90},", re.IGNORECASE)),
    ("Balanced contrast opener", re.compile(r"\b(?:at the same time|on the flip side|in turn|as a result),\s", re.IGNORECASE)),
    ("Synthesizing conclusion opener", re.compile(r"\b(?:ultimately|overall|importantly|notably|crucially|essentially),\s", re.IGNORECASE)),
    ("Scope framing opener", re.compile(r"\b(?:at its core|in practice|when it comes to|to address this)\b", re.IGNORECASE)),
    ("Causal enable/allow/ensure clause", re.compile(r"\b(?:enables?|allows?)\s+[a-z0-9\-\s]{2,36}\s+to\b|\bensures?\s+that\b", re.IGNORECASE)),
    ("Dual-pole 'both X and Y' balance", re.compile(r"\bboth\s+[a-z\-]{3,24}\s+and\s+[a-z\-]{3,24}\b", re.IGNORECASE)),
    ("Horizon/scope cliché", re.compile(r"\b(?:over the long term|across the board|in the long run)\b", re.IGNORECASE)),
    ("Role/importance cliché", re.compile(r"\bplays\s+a\s+(?:key|vital|central|critical|major|pivotal)\s+role\b", re.IGNORECASE)),
]

# Coursework & structured Q&A LLM forensics
_TAXONOMY_HEADING_RE = re.compile(
    r"(?:^|\n|(?<=[a-z0-9.)]))[ \t]*([A-Z][A-Za-z0-9 &/\-()’'–]{2,52}):[ \t]*(?=[A-Z\"“])"
)
_WHEN_ANAPHORA_RE = re.compile(r":\s*When\b")
_VERB_FIRST_DEF_RE = re.compile(
    r":\s*(?:Gives|Allows|Lets|Provides|Focuses|Helps|Tracks|Tools that|Built-in|"
    r"Centred on|Centered on|Perfect for|Great for|Ideal for|Use |Look at )\b"
)
_IMPERATIVE_STEP_RE = re.compile(
    r"^(?:Log into|Open|Go to|Click|Tap|Choose|Pick|Write|Type|Upload|Attach|Add|Adjust|Check|Review|Select)\b"
)
_OXFORD_TRIAD_RE = re.compile(
    r"\b[A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,3},\s+[A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,3},\s+(?:and|or)\s+[A-Za-z0-9\-]+",
    re.IGNORECASE,
)
_UK_SPELLING_RE = re.compile(
    r"\b(?:colour|colours|recognisable|disorganised|centred|organisation|organisations|behaviour|analyse|programme|advert|adverts|favourite)\b",
    re.IGNORECASE,
)
_MAKES_LOOK_RE = re.compile(
    r"\bmakes?\s+(?:a\s+|the\s+|your\s+)?(?:business|company|brand|profile|page|organisation|organization|them|it)\s+(?:look|appear|seem)\b",
    re.IGNORECASE,
)
_WHICH_EVAL_RE = re.compile(
    r",\s+which\s+(?:makes|damages|puts|helps|shows|matches|leads|creates|builds|turns|means|allows|gives)\b",
    re.IGNORECASE,
)
_EXPLAINER_STOCK_RE = re.compile(
    r"\b(?:so\s+(?:customers|users|people|followers)\s+can|potential\s+(?:new\s+)?customers|"
    r"turns\s+a\s+negative\s+situation\s+into\s+a\s+positive|"
    r"before\s+it\s+(?:turns\s+into|escalates)|when\s+things\s+go\s+wrong|in\s+real\s+time|"
    r"thought[- ]leadership|call-to-action|social\s+proof|organic\s+reach|target\s+audience|"
    r"brand\s+sentiment|behind-the-scenes|decision-makers)\b",
    re.IGNORECASE,
)

# Soft instructional / corporate / RLHF expository vocabulary
_MODERN_RLHF_WORDS: set[str] = {
    "collaboration", "well-defined", "milestones", "distributed", "expectations",
    "handoffs", "predictable", "synchronous", "asynchronous", "asynchronously",
    "high-performing", "centralized", "cohesion", "intentional", "deliverables",
    "retrospectives", "blockers", "structured", "purposeful", "velocity",
    "morale", "alignment", "stakeholders", "initiatives", "outcomes",
    "capabilities", "scalability", "streamline", "workflow", "workflows",
    "proactive", "proactively", "framework", "frameworks", "prioritize",
    "cultivate", "meaningful", "actionable", "consistency", "accountability",
    "visibility", "efficiency", "productivity", "sustainability", "implications",
    "considerations", "perspectives", "environments", "trajectories", "norms",
    "establishing", "maintaining", "balancing", "relying", "ensuring",
    "enhancing", "mitigating", "evaluating", "integrating", "facilitating",
}

_NOMINALIZATION_RE = re.compile(
    r"^[a-z]{4,}(?:tions?|sions?|ments?|ities|ity|ances?|ences?|ships?)$",
    re.IGNORECASE,
)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

_CASUAL_HUMAN_MARKERS = re.compile(
    r"\b(?:i|i'm|i've|i'd|i'll|my|me|yeah|nah|honestly|tbh|kinda|sorta|lol|haha|tho|gonna|wanna|lemme)\b",
    re.IGNORECASE,
)
_CITATION_OR_DATA_RE = re.compile(
    r"(?:\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?,?\s*\d{4}[^)]*\)|"
    r"\[\d+(?:\s*[-,]\s*\d+)*\]|"
    r"\b(?:p|n)\s*[<=>]\s*\d|"
    r"\b\d+(?:\.\d+)?%|\b(?:19|20)\d{2}\b)",
)

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


def _count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    groups = _VOWEL_GROUP_RE.findall(w)
    return max(1, len(groups))


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
    sentences = split_sentences_recover_collapsed(cleaned)
    sent_lengths = [len(s.split()) for s in sentences if s.split()]

    # 1. Burstiness CV & length range
    if len(sent_lengths) >= 2:
        mean_l = statistics.mean(sent_lengths)
        std_l = statistics.pstdev(sent_lengths)
        cv = (std_l / mean_l) if mean_l > 0 else 0.0
        adjacent_deltas = [abs(sent_lengths[i] - sent_lengths[i - 1]) for i in range(1, len(sent_lengths))]
        avg_delta = statistics.mean(adjacent_deltas) if adjacent_deltas else 0.0
        min_sent_len = min(sent_lengths)
    else:
        cv = 0.35
        avg_delta = 5.0
        min_sent_len = sent_lengths[0] if sent_lengths else 0

    # 2. Contraction ratio
    c_cnt, e_cnt, contraction_rate = compute_contraction_stats(cleaned)
    has_contraction_ops = (c_cnt + e_cnt) > 0

    # 3. Lexical diversity (MATTR) & morphology
    mattr = compute_mattr(words, window_size=40)
    alpha_words = [re.sub(r"[^A-Za-z]", "", w) for w in words]
    alpha_words = [w for w in alpha_words if w]
    avg_word_len = (sum(len(w) for w in alpha_words) / float(max(1, len(alpha_words))))
    nom_count = sum(1 for w in alpha_words if _NOMINALIZATION_RE.match(w))
    nom_rate = nom_count / float(max(1, len(alpha_words)))
    poly_count = sum(1 for w in alpha_words if _count_syllables(w) >= 3)
    poly_rate = poly_count / float(max(1, len(alpha_words)))

    # 4. Classic AI vocabulary & robotic transitions
    vocab_hits: list[str] = []
    for word in HIGH_RISK_AI_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", cleaned, re.IGNORECASE):
            vocab_hits.append(word)

    transition_hits: list[str] = []
    for phrase in ROBOTIC_TRANSITIONS:
        if re.search(r"\b" + re.escape(phrase) + r"\b", cleaned, re.IGNORECASE):
            transition_hits.append(phrase)

    # 5. Modern RLHF discourse framing & soft vocabulary
    framing_hits: list[str] = []
    for label, pat in _RLHF_FRAMING_PATTERNS:
        if pat.search(cleaned):
            framing_hits.append(label)

    triad_hits = _OXFORD_TRIAD_RE.findall(cleaned)
    rlhf_soft_hits = [w.lower() for w in words if w.lower() in _MODERN_RLHF_WORDS]
    rlhf_density_per_100 = (len(rlhf_soft_hits) / float(word_count)) * 100.0

    # 6. Structured Coursework & Plain-English Explainer Forensics
    raw_headings = [
        m.group(1).strip()
        for m in _TAXONOMY_HEADING_RE.finditer(cleaned)
        if m.group(1).strip().lower() not in ("your answer", "http", "https")
    ]
    taxonomy_count = len(raw_headings)
    when_anaphora_count = len(_WHEN_ANAPHORA_RE.findall(cleaned))
    verb_def_count = len(_VERB_FIRST_DEF_RE.findall(cleaned))

    # Count imperative UI/process checklist steps across sentences & colon splits
    step_clauses = []
    for sent in sentences:
        if ":" in sent:
            step_clauses.append(sent.split(":", 1)[1].strip())
        step_clauses.append(sent.strip())
    imperative_step_count = sum(1 for cl in step_clauses if _IMPERATIVE_STEP_RE.match(cl))

    makes_look_count = len(_MAKES_LOOK_RE.findall(cleaned))
    which_eval_count = len(_WHICH_EVAL_RE.findall(cleaned))
    explainer_stock_count = len(_EXPLAINER_STOCK_RE.findall(cleaned))

    has_uk_spelling = bool(_UK_SPELLING_RE.search(cleaned))
    has_oxford_comma = bool(re.search(r",\s+(?:and|or)\s+", cleaned))

    em_dash_count = cleaned.count("\u2014")
    en_dash_count = cleaned.count("\u2013")
    has_not_only = bool(_NOT_ONLY_RE.search(cleaned))

    # 7. Unicode / watermark / homoglyph checks
    uni = detect_unicode_and_homoglyphs(cleaned)

    # 8. Humanizer artifact heuristics
    artifact_hits = [
        art for art in _HUMANIZER_ARTIFACTS if re.search(r"\b" + re.escape(art) + r"\b", cleaned, re.IGNORECASE)
    ]

    # Combine deterministic AI signals into [0.0, 1.0]
    ai_score = 0.14
    reasons: list[str] = []

    if vocab_hits:
        ai_score += min(0.42, len(vocab_hits) * 0.14)
        reasons.append(f"AI lexicon markers ({len(vocab_hits)}): {', '.join(vocab_hits[:4])}")
    if transition_hits:
        ai_score += min(0.30, len(transition_hits) * 0.15)
        reasons.append(f"Formulaic transitions: {', '.join(transition_hits[:3])}")

    if framing_hits:
        ai_score += min(0.36, len(framing_hits) * 0.09)
        reasons.append(f"RLHF discourse scaffolding ({len(framing_hits)}): {', '.join(framing_hits[:3])}")

    if triad_hits:
        triad_add = 0.34 if len(triad_hits) >= 3 else (0.22 if len(triad_hits) == 2 else 0.10)
        ai_score += triad_add
        reasons.append(f"Parallel Oxford-comma triads ({len(triad_hits)})")

    if has_uk_spelling and has_oxford_comma and (len(triad_hits) >= 1 or taxonomy_count >= 2 or makes_look_count >= 1):
        ai_score += 0.10
        reasons.append("UK spelling paired with systematic US Oxford-comma cadence")

    if taxonomy_count >= 2:
        tax_add = 0.54 if taxonomy_count >= 5 else (0.40 if taxonomy_count >= 3 else 0.26)
        ai_score += tax_add
        reasons.append(f"Structured 'Heading: Explanation' taxonomy ({taxonomy_count} items)")

    if when_anaphora_count >= 2:
        ai_score += 0.26
        reasons.append(f"Repetitive ': When ...' example template ({when_anaphora_count}x)")

    if verb_def_count >= 2:
        ai_score += 0.26
        reasons.append(f"Verb-first textbook definitions ({verb_def_count}x)")

    if imperative_step_count >= 3:
        ai_score += 0.38
        reasons.append(f"Sequential imperative process checklist ({imperative_step_count} steps)")

    if makes_look_count >= 1:
        ai_score += 0.32 if makes_look_count >= 2 else 0.16
        reasons.append(f"Explainer 'makes [entity] look/appear' pattern ({makes_look_count}x)")

    if which_eval_count >= 1:
        ai_score += 0.24 if which_eval_count >= 2 else 0.12
        reasons.append(f"Evaluative ', which [verb]s' clause pattern ({which_eval_count}x)")

    if explainer_stock_count >= 2:
        ai_score += 0.30 if explainer_stock_count >= 4 else 0.18
        reasons.append(f"Plain-English explainer stock phrases ({explainer_stock_count})")

    if len(rlhf_soft_hits) >= 4 or rlhf_density_per_100 >= 7.0:
        add_soft = 0.22 if rlhf_density_per_100 >= 10.0 else 0.14
        ai_score += add_soft
        uniq_soft = list(dict.fromkeys(rlhf_soft_hits))[:4]
        reasons.append(f"Dense expository RLHF vocabulary ({rlhf_density_per_100:.1f}/100w: {', '.join(uniq_soft)})")

    if nom_rate >= 0.075 and word_count >= 35:
        ai_score += 0.14 if nom_rate >= 0.11 else 0.09
        reasons.append(f"Elevated abstract nominalization rate ({nom_rate * 100:.1f}%)")

    if poly_rate >= 0.23 and avg_word_len >= 5.3 and word_count >= 35:
        ai_score += 0.12
        reasons.append(f"Uniformly high polysyllabic density ({poly_rate * 100:.0f}% 3+ syllables, avg {avg_word_len:.1f} chars)")

    # Cadence & frictionless punctuation checks
    if len(sent_lengths) >= 3 and cv < 0.26:
        ai_score += 0.18
        reasons.append(f"Uniform sentence cadence (burstiness CV={cv:.2f})")
    elif len(sent_lengths) >= 2 and cv < 0.34 and min_sent_len >= 11:
        ai_score += 0.12
        reasons.append(f"Steady medium-long sentence band (min {min_sent_len}w, CV={cv:.2f})")
    elif len(sent_lengths) >= 3 and avg_delta <= 2.5:
        ai_score += 0.10
        reasons.append(f"Low inter-sentence length delta ({avg_delta:.1f}w)")

    has_expressive_punct = bool(re.search(r"[?!()]|--|\.\.\.", cleaned))
    if len(sent_lengths) >= 3 and not has_expressive_punct and min_sent_len >= 10 and c_cnt == 0:
        ai_score += 0.10
        reasons.append("Frictionless declarative rhythm (no contractions, questions, or asides)")

    if em_dash_count >= 2 or (em_dash_count >= 1 and word_count <= 140):
        ai_score += 0.12
        reasons.append(f"Unicode em-dash typography ({em_dash_count})")
    elif en_dash_count >= 1:
        ai_score += 0.12
        reasons.append(f"Unicode en-dash range formatting ({en_dash_count})")

    if has_not_only:
        ai_score += 0.10
        reasons.append("Correlative 'Not only ... but also ...' template")
    if has_contraction_ops and contraction_rate == 0.0 and word_count >= 35:
        ai_score += 0.08
        reasons.append("Zero contractions despite formal expansion opportunities")
    if mattr < 0.62 and word_count >= 45:
        ai_score += 0.08
        reasons.append(f"Low moving-average lexical diversity (MATTR={mattr:.2f})")

    # Human & empirical academic counter-signals (protect natural writing from false positives)
    casual_hits = len(_CASUAL_HUMAN_MARKERS.findall(cleaned))
    citation_hits = len(_CITATION_OR_DATA_RE.findall(cleaned))
    has_heavy_coursework_ai = taxonomy_count >= 2 or imperative_step_count >= 3 or makes_look_count >= 2

    if not has_heavy_coursework_ai:
        if c_cnt >= 3 or (c_cnt >= 1 and casual_hits >= 1 and len(triad_hits) <= 1):
            ai_score -= 0.14
        if casual_hits >= 2:
            ai_score -= min(0.22, casual_hits * 0.07)
        if len(sent_lengths) >= 2 and min_sent_len <= 7 and cv >= 0.40:
            ai_score -= 0.14

    if citation_hits >= 2 and len(vocab_hits) == 0 and len(framing_hits) <= 2 and taxonomy_count < 2:
        ai_score -= min(0.26, citation_hits * 0.08)
        reasons.append(f"Empirical data / explicit citation markers present ({citation_hits})")

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
        humanizer_score += 0.28
        reasons.append("Uniform sentence length masked with forced contractions")

    if not reasons:
        reasons.append("Natural lexical diversity and human sentence cadence")
    elif ai_score < 0.38 and not any("Natural" in r or "Empirical" in r for r in reasons):
        reasons.insert(0, "Human sentence variation and voice outweigh minor stylistic cues")

    return {
        "stylometric_ai_score": round(min(1.0, max(0.0, ai_score)), 4),
        "stylometric_humanizer_score": round(min(1.0, max(0.0, humanizer_score)), 4),
        "burstiness_cv": round(cv, 3),
        "contraction_rate": round(contraction_rate, 3),
        "mattr": round(mattr, 3),
        "ai_lexicon_hits": len(vocab_hits) + len(transition_hits) + len(framing_hits) + taxonomy_count + len(triad_hits),
        "unicode_anomaly_count": uni["unicode_anomaly_count"],
        "homoglyph_count": uni["homoglyph_count"],
        "has_personal_counter": (casual_hits >= 2 and not has_heavy_coursework_ai) or (citation_hits >= 2 and taxonomy_count < 2),
        "reasons": reasons,
    }
