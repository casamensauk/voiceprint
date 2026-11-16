"""Built-in AI-tell scanner, statistical burstiness scorer, and Voice DNA alignment analyzer."""

from __future__ import annotations

import re
import statistics
from typing import Any

# Categorized AI vocabulary & cliché dictionary (100+ items)
HIGH_RISK_AI_WORDS: dict[str, str] = {
    "delve": "dig into",
    "delves": "looks at",
    "delving": "digging into",
    "tapestry": "mix",
    "testament": "proof",
    "underscore": "show",
    "underscores": "shows",
    "underscoring": "showing",
    "multifaceted": "layered",
    "foster": "build",
    "fosters": "builds",
    "fostering": "building",
    "realm": "area",
    "beacon": "example",
    "pivotal": "key",
    "crucial": "important",
    "intricate": "detailed",
    "intricately": "closely",
    "meticulous": "careful",
    "meticulously": "carefully",
    "commendable": "solid",
    "vibrant": "lively",
    "landscape": "space",
    "embark": "start",
    "embarking": "starting",
    "ever-evolving": "changing",
    "game-changer": "big shift",
    "game changer": "big shift",
    "unleash": "open up",
    "unleashing": "opening up",
    "elevate": "improve",
    "elevating": "improving",
    "harness": "use",
    "harnessing": "using",
    "leverage": "use",
    "leveraging": "using",
    "utilize": "use",
    "utilizes": "uses",
    "utilizing": "using",
    "synergy": "teamwork",
    "paradigm": "model",
    "seamless": "smooth",
    "seamlessly": "smoothly",
    "robust": "solid",
    "comprehensive": "full",
    "transformative": "major",
    "groundbreaking": "new",
    "indelible": "lasting",
    "catalyze": "spark",
    "nuanced": "subtle",
    "overarching": "main",
    "quintessential": "classic",
    "symphony": "blend",
    "mosaic": "mix",
    "cornerstone": "base",
    "linchpin": "key part",
    "spearhead": "lead",
    "spearheading": "leading",
    "bolster": "back up",
    "bolstering": "backing up",
    "paramount": "top priority",
    "myriad": "tons of",
    "plethora": "bunch of",
    "reimagine": "rethink",
    "resonate": "click",
    "resonates": "clicks",
    "ecosystem": "setup",
    "holistic": "full",
    "streamline": "simplify",
    "streamlining": "simplifying",
    "empower": "help",
    "empowering": "helping",
    "innovative": "fresh",
    "cutting-edge": "modern",
    "state-of-the-art": "latest",
    "unprecedented": "rare",
    "profound": "deep",
    "profoundly": "deeply",
    "indispensable": "needed",
    "ubiquitous": "everywhere",
    "dynamic": "active",
    "bespoke": "custom",
    "curate": "pick",
    "curated": "picked",
    "navigate": "handle",
    "navigating": "handling",
    "illuminate": "clarify",
    "illuminates": "clarifies",
    "exemplify": "show",
    "exemplifies": "shows",
    "interplay": "link",
    "juxtaposition": "contrast",
    "endeavor": "effort",
    "endeavors": "efforts",
    "facilitate": "help",
    "facilitates": "helps",
    "augment": "boost",
    "culminate": "end",
    "culminating": "ending",
    "epitomize": "sum up",
    "burgeoning": "growing",
    "resilient": "tough",
    "amplify": "boost",
    "optimize": "tune",
    "optimizes": "tunes",
    "optimizing": "tuning",
    "actionable": "practical",
    "scalable": "flexible",
    "synergistic": "combined",
    "demystify": "explain",
    "unlock": "open",
    "unlocking": "opening",
}

ROBOTIC_TRANSITIONS: dict[str, str] = {
    "it is important to note that": "",
    "it's important to note that": "",
    "it is worth noting that": "",
    "it's worth noting that": "",
    "it is worth noting": "",
    "it's worth mentioning that": "",
    "in today's fast-paced digital age": "these days",
    "in today's fast-paced world": "nowadays",
    "in today's fast-paced": "today's",
    "in today's digital age": "these days",
    "in the realm of": "in",
    "at the end of the day": "honestly",
    "in conclusion": "so yeah",
    "to summarize": "short version",
    "in summary": "basically",
    "first and foremost": "first",
    "having said that": "that said",
    "with that being said": "still",
    "by and large": "mostly",
    "needless to say": "obviously",
    "moreover": "plus",
    "furthermore": "also",
    "consequently": "so",
    "nevertheless": "still",
    "nonetheless": "still",
    "additionally": "also",
    "subsequently": "then",
    "dive deep into": "dig into",
    "let's explore": "let's look at",
    "let us explore": "let's look at",
    "whether you're a": "if you're a",
    "serves as a testament to": "shows",
    "stands as a testament to": "shows",
    "stands as a testament": "shows",
    "serves as a reminder": "reminds us",
    "pave the way for": "open the door to",
    "on the other hand": "then again",
    "in light of": "given",
    "with respect to": "about",
    "as a result": "so",
}

CONTRACTION_PAIRS: list[tuple[str, str]] = [
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("cannot", "can't"),
    ("can not", "can't"),
    ("will not", "won't"),
    ("would not", "wouldn't"),
    ("could not", "couldn't"),
    ("should not", "shouldn't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("was not", "wasn't"),
    ("were not", "weren't"),
    ("have not", "haven't"),
    ("has not", "hasn't"),
    ("had not", "hadn't"),
    ("i am", "i'm"),
    ("i have", "i've"),
    ("i will", "i'll"),
    ("i would", "i'd"),
    ("you are", "you're"),
    ("you have", "you've"),
    ("you will", "you'll"),
    ("we are", "we're"),
    ("we have", "we've"),
    ("we will", "we'll"),
    ("they are", "they're"),
    ("they have", "they've"),
    ("it is", "it's"),
    ("that is", "that's"),
    ("there is", "there's"),
    ("here is", "here's"),
    ("what is", "what's"),
    ("let us", "let's"),
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_COLON_BULLET_RE = re.compile(r"(?m)^(?:\s*[-*•]\s+|\s*\d+\.\s+)\*\*[^*\n]{2,50}\*\*\s*:")
_NOT_ONLY_RE = re.compile(r"\bnot only\b.{2,120}\bbut also\b", re.IGNORECASE | re.DOTALL)


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentence-like segments."""
    stripped = text.strip()
    if not stripped:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(stripped) if p.strip()]
    return parts


def compute_contraction_stats(text: str) -> tuple[int, int, float]:
    """Return (contracted_count, expanded_count, contraction_rate)."""
    lower = text.lower()
    contracted = 0
    expanded = 0
    for exp_form, con_form in CONTRACTION_PAIRS:
        expanded += len(re.findall(r"\b" + re.escape(exp_form) + r"\b", lower))
        contracted += len(re.findall(r"\b" + re.escape(con_form) + r"\b", lower))
    total = contracted + expanded
    rate = (contracted / total) if total > 0 else 0.5
    return contracted, expanded, rate


def compute_burstiness(text: str) -> dict[str, Any]:
    """Measure sentence-length variation and map it to a 0-100 burstiness score."""
    sentences = split_sentences(text)
    lengths = [len(s.split()) for s in sentences if s.split()]
    if not lengths:
        return {
            "sentence_count": 0,
            "sentence_lengths": [],
            "mean_sentence_len": 0.0,
            "stdev_sentence_len": 0.0,
            "coeff_of_variation": 0.0,
            "burstiness_score": 50,
        }

    mean_len = statistics.mean(lengths)
    if len(lengths) == 1:
        # Single-sentence rhythm check: clauses/naturalness vs stiff 22+ word monolith
        w = lengths[0]
        punct_breaks = len(re.findall(r"[,;!?]|--|\.\.\.", text))
        if w <= 16:
            cv = 0.45
            score = 84
        elif punct_breaks >= 2:
            cv = 0.35
            score = 72
        else:
            cv = 0.18
            score = 45
        return {
            "sentence_count": 1,
            "sentence_lengths": lengths,
            "mean_sentence_len": round(mean_len, 2),
            "stdev_sentence_len": 0.0,
            "coeff_of_variation": round(cv, 3),
            "burstiness_score": score,
        }

    stdev_len = statistics.pstdev(lengths)
    cv = (stdev_len / mean_len) if mean_len > 0 else 0.0
    # Human writing typically sits at CV >= 0.45; raw AI often sits around 0.12 - 0.25
    # Also reward having at least one short punchy sentence (<= 8 words) when multi-sentence
    raw_score = 18.0 + (cv / 0.55) * 72.0
    if min(lengths) <= 7 and max(lengths) >= 14:
        raw_score += 10.0
    if len(lengths) >= 3 and cv < 0.22:
        raw_score -= 12.0
    score = int(max(5, min(100, round(raw_score))))
    return {
        "sentence_count": len(lengths),
        "sentence_lengths": lengths,
        "mean_sentence_len": round(mean_len, 2),
        "stdev_sentence_len": round(stdev_len, 2),
        "coeff_of_variation": round(cv, 3),
        "burstiness_score": score,
    }


def scan_ai_tells(text: str) -> dict[str, Any]:
    """Find AI vocabulary, robotic transitions, and structural/punctuation tells."""
    flags: list[dict[str, str]] = []
    seen_phrases: set[str] = set()
    lower = text.lower()
    word_count = max(1, len(text.split()))

    # 1. Robotic transitions (check longer phrases first)
    for phrase, suggestion in sorted(
        ROBOTIC_TRANSITIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if matches:
            key = phrase.lower()
            if key not in seen_phrases:
                seen_phrases.add(key)
                flags.append(
                    {
                        "phrase": matches[0].group(0),
                        "category": "robotic_transition",
                        "count": str(len(matches)),
                        "suggestion": suggestion or "remove / rephrase",
                    }
                )

    # 2. High-risk AI vocabulary
    for word, suggestion in HIGH_RISK_AI_WORDS.items():
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if matches:
            key = word.lower()
            if key not in seen_phrases:
                seen_phrases.add(key)
                flags.append(
                    {
                        "phrase": matches[0].group(0),
                        "category": "ai_vocab",
                        "count": str(len(matches)),
                        "suggestion": suggestion,
                    }
                )

    # 3. Structural & punctuation tells
    em_dash_count = text.count("—")
    em_dash_per_100 = (em_dash_count / word_count) * 100.0
    if em_dash_count > 0:
        flags.append(
            {
                "phrase": "— (em-dash)",
                "category": "punctuation_tell",
                "count": str(em_dash_count),
                "suggestion": "use comma, period, or --",
            }
        )

    not_only_matches = _NOT_ONLY_RE.findall(text)
    if not_only_matches:
        flags.append(
            {
                "phrase": "Not only ... but also ...",
                "category": "structural_tell",
                "count": str(len(not_only_matches)),
                "suggestion": "split into direct statement",
            }
        )

    colon_bullets = _COLON_BULLET_RE.findall(text)
    if colon_bullets:
        flags.append(
            {
                "phrase": "- **Label**: summary bullets",
                "category": "structural_tell",
                "count": str(len(colon_bullets)),
                "suggestion": "write natural sentences or plain bullets",
            }
        )

    return {
        "flags": flags,
        "em_dash_count": em_dash_count,
        "em_dash_per_100": round(em_dash_per_100, 2),
        "not_only_count": len(not_only_matches),
        "colon_bullet_count": len(colon_bullets),
        "vocab_flag_count": sum(1 for f in flags if f["category"] == "ai_vocab"),
        "transition_flag_count": sum(1 for f in flags if f["category"] == "robotic_transition"),
        "total_tell_instances": sum(int(f["count"]) for f in flags),
        "lower_preview": lower[:60],
    }


def compute_style_alignment(text: str, target_dna: dict[str, Any] | None) -> int:
    """Score 0-100 alignment between text surface traits and target Voice DNA."""
    if not target_dna or target_dna.get("sample_count", 0) == 0:
        return 75

    score = 100.0
    sentences = split_sentences(text)
    mean_len = (
        statistics.mean([len(s.split()) for s in sentences])
        if sentences
        else float(len(text.split()))
    )
    target_len = float(target_dna.get("avg_sentence_length", 12.0))
    len_diff = abs(mean_len - target_len)
    score -= min(28.0, len_diff * 1.6)

    # Contraction alignment
    contracted, expanded, c_rate = compute_contraction_stats(text)
    if contracted + expanded > 0:
        target_c = float(target_dna.get("contraction_rate", 0.6))
        score -= min(22.0, abs(c_rate - target_c) * 32.0)

    # Lowercase start alignment
    if sentences:
        lower_starts = sum(1 for s in sentences if s and s[0].islower()) / len(sentences)
        target_lower = float(target_dna.get("lowercase_start_rate", 0.0))
        score -= min(22.0, abs(lower_starts - target_lower) * 28.0)

    # Em-dash penalty if user DNA never uses em-dashes
    if text.count("—") > 0 and float(target_dna.get("punctuation_profile", {}).get("em_dash", 0.0)) < 0.2:
        score -= 15.0

    # Trailing period habit for short texts
    target_no_period = float(target_dna.get("no_trailing_period_rate", 0.0))
    if len(sentences) <= 2 and target_no_period > 0.55 and text.rstrip().endswith("."):
        score -= 10.0

    return int(max(10, min(100, round(score))))


def analyze_text(text: str, target_dna: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run full AI-tell scan, statistical burstiness, and Voice DNA alignment."""
    cleaned = text.strip()
    if not cleaned:
        return {
            "word_count": 0,
            "char_count": 0,
            "human_score": 0,
            "ai_risk_score": 0,
            "burstiness_score": 0,
            "style_match_score": 0,
            "sentence_lengths": [],
            "flags": [],
            "metrics": {
                "mean_sentence_len": 0.0,
                "stdev_sentence_len": 0.0,
                "coeff_of_variation": 0.0,
                "contraction_rate": 0.0,
                "em_dash_count": 0,
                "vocab_flag_count": 0,
                "transition_flag_count": 0,
            },
        }

    word_count = len(cleaned.split())
    burst = compute_burstiness(cleaned)
    tells = scan_ai_tells(cleaned)
    _, _, contraction_rate = compute_contraction_stats(cleaned)
    style_match = compute_style_alignment(cleaned, target_dna)

    # 1. Absence of AI tells (0 - 100, 45% weight)
    tell_penalty = (
        tells["vocab_flag_count"] * 17.0
        + tells["transition_flag_count"] * 15.0
        + (16.0 if tells["em_dash_per_100"] > 1.0 else (8.0 if tells["em_dash_count"] > 0 else 0.0))
        + tells["not_only_count"] * 12.0
        + tells["colon_bullet_count"] * 12.0
    )
    ai_absence_score = max(0.0, 100.0 - tell_penalty)

    # 2. Burstiness score (0 - 100, 30% weight)
    burstiness_score = float(burst["burstiness_score"])

    # 3. Structural naturalness (0 - 100, 15% weight)
    structural_score = 100.0
    if burst["sentence_count"] >= 3 and burst["coeff_of_variation"] < 0.25:
        structural_score -= 45.0
    if tells["em_dash_per_100"] > 1.0:
        structural_score -= 25.0
    if tells["not_only_count"] > 0:
        structural_score -= 25.0
    if tells["colon_bullet_count"] > 0:
        structural_score -= 25.0
    if contraction_rate == 0.0 and word_count >= 25:
        structural_score -= 15.0
    structural_score = max(0.0, structural_score)

    # Weighted blend
    human_float = (
        0.45 * ai_absence_score
        + 0.30 * burstiness_score
        + 0.15 * structural_score
        + 0.10 * float(style_match)
    )
    # Strong cap if multiple AI vocabulary tells appear simultaneously
    if tells["vocab_flag_count"] + tells["transition_flag_count"] >= 4:
        human_float = min(human_float, 38.0)

    human_score = int(max(1, min(100, round(human_float))))
    ai_risk_score = 100 - human_score

    return {
        "word_count": word_count,
        "char_count": len(cleaned),
        "human_score": human_score,
        "ai_risk_score": ai_risk_score,
        "burstiness_score": int(burstiness_score),
        "style_match_score": style_match,
        "sentence_lengths": burst["sentence_lengths"],
        "flags": tells["flags"],
        "metrics": {
            "mean_sentence_len": burst["mean_sentence_len"],
            "stdev_sentence_len": burst["stdev_sentence_len"],
            "coeff_of_variation": burst["coeff_of_variation"],
            "contraction_rate": round(contraction_rate, 2),
            "em_dash_count": tells["em_dash_count"],
            "vocab_flag_count": tells["vocab_flag_count"],
            "transition_flag_count": tells["transition_flag_count"],
        },
    }
