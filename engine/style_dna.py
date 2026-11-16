"""Voice DNA extraction, TF-IDF style sample retrieval, and final-edit diff learning."""

from __future__ import annotations

import difflib
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .analyzer import compute_contraction_stats, split_sentences
from .db import (
    add_edit_pair,
    add_sample,
    upsert_learned_replacement,
)

SLANG_MARKERS = [
    "tbh",
    "imo",
    "imho",
    "ngl",
    "lol",
    "haha",
    "tho",
    "gonna",
    "wanna",
    "kinda",
    "sorta",
    "lemme",
    "def",
    "prob",
    "btw",
    "fyi",
    "cheers",
    "best",
    "thanks",
    "thx",
    "yeah",
    "nah",
    "honestly",
    "basically",
    "literally",
    "folks",
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9'’-]+|[^\w\s]")
_STOP_DIFF_WORDS = {"a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "or", "is", "are"}


def extract_voice_dna(
    samples: list[dict[str, Any]],
    learned_replacements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute quantitative style DNA across all weighted samples in a mode."""
    learned_replacements = learned_replacements or []
    if not samples:
        return {
            "sample_count": 0,
            "total_words": 0,
            "gold_edit_count": 0,
            "avg_sentence_length": 13.0,
            "sentence_length_std": 5.5,
            "contraction_rate": 0.65,
            "lowercase_start_rate": 0.0,
            "lowercase_i_rate": 0.0,
            "no_trailing_period_rate": 0.0,
            "oxford_comma_rate": 0.15,
            "personal_pronoun_rate": 2.0,
            "punctuation_profile": {
                "ellipsis": 0.0,
                "double_dash": 0.0,
                "em_dash": 0.0,
                "exclamation": 0.5,
                "question": 0.8,
                "semicolon": 0.0,
                "parens": 0.3,
                "comma": 4.0,
            },
            "common_openers": [],
            "common_fillers_and_slang": [],
            "learned_rules_count": len(learned_replacements),
            "directives_summary": [
                "No personal samples uploaded yet for this mode — using crisp natural human defaults.",
                "Upload chats or emails in 'Grow With Me' so VoicePrint learns your exact cadence.",
            ],
        }

    total_words = sum(max(1, len(s["content"].split())) for s in samples)
    gold_count = sum(1 for s in samples if float(s.get("weight", 1.0)) > 1.5)

    weighted_sentence_lengths: list[float] = []
    contraction_weighted_sum = 0.0
    contraction_weight_total = 0.0
    lower_starts = 0.0
    total_sentences_w = 0.0
    lower_i_count = 0.0
    upper_i_count = 0.0
    no_period_w = 0.0
    short_sample_w = 0.0

    punct_counts: Counter[str] = Counter()
    opener_counts: Counter[str] = Counter()
    slang_counts: Counter[str] = Counter()
    total_weight_words = 0.0
    oxford_joins = 0.0
    non_oxford_joins = 0.0
    personal_pronouns = 0.0

    for sample in samples:
        text = sample["content"].strip()
        if not text:
            continue
        w = float(sample.get("weight", 1.0))
        words = text.split()
        total_weight_words += len(words) * w

        sents = split_sentences(text)
        for sent in sents:
            sw = len(sent.split())
            if sw > 0:
                repeats = max(1, round(w))
                weighted_sentence_lengths.extend([float(sw)] * repeats)
                total_sentences_w += w
                if sent[0].islower():
                    lower_starts += w

        # Standalone i vs I
        lower_i_count += len(re.findall(r"\bi\b", text)) * w
        upper_i_count += len(re.findall(r"\bI\b", text)) * w

        # Contraction rate
        c_cnt, e_cnt, c_rate = compute_contraction_stats(text)
        if c_cnt + e_cnt > 0:
            contraction_weighted_sum += c_rate * w
            contraction_weight_total += w

        # Trailing period habit on shorter messages
        if len(sents) <= 3:
            short_sample_w += w
            last_char = text.rstrip()[-1:]
            if last_char not in ".!?":
                no_period_w += w

        # Punctuation per 100 words
        punct_counts["ellipsis"] += len(re.findall(r"\.\.\.", text)) * w
        punct_counts["double_dash"] += len(re.findall(r"--", text)) * w
        punct_counts["em_dash"] += text.count("—") * w
        punct_counts["exclamation"] += text.count("!") * w
        punct_counts["question"] += text.count("?") * w
        punct_counts["semicolon"] += text.count(";") * w
        punct_counts["parens"] += text.count("(") * w
        punct_counts["comma"] += text.count(",") * w

        # Opener detection (first 1-2 words)
        first_line = text.splitlines()[0].strip()
        first_words = re.findall(r"[A-Za-z']+", first_line.lower())
        if first_words:
            opener_counts[first_words[0]] += w
            if len(first_words) >= 2 and first_words[0] in {"quick", "no", "just", "hot", "short", "hey", "hi"}:
                opener_counts[f"{first_words[0]} {first_words[1]}"] += w

        # Slang / signature markers & list/pronoun voice habits
        lower_text = text.lower()
        for marker in SLANG_MARKERS:
            found = len(re.findall(r"\b" + re.escape(marker) + r"\b", lower_text))
            if found:
                slang_counts[marker] += found * w

        ox_cnt = len(re.findall(r",\s+(?:and|or)\s+", lower_text))
        total_conj = len(re.findall(r"\b(?:and|or)\b", lower_text))
        oxford_joins += ox_cnt * w
        non_oxford_joins += max(0, total_conj - ox_cnt) * w
        personal_pronouns += len(re.findall(r"\b(?:i|i'm|i've|i'd|i'll|me|my|we|we're|our|you|you're|you'll|your)\b", lower_text)) * w

    avg_sent_len = statistics.mean(weighted_sentence_lengths) if weighted_sentence_lengths else 12.0
    std_sent_len = (
        statistics.pstdev(weighted_sentence_lengths)
        if len(weighted_sentence_lengths) >= 2
        else 4.0
    )
    contraction_rate = (
        contraction_weighted_sum / contraction_weight_total
        if contraction_weight_total > 0
        else 0.65
    )
    lowercase_start_rate = (lower_starts / total_sentences_w) if total_sentences_w > 0 else 0.0
    total_i = lower_i_count + upper_i_count
    lowercase_i_rate = (lower_i_count / total_i) if total_i > 0 else 0.0
    no_trailing_period_rate = (no_period_w / short_sample_w) if short_sample_w > 0 else 0.0

    norm_denom = max(1.0, total_weight_words) / 100.0
    oxford_total = oxford_joins + non_oxford_joins
    oxford_comma_rate = (oxford_joins / oxford_total) if oxford_total > 0 else 0.15
    personal_pronoun_rate = personal_pronouns / norm_denom
    punctuation_profile = {
        k: round(punct_counts[k] / norm_denom, 2)
        for k in ("ellipsis", "double_dash", "em_dash", "exclamation", "question", "semicolon", "parens", "comma")
    }

    common_openers = [k for k, _ in opener_counts.most_common(6)]
    common_fillers = [k for k, _ in slang_counts.most_common(8)]

    directives: list[str] = []
    directives.append(
        f"Target sentence length around {avg_sent_len:.1f} words with +/- {std_sent_len:.1f} word variation (mix short punches with medium thoughts)."
    )
    if contraction_rate >= 0.6:
        directives.append(f"Use contractions heavily ({int(contraction_rate * 100)}% of opportunities: don't, i'm/I'm, it's, won't).")
    elif contraction_rate <= 0.2:
        directives.append("Keep contractions minimal; prefer full clean phrasing.")
    else:
        directives.append("Use natural everyday contractions when speaking smoothly.")

    if oxford_comma_rate < 0.35:
        directives.append("Avoid stiff Oxford-comma triads ('a, b, and c'); connect thoughts naturally without list templates.")
    if personal_pronoun_rate >= 2.0:
        directives.append("Write directly to the reader ('you'/'we'/'I') instead of third-person textbook phrasing like 'allows businesses to'.")

    if lowercase_start_rate >= 0.45:
        directives.append("Prefer lowercase sentence starts for a relaxed, natural cadence.")
    if lowercase_i_rate >= 0.5:
        directives.append("Use lowercase 'i' when referring to yourself casually.")
    if no_trailing_period_rate >= 0.45:
        directives.append("Skip the final trailing period on short standalone messages.")
    if punctuation_profile["double_dash"] > 0.3 and punctuation_profile["em_dash"] < 0.2:
        directives.append("Use double-hyphens (--) for mid-thought asides instead of Unicode em-dashes (—).")
    elif punctuation_profile["em_dash"] < 0.2:
        directives.append("Never use em-dashes (—); break thoughts with commas, parentheses, or short periods.")
    if common_fillers:
        directives.append(f"Personal markers present in your writing: {', '.join(common_fillers[:6])}.")
    if learned_replacements:
        top_rules = [f"'{r['avoid_text']}' -> '{r['prefer_text']}'" for r in learned_replacements[:4]]
        directives.append(f"Learned word swaps from your edits: {'; '.join(top_rules)}.")

    return {
        "sample_count": len(samples),
        "total_words": total_words,
        "gold_edit_count": gold_count,
        "avg_sentence_length": round(avg_sent_len, 1),
        "sentence_length_std": round(std_sent_len, 1),
        "contraction_rate": round(contraction_rate, 2),
        "lowercase_start_rate": round(lowercase_start_rate, 2),
        "lowercase_i_rate": round(lowercase_i_rate, 2),
        "no_trailing_period_rate": round(no_trailing_period_rate, 2),
        "oxford_comma_rate": round(oxford_comma_rate, 2),
        "personal_pronoun_rate": round(personal_pronoun_rate, 2),
        "punctuation_profile": punctuation_profile,
        "common_openers": common_openers,
        "common_fillers_and_slang": common_fillers,
        "learned_rules_count": len(learned_replacements),
        "directives_summary": directives,
    }


def _ngrams_and_words(text: str) -> Counter[str]:
    lower = text.lower()
    words = re.findall(r"[a-z0-9']+", lower)
    feats: Counter[str] = Counter(f"w:{w}" for w in words)
    padded = f" {lower} "
    for i in range(max(0, len(padded) - 2)):
        feats[f"c:{padded[i:i+3]}"] += 1
    return feats


def retrieve_similar_samples(
    query_text: str,
    samples: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve the top-k most stylistically relevant samples via TF-IDF cosine + gold weighting."""
    if not samples:
        return []
    if len(samples) <= top_k:
        return sorted(samples, key=lambda s: float(s.get("weight", 1.0)), reverse=True)

    docs = [_ngrams_and_words(s["content"]) for s in samples]
    df: Counter[str] = Counter()
    for doc in docs:
        for term in doc.keys():
            df[term] += 1

    n_docs = len(docs)
    q_feats = _ngrams_and_words(query_text)
    q_vec: dict[str, float] = {
        t: cnt * (math.log((n_docs + 1) / (df[t] + 1)) + 1.0)
        for t, cnt in q_feats.items()
    }
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
    q_words = max(1, len(query_text.split()))

    scored: list[tuple[float, dict[str, Any]]] = []
    for sample, doc in zip(samples, docs):
        d_vec = {
            t: cnt * (math.log((n_docs + 1) / (df[t] + 1)) + 1.0)
            for t, cnt in doc.items()
        }
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec[t] * val for t, val in d_vec.items() if t in q_vec)
        cos = dot / (q_norm * d_norm)

        # Length proximity & gold-edit boost
        s_words = max(1, len(sample["content"].split()))
        len_sim = min(q_words, s_words) / max(q_words, s_words)
        weight_boost = 1.0 + (float(sample.get("weight", 1.0)) - 1.0) * 0.25
        score = (0.65 * cos + 0.35 * len_sim) * weight_boost
        scored.append((score, sample))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def learn_from_user_edit(
    context_mode: str,
    original_input: str,
    ai_draft: str,
    user_final_edit: str,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Diff the draft vs user's final edit to extract concrete word/style habits."""
    draft_clean = ai_draft.strip()
    final_clean = user_final_edit.strip()
    if not final_clean:
        return {"learned_habits": [], "rules_added": 0, "changed": False}

    learned_habits: list[str] = []
    rules_added = 0

    if draft_clean == final_clean:
        # Even without edits, reinforcing an approved rewrite slightly builds confidence
        add_sample(
            final_clean,
            context_mode=context_mode,
            source_type="learned_edit",
            weight=1.5,
            db_path=db_path,
        )
        return {
            "learned_habits": ["Confirmed draft matched your voice as-is (saved at 1.5x weight)."],
            "rules_added": 0,
            "changed": False,
        }

    draft_tokens = _TOKEN_RE.findall(draft_clean)
    final_tokens = _TOKEN_RE.findall(final_clean)
    matcher = difflib.SequenceMatcher(
        None,
        [t.lower() for t in draft_tokens],
        [t.lower() for t in final_tokens],
    )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            old_slice = draft_tokens[i1:i2]
            new_slice = final_tokens[j1:j2]
            if 1 <= len(old_slice) <= 4 and 1 <= len(new_slice) <= 4:
                old_phrase = " ".join(old_slice).strip(" ,.;:!?\"'")
                new_phrase = " ".join(new_slice).strip(" ,.;:!?\"'")
                if (
                    old_phrase
                    and new_phrase
                    and old_phrase.lower() != new_phrase.lower()
                    and old_phrase.lower() not in _STOP_DIFF_WORDS
                    and any(ch.isalpha() for ch in old_phrase)
                    and any(ch.isalpha() for ch in new_phrase)
                ):
                    upsert_learned_replacement(
                        context_mode=context_mode,
                        avoid_text=old_phrase,
                        prefer_text=new_phrase,
                        source="auto_diff",
                        db_path=db_path,
                    )
                    rules_added += 1
                    learned_habits.append(f"Replaced '{old_phrase.lower()}' -> '{new_phrase}'")
        elif tag == "delete":
            deleted = [t for t in draft_tokens[i1:i2] if any(c.isalpha() for c in t)]
            if 1 <= len(deleted) <= 3:
                phrase = " ".join(deleted).lower()
                if phrase not in _STOP_DIFF_WORDS and len(phrase) >= 4:
                    upsert_learned_replacement(
                        context_mode=context_mode,
                        avoid_text=phrase,
                        prefer_text="",
                        source="auto_diff_cut",
                        db_path=db_path,
                    )
                    rules_added += 1
                    learned_habits.append(f"Trimmed filler '{phrase}'")

    # Surface style diff observations
    if draft_clean.endswith(".") and not final_clean.endswith((".", "!", "?")):
        learned_habits.append("Dropped trailing period")
    if "—" in draft_clean and "—" not in final_clean:
        learned_habits.append("Removed em-dash (—)")

    draft_words = len(draft_clean.split())
    final_words = len(final_clean.split())
    if abs(draft_words - final_words) >= 3:
        direction = "Shortened" if final_words < draft_words else "Expanded"
        learned_habits.append(f"{direction} length from {draft_words} to {final_words} words")

    draft_lower = sum(1 for s in split_sentences(draft_clean) if s and s[0].islower())
    final_lower = sum(1 for s in split_sentences(final_clean) if s and s[0].islower())
    if final_lower > draft_lower:
        learned_habits.append("Shifted sentence starts toward lowercase")

    if not learned_habits:
        learned_habits.append("Saved phrasing tweak as a 2.5x gold voice sample")

    add_edit_pair(
        context_mode=context_mode,
        original_input=original_input,
        ai_draft=draft_clean,
        user_final_edit=final_clean,
        extracted_rules=learned_habits,
        db_path=db_path,
    )
    add_sample(
        final_clean,
        context_mode=context_mode,
        source_type="learned_edit",
        weight=2.5,
        db_path=db_path,
    )

    return {
        "learned_habits": learned_habits,
        "rules_added": rules_added,
        "changed": True,
    }
