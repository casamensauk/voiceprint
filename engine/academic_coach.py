"""Author-led academic writing aids: source/citation overlap checker & revision coach."""

from __future__ import annotations

import difflib
import re
from typing import Any

from .analyzer import split_sentences

_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")
_QUOTE_RE = re.compile(r'["“]([^"”]{8,600})["”]')
_CITATION_NEARBY_RE = re.compile(
    r"(?:\([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?,?\s*\d{4}[^)]*\)|"
    r"\([A-Z][A-Za-z\-]+\s+\d+(?:-\d+)?\)|"
    r"\[\d+(?:\s*[-,]\s*\d+)*\]|"
    r"\bet\s+al\.)",
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "she", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "which", "while",
    "who", "will", "with", "would", "can", "could", "may", "might", "also", "such",
}

_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+(?:[a-z]+ly\s+)?"
    r"([a-z]{3,}ed|shown|known|made|found|given|taken|seen|done|written|built|held|"
    r"brought|thought|kept|led|sent|spent|understood|drawn|chosen|driven|spoken|broken)\b",
    re.IGNORECASE,
)
_PASSIVE_EXCLUSIONS = {
    "interested", "tired", "concerned", "involved", "based", "located", "related",
    "supposed", "used", "excited", "bored", "worried", "pleased", "satisfied",
}

_WEASEL_CLAIMS: list[str] = [
    "many scholars argue",
    "researchers have shown",
    "studies have shown",
    "studies show",
    "some people say",
    "it is widely believed",
    "experts agree",
    "everyone knows",
    "it has been argued",
    "many people think",
    "research suggests",
]

_VAGUE_FILLERS: list[str] = [
    "due to the fact that",
    "in order to",
    "at this point in time",
    "it is interesting to note",
    "plays an important role",
    "throughout history",
    "in modern society",
    "various different",
    "a myriad of",
    "very",
    "really",
    "extremely",
    "things",
    "stuff",
]


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _content_words(text: str) -> list[str]:
    return [
        m.group(0).lower()
        for m in _WORD_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS and len(m.group(0)) >= 3
    ]


def _sentence_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for s in split_sentences(text):
        idx = text.find(s, pos)
        if idx == -1:
            idx = pos
        end = idx + len(s)
        spans.append((s, idx, end))
        pos = end
    return spans


def _has_citation_near(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 50) : min(len(text), end + 80)]
    return bool(_CITATION_NEARBY_RE.search(window))


def check_source_citations(draft_text: str, source_text: str) -> dict[str, Any]:
    """Compare draft against reference notes/excerpts to flag unquoted or uncited overlap."""
    draft = draft_text or ""
    source = source_text or ""
    if not draft.strip() or not source.strip():
        return {
            "overall_overlap_pct": 0.0,
            "status_label": "Waiting for Draft & Source",
            "quoted_cited_count": 0,
            "matches": [],
            "annotated_segments": [{"text": draft, "type": "normal"}] if draft else [],
        }

    # 1. Identify direct quotes in draft and verify nearby citations
    quote_ranges: list[tuple[int, int]] = []
    properly_quoted_ranges: list[tuple[int, int]] = []
    for qm in _QUOTE_RE.finditer(draft):
        qs, qe = qm.start(), qm.end()
        quote_ranges.append((qs, qe))
        if _has_citation_near(draft, qs, qe):
            properly_quoted_ranges.append((qs, qe))

    def _in_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start >= rs and end <= re_ for rs, re_ in ranges)

    # 2. Detect >= 5 consecutive unquoted verbatim word matches
    draft_tokens = _tokenize_with_spans(draft)
    source_tokens = _tokenize_with_spans(source)
    draft_words = [t[0] for t in draft_tokens]
    source_words = [t[0] for t in source_tokens]

    matcher = difflib.SequenceMatcher(None, draft_words, source_words, autojunk=False)
    verbatim_spans: list[tuple[int, int, str, str, int]] = []
    for block in matcher.get_matching_blocks():
        if block.size < 5:
            continue
        d_start = draft_tokens[block.a][1]
        d_end = draft_tokens[block.a + block.size - 1][2]
        if _in_ranges(d_start, d_end, quote_ranges):
            continue
        s_start = source_tokens[block.b][1]
        s_end = source_tokens[block.b + block.size - 1][2]
        verbatim_spans.append(
            (d_start, d_end, draft[d_start:d_end], source[s_start:s_end], block.size)
        )

    # 3. Detect close paraphrase / patchwriting per non-verbatim sentence
    src_sentences = _sentence_spans(source)
    src_content_sets = [(ss, set(_content_words(ss)), _content_words(ss)) for ss, _, _ in src_sentences]
    paraphrase_spans: list[tuple[int, int, str, str, bool, float]] = []

    for d_sent, ds, de in _sentence_spans(draft):
        if _in_ranges(ds, de, quote_ranges):
            continue
        if any(not (de <= vs or ds >= ve) for vs, ve, *_ in verbatim_spans):
            continue
        d_words_raw = _WORD_RE.findall(d_sent)
        if len(d_words_raw) < 6:
            continue
        d_content_list = _content_words(d_sent)
        d_content = set(d_content_list)
        if len(d_content) < 4:
            continue
        d_trigrams = {
            tuple(d_content_list[i : i + 3]) for i in range(max(0, len(d_content_list) - 2))
        }

        best_ratio = 0.0
        best_src = ""
        for src_sent, s_content, s_content_list in src_content_sets:
            if not s_content:
                continue
            overlap = len(d_content & s_content) / min(len(d_content), len(s_content))
            s_trigrams = {
                tuple(s_content_list[i : i + 3]) for i in range(max(0, len(s_content_list) - 2))
            }
            shared_trigrams = len(d_trigrams & s_trigrams)
            if overlap >= 0.45 or shared_trigrams >= 2:
                if overlap > best_ratio:
                    best_ratio = overlap
                    best_src = src_sent

        if best_src:
            cited = _has_citation_near(draft, ds, de)
            paraphrase_spans.append((ds, de, d_sent, best_src, cited, best_ratio))

    # 4. Build non-overlapping highlight regions and matches list
    regions: list[tuple[int, int, str]] = []
    for qs, qe in properly_quoted_ranges:
        regions.append((qs, qe, "properly_quoted"))
    for vs, ve, *_ in verbatim_spans:
        regions.append((vs, ve, "verbatim_unquoted"))
    for ps, pe, *_ in paraphrase_spans:
        regions.append((ps, pe, "close_paraphrase"))

    # Resolve overlaps giving priority: properly_quoted > verbatim_unquoted > close_paraphrase
    priority = {"properly_quoted": 3, "verbatim_unquoted": 2, "close_paraphrase": 1}
    regions.sort(key=lambda r: (r[0], -priority[r[2]]))
    resolved: list[tuple[int, int, str]] = []
    for start, end, rtype in regions:
        if resolved and start < resolved[-1][1]:
            if priority[rtype] > priority[resolved[-1][2]]:
                resolved[-1] = (start, end, rtype)
            continue
        resolved.append((start, end, rtype))

    annotated_segments: list[dict[str, str]] = []
    cursor = 0
    for start, end, rtype in resolved:
        if start > cursor:
            annotated_segments.append({"text": draft[cursor:start], "type": "normal"})
        annotated_segments.append({"text": draft[start:end], "type": rtype})
        cursor = end
    if cursor < len(draft):
        annotated_segments.append({"text": draft[cursor:], "type": "normal"})

    matches: list[dict[str, Any]] = []
    flagged_words = 0
    for vs, ve, d_ex, s_ex, w_len in verbatim_spans:
        flagged_words += w_len
        matches.append(
            {
                "type": "verbatim_unquoted",
                "severity": "high",
                "word_length": w_len,
                "has_citation_nearby": _has_citation_near(draft, vs, ve),
                "draft_excerpt": d_ex,
                "source_excerpt": s_ex,
                "guidance": "Wrap verbatim phrasing in quotation marks and add a citation, or rewrite the idea completely from scratch.",
            }
        )
    for ps, pe, d_ex, s_ex, cited, ratio in paraphrase_spans:
        w_len = len(_WORD_RE.findall(d_ex))
        if not cited:
            flagged_words += w_len
        matches.append(
            {
                "type": "close_paraphrase",
                "severity": "medium" if cited else "high",
                "word_length": w_len,
                "overlap_ratio": round(ratio, 2),
                "has_citation_nearby": cited,
                "draft_excerpt": d_ex,
                "source_excerpt": s_ex,
                "guidance": (
                    "Cited, but sentence structure still closely mirrors the source; synthesize the idea in your own voice."
                    if cited
                    else "Close paraphrase without an in-text citation — add attribution and restate the concept in your own structure."
                ),
            }
        )

    total_words = max(1, len(_WORD_RE.findall(draft)))
    overlap_pct = round(min(100.0, (flagged_words / total_words) * 100.0), 1)
    if overlap_pct < 8.0:
        status_label = "Clean Attribution"
    elif overlap_pct <= 25.0:
        status_label = "Needs Citation / Review"
    else:
        status_label = "High Unquoted Overlap"

    return {
        "overall_overlap_pct": overlap_pct,
        "status_label": status_label,
        "quoted_cited_count": len(properly_quoted_ranges),
        "matches": matches,
        "annotated_segments": annotated_segments,
    }


def coach_academic_draft(draft_text: str) -> dict[str, Any]:
    """Inspect draft for academic clarity issues without rewriting text for the author."""
    draft = (draft_text or "").strip()
    if not draft:
        return {
            "clarity_score": 0,
            "sentence_count": 0,
            "counts": {
                "passive_voice": 0,
                "unattributed_claim": 0,
                "vague_filler": 0,
                "repetitive_opener": 0,
                "long_sentence": 0,
            },
            "sentence_feedback": [],
            "priority_actions": ["Paste your draft to receive structural and clarity coaching."],
        }

    sentences = split_sentences(draft)
    counts = {
        "passive_voice": 0,
        "unattributed_claim": 0,
        "vague_filler": 0,
        "repetitive_opener": 0,
        "long_sentence": 0,
    }
    sentence_feedback: list[dict[str, Any]] = []
    openers = []
    for sent in sentences:
        words = _WORD_RE.findall(sent)
        openers.append(words[0].lower() if words else "")

    for idx, sent in enumerate(sentences):
        issues: list[dict[str, str]] = []
        lower = sent.lower()
        words = _WORD_RE.findall(sent)

        # 1. Passive voice
        for pm in _PASSIVE_RE.finditer(sent):
            participle = pm.group(1).lower()
            if participle in _PASSIVE_EXCLUSIONS:
                continue
            counts["passive_voice"] += 1
            issues.append(
                {
                    "category": "passive_voice",
                    "phrase": pm.group(0),
                    "tip": "Identify the actor performing this action and consider rewriting in active voice.",
                }
            )
            break

        # 2. Unattributed claims
        for claim in _WEASEL_CLAIMS:
            if re.search(r"\b" + re.escape(claim) + r"\b", lower):
                counts["unattributed_claim"] += 1
                issues.append(
                    {
                        "category": "unattributed_claim",
                        "phrase": claim,
                        "tip": "Name the specific study, author, year, or dataset supporting this claim.",
                    }
                )
                break

        # 3. Vague academic filler
        for filler in _VAGUE_FILLERS:
            if re.search(r"\b" + re.escape(filler) + r"\b", lower):
                counts["vague_filler"] += 1
                issues.append(
                    {
                        "category": "vague_filler",
                        "phrase": filler,
                        "tip": "Cut unnecessary throat-clearing or replace vague modifiers with concrete evidence.",
                    }
                )
                break

        # 4. Repetitive opener (consecutive identical first word)
        opener = openers[idx]
        if opener and idx > 0 and openers[idx - 1] == opener:
            counts["repetitive_opener"] += 1
            issues.append(
                {
                    "category": "repetitive_opener",
                    "phrase": opener,
                    "tip": f"Consecutive sentences start with '{opener}' — vary your opening transition or clause structure.",
                }
            )

        # 5. Long winding sentences (> 35 words)
        if len(words) > 35:
            counts["long_sentence"] += 1
            issues.append(
                {
                    "category": "long_sentence",
                    "phrase": f"{len(words)} words",
                    "tip": "This sentence exceeds 35 words; check whether two distinct arguments can be separated.",
                }
            )

        if issues:
            sentence_feedback.append(
                {
                    "sentence_index": idx + 1,
                    "sentence": sent,
                    "word_count": len(words),
                    "issues": issues,
                }
            )

    total_issues = sum(counts.values())
    penalty = (total_issues / max(1, len(sentences))) * 22.0
    clarity_score = int(max(15, min(100, round(100.0 - penalty))))

    priority_actions: list[str] = []
    if counts["unattributed_claim"]:
        priority_actions.append(
            f"Ground {counts['unattributed_claim']} broad claim(s) with explicit author/year citations."
        )
    if counts["passive_voice"]:
        priority_actions.append(
            f"Clarify agency in {counts['passive_voice']} passive-voice sentence(s) by naming who performed the action."
        )
    if counts["vague_filler"]:
        priority_actions.append(
            f"Tighten {counts['vague_filler']} sentence(s) containing filler or vague intensifiers."
        )
    if counts["repetitive_opener"]:
        priority_actions.append(
            f"Vary sentence openings where '{openers[1] if len(openers) > 1 else 'the same word'}' repeats consecutively."
        )
    if counts["long_sentence"]:
        priority_actions.append(
            f"Split or trim {counts['long_sentence']} sentence(s) over 35 words so each claim stands clearly."
        )
    if not priority_actions:
        priority_actions.append("Clean structure — no passive voice, vague filler, or unattributed claims detected.")

    return {
        "clarity_score": clarity_score,
        "sentence_count": len(sentences),
        "counts": counts,
        "sentence_feedback": sentence_feedback,
        "priority_actions": priority_actions[:5],
    }
