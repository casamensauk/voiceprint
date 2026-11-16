"""Text normalization, reference stripping, file extraction, and exact-offset windowing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from engine.pangram_client import extract_docx_text

_REF_HEADER_RE = re.compile(
    r"(?im)^(?:#+\s*)?(?:references|bibliography|works\s+cited|literature\s+cited|sources)\s*:?\s*$"
)
_SENTENCE_SPAN_RE = re.compile(r"[^\n.!?]+(?:[.!?]+(?=\s|$)|\n+|$)")


@dataclass
class SentenceSpan:
    """Single sentence or line span with exact offsets into the source text."""

    text: str
    start_index: int
    end_index: int
    word_count: int


@dataclass
class PreprocessedSegment:
    """Sentence-aligned sliding window preserving exact character offsets."""

    text: str
    start_index: int
    end_index: int
    word_count: int
    sentence_count: int
    is_excluded_reference: bool
    sentences: list[SentenceSpan]


def detect_reference_section_start(text: str) -> int | None:
    """Detect the start character offset of a trailing Bibliography / References section."""
    if not text.strip():
        return None
    min_pos = int(len(text) * 0.30)
    for match in _REF_HEADER_RE.finditer(text):
        if match.start() >= min_pos:
            return match.start()
    return None


def _extract_sentence_spans(text: str, base_offset: int = 0) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    for m in _SENTENCE_SPAN_RE.finditer(text):
        s_start, s_end = m.start(), m.end()
        while s_start < s_end and text[s_start].isspace():
            s_start += 1
        while s_end > s_start and text[s_end - 1].isspace():
            s_end -= 1
        if s_end <= s_start:
            continue
        frag = text[s_start:s_end]
        wc = len(frag.split())
        if wc == 0:
            continue
        spans.append(
            SentenceSpan(
                text=frag,
                start_index=base_offset + s_start,
                end_index=base_offset + s_end,
                word_count=wc,
            )
        )
    return spans


def segment_document(
    text: str,
    target_words: int = 75,
    max_words: int = 120,
) -> list[PreprocessedSegment]:
    """Slice document into sentence-aligned windows with exact [start_index, end_index] offsets."""
    if not text or not text.strip():
        return []

    ref_start = detect_reference_section_start(text)
    body_text = text[:ref_start] if ref_start is not None else text
    ref_text = text[ref_start:] if ref_start is not None else ""

    segments: list[PreprocessedSegment] = []
    body_sents = _extract_sentence_spans(body_text, base_offset=0)

    current: list[SentenceSpan] = []
    current_words = 0

    for sent in body_sents:
        if current and (current_words + sent.word_count > max_words or current_words >= target_words):
            seg_start = current[0].start_index
            seg_end = current[-1].end_index
            segments.append(
                PreprocessedSegment(
                    text=text[seg_start:seg_end],
                    start_index=seg_start,
                    end_index=seg_end,
                    word_count=current_words,
                    sentence_count=len(current),
                    is_excluded_reference=False,
                    sentences=list(current),
                )
            )
            current = []
            current_words = 0

        current.append(sent)
        current_words += sent.word_count

    if current:
        seg_start = current[0].start_index
        seg_end = current[-1].end_index
        segments.append(
            PreprocessedSegment(
                text=text[seg_start:seg_end],
                start_index=seg_start,
                end_index=seg_end,
                word_count=current_words,
                sentence_count=len(current),
                is_excluded_reference=False,
                sentences=list(current),
            )
        )

    if ref_text.strip() and ref_start is not None:
        r_start = ref_start
        r_end = len(text)
        while r_start < r_end and text[r_start].isspace():
            r_start += 1
        while r_end > r_start and text[r_end - 1].isspace():
            r_end -= 1
        if r_end > r_start:
            ref_slice = text[r_start:r_end]
            segments.append(
                PreprocessedSegment(
                    text=ref_slice,
                    start_index=r_start,
                    end_index=r_end,
                    word_count=len(ref_slice.split()),
                    sentence_count=len(_extract_sentence_spans(ref_slice, r_start)),
                    is_excluded_reference=True,
                    sentences=[],
                )
            )

    return segments


def extract_document_text(filename: str, file_bytes: bytes) -> str:
    """Extract readable plain text from .docx, .rtf, .txt, or .md bytes."""
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        return extract_docx_text(file_bytes)
    raw = file_bytes.decode("utf-8", errors="replace")
    if ext == ".rtf":
        # Strip RTF control groups and commands cleanly
        cleaned = re.sub(r"\\par\b", "\n", raw)
        cleaned = re.sub(r"\\'[0-9a-fA-F]{2}", " ", cleaned)
        cleaned = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", cleaned)
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    return raw
