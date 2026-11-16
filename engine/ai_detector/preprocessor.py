"""Text normalization, LMS/bibliography exclusion, collapsed-DOM recovery, and exact-offset windowing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from engine.pangram_client import extract_docx_text

_REF_HEADER_RE = re.compile(
    r"(?im)^(?:#+\s*)?(?:references|bibliography|works\s+cited|literature\s+cited|sources)\s*:?\s*$"
)

# LMS assessment lines (question prompts, 'Your Answer:', '[Click to edit](...)')
_LMS_EXCLUDE_LINE_RE = re.compile(
    r"^(?:"
    r"(?:Q\d+|Question\s+\d+)\b[^\n]*|"
    r"Your\s+Answer\s*:|"
    r"\[Click to edit\]\(https?://[^)\n]+\)"
    r")\s*$",
    re.IGNORECASE,
)

# Mixed-case brands/tech names that must NOT be split on internal camelCase capitals
_MIXED_CASE_BRAND_RE = re.compile(
    r"\b(?:TikTok|LinkedIn|YouTube|WhatsApp|Hootsuite|Trustpilot|WordPress|HubSpot|"
    r"PayPal|iPhone|iPad|iOS|macOS|MacBook|JavaScript|TypeScript|OpenAI|ChatGPT|"
    r"PowerPoint|OneDrive|SharePoint|QuickBooks|eBay|McDonald|Facebook|Instagram)\b"
)

_PERIOD_SPLIT_RE = re.compile(r"[.!?]+(?=\s|[A-Z][a-z\"“]|\"|$)|\n+")
_CAMEL_LIST_JOIN_RE = re.compile(r"(?<=[a-z0-9)”\"'])(?=[A-Z][A-Za-z]{2,})")


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
    exclusion_reason: str = ""


def detect_reference_section_start(text: str) -> int | None:
    """Detect the start character offset of a trailing Bibliography / References section."""
    if not text.strip():
        return None
    min_pos = int(len(text) * 0.30)
    for match in _REF_HEADER_RE.finditer(text):
        if match.start() >= min_pos:
            return match.start()
    return None


def split_sentences_recover_collapsed(text: str) -> list[str]:
    """Split text into sentences while recovering missing post-period spaces and collapsed HTML list joins."""
    if not text or not text.strip():
        return []
    spans = _extract_sentence_spans(text, base_offset=0)
    return [s.text for s in spans]


def _extract_sentence_spans(text: str, base_offset: int = 0) -> list[SentenceSpan]:
    """Find sentence spans preserving exact offsets, splitting collapsed DOM joins safely."""
    brand_spans = [(m.start(), m.end()) for m in _MIXED_CASE_BRAND_RE.finditer(text)]

    def _inside_brand(pos: int) -> bool:
        return any(bs < pos < be for bs, be in brand_spans)

    split_points: list[tuple[int, int]] = []
    for m in _PERIOD_SPLIT_RE.finditer(text):
        # Keep punctuation attached to preceding sentence, skip whitespace after
        end_of_punct = m.start()
        while end_of_punct < len(text) and text[end_of_punct] in ".!?":
            end_of_punct += 1
        split_points.append((end_of_punct, m.end()))

    for m in _CAMEL_LIST_JOIN_RE.finditer(text):
        pos = m.start()
        if not _inside_brand(pos):
            split_points.append((pos, pos))

    split_points.sort(key=lambda pair: pair[0])

    # Deduplicate overlapping split points
    merged_splits: list[tuple[int, int]] = []
    for sp_end, sp_next in split_points:
        if merged_splits and sp_end <= merged_splits[-1][1]:
            if sp_next > merged_splits[-1][1]:
                merged_splits[-1] = (merged_splits[-1][0], sp_next)
            continue
        merged_splits.append((sp_end, sp_next))

    raw_ranges: list[tuple[int, int]] = []
    cursor = 0
    for sp_end, sp_next in merged_splits:
        if sp_end > cursor:
            raw_ranges.append((cursor, sp_end))
        cursor = max(cursor, sp_next)
    if cursor < len(text):
        raw_ranges.append((cursor, len(text)))

    spans: list[SentenceSpan] = []
    for r_start, r_end in raw_ranges:
        s_start, s_end = r_start, r_end
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


def _split_body_and_lms_blocks(body_text: str) -> list[tuple[int, int, bool]]:
    """Split body text into contiguous (start, end, is_lms_excluded) regions by line."""
    blocks: list[tuple[int, int, bool]] = []
    pos = 0
    lines = body_text.splitlines(keepends=True)
    for ln in lines:
        ln_len = len(ln)
        stripped = ln.strip()
        if not stripped:
            pos += ln_len
            continue
        is_lms = bool(_LMS_EXCLUDE_LINE_RE.match(stripped))
        if blocks and blocks[-1][2] == is_lms:
            blocks[-1] = (blocks[-1][0], pos + ln_len, is_lms)
        else:
            blocks.append((pos, pos + ln_len, is_lms))
        pos += ln_len
    return blocks


def segment_document(
    text: str,
    target_words: int = 85,
    max_words: int = 135,
) -> list[PreprocessedSegment]:
    """Slice document into sentence-aligned windows with exact [start_index, end_index] offsets."""
    if not text or not text.strip():
        return []

    ref_start = detect_reference_section_start(text)
    body_text = text[:ref_start] if ref_start is not None else text
    ref_text = text[ref_start:] if ref_start is not None else ""

    segments: list[PreprocessedSegment] = []
    blocks = _split_body_and_lms_blocks(body_text)

    for b_start, b_end, is_lms in blocks:
        if is_lms:
            s0, s1 = b_start, b_end
            while s0 < s1 and text[s0].isspace():
                s0 += 1
            while s1 > s0 and text[s1 - 1].isspace():
                s1 -= 1
            if s1 > s0:
                slice_txt = text[s0:s1]
                segments.append(
                    PreprocessedSegment(
                        text=slice_txt,
                        start_index=s0,
                        end_index=s1,
                        word_count=len(slice_txt.split()),
                        sentence_count=1,
                        is_excluded_reference=True,
                        sentences=[],
                        exclusion_reason="Excluded assessment prompt / LMS chrome",
                    )
                )
            continue

        block_slice = text[b_start:b_end]
        block_sents = _extract_sentence_spans(block_slice, base_offset=b_start)
        block_segs: list[PreprocessedSegment] = []
        current: list[SentenceSpan] = []
        current_words = 0

        for sent in block_sents:
            if current and (current_words + sent.word_count > max_words or current_words >= target_words):
                seg_start = current[0].start_index
                seg_end = current[-1].end_index
                block_segs.append(
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
            block_segs.append(
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

        # Merge short trailing window (< 45 words) within the same answer block
        if len(block_segs) >= 2 and block_segs[-1].word_count < 45:
            last_seg = block_segs.pop()
            prev_seg = block_segs[-1]
            merged_start = prev_seg.start_index
            merged_end = last_seg.end_index
            merged_sents = list(prev_seg.sentences) + list(last_seg.sentences)
            block_segs[-1] = PreprocessedSegment(
                text=text[merged_start:merged_end],
                start_index=merged_start,
                end_index=merged_end,
                word_count=prev_seg.word_count + last_seg.word_count,
                sentence_count=len(merged_sents),
                is_excluded_reference=False,
                sentences=merged_sents,
            )

        segments.extend(block_segs)

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
                    exclusion_reason="Excluded reference/bibliography section (protected from false positives)",
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
        cleaned = re.sub(r"\\par\b", "\n", raw)
        cleaned = re.sub(r"\\'[0-9a-fA-F]{2}", " ", cleaned)
        cleaned = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", cleaned)
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    return raw
