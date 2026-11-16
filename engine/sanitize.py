"""Hidden-character clean-up: strips invisible/spoofing Unicode that AI detectors flag."""

from __future__ import annotations

import re
import unicodedata

from .ai_detector.stylometrics import _HOMOGLYPH_CHARS

_ZERO_WIDTH = set("​‌‍⁠﻿᠎­")
_EXOTIC_SPACES = {" ", " ", " ", "　"} | {chr(c) for c in range(0x2000, 0x200B)}
_BIDI = set("‎‏؜") | {chr(c) for c in range(0x202A, 0x202F)} | {chr(c) for c in range(0x2066, 0x206A)}

# Cyrillic/Greek look-alikes -> Latin equivalents (covers every char in the detector's list).
HOMOGLYPH_TO_LATIN: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X", "І": "I",
    "К": "K", "М": "M", "Н": "H", "Т": "T", "В": "B",
    "α": "a", "β": "b", "γ": "y", "ε": "e", "ζ": "Z", "η": "n", "ι": "i", "κ": "k",
    "μ": "u", "ν": "v", "ο": "o", "ρ": "p", "σ": "o", "τ": "t", "υ": "u", "χ": "x", "ω": "w",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X",
}

_TOKEN_RE = re.compile(r"\S+")


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2190 <= cp <= 0x21FF
        or 0x2300 <= cp <= 0x23FF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B00 <= cp <= 0x2BFF
        or cp in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D)
    )


def _fix_homoglyphs(text: str, counts: dict[str, int]) -> str:
    def _sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if not any("a" <= c.lower() <= "z" for c in tok):
            return tok
        if not any(c in _HOMOGLYPH_CHARS for c in tok):
            return tok
        out = []
        for c in tok:
            if c in _HOMOGLYPH_CHARS and c in HOMOGLYPH_TO_LATIN:
                counts["homoglyph"] = counts.get("homoglyph", 0) + 1
                out.append(HOMOGLYPH_TO_LATIN[c])
            else:
                out.append(c)
        return "".join(out)

    return _TOKEN_RE.sub(_sub, text)


def strip_hidden_characters(text: str) -> tuple[str, dict]:
    """Return (clean_text, {"removed", "replaced", "by_category"})."""
    removed: dict[str, int] = {}
    replaced: dict[str, int] = {}
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        cp = ord(ch)
        if ch in ("‍", "️") and (
            (i > 0 and (_is_emoji(text[i - 1]) or text[i - 1] in ("‍", "️")))
            or (ch == "‍" and i + 1 < n and _is_emoji(text[i + 1]))
        ):
            out.append(ch)
            continue
        if ch in _ZERO_WIDTH:
            removed["zero_width"] = removed.get("zero_width", 0) + 1
        elif ch in _BIDI:
            removed["bidi_control"] = removed.get("bidi_control", 0) + 1
        elif ch in _EXOTIC_SPACES:
            replaced["exotic_space"] = replaced.get("exotic_space", 0) + 1
            out.append(" ")
        elif 0xE0000 <= cp <= 0xE007F:
            removed["tag_character"] = removed.get("tag_character", 0) + 1
        elif 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
            removed["variation_selector"] = removed.get("variation_selector", 0) + 1
        elif 0xE000 <= cp <= 0xF8FF or 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD:
            removed["private_use"] = removed.get("private_use", 0) + 1
        elif (cp < 0x20 and ch not in "\n\t") or 0x7F <= cp <= 0x9F:
            removed["control_code"] = removed.get("control_code", 0) + 1
        else:
            out.append(ch)

    result = _fix_homoglyphs("".join(out), replaced)
    result = unicodedata.normalize("NFC", result)
    by_category = dict(removed)
    for k, v in replaced.items():
        by_category[k] = by_category.get(k, 0) + v
    return result, {
        "removed": sum(removed.values()),
        "replaced": sum(replaced.values()),
        "by_category": by_category,
    }
