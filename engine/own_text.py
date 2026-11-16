"""Pure functions that reduce an email/chat body to only what the user wrote, and mask private details."""

from __future__ import annotations

import re

_ORIGINAL_RE = re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.I)
_FORWARDED_RE = re.compile(r"^\s*-{3,}\s*Forwarded message\s*-{3,}\s*$", re.I)
_BEGIN_FWD_RE = re.compile(r"^\s*Begin forwarded message:?\s*$", re.I)
_OUTLOOK_FROM_RE = re.compile(r"^\s*[*_]*From:[*_]*\s+\S")
_OUTLOOK_NEXT_RE = re.compile(r"^\s*[*_]*(Sent|To|Subject|Date|Cc):", re.I)
_ON_START_RE = re.compile(r"^\s*On\s+\S")
_ON_WROTE_RE = re.compile(r"^On\s.{3,400}\bwrote:$")

_DEVICE_RES = (
    re.compile(r"^\s*sent from my [\w ]{1,40}\.?\s*$", re.I),
    re.compile(r"^\s*sent from (mail|outlook|yahoo mail|samsung|gmail|proton ?mail)\b.{0,60}$", re.I),
    re.compile(r"^\s*sent (via|using|with) [\w .]{1,40}\.?\s*$", re.I),
    re.compile(r"^\s*get outlook for (ios|android)\s*$", re.I),
)

_NAME_TOKEN = r"[A-Z][\w'’.-]*"
_GREETING_RE = re.compile(
    rf"^(?i:hi|hello|hey|dear|morning|good (?:morning|afternoon|evening))"
    rf"(?:\s+(?i:there|all|everyone|team)|\s+{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}})?\s*[,!:]?\s*$"
)
_SIGNOFF_RE = re.compile(
    r"^(?:many thanks|thanks a lot|thanks again|thank you|thanks|thx|ta|best regards|best wishes|"
    r"kind regards|warm regards|warmest regards|regards|all the best|best|cheers|sincerely|"
    r"yours sincerely|yours faithfully|yours|talk soon|speak soon|take care)\s*[,.!]*$",
    re.I,
)
_NAME_LINE_RE = re.compile(rf"^{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,2}}$")


def _is_on_wrote(lines: list[str], i: int) -> bool:
    if not _ON_START_RE.match(lines[i]):
        return False
    joined = ""
    for k in range(3):
        if i + k >= len(lines):
            return False
        piece = lines[i + k].strip()
        if not piece:
            return False
        joined = f"{joined} {piece}".strip()
        if _ON_WROTE_RE.match(joined):
            return True
    return False


def _is_outlook_block(lines: list[str], i: int) -> bool:
    if not _OUTLOOK_FROM_RE.match(lines[i]):
        return False
    seen = 0
    for nxt in lines[i + 1 : i + 5]:
        if not nxt.strip():
            continue
        if _OUTLOOK_NEXT_RE.match(nxt):
            return True
        seen += 1
        if seen >= 2:
            break
    return False


def _find_cut(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if (
            _ORIGINAL_RE.match(line)
            or _FORWARDED_RE.match(line)
            or _BEGIN_FWD_RE.match(line)
            or _is_on_wrote(lines, i)
            or _is_outlook_block(lines, i)
        ):
            return i
    return None


def _strip_greeting(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if _GREETING_RE.match(line.strip()):
            return lines[idx + 1 :]
        break
    return lines


def _strip_signoff(lines: list[str]) -> list[str]:
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if not lines:
        return lines
    last = lines[-1].strip()
    if _NAME_LINE_RE.match(last) and len(last) <= 30:
        j = len(lines) - 2
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and _SIGNOFF_RE.match(lines[j].strip()):
            return lines[:j]
    if _SIGNOFF_RE.match(last) and last.endswith(","):
        return lines[:-1]
    return lines


def extract_own_text(body: str) -> str:
    """Only what the user wrote: no quoted replies, forwards, signature, greeting or sign-off."""
    if not body:
        return ""
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cut = _find_cut(lines)
    if cut is not None:
        lines = lines[:cut]
    lines = [ln for ln in lines if not ln.lstrip().startswith(">")]
    for i, ln in enumerate(lines):
        if ln in ("-- ", "--") or ln.rstrip() == "--":
            lines = lines[:i]
            break
    lines = [ln for ln in lines if not any(r.match(ln) for r in _DEVICE_RES)]
    lines = _strip_greeting(lines)
    lines = _strip_signoff(lines)
    text = "\n".join(re.sub(r"[ \t ]+", " ", ln).strip() for ln in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- masking

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]*[^\s<>\"')\].,;:!?]")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
_LONG_NUMBER_RE = re.compile(r"(?<![\d])\d(?:[ -]?\d){12,18}(?![\d])")
_PHONE_RES = (
    re.compile(r"(?<![\w])(?:\+|00)\d[\d ()-]{7,17}\d(?!\d)"),
    re.compile(r"(?<![\w])\(?0\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}(?!\d)"),
    re.compile(r"(?<![\w])\(\d{3}\)\s?\d{3}[ -]\d{4}(?!\d)"),
    re.compile(r"(?<![\w])\d{3}[ -]\d{3}[ -]\d{4}(?!\d)"),
)


def mask_private_details(text: str) -> str:
    text = _URL_RE.sub("[link]", text)
    text = _EMAIL_RE.sub("[email]", text)
    text = _POSTCODE_RE.sub("[postcode]", text)
    text = _LONG_NUMBER_RE.sub("[number]", text)
    for rx in _PHONE_RES:
        text = rx.sub("[phone]", text)
    return text


_PLACEHOLDER_RE = re.compile(r"\[(?:email|link|phone|postcode|number)\]")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def is_substantive(text: str) -> bool:
    """At least 8 real words once masking placeholders are ignored."""
    return len(_WORD_RE.findall(_PLACEHOLDER_RE.sub(" ", text or ""))) >= 8
