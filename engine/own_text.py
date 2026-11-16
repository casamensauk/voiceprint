"""Pure functions that reduce an email/chat body to only what the user wrote, and mask private details."""

from __future__ import annotations

import re

_ORIGINAL_RE = re.compile(
    r"^\s*-{2,}\s*(?:Original Message|Message d'origine|Ursprüngliche Nachricht|Mensaje original|"
    r"Messaggio originale|Oorspronkelijk bericht|Mensagem original)\s*-{2,}\s*$",
    re.I,
)
_FORWARDED_RE = re.compile(
    r"^\s*-{3,}\s*(?:Forwarded message|Message transféré|Weitergeleitete Nachricht|Mensaje reenviado|"
    r"Messaggio inoltrato|Doorgestuurd bericht|Mensagem encaminhada)\s*-{3,}\s*$",
    re.I,
)
_BEGIN_FWD_RE = re.compile(r"^\s*Begin forwarded message:?\s*$", re.I)

# Header labels in English, French, German, Spanish, Italian, Dutch and Portuguese.
_FROM_LABELS = r"from|de|von|da|van"
_OTHER_LABELS = (
    r"sent|date|to|cc|bcc|subject|"
    r"envoyé|à|objet|"
    r"gesendet|datum|an|betreff|"
    r"enviado|fecha|para|asunto|"
    r"inviato|data|a|oggetto|"
    r"verzonden|aan|onderwerp|"
    r"enviada|assunto"
)
_HEADER_TAIL = r"[*_]*\s*:[*_]*\s*\S"
_FROM_LINE_RE = re.compile(rf"^\s*[*_]*(?:{_FROM_LABELS}){_HEADER_TAIL}", re.I)
_HEADER_LINE_RE = re.compile(rf"^\s*[*_]*(?:{_FROM_LABELS}|{_OTHER_LABELS}){_HEADER_TAIL}", re.I)
_ADDRESS_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|<[^<>\s]+>")

_ON_START_RE = re.compile(r"^\s*(?:On|Le|Am|El|Il|Op|Em)\s+\S")
_WROTE_SIMPLE_RE = re.compile(
    r"^(?:On|Le|El|Il|Em)\s.{3,400}?\b(?:wrote|a écrit|escribió|ha scritto|escreveu)\s*:$"
)
_WROTE_NAMED_RE = re.compile(r"^(?:Am|Op)\s.{3,400}?\b(?:schrieb|schreef)\b(?:\s.{0,200})?:$")

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
_CAP_WORD = r"[A-Z\u00C0-\u00DE][^\W\d_]*"
_NAME_LINE_RE = re.compile(rf"^{_CAP_WORD}(?:\s+{_CAP_WORD})?$")
_RULE_RE = re.compile(r"^\s*[_=-]{5,}\s*$")
_SIGNATURE_MAX_LINES = 8


def _wrote_marker_end(lines: list[str], i: int) -> int | None:
    """Index of the last line of an "On ... wrote:" style marker that starts at i, in seven languages."""
    if not _ON_START_RE.match(lines[i]):
        return None
    joined = ""
    for k in range(3):
        if i + k >= len(lines):
            return None
        piece = lines[i + k].strip()
        if not piece:
            return None
        joined = f"{joined} {piece}".strip()
        if _WROTE_SIMPLE_RE.match(joined) or _WROTE_NAMED_RE.match(joined):
            return i + k
    return None


def _is_quoted(line: str) -> bool:
    return line.lstrip().startswith(">")


def _is_quote_marker(lines: list[str], i: int) -> bool:
    """A real quote header: the marker is followed by quoted text, header text, or nothing at all."""
    end = _wrote_marker_end(lines, i)
    if end is None:
        return False
    rest = [ln for ln in lines[end + 1 :] if ln.strip()]
    if not rest or _is_quoted(rest[0]):
        return True
    return all(_is_quoted(ln) or _HEADER_LINE_RE.match(ln) for ln in rest)


def _is_header_block(lines: list[str], i: int) -> bool:
    """Outlook-style From/Sent/To/Subject block (and its translations) that starts a quoted message."""
    if not _FROM_LINE_RE.match(lines[i]):
        return False
    run = 0
    while i + run < len(lines) and _HEADER_LINE_RE.match(lines[i + run]):
        run += 1
    if _ADDRESS_RE.search(lines[i]):
        for nxt in lines[i + 1 : i + 6]:
            if nxt.strip():
                return bool(_HEADER_LINE_RE.match(nxt))
        return False
    after = i + run
    return run >= 2 and after < len(lines) and not lines[after].strip()


def _find_cut(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if (
            _ORIGINAL_RE.match(line)
            or _FORWARDED_RE.match(line)
            or _BEGIN_FWD_RE.match(line)
            or _is_quote_marker(lines, i)
            or _is_header_block(lines, i)
        ):
            return i
    return None


def _drop_quoted(lines: list[str]) -> list[str]:
    """Drop runs of two or more ">" lines, and a single ">" line that directly follows a quote marker."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not _is_quoted(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and _is_quoted(lines[j]):
            j += 1
        follows_marker = i > 0 and any(_wrote_marker_end(lines, k) == i - 1 for k in range(max(0, i - 3), i))
        if j - i < 2 and not follows_marker:
            out.extend(lines[i:j])
        i = j
    return out


def _strip_greeting(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if _GREETING_RE.match(line.strip()):
            return lines[idx + 1 :]
        break
    return lines


def _strip_signoff(lines: list[str]) -> list[str]:
    """Remove a trailing "Thanks,\nName" or "Thanks," pair, but never the whole message."""
    kept = list(lines)
    while kept and not kept[-1].strip():
        kept.pop()
    if not kept:
        return lines
    result = kept
    last = kept[-1].strip()
    if _NAME_LINE_RE.match(last):
        j = len(kept) - 2
        while j >= 0 and not kept[j].strip():
            j -= 1
        if j >= 0 and _SIGNOFF_RE.match(kept[j].strip()):
            result = kept[:j]
    elif _SIGNOFF_RE.match(last) and last.endswith(","):
        result = kept[:-1]
    if not any(ln.strip() for ln in result):
        return lines
    return result


def _cut_signature(lines: list[str]) -> list[str]:
    for i, ln in enumerate(lines):
        if ln == "-- ":
            following = [x for x in lines[i + 1 :] if x.strip()]
            if len(following) <= _SIGNATURE_MAX_LINES:
                return lines[:i]
            return lines[:i] + lines[i + 1 :]
    return lines


def extract_own_text(body: str) -> str:
    """Only what the user wrote: no quoted replies, forwards, signature, greeting or sign-off."""
    if not body:
        return ""
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cut = _find_cut(lines)
    if cut is not None:
        lines = lines[:cut]
        while lines and (not lines[-1].strip() or _RULE_RE.match(lines[-1])):
            lines.pop()
    lines = _drop_quoted(lines)
    lines = _cut_signature(lines)
    lines = [ln for ln in lines if not any(r.match(ln) for r in _DEVICE_RES)]
    lines = _strip_greeting(lines)
    lines = _strip_signoff(lines)
    text = "\n".join(re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in lines)
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
