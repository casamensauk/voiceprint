"""Multi-provider LLM rewriter + deterministic De-AI & personal Voice DNA transformer."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .analyzer import (
    CONTRACTION_PAIRS,
    HIGH_RISK_AI_WORDS,
    ROBOTIC_TRANSITIONS,
    analyze_text,
    compute_burstiness,
    split_sentences,
)
from .db import (
    add_rewrite_history,
    get_edit_pairs,
    get_learned_replacements,
    get_samples,
    get_settings,
)
from .style_dna import extract_voice_dna, retrieve_similar_samples

# Extra casual / conversational swaps layered on top of HIGH_RISK_AI_WORDS
CASUAL_UPGRADES: dict[str, str] = {
    "however": "though",
    "therefore": "so",
    "thus": "so",
    "ensure": "make sure",
    "ensures": "makes sure",
    "ensuring": "making sure",
    "obtain": "get",
    "provide": "give",
    "provides": "gives",
    "demonstrate": "show",
    "demonstrates": "shows",
    "regarding": "about",
    "concerning": "about",
    "approximately": "around",
    "immediately": "right away",
    "assistance": "help",
    "inform": "let know",
    "inquire": "ask",
    "commence": "start",
    "terminate": "end",
    "sufficient": "enough",
    "numerous": "lots of",
    "significant": "big",
    "significantly": "a lot",
    "substantial": "big",
    "individuals": "people",
    "implement": "set up",
    "implementing": "setting up",
    "methodology": "approach",
    "subsequent": "next",
    "prior to": "before",
    "in order to": "to",
    "due to the fact that": "because",
    "at this point in time": "right now",
    "in the near future": "soon",
}


def _match_case(replacement: str, original: str, force_lowercase: bool = False) -> str:
    if not replacement:
        return ""
    if force_lowercase:
        return replacement.lower()
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float = 25.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_ollama_ready(ollama_url: str) -> bool:
    try:
        req = urllib.request.Request(f"{ollama_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=0.45) as resp:
            return resp.status == 200
    except Exception:
        return False


def resolve_voice_corpus(
    mode: str,
    db_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return active samples, rules, and edit pairs with cross-mode fallback if <3 samples in mode."""
    mode_samples = get_samples(context_mode=mode, db_path=db_path)
    if len(mode_samples) < 3:
        all_samples = get_samples(context_mode=None, db_path=db_path)
        seen_ids = {s.get("id") for s in mode_samples if s.get("id") is not None}
        for s in all_samples:
            if s.get("id") in seen_ids:
                continue
            cross = dict(s)
            cross["weight"] = round(float(s.get("weight", 1.0)) * 0.85, 3)
            mode_samples.append(cross)
            if s.get("id") is not None:
                seen_ids.add(s.get("id"))

    mode_rules = get_learned_replacements(context_mode=mode, db_path=db_path)
    if not mode_rules:
        mode_rules = get_learned_replacements(context_mode=None, db_path=db_path)

    mode_edits = get_edit_pairs(context_mode=mode, limit=5, db_path=db_path)
    if not mode_edits:
        mode_edits = get_edit_pairs(context_mode=None, limit=5, db_path=db_path)

    return mode_samples, mode_rules, mode_edits


def _build_llm_prompts(
    text: str,
    context_mode: str,
    intensity: str,
    dna: dict[str, Any],
    learned_replacements: list[dict[str, Any]],
    top_samples: list[dict[str, Any]],
    edit_pairs: list[dict[str, Any]],
) -> tuple[str, str]:
    directives = "\n".join(f"- {d}" for d in dna.get("directives_summary", []))
    rules_block = ""
    if learned_replacements:
        rule_lines = [
            f"- Avoid '{r['avoid_text']}' -> use '{r['prefer_text'] or '[omit]'}'"
            for r in learned_replacements[:15]
        ]
        rules_block = "\nUSER'S LEARNED WORD/PHRASE PREFERENCES (strictly follow):\n" + "\n".join(rule_lines)

    samples_block = ""
    if top_samples:
        formatted = "\n---\n".join(s["content"].strip()[:400] for s in top_samples[:6])
        samples_block = f"\nAUTHENTIC WRITING SAMPLES FROM THIS USER ({context_mode} & personal voice — mirror this exact tone and rhythm):\n{formatted}"

    edits_block = ""
    if edit_pairs:
        pairs_txt = "\n".join(
            f"Draft: {p['ai_draft'][:180]}\nUser's actual edit: {p['user_final_edit'][:180]}"
            for p in edit_pairs[:3]
        )
        edits_block = f"\nPAST CORRECTIONS SHOWING HOW THIS USER TWEAKS DRAFTS:\n{pairs_txt}"

    system_prompt = (
        f"You are VoicePrint, a personal writing assistant that edits and rephrases the user's draft notes, "
        f"messages, and emails into their natural voice from the samples below, tailored for '{context_mode}' mode.\n\n"
        f"RULES:\n"
        f"1. Preserve the core facts and meaning accurately, without inventing facts or adding fluff.\n"
        f"2. Use varied, natural sentence cadence (mix crisp 4-9 word sentences with longer thoughts) and straight keyboard apostrophes/quotes.\n"
        f"3. Match the user's Voice DNA level of contractions (it's, don't, you'll, can't) and direct tone without third-person textbook framing.\n"
        f"4. NEVER tack on email sign-offs ('Best.', 'Cheers.', 'Thanks.') unless the input text itself is an email or already ends with a sign-off.\n"
        f"5. NEVER use faux-casual list transitions ('Another good way to use X is...', 'X is really good for...'), parallel 3-or-4-item Oxford-comma lists, RLHF framing ('Instead of...', 'By ...ing,', 'At its core,', 'Ultimately,'), 'Heading: Explanation' colons, Unicode em-dashes (—), 'Not only ... but also', or '**Label**:' bullets.\n"
        f"6. NEVER use AI clichés (delve, tapestry, testament, underscore, multifaceted, foster, realm, pivotal, crucial, intricate, meticulous, vibrant, landscape, embark, unleash, elevate, leverage, synergy, seamless, robust, comprehensive, transformative, moreover, furthermore, in conclusion, it's worth noting, at the end of the day).\n"
        f"7. Rewrite intensity: {intensity}.\n"
        f"8. VOICE DNA TARGETS:\n{directives}"
        f"{rules_block}"
        f"{samples_block}"
        f"{edits_block}\n\n"
        f"Return ONLY the rewritten text — no preamble, no quotes, no explanation."
    )
    user_prompt = f"Edit this draft into my natural writing voice:\n\n{text}"
    return system_prompt, user_prompt


def _call_llm_provider(
    provider_choice: str,
    settings: dict[str, str],
    system_prompt: str,
    user_prompt: str,
) -> tuple[str | None, str]:
    """Attempt rewrite via configured LLM provider; return (text_or_none, provider_label)."""
    openrouter_key = (settings.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    candidates: list[str] = []
    if provider_choice == "auto":
        if openrouter_key:
            candidates.append("openrouter")
        if settings.get("gemini_api_key"):
            candidates.append("gemini")
        if settings.get("openai_api_key"):
            candidates.append("openai")
        if settings.get("anthropic_api_key"):
            candidates.append("anthropic")
        if _check_ollama_ready(settings.get("ollama_url", "http://localhost:11434")):
            candidates.append("ollama")
    elif provider_choice != "local_only":
        candidates.append(provider_choice)

    for prov in candidates:
        try:
            if prov == "openrouter" and openrouter_key:
                base_url = (
                    settings.get("openrouter_base_url")
                    or os.environ.get("OPENROUTER_BASE_URL")
                    or "https://openrouter.ai/api/v1"
                ).rstrip("/")
                model = (
                    settings.get("openrouter_rewrite_model")
                    or settings.get("ai_detect_arbiter_model")
                    or "google/gemini-2.5-flash"
                ).strip()
                res = _post_json(
                    f"{base_url}/chat/completions",
                    {
                        "model": model,
                        "temperature": 0.88,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "HTTP-Referer": "http://127.0.0.1:18940",
                        "X-Title": "VoicePrint",
                    },
                )
                txt = res["choices"][0]["message"]["content"].strip()
                if txt:
                    return txt, f"openrouter ({model})"

            elif prov == "gemini" and settings.get("gemini_api_key"):
                model = settings.get("gemini_model") or "gemini-2.5-flash"
                key = settings["gemini_api_key"]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                res = _post_json(
                    url,
                    {
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                        "generationConfig": {"temperature": 0.85},
                    },
                    headers={},
                )
                txt = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                if txt:
                    return txt, f"gemini ({model})"

            elif prov == "openai" and settings.get("openai_api_key"):
                model = settings.get("openai_model") or "gpt-4o-mini"
                res = _post_json(
                    "https://api.openai.com/v1/chat/completions",
                    {
                        "model": model,
                        "temperature": 0.85,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    headers={"Authorization": f"Bearer {settings['openai_api_key']}"},
                )
                txt = res["choices"][0]["message"]["content"].strip()
                if txt:
                    return txt, f"openai ({model})"

            elif prov == "anthropic" and settings.get("anthropic_api_key"):
                model = settings.get("anthropic_model") or "claude-3-5-haiku-latest"
                res = _post_json(
                    "https://api.anthropic.com/v1/messages",
                    {
                        "model": model,
                        "max_tokens": 1800,
                        "temperature": 0.85,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                    headers={
                        "x-api-key": settings["anthropic_api_key"],
                        "anthropic-version": "2023-06-01",
                    },
                )
                txt = "".join(b.get("text", "") for b in res.get("content", [])).strip()
                if txt:
                    return txt, f"anthropic ({model})"

            elif prov == "ollama":
                base = (settings.get("ollama_url") or "http://localhost:11434").rstrip("/")
                model = settings.get("ollama_model") or "llama3.2"
                res = _post_json(
                    f"{base}/api/chat",
                    {
                        "model": model,
                        "stream": False,
                        "options": {"temperature": 0.85},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    headers={},
                    timeout=35.0,
                )
                txt = (res.get("message") or {}).get("content", "").strip()
                if txt:
                    return txt, f"ollama ({model})"
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, TimeoutError, ValueError):
            continue

    return None, "local-dna-engine"


def _conjugate_3s(verb: str) -> str:
    """Conjugate a base English verb into 3rd-person singular present tense."""
    low = verb.lower()
    if low in {"have", "has"}:
        return "has"
    if low.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if len(low) >= 2 and low.endswith("y") and low[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _replace_not_only(text: str) -> str:
    does_pattern = re.compile(
        r"\bnot only\s+does\s+(it|this|that|he|she)\s+([a-z]+)([^,.;\n]*),?\s*but\s+(?:it\s+|they\s+|this\s+)?also\s+([^.;\n]{2,120})",
        re.IGNORECASE,
    )
    text = does_pattern.sub(
        lambda m: f"{m.group(1)} {_conjugate_3s(m.group(2))}{m.group(3).rstrip()}, and it also {m.group(4).strip()}",
        text,
    )
    inv_pattern = re.compile(
        r"\bnot only\s+(?:does|do|did)\s+([^,.;\n]{2,100}),?\s*but\s+(?:it\s+|they\s+|this\s+)?also\s+([^.;\n]{2,120})",
        re.IGNORECASE,
    )
    text = inv_pattern.sub(lambda m: f"{m.group(1).strip()}, and it also {m.group(2).strip()}", text)
    pattern = re.compile(r"\bnot only\s+([^,.;\n]{2,100}),?\s*but\s+(?:it\s+)?also\s+([^.;\n]{2,120})", re.IGNORECASE)
    return pattern.sub(lambda m: f"{m.group(1).strip()} and {m.group(2).strip()}", text)


def _modulate_burstiness(text: str, target_avg: float, intensity: str, from_llm: bool = False) -> str:
    """Break up uniform AI sentence rhythm by splitting long compound clauses into natural bursts."""
    if intensity == "subtle":
        return text

    paragraphs = text.split("\n")
    out_paragraphs: list[str] = []

    for para in paragraphs:
        if not para.strip():
            out_paragraphs.append(para)
            continue
        burst = compute_burstiness(para)
        if from_llm and burst["coeff_of_variation"] >= 0.30:
            out_paragraphs.append(para)
            continue
        sents = split_sentences(para)
        needs_variation = len(sents) >= 2 and (
            burst["coeff_of_variation"] < 0.35 or max(len(s.split()) for s in sents) >= 22
        )
        new_sents: list[str] = []
        for idx, sent in enumerate(sents):
            words = sent.split()
            should_split = (
                len(words) >= 18
                or (needs_variation and len(words) >= 13 and idx % 2 == 1)
                or (target_avg <= 11.5 and len(words) >= 15)
            )
            if should_split:
                # Split only on safe clause boundaries (never mid-enumeration noun lists)
                split_match = re.search(
                    r"(?:"
                    r";\s+"
                    r"|,\s+(?:while|whereas|though|although)\s+"
                    r"|,\s+which\s+(?=(?:gives|makes|lets|helps|means|allows|keeps|turns|shows|is|was|can|will)\b)"
                    r"|,\s+and\s+(?=(?:it|they|you|we|this|that|there)\s+(?:can|will|also|get|have|use|need|don't|won't|are|is)\b)"
                    r")",
                    sent,
                    re.IGNORECASE,
                )
                is_so_purpose = False
                if not split_match and len(words) >= 22:
                    split_match = re.search(
                        r"\s+so\s+(?=(?:customers|users|people|you|we|they|teams|businesses|anyone)\s+(?:can|will|don't|won't|get|find|know|see)\b)",
                        sent,
                        re.IGNORECASE,
                    )
                    if split_match:
                        is_so_purpose = True

                if split_match and split_match.start() >= 14 and len(sent) - split_match.end() >= 12:
                    left = sent[: split_match.start()].strip().rstrip(",;")
                    right = sent[split_match.end() :].strip()
                    conj = split_match.group(0).strip(" ,;").lower()
                    if not right.lower().startswith("to ") and len(left.split()) >= 3 and len(right.split()) >= 4:
                        if conj == "which":
                            right = f"That {right}"
                        elif is_so_purpose:
                            right = f"That way {right}"
                        else:
                            right = right[:1].upper() + right[1:] if right else right
                        if left and not left.endswith((".", "!", "?")):
                            left += "."
                        new_sents.append(left)
                        new_sents.append(right)
                        continue
            new_sents.append(sent)
        out_paragraphs.append(" ".join(new_sents))

    return "\n".join(out_paragraphs)


def _apply_contractions(text: str, target_rate: float) -> str:
    out = text
    if target_rate >= 0.45:
        for exp_form, con_form in CONTRACTION_PAIRS:
            pat = re.compile(r"\b" + re.escape(exp_form) + r"\b", re.IGNORECASE)
            out = pat.sub(lambda m, c=con_form: _match_case(c, m.group(0)), out)
    elif target_rate < 0.15:
        for exp_form, con_form in CONTRACTION_PAIRS:
            if exp_form == "can not":
                continue
            pat = re.compile(r"\b" + re.escape(con_form) + r"\b", re.IGNORECASE)
            out = pat.sub(lambda m, e=exp_form: _match_case(e, m.group(0)), out)
    return out


def _apply_surface_dna(
    text: str,
    context_mode: str,
    dna: dict[str, Any],
    intensity: str,
    original_text: str = "",
) -> str:
    """Apply user's personal casing, punctuation, and trailing-period habits."""
    out = (
        text.strip()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    if not out:
        return out

    # Strip hallucinated email sign-offs appended to non-email prose
    if context_mode != "work_email" and not re.search(
        r"(?:best|cheers|regards)[.!]*\s*$", original_text.strip(), re.IGNORECASE
    ):
        out = re.sub(r"(?<=[.!?])\s+(?:Best|Cheers|Warm regards|Kind regards)\.?\s*$", "", out, flags=re.IGNORECASE)

    # Strip bold-colon AI bullet labels: "- **Security**: It uses..." -> "- Security: it uses..."
    out = re.sub(r"(?m)^(\s*[-*•]\s+|\s*\d+\.\s+)\*\*([^*\n]{2,50})\*\*\s*:", r"\1\2:", out)

    # Em-dash replacement based on user's punctuation profile
    punct = dna.get("punctuation_profile", {})
    if "—" in out and float(punct.get("em_dash", 0.0)) < 0.25:
        if float(punct.get("double_dash", 0.0)) >= 0.25:
            out = re.sub(r"\s*—\s*", " -- ", out)
        else:
            out = re.sub(r"\s*—\s*", ", ", out)

    # Contraction profile
    c_rate = float(dna.get("contraction_rate", 0.65))
    if context_mode in ("casual_chat", "social_post") and dna.get("sample_count", 0) == 0:
        c_rate = max(c_rate, 0.75)
    out = _apply_contractions(out, c_rate)

    # Capitalize cleanly after stripped sentence-initial transitions
    out = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)

    # Lowercase habits (e.g., user types 'yeah i think...' in casual chat)
    lower_start_rate = float(dna.get("lowercase_start_rate", 0.0))
    lower_i_rate = float(dna.get("lowercase_i_rate", 0.0))

    if lower_start_rate >= 0.5 and context_mode in ("casual_chat", "social_post"):
        lines = []
        for line in out.splitlines():
            sents = split_sentences(line)
            lowered_sents = []
            for s in sents:
                if len(s) >= 2 and s[0].isupper() and not s[:2].isupper():
                    s = s[0].lower() + s[1:]
                lowered_sents.append(s)
            lines.append(" ".join(lowered_sents))
        out = "\n".join(lines)

    if lower_i_rate >= 0.5 and context_mode in ("casual_chat", "social_post"):
        out = re.sub(r"\bI\b", "i", out)
        out = re.sub(r"\bI(['’](?:m|ve|ll|d))\b", r"i\1", out)

    # Trailing period habit on short casual/social messages
    no_period_rate = float(dna.get("no_trailing_period_rate", 0.0))
    sents = split_sentences(out)
    if (
        intensity != "subtle"
        and context_mode in ("casual_chat", "social_post")
        and no_period_rate >= 0.45
        and len(sents) <= 2
        and out.endswith(".")
        and not out.endswith("...")
    ):
        out = out[:-1].rstrip()

    # Final whitespace tidy
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([.!?])\s*([.!?])+", r"\1", out)
    return out.strip()


def apply_deterministic_de_ai(
    text: str,
    context_mode: str,
    dna: dict[str, Any],
    learned_replacements: list[dict[str, Any]],
    intensity: str = "deep",
    from_llm: bool = False,
    original_text: str = "",
) -> str:
    """Deterministic De-AI + Voice DNA pass (works standalone or after LLM generation)."""
    out = text.strip()
    if not out:
        return out

    # 1. Apply user's learned & custom replacements FIRST (longest phrases first)
    sorted_rules = sorted(
        learned_replacements,
        key=lambda r: len(r.get("avoid_text", "")),
        reverse=True,
    )
    for rule in sorted_rules:
        avoid = (rule.get("avoid_text") or "").strip()
        prefer = (rule.get("prefer_text") or "").strip()
        if not avoid:
            continue
        pat = re.compile(r"\b" + re.escape(avoid) + r"\b", re.IGNORECASE)
        if prefer == "":
            out = pat.sub("", out)
        else:
            out = pat.sub(lambda m, p=prefer: _match_case(p, m.group(0)), out)

    # 2. Strip/replace robotic transitions (longest first)
    for phrase, replacement in sorted(
        ROBOTIC_TRANSITIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        pat = re.compile(r"\b" + re.escape(phrase) + r"\b(?:\s*,)?\s*", re.IGNORECASE)
        if replacement == "":
            out = pat.sub("", out)
        else:
            out = pat.sub(lambda m, r=replacement: _match_case(r, m.group(0)) + " ", out)

    # 3. Replace high-risk AI words and collapse duplicate prepositions (e.g. "delve into" -> "dig into")
    for ai_word, plain_word in HIGH_RISK_AI_WORDS.items():
        pat = re.compile(r"\b" + re.escape(ai_word) + r"\b", re.IGNORECASE)
        out = pat.sub(lambda m, p=plain_word: _match_case(p, m.group(0)), out)
    out = re.sub(r"\b(into)\s+into\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(to)\s+to\b", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"^(?:Plus|Also)\s+(?=[a-zA-Z])", "", out)

    # 4. In casual_chat / social_post or deep mode, swap stiff formal connectors & textbook framing
    if context_mode in ("casual_chat", "social_post") or intensity == "deep":
        for formal, natural in sorted(
            CASUAL_UPGRADES.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            pat = re.compile(r"\b" + re.escape(formal) + r"\b", re.IGNORECASE)
            out = pat.sub(lambda m, n=natural: _match_case(n, m.group(0)), out)

    if not from_llm and (context_mode in ("casual_chat", "social_post") or intensity in ("natural", "deep")):
        is_casual = context_mode in ("casual_chat", "social_post")
        out = re.sub(
            r"\bwhich gives businesses a dedicated\b",
            "that gives you a" if is_casual else "giving teams a",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"\bgives businesses a dedicated\b",
            "gives you a" if is_casual else "gives teams a",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"\ballow businesses to\b",
            "let you" if is_casual else "let teams",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"\ballows businesses to\b",
            "lets you" if is_casual else "lets teams",
            out,
            flags=re.IGNORECASE,
        )
        if intensity == "deep" or is_casual:
            out = re.sub(
                r"\bso customers can find them easily\b",
                "so people can find them easily",
                out,
                flags=re.IGNORECASE,
            )

    # 5. Structural pattern removal
    out = _replace_not_only(out)
    if intensity == "deep" and float(dna.get("oxford_comma_rate", 0.15)) < 0.40:
        out = re.sub(
            r",\s+(and|or)\s+(?!\s*(?:it|they|you|we|i|he|she|this|that|there)\b)",
            r" \1 ",
            out,
            flags=re.IGNORECASE,
        )
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"(?m)^\s*,\s*", "", out)

    # 6. Burstiness rhythm modulation
    out = _modulate_burstiness(out, float(dna.get("avg_sentence_length", 12.0)), intensity, from_llm=from_llm)

    # 7. Personal Voice DNA surface polish
    out = _apply_surface_dna(out, context_mode, dna, intensity, original_text=original_text or text)
    return out


def rewrite_text(
    text: str,
    context_mode: str | None = None,
    intensity: str | None = None,
    save_history: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Rewrite input text to erase AI tells and match the user's personal Voice DNA."""
    cleaned = text.strip()
    settings = get_settings(db_path)
    mode = context_mode or settings.get("active_context_mode") or "casual_chat"
    eff_intensity = intensity or settings.get("intensity") or "deep"

    samples, learned_rules, edit_pairs = resolve_voice_corpus(mode, db_path=db_path)

    dna = extract_voice_dna(samples, learned_rules)
    before_analysis = analyze_text(cleaned, dna)

    if not cleaned:
        return {
            "original": text,
            "rewritten": "",
            "context_mode": mode,
            "intensity": eff_intensity,
            "before": before_analysis,
            "after": before_analysis,
            "dna_used": dna,
            "provider_used": "none",
            "samples_referenced": 0,
        }

    top_samples = retrieve_similar_samples(cleaned, samples, top_k=6)
    sys_prompt, usr_prompt = _build_llm_prompts(
        cleaned, mode, eff_intensity, dna, learned_rules, top_samples, edit_pairs
    )
    provider_setting = settings.get("provider", "auto")
    llm_draft, provider_used = _call_llm_provider(
        provider_setting, settings, sys_prompt, usr_prompt
    )

    base_text = llm_draft if llm_draft else cleaned
    rewritten = apply_deterministic_de_ai(
        base_text,
        context_mode=mode,
        dna=dna,
        learned_replacements=learned_rules,
        intensity=eff_intensity,
        from_llm=bool(llm_draft),
        original_text=cleaned,
    )

    # 1-pass Voice Fidelity self-check when LLM generated text
    if llm_draft and len(rewritten.split()) >= 20:
        from .ai_detector.stylometrics import score_window_stylometrics

        stylo = score_window_stylometrics(rewritten)
        if float(stylo["stylometric_ai_score"]) >= 0.35:
            reasons_txt = "; ".join((stylo.get("reasons") or [])[:4])
            critique_prompt = (
                f"Revise this draft to better reflect the user's writing style in the samples above while addressing these stylistic cues: {reasons_txt}. "
                f"Keep the core meaning accurate, use natural contractions, vary sentence length naturally, avoid tacked-on email closings, and return ONLY the revised text.\n\nDraft:\n{rewritten}"
            )
            refined_raw, refined_prov = _call_llm_provider(
                provider_setting, settings, sys_prompt, critique_prompt
            )
            if refined_raw:
                refined = apply_deterministic_de_ai(
                    refined_raw,
                    context_mode=mode,
                    dna=dna,
                    learned_replacements=learned_rules,
                    intensity=eff_intensity,
                    from_llm=True,
                    original_text=cleaned,
                )
                if float(score_window_stylometrics(refined)["stylometric_ai_score"]) < float(stylo["stylometric_ai_score"]):
                    rewritten = refined
                    provider_used = refined_prov

    after_analysis = analyze_text(rewritten, dna)
    if save_history:
        add_rewrite_history(
            context_mode=mode,
            input_text=cleaned,
            output_text=rewritten,
            before_human_score=before_analysis["human_score"],
            after_human_score=after_analysis["human_score"],
            provider_used=provider_used,
            db_path=db_path,
        )

    return {
        "original": cleaned,
        "rewritten": rewritten,
        "context_mode": mode,
        "intensity": eff_intensity,
        "before": before_analysis,
        "after": after_analysis,
        "dna_used": dna,
        "provider_used": provider_used,
        "samples_referenced": len(top_samples),
    }
