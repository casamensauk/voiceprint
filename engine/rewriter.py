"""Multi-provider LLM rewriter + deterministic De-AI & personal Voice DNA transformer."""

from __future__ import annotations

import json
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
        formatted = "\n---\n".join(s["content"].strip() for s in top_samples[:5])
        samples_block = f"\nAUTHENTIC WRITING SAMPLES FROM THIS USER ({context_mode}):\n{formatted}"

    edits_block = ""
    if edit_pairs:
        pairs_txt = "\n".join(
            f"Draft: {p['ai_draft'][:180]}\nUser's actual edit: {p['user_final_edit'][:180]}"
            for p in edit_pairs[:3]
        )
        edits_block = f"\nPAST CORRECTIONS SHOWING HOW THIS USER TWEAKS DRAFTS:\n{pairs_txt}"

    system_prompt = (
        f"You are VoicePrint, a ghostwriter that rewrites text so it sounds 100% written by the user "
        f"in '{context_mode}' mode and leaves zero AI statistical fingerprints.\n\n"
        f"RULES:\n"
        f"1. Preserve the core facts and intent accurately, without inventing facts.\n"
        f"2. Dismantle AI rhythm: vary sentence lengths dramatically (high burstiness — mix short 3-7 word lines with medium/longer ones).\n"
        f"3. NEVER use AI clichés (delve, tapestry, testament, underscore, multifaceted, foster, realm, pivotal, crucial, intricate, meticulous, vibrant, landscape, embark, unleash, elevate, leverage, synergy, seamless, robust, comprehensive, transformative, moreover, furthermore, in conclusion, it's worth noting, at the end of the day).\n"
        f"4. NEVER use Unicode em-dashes (—) or 'Not only X, but also Y' templates or '**Label**:' bullet formatting.\n"
        f"5. Rewrite intensity: {intensity}.\n"
        f"6. VOICE DNA TARGETS:\n{directives}"
        f"{rules_block}"
        f"{samples_block}"
        f"{edits_block}\n\n"
        f"Return ONLY the rewritten text — no preamble, no quotes, no explanation."
    )
    user_prompt = f"Rewrite this text in my exact voice:\n\n{text}"
    return system_prompt, user_prompt


def _call_llm_provider(
    provider_choice: str,
    settings: dict[str, str],
    system_prompt: str,
    user_prompt: str,
) -> tuple[str | None, str]:
    """Attempt rewrite via configured LLM provider; return (text_or_none, provider_label)."""
    candidates: list[str] = []
    if provider_choice == "auto":
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
            if prov == "gemini" and settings.get("gemini_api_key"):
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


def _modulate_burstiness(text: str, target_avg: float, intensity: str) -> str:
    """Break up uniform AI sentence rhythm by splitting long compound clauses into natural bursts."""
    if intensity == "subtle":
        return text

    paragraphs = text.split("\n")
    out_paragraphs: list[str] = []

    for para in paragraphs:
        if not para.strip():
            out_paragraphs.append(para)
            continue
        sents = split_sentences(para)
        burst = compute_burstiness(para)
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
                # Split cleanly on ", and ", ", while ", ", which ", "; ", or ", " when bursting uniform prose
                split_match = re.search(r"(?:,\s+(?:and|while|which|whereas|so)\s+|;\s+)", sent)
                if not split_match and needs_variation and len(words) >= 12:
                    split_match = re.search(r",\s+(?=[a-zA-Z])", sent)
                if split_match and split_match.start() >= 14 and len(sent) - split_match.end() >= 12:
                    left = sent[: split_match.start()].strip().rstrip(",;")
                    right = sent[split_match.end() :].strip()
                    conj = split_match.group(0).strip(" ,;").lower()
                    if not right.lower().startswith("to ") and len(left.split()) >= 3 and len(right.split()) >= 4:
                        if conj == "which":
                            right = f"That {right}"
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


def _apply_surface_dna(text: str, context_mode: str, dna: dict[str, Any], intensity: str) -> str:
    """Apply user's personal casing, punctuation, and trailing-period habits."""
    out = text.strip()
    if not out:
        return out

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

    # 4. In casual_chat / social_post or deep mode, swap stiff formal connectors
    if context_mode in ("casual_chat", "social_post") or intensity == "deep":
        for formal, natural in sorted(
            CASUAL_UPGRADES.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            pat = re.compile(r"\b" + re.escape(formal) + r"\b", re.IGNORECASE)
            out = pat.sub(lambda m, n=natural: _match_case(n, m.group(0)), out)

    # 5. Structural pattern removal
    out = _replace_not_only(out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"(?m)^\s*,\s*", "", out)

    # 6. Burstiness rhythm modulation
    out = _modulate_burstiness(out, float(dna.get("avg_sentence_length", 12.0)), intensity)

    # 7. Personal Voice DNA surface polish
    out = _apply_surface_dna(out, context_mode, dna, intensity)
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

    samples = get_samples(context_mode=mode, db_path=db_path)
    learned_rules = get_learned_replacements(context_mode=mode, db_path=db_path)
    edit_pairs = get_edit_pairs(context_mode=mode, limit=5, db_path=db_path)

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

    top_samples = retrieve_similar_samples(cleaned, samples, top_k=5)
    sys_prompt, usr_prompt = _build_llm_prompts(
        cleaned, mode, eff_intensity, dna, learned_rules, top_samples, edit_pairs
    )
    llm_draft, provider_used = _call_llm_provider(
        settings.get("provider", "auto"), settings, sys_prompt, usr_prompt
    )

    base_text = llm_draft if llm_draft else cleaned
    rewritten = apply_deterministic_de_ai(
        base_text,
        context_mode=mode,
        dna=dna,
        learned_replacements=learned_rules,
        intensity=eff_intensity,
    )

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
