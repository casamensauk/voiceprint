"""Stage 2: DNA-GPT greedy prefix-truncation & regeneration BScore scorer (OpenRouter + multi-LLM fallback)."""

from __future__ import annotations

import difflib
import json
import math
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .preprocessor import PreprocessedSegment

_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_REGEN_MODELS = "meta-llama/llama-3.3-70b-instruct,google/gemini-2.5-flash"

_BSCORE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "in", "into", "is", "it",
    "its", "of", "on", "or", "our", "she", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "which", "while",
    "who", "will", "with", "would", "can", "could", "may", "might",
}


class OpenRouterError(Exception):
    """Error communicating with OpenRouter or fallback detector LLM API."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _http_json_post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    model: str,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = str(err)
        raise OpenRouterError(f"LLM API error ({err.code}) for {model}: {detail}", int(err.code or 502)) from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise OpenRouterError(f"LLM connection error for {model}: {err}", 502) from err


def openrouter_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 120,
    logprobs: bool = False,
    response_format: dict[str, Any] | None = None,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    """Call OpenRouter /chat/completions via stdlib urllib and return parsed JSON."""
    url = f"{(base_url or DEFAULT_OPENROUTER_URL).rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 3
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "HTTP-Referer": "https://github.com/casamensauk/voiceprint",
        "X-Title": "VoicePrint AI Detector",
    }
    return _http_json_post(url, payload, headers, model=model, timeout_s=timeout_s)


def call_provider_chat(
    provider_cfg: dict[str, str],
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 120,
    logprobs: bool = False,
    response_format: dict[str, Any] | None = None,
    timeout_s: float = 25.0,
) -> dict[str, Any]:
    """Dispatch chat completion to OpenRouter or configured fallback provider (Gemini/OpenAI/Anthropic/Ollama)."""
    provider = provider_cfg.get("provider", "openrouter")
    api_key = (provider_cfg.get("api_key") or "").strip()
    base_url = (provider_cfg.get("base_url") or DEFAULT_OPENROUTER_URL).strip()

    if provider == "openrouter":
        return openrouter_chat_completion(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            response_format=response_format,
            timeout_s=timeout_s,
        )

    if provider == "openai":
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
        if response_format is not None:
            payload["response_format"] = response_format
        return _http_json_post(
            "https://api.openai.com/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {api_key}"},
            model=model,
            timeout_s=timeout_s,
        )

    if provider == "gemini":
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_parts = [m["content"] for m in messages if m.get("role") != "system"]
        gen_cfg: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if response_format is not None:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}],
            "generationConfig": gen_cfg,
        }
        if sys_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_parts)}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        raw = _http_json_post(url, payload, {}, model=model, timeout_s=timeout_s)
        cands = raw.get("candidates") or []
        txt = ""
        if cands and isinstance(cands[0], dict):
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            txt = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
        usage_meta = raw.get("usageMetadata") or {}
        total_t = int(usage_meta.get("totalTokenCount") or 0)
        return {
            "choices": [{"message": {"role": "assistant", "content": txt}}],
            "usage": {"total_tokens": total_t, "cost": (total_t / 1_000_000.0) * 0.15},
        }

    if provider == "anthropic":
        sys_txt = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        usr_msgs = [{"role": "user", "content": m["content"]} for m in messages if m.get("role") != "system"]
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": sys_txt,
            "messages": usr_msgs,
        }
        raw = _http_json_post(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            model=model,
            timeout_s=timeout_s,
        )
        txt = "".join(str(b.get("text") or "") for b in (raw.get("content") or []) if isinstance(b, dict)).strip()
        usage = raw.get("usage") or {}
        total_t = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        return {
            "choices": [{"message": {"role": "assistant", "content": txt}}],
            "usage": {"total_tokens": total_t, "cost": (total_t / 1_000_000.0) * 0.50},
        }

    if provider == "ollama":
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": messages,
        }
        if response_format is not None:
            payload["format"] = "json"
        raw = _http_json_post(url, payload, {}, model=model, timeout_s=max(timeout_s, 30.0))
        txt = str((raw.get("message") or {}).get("content") or "").strip()
        return {
            "choices": [{"message": {"role": "assistant", "content": txt}}],
            "usage": {"total_tokens": 0, "cost": 0.0},
        }

    raise OpenRouterError(f"Unsupported detector provider: {provider}", 400)


def test_openrouter_key(api_key: str, base_url: str = DEFAULT_OPENROUTER_URL) -> dict[str, Any]:
    """Verify an OpenRouter API key via GET /models."""
    key = api_key.strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key.", 400)
    url = f"{(base_url or DEFAULT_OPENROUTER_URL).rstrip('/')}/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            count = len(data.get("data") or []) if isinstance(data, dict) else 0
            return {"ok": True, "models_available": count}
    except urllib.error.HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = str(err)
        raise OpenRouterError(f"OpenRouter key rejected ({err.code}): {detail}", int(err.code or 401)) from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise OpenRouterError(f"OpenRouter check failed: {err}", 502) from err


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def compute_bscore(actual_suffix: str, continuations: list[str]) -> float:
    """Compute DNA-GPT BScore using content-word 1/2-grams, all-word 2/3-grams, and LCS."""
    actual_tokens = [w.lower() for w in _WORD_RE.findall(actual_suffix)]
    actual_content = [w for w in actual_tokens if w not in _BSCORE_STOPWORDS and len(w) >= 3]
    if len(actual_tokens) < 4 or not continuations:
        return 0.0

    best_scores: list[float] = []

    for cont in continuations:
        gen_tokens = [w.lower() for w in _WORD_RE.findall(cont)]
        if len(gen_tokens) < 3:
            continue
        gen_content = [w for w in gen_tokens if w not in _BSCORE_STOPWORDS and len(w) >= 3]

        components: list[tuple[float, float]] = []
        # Content-word 1-gram & 2-gram overlap (captures topic/vocabulary continuation without stopword noise)
        if actual_content and gen_content:
            c1_a, c1_g = _ngrams(actual_content, 1), _ngrams(gen_content, 1)
            if c1_a and c1_g:
                components.append((len(c1_a & c1_g) / float(min(len(c1_a), len(c1_g))), 0.35))
            c2_a, c2_g = _ngrams(actual_content, 2), _ngrams(gen_content, 2)
            if c2_a and c2_g:
                components.append((len(c2_a & c2_g) / float(min(len(c2_a), len(c2_g))), 0.25))

        # All-word 2-gram & 3-gram overlap (captures phrasal transitions)
        for n, w in ((2, 0.20), (3, 0.10)):
            a_set, g_set = _ngrams(actual_tokens, n), _ngrams(gen_tokens, n)
            if a_set and g_set:
                components.append((len(a_set & g_set) / float(min(len(a_set), len(g_set))), w))

        ngram_total = sum(val * wt for val, wt in components)
        wt_total = sum(wt for _, wt in components)
        norm_ngram = (ngram_total / wt_total) if wt_total > 0 else 0.0

        seq_ratio = difflib.SequenceMatcher(None, actual_tokens[:60], gen_tokens[:60]).ratio()
        combined = min(1.0, 0.72 * norm_ngram + 0.28 * seq_ratio)
        best_scores.append(combined)

    if not best_scores:
        return 0.0

    peak = max(best_scores)
    mean_top = sum(sorted(best_scores, reverse=True)[:2]) / float(min(2, len(best_scores)))
    raw = 0.6 * peak + 0.4 * mean_top
    calibrated = 1.0 / (1.0 + math.exp(-15.0 * (raw - 0.13)))
    return round(min(1.0, max(0.0, calibrated)), 4)


def _extract_entropy_from_choice(choice: dict[str, Any]) -> float | None:
    lp_obj = choice.get("logprobs")
    if not isinstance(lp_obj, dict):
        return None
    content_lps = lp_obj.get("content")
    if not isinstance(content_lps, list) or not content_lps:
        return None
    entropies: list[float] = []
    for tok in content_lps[:40]:
        if not isinstance(tok, dict):
            continue
        top_lps = tok.get("top_logprobs")
        if isinstance(top_lps, list) and top_lps:
            h = 0.0
            for item in top_lps:
                lp = float(item.get("logprob") or -10.0)
                p = min(1.0, max(1e-9, math.exp(lp)))
                h -= p * math.log(p)
            entropies.append(h)
        elif "logprob" in tok:
            lp = float(tok.get("logprob") or -1.0)
            entropies.append(max(0.0, -lp))
    if not entropies:
        return None
    return round(sum(entropies) / float(len(entropies)), 3)


def _extract_usage_cost(resp: dict[str, Any]) -> tuple[int, float]:
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return 0, 0.0
    prompt_t = int(usage.get("prompt_tokens") or 0)
    comp_t = int(usage.get("completion_tokens") or 0)
    total_t = int(usage.get("total_tokens") or (prompt_t + comp_t))
    explicit_cost = usage.get("cost")
    if explicit_cost is not None:
        return total_t, float(explicit_cost)
    return total_t, (total_t / 1_000_000.0) * 0.14


def _score_single_window_dna(
    segment: PreprocessedSegment,
    provider_cfg: dict[str, str],
    models: list[str],
) -> dict[str, Any]:
    words = segment.text.split()
    if len(words) < 20:
        return {"dna_bscore": None, "dna_entropy": None, "tokens": 0, "cost_usd": 0.0}

    provider = provider_cfg.get("provider", "openrouter")
    split_ratios = (0.40, 0.65)
    bscores: list[float] = []
    entropies: list[float] = []
    total_tokens = 0
    total_cost = 0.0

    for idx_ratio, ratio in enumerate(split_ratios):
        split_idx = max(8, min(len(words) - 6, int(len(words) * ratio)))
        prefix = " ".join(words[:split_idx])
        actual_suffix = " ".join(words[split_idx:])
        max_comp_tokens = min(110, max(32, int(len(words[split_idx:]) * 1.4)))

        target_models = models[:2] if (provider == "openrouter" and idx_ratio == 0) else models[:1]
        continuations: list[str] = []

        for model in target_models:
            try:
                resp = call_provider_chat(
                    provider_cfg=provider_cfg,
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Continue the text below seamlessly from exactly where it stops, matching its exact tone and register. Output only the continuation.",
                        },
                        {"role": "user", "content": prefix},
                    ],
                    temperature=0.0,
                    max_tokens=max_comp_tokens,
                    logprobs=(provider in ("openrouter", "openai")),
                    timeout_s=18.0,
                )
            except OpenRouterError:
                try:
                    resp = call_provider_chat(
                        provider_cfg=provider_cfg,
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Continue the text below seamlessly from exactly where it stops. Output only the continuation.",
                            },
                            {"role": "user", "content": prefix},
                        ],
                        temperature=0.0,
                        max_tokens=max_comp_tokens,
                        logprobs=False,
                        timeout_s=18.0,
                    )
                except OpenRouterError:
                    continue

            choices = resp.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                content = str(msg.get("content") or "").strip()
                if content:
                    continuations.append(content)
                ent = _extract_entropy_from_choice(choices[0])
                if ent is not None:
                    entropies.append(ent)
            tks, cost = _extract_usage_cost(resp)
            total_tokens += tks
            total_cost += cost

        if continuations:
            bscores.append(compute_bscore(actual_suffix, continuations))

    if not bscores:
        return {"dna_bscore": None, "dna_entropy": None, "tokens": total_tokens, "cost_usd": total_cost}

    avg_bscore = round(sum(bscores) / float(len(bscores)), 4)
    avg_entropy = round(sum(entropies) / float(len(entropies)), 3) if entropies else None
    return {
        "dna_bscore": avg_bscore,
        "dna_entropy": avg_entropy,
        "tokens": total_tokens,
        "cost_usd": total_cost,
    }


def run_dna_regeneration(
    segments: list[PreprocessedSegment],
    api_key: str = "",
    base_url: str = DEFAULT_OPENROUTER_URL,
    regen_models_csv: str = DEFAULT_REGEN_MODELS,
    max_windows: int = 8,
    provider_cfg: dict[str, str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run Stage 2 DNA-GPT prefix continuations across non-excluded windows in parallel."""
    cfg = provider_cfg or {"provider": "openrouter", "api_key": api_key, "base_url": base_url}
    eff_key = (cfg.get("api_key") or "").strip()
    if cfg.get("provider") != "ollama" and not eff_key:
        return {}

    models = [m.strip() for m in (regen_models_csv or cfg.get("regen_models") or DEFAULT_REGEN_MODELS).split(",") if m.strip()]
    if not models:
        return {}

    target_indices = [
        idx for idx, seg in enumerate(segments) if not seg.is_excluded_reference and seg.word_count >= 20
    ][:max_windows]

    if not target_indices:
        return {}

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(target_indices))) as pool:
        futures = {
            pool.submit(_score_single_window_dna, segments[idx], cfg, models): idx
            for idx in target_indices
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = {"dna_bscore": None, "dna_entropy": None, "tokens": 0, "cost_usd": 0.0}
    return results
