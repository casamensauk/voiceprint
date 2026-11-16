"""Stage 2: OpenRouter DNA-GPT black-box prefix-truncation & regeneration BScore scorer."""

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


class OpenRouterError(Exception):
    """Error communicating with OpenRouter API."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def openrouter_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
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
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/casamensauk/voiceprint",
        "X-Title": "VoicePrint AI Detector",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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
        raise OpenRouterError(f"OpenRouter error ({err.code}) for {model}: {detail}", int(err.code or 502)) from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise OpenRouterError(f"OpenRouter connection error for {model}: {err}", 502) from err


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
    """Compute DNA-GPT BScore via weighted n-gram overlap & LCS similarity across continuations."""
    actual_tokens = [w.lower() for w in _WORD_RE.findall(actual_suffix)]
    if len(actual_tokens) < 4 or not continuations:
        return 0.0

    best_scores: list[float] = []
    weights = ((1, 0.20), (2, 0.30), (3, 0.30), (4, 0.20))

    for cont in continuations:
        gen_tokens = [w.lower() for w in _WORD_RE.findall(cont)]
        if len(gen_tokens) < 3:
            continue
        ngram_score = 0.0
        weight_sum = 0.0
        for n, w in weights:
            a_set = _ngrams(actual_tokens, n)
            g_set = _ngrams(gen_tokens, n)
            if not a_set or not g_set:
                continue
            overlap = len(a_set & g_set) / float(min(len(a_set), len(g_set)))
            ngram_score += overlap * w
            weight_sum += w
        norm_ngram = (ngram_score / weight_sum) if weight_sum > 0 else 0.0
        seq_ratio = difflib.SequenceMatcher(None, actual_tokens[:60], gen_tokens[:60]).ratio()
        combined = min(1.0, 0.72 * norm_ngram + 0.28 * seq_ratio)
        best_scores.append(combined)

    if not best_scores:
        return 0.0
    # Calibration curve: DNA-GPT BScore > 0.26 strongly signals LLM distribution match
    peak = max(best_scores)
    mean_top = sum(sorted(best_scores, reverse=True)[:2]) / float(min(2, len(best_scores)))
    raw = 0.6 * peak + 0.4 * mean_top
    calibrated = 1.0 / (1.0 + math.exp(-11.0 * (raw - 0.24)))
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
    # Approximate ~$0.14 per 1M tokens for fast OpenRouter regeneration models
    return total_t, (total_t / 1_000_000.0) * 0.14


def _score_single_window_dna(
    segment: PreprocessedSegment,
    api_key: str,
    base_url: str,
    models: list[str],
) -> dict[str, Any]:
    words = segment.text.split()
    if len(words) < 20:
        return {"dna_bscore": None, "dna_entropy": None, "tokens": 0, "cost_usd": 0.0}

    split_idx = max(10, int(len(words) * 0.50))
    prefix = " ".join(words[:split_idx])
    actual_suffix = " ".join(words[split_idx:])
    max_comp_tokens = min(110, max(32, int(len(words[split_idx:]) * 1.4)))

    continuations: list[str] = []
    entropies: list[float] = []
    total_tokens = 0
    total_cost = 0.0

    for model in models[:2]:
        try:
            resp = openrouter_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Continue the text below seamlessly from exactly where it stops, matching its tone and register. Output only the continuation.",
                    },
                    {"role": "user", "content": prefix},
                ],
                temperature=0.7,
                max_tokens=max_comp_tokens,
                logprobs=True,
                timeout_s=18.0,
            )
        except OpenRouterError:
            # Retry once without logprobs if provider doesn't support top_logprobs
            try:
                resp = openrouter_chat_completion(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Continue the text below seamlessly from exactly where it stops. Output only the continuation.",
                        },
                        {"role": "user", "content": prefix},
                    ],
                    temperature=0.7,
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

    if not continuations:
        return {"dna_bscore": None, "dna_entropy": None, "tokens": total_tokens, "cost_usd": total_cost}

    bscore = compute_bscore(actual_suffix, continuations)
    avg_entropy = round(sum(entropies) / float(len(entropies)), 3) if entropies else None
    return {
        "dna_bscore": bscore,
        "dna_entropy": avg_entropy,
        "tokens": total_tokens,
        "cost_usd": total_cost,
    }


def run_dna_regeneration(
    segments: list[PreprocessedSegment],
    api_key: str,
    base_url: str = DEFAULT_OPENROUTER_URL,
    regen_models_csv: str = DEFAULT_REGEN_MODELS,
    max_windows: int = 8,
) -> dict[int, dict[str, Any]]:
    """Run Stage 2 DNA-GPT prefix continuations across non-excluded windows in parallel."""
    models = [m.strip() for m in (regen_models_csv or DEFAULT_REGEN_MODELS).split(",") if m.strip()]
    if not models or not api_key.strip():
        return {}

    target_indices = [
        idx for idx, seg in enumerate(segments) if not seg.is_excluded_reference and seg.word_count >= 20
    ][:max_windows]

    if not target_indices:
        return {}

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(target_indices))) as pool:
        futures = {
            pool.submit(_score_single_window_dna, segments[idx], api_key, base_url, models): idx
            for idx in target_indices
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = {"dna_bscore": None, "dna_entropy": None, "tokens": 0, "cost_usd": 0.0}
    return results
