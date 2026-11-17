#!/usr/bin/env python3
"""Render the chat-format dataset into the exact text the model sees at generation time.

Run INSIDE the mlx-lm venv by train.py (it needs the model's tokenizer). Why it exists:
thinking models such as Gemma 4 end a non-thinking generation prompt with an empty thought
channel, which a plainly rendered training conversation does not contain. Training on plain
chat rows would therefore teach a format the model is never asked for, and mlx-lm's
--mask-prompt offset would hide the first words of every answer. Rendering each example as
`generation prompt + answer + end of turn` keeps training and writing identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROBE = "⁣ANSWER⁣"  # unlikely to be altered by a chat template


def render_example(tokenizer, messages: list[dict]) -> str:
    """Generation prompt (thinking off) + the assistant text + the template's end-of-turn suffix."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("each example needs a final assistant message")
    head, answer = messages[:-1], str(messages[-1].get("content") or "")
    prompt = tokenizer.apply_chat_template(head, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    probe_full = tokenizer.apply_chat_template(head + [{"role": "assistant", "content": PROBE}], tokenize=False)
    suffix = probe_full[probe_full.rindex(PROBE) + len(PROBE):] if PROBE in probe_full else ""
    text = prompt + answer + suffix
    bos = getattr(tokenizer, "bos_token", None)
    # mlx-lm's text dataset calls tokenizer.encode(text). Keep exactly one BOS: drop the
    # literal one only when encoding adds its own.
    if bos and text.startswith(bos) and _encode_adds_bos(tokenizer):
        text = text[len(bos):]
    return text


def _encode_adds_bos(tokenizer) -> bool:
    encode = getattr(tokenizer, "encode", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if encode is None or bos_id is None:
        return False
    ids = encode("x")
    return bool(ids) and ids[0] == bos_id


def render_dir(tokenizer, src: Path, dst: Path) -> dict[str, int]:
    dst.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name in ("train", "valid", "test"):
        src_file = src / f"{name}.jsonl"
        if not src_file.exists():
            continue
        n = 0
        with src_file.open(encoding="utf-8") as fin, (dst / f"{name}.jsonl").open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                text = render_example(tokenizer, json.loads(line)["messages"])
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
        counts[name] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer  # only available inside the venv

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    counts = render_dir(tokenizer, args.src, args.dst)
    sample = (args.dst / "train.jsonl").open(encoding="utf-8").readline()
    ids = tokenizer.encode(json.loads(sample)["text"]) if sample else []
    bos_id = getattr(tokenizer, "bos_token_id", None)
    leading_bos = sum(1 for t in ids[:3] if t == bos_id)
    print(f"Rendered {counts} into {args.dst}; leading BOS tokens after encoding: {leading_bos}")
    return 0 if leading_bos == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
