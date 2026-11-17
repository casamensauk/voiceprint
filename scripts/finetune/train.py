#!/usr/bin/env python3
"""Train a LoRA adapter on the prepared dataset with Apple MLX (mlx-lm).

Dry run by default: prints the exact command and the numbers, runs nothing.
  --setup   create data/finetune/.venv and install the pinned mlx-lm (uv)
  --trial   40 iterations into adapter-trial, then estimate the full run
  --run     the full training; output also appended to train.log
The first --trial/--run downloads the base model (about 6.8 GB) from huggingface.co.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

MLX_LM_VERSION = "0.32.2"
DEFAULT_MODEL = "mlx-community/gemma-4-12B-it-4bit"
MODEL_SIZES = {DEFAULT_MODEL: "about 6.8 GB"}
TRIAL_ITERS = 40
STEPS_PER_EVAL = 200
SAVE_EVERY = 200
MAX_SEQ_LENGTH = 1024

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# mlx-lm 0.32.2 prints rows: "  <iter>    <loss> <arrow>    <tokens/sec>    <trained tokens>k"
_ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▼▲]?\s+([\d,]+)\s+([\d.]+)k\s*$")
# older mlx-lm: "Iter 10: Train loss 2.1, Learning Rate ..., It/sec 0.5, Tokens/sec 120.3, ..."
_CLASSIC_RE = re.compile(r"Iter\s+(\d+):.*?It/sec\s+([\d.]+).*?Tokens/sec\s+([\d.]+)")


def compute_iters(train_examples: int, epochs: float, batch_size: int) -> int:
    return max(1, math.ceil(epochs * train_examples / batch_size))


def build_command(layout: common.Layout, model: str, iters: int, batch_size: int, adapter_dir: Path, trial: bool) -> list[str]:
    """The `mlx_lm lora` command, run from the work dir (so `dataset` and `adapter*` are relative)."""
    return [
        str(layout.venv_python), "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--fine-tune-type", "lora",
        "--data", "dataset",
        "--adapter-path", adapter_dir.name,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--mask-prompt",
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--val-batches", "25",
        "--steps-per-eval", str(iters + 1 if trial else STEPS_PER_EVAL),
        "--steps-per-report", "5" if trial else "10",
        "--save-every", str(iters if trial else SAVE_EVERY),
        "--seed", "0",
    ]


def parse_speed(log: str) -> dict:
    """Pull tokens/sec and iterations/sec out of mlx-lm's training log. Empty dict when nothing parses."""
    log = _ANSI.sub("", log)
    pairs: list[tuple[float, float | None]] = []  # (tokens/sec, iterations/sec)
    for line in log.splitlines():
        m = _CLASSIC_RE.search(line)
        if m:
            pairs.append((float(m.group(3)), float(m.group(2))))
            continue
        m = _ROW_RE.match(line)
        if m:
            it, tok_s, trained_k = int(m.group(1)), float(m.group(3).replace(",", "")), float(m.group(4))
            # iterations/sec = tokens/sec divided by the tokens trained per iteration
            pairs.append((tok_s, tok_s / (trained_k * 1000.0 / it) if tok_s > 0 and trained_k > 0 else None))
    if not pairs:
        return {}
    if len(pairs) > 1:  # the first report includes warm-up
        pairs = pairs[1:]
    its = [p[1] for p in pairs if p[1]]
    return {
        "tokens_per_sec": sum(p[0] for p in pairs) / len(pairs),
        "iters_per_sec": (sum(its) / len(its)) if its else None,
    }


def _fmt_duration(seconds: float) -> str:
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> tuple[int, str]:
    """Run one subprocess, streaming output. Patched out in tests."""
    return common.run_command_streaming(cmd, cwd=cwd, log_path=log_path)


def setup_venv(layout: common.Layout) -> int:
    layout.work.mkdir(parents=True, exist_ok=True)
    if not layout.venv_python.exists():
        code, _ = run_command(["uv", "venv", "--python", "3.12", str(layout.venv)], cwd=layout.work)
        if code:
            print("uv venv failed.")
            return code
    code, out = run_command(
        [str(layout.venv_python), "-c", "import importlib.metadata as m; print(m.version('mlx-lm'))"], cwd=layout.work
    )
    if code == 0 and out.strip().splitlines()[-1:] == [MLX_LM_VERSION]:
        print(f"mlx-lm {MLX_LM_VERSION} already installed in {layout.venv}")
        return 0
    code, _ = run_command(
        ["uv", "pip", "install", "--python", str(layout.venv_python), f"mlx-lm=={MLX_LM_VERSION}"], cwd=layout.work
    )
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=None, help="Base directory (default data/finetune)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Base model (Hugging Face repo or local path)")
    ap.add_argument("--epochs", type=float, default=3, help="Passes over the training set (default 3)")
    ap.add_argument("--batch-size", type=int, default=2, help="Examples per step (default 2)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--setup", action="store_true", help="Create the venv and install mlx-lm")
    mode.add_argument("--trial", action="store_true", help=f"Run {TRIAL_ITERS} iterations and estimate the full run")
    mode.add_argument("--run", action="store_true", help="Run the full training")
    args = ap.parse_args(argv)
    layout = common.Layout(args.work_dir)

    if args.setup:
        return setup_venv(layout)

    manifest = {}
    if layout.manifest.exists():
        manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    n_train = int((manifest.get("examples") or {}).get("train") or 0)
    iters = compute_iters(n_train, args.epochs, args.batch_size)
    cmd = build_command(layout, args.model, iters, args.batch_size, layout.adapter, trial=False)

    if not (args.trial or args.run):
        size = MODEL_SIZES.get(args.model, "size unknown")
        print("DRY RUN (nothing run). Steps: --setup, then --trial, then --run.")
        print(f"Model: {args.model}")
        print(f"Downloaded from: https://huggingface.co/{args.model} on first use ({size})")
        print(f"Training examples: {n_train or 'unknown (run prepare.py --run first)'}")
        print(f"Iterations: {iters} = {args.epochs:g} epochs x {n_train} examples / batch size {args.batch_size}")
        print(f"Command (run from {layout.work}):")
        print("  " + " ".join(cmd))
        return 0

    if not layout.venv_python.exists():
        print(f"No venv at {layout.venv}. Run: train.py --setup")
        return 1
    if not n_train or not (layout.dataset / "train.jsonl").exists():
        print(f"No dataset at {layout.dataset}. Run prepare.py --run first.")
        return 1

    if args.trial:
        cmd = build_command(layout, args.model, TRIAL_ITERS, args.batch_size, layout.adapter_trial, trial=True)
        code, out = run_command(cmd, cwd=layout.work)
        if code:
            print(f"mlx-lm exited with code {code}.")
            return code
        speed = parse_speed(out)
        if not speed:
            print("Could not find speed figures in the mlx-lm output; see the lines above.")
            return 1
        print(f"Trial speed: {speed['tokens_per_sec']:.0f} tokens/sec", end="")
        if speed.get("iters_per_sec"):
            full = iters / speed["iters_per_sec"]
            print(f", {speed['iters_per_sec']:.3f} iterations/sec")
            print(f"Estimate for the full run: {iters} iterations, about {_fmt_duration(full)}")
        else:
            print()
        return 0

    layout.work.mkdir(parents=True, exist_ok=True)
    code, _ = run_command(cmd, cwd=layout.work, log_path=layout.train_log)
    if code:
        print(f"mlx-lm exited with code {code}. See {layout.train_log}")
        return code
    print(f"Adapter saved to {layout.adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
