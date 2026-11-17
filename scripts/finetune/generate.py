#!/usr/bin/env python3
"""Write test answers with the trained adapter and sanity-check them.

Dry run by default: lists the drafts (word counts only) and what would run.
--run makes content notes with the LOCAL Ollama model and generates with mlx-lm (no paid service).
Draft texts are never written to disk or printed; output texts are printed only with --show.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

from engine.analyzer import analyze_text  # noqa: E402

MIN_DRAFT_CHARS = 50
DEFAULT_BASE_MODEL = "mlx-community/gemma-4-12B-it-4bit"
DEFAULT_HISTORY_LATEST = 5
KIND = "a piece of writing"

_GREETING_RE = re.compile(r"^(hi|hello|hey|dear|good (morning|afternoon|evening))\b[^\n]{0,40}$", re.IGNORECASE)
_SIGNOFF_RE = re.compile(
    r"^(best|best wishes|kind regards|warm regards|regards|many thanks|thanks|thank you|cheers|sincerely|yours|"
    r"all the best|speak soon|take care)\b[^\n]{0,30}$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# ---------------------------------------------------------------- drafts


def load_drafts(files: list[Path], history_latest: int | None, db_path: Path | None) -> list[tuple[str, str]]:
    """Return [(source description, text)], skipping drafts under MIN_DRAFT_CHARS."""
    drafts: list[tuple[str, str]] = []
    for path in files:
        p = Path(path)
        if not p.is_file():
            raise ValueError(f"No such file: {p}")
        drafts.append((f"file {p.name}", p.read_text(encoding="utf-8")))
    if history_latest:
        if db_path is None or not Path(db_path).exists():
            raise ValueError(f"No database at {db_path}")
        with closing(common.open_readonly(db_path)) as conn:
            rows = conn.execute("SELECT id, input_text FROM rewrite_history ORDER BY id DESC").fetchall()
        seen: set[str] = set()
        picked = 0
        for r in rows:
            text = (r["input_text"] or "").strip()
            if text in seen:
                continue
            seen.add(text)
            if len(text) < MIN_DRAFT_CHARS:
                continue
            drafts.append((f"rewrite_history id {r['id']}", text))
            picked += 1
            if picked >= history_latest:
                break
    return [(s, t.strip()) for s, t in drafts if len(t.strip()) >= MIN_DRAFT_CHARS]


# ---------------------------------------------------------------- mlx-lm

# A thinking model may still open with a thought channel; never let it into an answer.
_THOUGHT_RE = re.compile(r"<\|channel>thought.*?(?:<channel\|>|\Z)", re.DOTALL)


def build_generate_command(
    layout: common.Layout, model: str, adapter: Path | None, max_tokens: int, temp: float, seed: int
) -> list[str]:
    """The `mlx_lm generate` command; the prompt is passed on stdin (`--prompt -`) and the chat template is applied."""
    cmd = [
        str(layout.venv_python), "-m", "mlx_lm", "generate",
        "--model", model,
        "--prompt", "-",
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--seed", str(seed),
        "--verbose", "True",
        # Thinking off: the prompt then ends exactly as the rendered training rows do.
        "--chat-template-config", json.dumps({"enable_thinking": False}),
    ]
    if adapter is not None:
        cmd += ["--adapter-path", str(adapter)]
    return cmd


def run_mlx_generate(cmd: list[str], prompt: str) -> str:
    """Run mlx-lm once and return its raw stdout. Patched out in tests."""
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError(f"mlx_lm generate exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
    return proc.stdout


def parse_generated(stdout: str) -> str:
    """Take only the generated text: what sits between mlx-lm's two '==========' lines."""
    lines = stdout.splitlines()
    marks = [i for i, line in enumerate(lines) if line.strip() == "=" * 10]
    text = "\n".join(lines[marks[0] + 1 : marks[1]]) if len(marks) >= 2 else stdout
    return _THOUGHT_RE.sub("", text).strip()


# ---------------------------------------------------------------- metrics


def _facts(draft: str) -> set[str]:
    facts = {n.lower() for n in _NUMBER_RE.findall(draft)}
    for m in _WORD_RE.finditer(draft):
        w = m.group(0)
        if not w[0].isupper():
            continue
        before = draft[: m.start()].rstrip(" \t\"'(")
        if not before or before[-1] in ".!?\n":
            continue  # sentence-initial
        facts.add(w.lower().removesuffix("'s"))
    return facts


def fact_coverage(draft: str, output: str) -> float | None:
    facts = _facts(draft)
    if not facts:
        return None
    out_tokens = set(_NUMBER_RE.findall(output.lower())) | {w.lower() for w in _WORD_RE.findall(output)}
    out_tokens |= {t.removesuffix("'s") for t in out_tokens}
    return round(sum(1 for f in facts if f in out_tokens) / len(facts), 3)


def longest_copied_run(output: str, chunks: list[list[str]]) -> int:
    """Longest run of consecutive words the output shares with any training chunk."""
    out = common.norm_words(output)
    if not out:
        return 0
    out_vocab = set(out)
    best = 0
    for words in chunks:
        if len(words) <= best:
            continue
        positions: dict[str, list[int]] = {}
        for j, w in enumerate(words):
            if w in out_vocab:
                positions.setdefault(w, []).append(j)
        prev: dict[int, int] = {}
        for w in out:
            cur: dict[int, int] = {}
            for j in positions.get(w, ()):
                cur[j] = prev.get(j - 1, 0) + 1
                if cur[j] > best:
                    best = cur[j]
            prev = cur
    return best


def email_like(text: str) -> bool:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    if _GREETING_RE.match(lines[0]):
        return True
    return any(_SIGNOFF_RE.match(ln) for ln in lines[-2:])


def compute_metrics(draft: str, output: str, train_chunks: list[list[str]]) -> dict:
    return {
        "words": common.word_count(output),
        "fact_coverage": fact_coverage(draft, output),
        "longest_copied_run": longest_copied_run(output, train_chunks),
        "email_like": email_like(output),
        "inhouse_human_score": analyze_text(output)["human_score"],
    }


def clean_output(text: str) -> str:
    cleaned, _ = common.strip_hidden_characters(text)
    return common.strip_placeholders(cleaned).strip()


def load_train_chunks(layout: common.Layout) -> list[list[str]]:
    chunks = []
    for row in common.read_jsonl(layout.dataset / "train.jsonl"):
        msgs = row.get("messages") or []
        if msgs:
            chunks.append(common.norm_words(msgs[-1].get("content", "")))
    return chunks


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="SQLite db holding rewrite_history")
    ap.add_argument("--work-dir", type=Path, default=None, help="Base directory (default data/finetune)")
    ap.add_argument("--file", type=Path, action="append", default=[], help="Draft text file (repeatable)")
    ap.add_argument("--history-latest", type=int, default=None, help="Use the N most recent distinct rewrite_history inputs")
    ap.add_argument("--notes-model", default=common.DEFAULT_NOTES_MODEL, help="Local Ollama model for content notes")
    ap.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base model the adapter was trained on")
    ap.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (default 0.7)")
    ap.add_argument("--seed", type=int, default=1234, help="Fixed sampling seed")
    ap.add_argument("--baseline", action="store_true", help="Also generate from the base model without the adapter")
    ap.add_argument("--show", action="store_true", help="Print the output texts")
    ap.add_argument("--run", action="store_true", help="Make notes and generate (local only)")
    args = ap.parse_args(argv)

    layout = common.Layout(args.work_dir)
    history_latest = args.history_latest
    if history_latest is None and not args.file:
        history_latest = DEFAULT_HISTORY_LATEST
    db_path = common.resolve_db(args.db) if history_latest else None
    try:
        drafts = load_drafts(args.file, history_latest, db_path)
    except (ValueError, OSError) as err:
        print(str(err))
        return 1
    if not drafts:
        print(f"No usable drafts (need at least {MIN_DRAFT_CHARS} characters).")
        return 1

    runs_per_draft = 2 if args.baseline else 1
    if not args.run:
        print("DRY RUN (nothing run). Add --run to generate.")
        for i, (source, text) in enumerate(drafts, 1):
            print(f"  draft {i}: {common.word_count(text)} words ({source})")
        print(f"Would make {len(drafts)} sets of notes with Ollama model {args.notes_model} (local)")
        print(f"Would run mlx_lm generate {len(drafts) * runs_per_draft} times (adapter{' + baseline' if args.baseline else ''})")
        print(f"Temperature {args.temp}, seed {args.seed}, adapter {layout.adapter}")
        return 0

    if not layout.venv_python.exists():
        print(f"No venv at {layout.venv}. Run: train.py --setup")
        return 1
    if not layout.adapter.exists():
        print(f"No adapter at {layout.adapter}. Run: train.py --run")
        return 1
    train_chunks = load_train_chunks(layout)

    results = []
    for i, (source, draft) in enumerate(drafts, 1):
        words = common.word_count(draft)
        notes = common.make_notes(draft, args.notes_model)
        if notes is None:
            print(f"draft {i}: could not make valid notes, skipped")
            continue
        prompt = common.build_user_prompt(KIND, words, notes)
        max_tokens = int(common.round10(words) * 2.5)
        entry: dict = {"source": source, "draft_words": words, "notes": notes}
        variants = [("adapter", layout.adapter)] + ([("baseline", None)] if args.baseline else [])
        for label, adapter in variants:
            cmd = build_generate_command(layout, args.model, adapter, max_tokens, args.temp, args.seed)
            try:
                text = clean_output(parse_generated(run_mlx_generate(cmd, prompt)))
            except Exception as err:
                entry[label] = {"error": str(err)[:200]}
                continue
            entry[label] = {"text": text, "metrics": compute_metrics(draft, text, train_chunks)}
        results.append(entry)

    layout.outputs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = layout.outputs / f"{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "model": args.model,
                "temperature": args.temp,
                "seed": args.seed,
                "notes_model": args.notes_model,
                "drafts": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"{'draft':<26} {'kind':<9} {'words':>5} {'facts':>6} {'copied':>6} {'email':>5} {'in-house*':>9}")
    for e in results:
        for label in ("adapter", "baseline"):
            v = e.get(label)
            if not v:
                continue
            if "error" in v:
                print(f"{e['source']:<26} {label:<9} ERROR: {v['error']}")
                continue
            m = v["metrics"]
            fc = "n/a" if m["fact_coverage"] is None else f"{m['fact_coverage'] * 100:.0f}%"
            print(
                f"{e['source']:<26} {label:<9} {m['words']:>5} {fc:>6} {m['longest_copied_run']:>6} "
                f"{'yes' if m['email_like'] else 'no':>5} {m['inhouse_human_score']:>9}"
            )
    print("* in-house score only: VoicePrint's own heuristic, not a Pangram result. 'copied' = longest run of words shared with the training set.")
    print(f"Saved: {out_path}")
    if args.show:
        for e in results:
            for label in ("adapter", "baseline"):
                if e.get(label, {}).get("text"):
                    print(f"\n--- {e['source']} / {label} ---\n{e[label]['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
