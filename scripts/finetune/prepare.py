#!/usr/bin/env python3
"""Build the fine-tuning training set from your own imported writing.

Dry run by default: applies the filters and chunking, prints every count and a time estimate,
writes nothing and calls nothing. --run asks a LOCAL Ollama model for content notes per chunk
(no paid service) and writes dataset/train.jsonl, valid.jsonl, test.jsonl and manifest.json.
Texts are never printed.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402

SOURCES = ("gmail", "google_chat", "document")
MIN_WORDS = 20
MAX_CHAT_MAIL_WORDS = 1500

_SIGNATURE_RE = re.compile(r"^[ \t>]*(sent from my\b.*|get outlook for\b.*)$", re.IGNORECASE | re.MULTILINE)
_AUTOMATED_RE = re.compile(
    r"unsubscribe|do not reply|do-not-reply|donotreply|no-reply|noreply|out of office|automatically generated",
    re.IGNORECASE,
)
_FORWARDED_RE = re.compile(
    r"-{2,}\s*original message\s*-{2,}"
    r"|-{2,}\s*forwarded message\s*-{2,}"
    r"|begin forwarded message"
    r"|^from:[^\n]*\n\s*(?:sent|date):"
    r"|^on [^\n]{5,120} wrote:\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def dedupe_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:200]


def clean_text(text: str) -> str:
    text = _SIGNATURE_RE.sub("", text or "")
    text = common.strip_placeholders(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_samples(db_path: Path, sources: tuple[str, ...]) -> list[tuple[str, str]]:
    marks = ",".join("?" for _ in sources)
    with closing(common.open_readonly(db_path)) as conn:
        rows = conn.execute(
            f"SELECT source_type, content FROM samples WHERE source_type IN ({marks}) ORDER BY id", sources
        ).fetchall()
    return [(r["source_type"], r["content"] or "") for r in rows]


def build_chunks(samples: list[tuple[str, str]]) -> tuple[list[dict], Counter]:
    """Apply the filters and chunking. Returns (chunks, counts)."""
    counts: Counter = Counter()
    seen: set[str] = set()
    chunks: list[dict] = []
    for source, raw in samples:
        counts[f"samples_{source}"] += 1
        text = clean_text(raw)
        words = common.word_count(text)
        if words < MIN_WORDS:
            counts["dropped_under_20_words"] += 1
            continue
        if source in ("gmail", "google_chat") and words > MAX_CHAT_MAIL_WORDS:
            counts["dropped_over_1500_words"] += 1
            continue
        if _AUTOMATED_RE.search(text):
            counts["dropped_automated"] += 1
            continue
        if _FORWARDED_RE.search(text):
            counts["dropped_forwarded_or_original_message"] += 1
            continue
        key = dedupe_key(text)
        if key in seen:
            counts["dropped_near_duplicate"] += 1
            continue
        seen.add(key)
        counts["kept_samples"] += 1
        pieces = common.split_long_text(text)
        if len(pieces) > 1:
            counts["samples_split_into_chunks"] += 1
        for piece in pieces:
            if common.word_count(piece) < MIN_WORDS:
                counts["dropped_short_chunk"] += 1
                continue
            chunks.append({"source": source, "text": piece, "hash": common.sha256_text(piece)})
    counts["chunks"] = len(chunks)
    return chunks, counts


def split_for(chunk_hash: str) -> str:
    bucket = int(chunk_hash[:8], 16) % 100
    return "train" if bucket < 90 else "valid" if bucket < 95 else "test"


def example_for(chunk: dict, notes: list[str]) -> dict:
    prompt = common.build_user_prompt(common.KINDS[chunk["source"]], common.word_count(chunk["text"]), notes)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chunk["text"]},
        ]
    }


def _eta(done: int, total: int, started: float) -> str:
    if not done:
        return "?"
    remaining = (time.monotonic() - started) / done * (total - done)
    return f"{int(remaining // 60)}m{int(remaining % 60):02d}s"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="SQLite db holding your samples")
    ap.add_argument("--work-dir", type=Path, default=None, help="Base directory (default data/finetune)")
    ap.add_argument("--no-documents", action="store_true", help="Leave out samples imported as documents")
    ap.add_argument("--notes-model", default=common.DEFAULT_NOTES_MODEL, help="Local Ollama model for content notes")
    ap.add_argument("--limit", type=int, default=None, help="Make notes for only the first N uncached chunks")
    ap.add_argument("--run", action="store_true", help="Do the work (local Ollama calls, writes files)")
    args = ap.parse_args(argv)

    layout = common.Layout(args.work_dir)
    db_path = common.resolve_db(args.db)
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1
    sources = tuple(s for s in SOURCES if not (args.no_documents and s == "document"))
    chunks, counts = build_chunks(load_samples(db_path, sources))

    cache = {r["hash"]: r["notes"] for r in common.read_jsonl(layout.notes) if r.get("hash") and r.get("notes")}
    pending = [c for c in chunks if c["hash"] not in cache]
    cached = len(chunks) - len(pending)
    to_process = pending[: args.limit] if args.limit is not None else pending

    print("DRY RUN (nothing written, no model called). Add --run to do the work." if not args.run else "RUN")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"  words in chunks: {sum(common.word_count(c['text']) for c in chunks)}")
    print(f"  notes cached: {cached}, still needed: {len(pending)}")
    est = len(to_process) * common.SECONDS_PER_NOTE
    print(
        f"  estimate: {len(to_process)} notes x {common.SECONDS_PER_NOTE}s = "
        f"{est // 3600}h{est % 3600 // 60:02d}m (local Ollama model {args.notes_model})"
    )
    if not args.run:
        return 0

    layout.work.mkdir(parents=True, exist_ok=True)
    failed = 0
    started = time.monotonic()
    with open(layout.notes, "a", encoding="utf-8") as fh:
        for i, chunk in enumerate(to_process, 1):
            notes = common.make_notes(chunk["text"], args.notes_model)
            if notes is None:
                failed += 1
            else:
                cache[chunk["hash"]] = notes
                fh.write(json.dumps({"hash": chunk["hash"], "notes": notes}, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(to_process)} chunks, {failed} failed, ETA {_eta(i, len(to_process), started)}")
    print(f"Notes made: {len(to_process) - failed}, failed and skipped: {failed}")

    layout.dataset.mkdir(parents=True, exist_ok=True)
    files = {name: [] for name in ("train", "valid", "test")}
    per_source: dict[str, Counter] = {s: Counter() for s in SOURCES}
    total_words = 0
    without_notes = 0
    for chunk in chunks:
        notes = cache.get(chunk["hash"])
        if not notes:
            without_notes += 1
            continue
        split = split_for(chunk["hash"])
        files[split].append(json.dumps(example_for(chunk, notes), ensure_ascii=False))
        per_source[chunk["source"]][split] += 1
        total_words += common.word_count(chunk["text"])
    for name, lines in files.items():
        (layout.dataset / f"{name}.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    manifest = {
        "examples": {name: len(lines) for name, lines in files.items()},
        "per_source": {s: dict(c) for s, c in per_source.items() if c},
        "total_words": total_words,
        "approx_tokens": int(total_words * 1.4),
        "chunks_without_notes": without_notes,
        "filters": {k: v for k, v in counts.items() if k.startswith("dropped")},
    }
    layout.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {layout.dataset}: train {len(files['train'])}, valid {len(files['valid'])}, "
        f"test {len(files['test'])}; about {manifest['approx_tokens']} tokens; {without_notes} chunks have no notes yet."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
