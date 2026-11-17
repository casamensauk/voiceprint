"""Shared helpers for the fine-tuning experiment scripts (standard library only)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.rewriter import strip_placeholders  # noqa: E402,F401  (re-exported for the scripts)
from engine.sanitize import strip_hidden_characters  # noqa: E402,F401

DEFAULT_WORK_DIR = ROOT / "data" / "finetune"
APP_SUPPORT_DB = Path.home() / "Library" / "Application Support" / "VoicePrint" / "voiceprint.db"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_NOTES_MODEL = "gemma4:12b"
MAX_CHUNK_WORDS = 400
SECONDS_PER_NOTE = 8

KINDS = {"gmail": "an email", "google_chat": "a chat message", "document": "a piece of writing"}

NOTES_PROMPT = (
    "List the content of the text below as 3 to 8 neutral bullet points. "
    "Include facts, requests and decisions only. Do not copy phrases from the text. "
    "Do not mention any greeting or sign-off, and do not describe the tone or style. "
    "Reply with the bullet points only, one per line, each starting with '- '.\n\nTEXT:\n"
)


def word_count(text: str) -> int:
    return len(text.split())


def round10(n: int) -> int:
    return int(round(n / 10.0)) * 10


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_db(arg: Path | None) -> Path:
    """Same default rule as scripts/try_models.py: the app's db when it exists, else the repo default."""
    if arg:
        return Path(arg)
    from engine.db import DEFAULT_DB_PATH

    return APP_SUPPORT_DB if APP_SUPPORT_DB.exists() else Path(DEFAULT_DB_PATH)


def open_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------- work dir layout


class Layout:
    """All files of the experiment live under one work directory."""

    def __init__(self, work_dir: Path | None = None) -> None:
        self.work = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
        self.dataset = self.work / "dataset"
        self.notes = self.work / "notes.jsonl"
        self.venv = self.work / ".venv"
        self.venv_python = self.venv / "bin" / "python"
        self.adapter = self.work / "adapter"
        self.adapter_trial = self.work / "adapter-trial"
        self.train_log = self.work / "train.log"
        self.outputs = self.work / "outputs"
        self.manifest = self.dataset / "manifest.json"


# ---------------------------------------------------------------- prompts


def build_user_prompt(kind: str, words: int, notes: list[str]) -> str:
    bullets = "\n".join(f"- {n}" for n in notes)
    return f"Write {kind} of about {round10(words)} words in my own voice, covering these points:\n{bullets}"


# ---------------------------------------------------------------- content notes (local Ollama)


def ollama_chat(model: str, prompt: str, url: str = OLLAMA_URL, timeout: float = 600.0) -> str:
    """One non-streaming chat call to the LOCAL Ollama server. Patched out in tests."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local server)
        data = json.loads(resp.read().decode("utf-8"))
    return str((data.get("message") or {}).get("content") or "")


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")


def parse_notes(reply: str) -> list[str] | None:
    """Return the bullet points (max 8) when the reply is valid, else None."""
    if word_count(reply or "") >= 200:
        return None
    bullets = []
    for line in (reply or "").splitlines():
        m = _BULLET_RE.match(line)
        if m:
            bullets.append(m.group(1).strip())
    if len(bullets) < 2:
        return None
    return bullets[:8]


def make_notes(text: str, model: str) -> list[str] | None:
    """Ask Ollama for content notes; validate; retry once; None on failure."""
    for _ in range(2):
        try:
            notes = parse_notes(ollama_chat(model, NOTES_PROMPT + text))
        except Exception:
            notes = None
        if notes:
            return notes
    return None


# ---------------------------------------------------------------- text helpers

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_long_text(text: str, limit: int = MAX_CHUNK_WORDS) -> list[str]:
    """Chunks of at most `limit` words, at paragraph boundaries, else sentence boundaries."""
    text = text.strip()
    if word_count(text) <= limit:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[tuple[str, str]] = []  # (text, separator used before it when joining)
    for para in paragraphs:
        if word_count(para) <= limit:
            pieces.append((para, "\n\n"))
            continue
        first = True
        for sent in _SENTENCE_SPLIT.split(para):
            words = sent.split()
            if len(words) <= limit:
                pieces.append((sent, "\n\n" if first else " "))
                first = False
            else:  # a single enormous sentence: the only case that cuts mid-sentence
                for i in range(0, len(words), limit):
                    pieces.append((" ".join(words[i : i + limit]), "\n\n" if first else " "))
                    first = False
    chunks: list[str] = []
    cur = ""
    cur_words = 0
    for piece, sep in pieces:
        w = word_count(piece)
        if cur and cur_words + w <= limit:
            cur += sep + piece
            cur_words += w
        else:
            if cur:
                chunks.append(cur)
            cur, cur_words = piece, w
    if cur:
        chunks.append(cur)
    return chunks


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def run_command_streaming(cmd: list[str], cwd: Path | None = None, log_path: Path | None = None) -> tuple[int, str]:
    """Run a command, echo its output live (and append to log_path); return (code, full output)."""
    import subprocess

    lines: list[str] = []
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log:
                log.write(line)
            lines.append(line)
        return proc.wait(), "".join(lines)
    finally:
        if log:
            log.close()
