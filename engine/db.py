"""SQLite persistence layer for VoicePrint samples, DNA rules, edits, and settings."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "voiceprint.db"

CONTEXT_MODES = ("casual_chat", "work_email", "long_form", "social_post")

DEFAULT_SETTINGS: dict[str, str] = {
    "provider": "auto",
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash",
    "openai_api_key": "",
    "openai_model": "gpt-4o-mini",
    "anthropic_api_key": "",
    "anthropic_model": "claude-3-5-haiku-latest",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "active_context_mode": "casual_chat",
    "intensity": "deep",
    "pangram_api_key": "",
    "pangram_model": "pangram-4",
    "pangram_public_link": "false",
    "inspector_engine": "inhouse",
    "openrouter_api_key": "",
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    "ai_detect_regen_models": "meta-llama/llama-3.3-70b-instruct,google/gemini-2.5-flash",
    "ai_detect_arbiter_model": "google/gemini-2.5-flash",
}


def get_db_path() -> Path:
    """Return the active SQLite database path, ensuring its parent directory exists."""
    env_path = os.environ.get("VOICEPRINT_DB_PATH")
    path = Path(env_path) if env_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClosingConnection(sqlite3.Connection):
    """SQLite connection that automatically closes upon exiting a `with` block."""

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row dictionary access."""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, factory=ClosingConnection)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create tables if missing and seed default settings."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_mode TEXT NOT NULL DEFAULT 'casual_chat',
                source_type TEXT NOT NULL DEFAULT 'manual',
                content TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edit_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_mode TEXT NOT NULL,
                original_input TEXT NOT NULL,
                ai_draft TEXT NOT NULL,
                user_final_edit TEXT NOT NULL,
                extracted_rules_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learned_replacements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_mode TEXT NOT NULL,
                avoid_text TEXT NOT NULL,
                prefer_text TEXT NOT NULL,
                times_seen INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT 'auto_diff',
                UNIQUE(context_mode, avoid_text)
            );

            CREATE TABLE IF NOT EXISTS rewrite_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_mode TEXT NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL,
                before_human_score INTEGER,
                after_human_score INTEGER,
                provider_used TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pangram_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type TEXT NOT NULL DEFAULT 'text',
                title TEXT NOT NULL,
                input_text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                headline TEXT NOT NULL,
                fraction_ai REAL NOT NULL DEFAULT 0.0,
                fraction_ai_assisted REAL NOT NULL DEFAULT 0.0,
                fraction_human REAL NOT NULL DEFAULT 0.0,
                max_humanizer_score REAL NOT NULL DEFAULT 0.0,
                dashboard_link TEXT NOT NULL DEFAULT '',
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_detector_cache (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.commit()


def get_settings(db_path: Path | None = None) -> dict[str, str]:
    """Return all application settings merged with defaults."""
    init_db(db_path)
    result = dict(DEFAULT_SETTINGS)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
        for row in rows:
            result[row["key"]] = row["value"]
    return result


def get_public_settings(db_path: Path | None = None) -> dict[str, Any]:
    """Return settings safe for UI rendering with API keys masked."""
    raw = get_settings(db_path)
    public: dict[str, Any] = {}
    for k, v in raw.items():
        if k.endswith("_api_key"):
            public[k] = ""
            public[f"{k}_configured"] = bool(v.strip())
            public[f"{k}_preview"] = (f"{v[:4]}...{v[-4:]}" if len(v) >= 10 else ("••••" if v else ""))
        else:
            public[k] = v
    return public


def update_settings(updates: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    """Persist provided settings keys (ignoring empty masked key updates unless cleared explicitly)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        for key, value in updates.items():
            if key not in DEFAULT_SETTINGS:
                continue
            str_val = "" if value is None else str(value).strip()
            # Keep existing API key if UI submitted empty string without explicit clear flag
            if key.endswith("_api_key") and str_val == "" and not updates.get(f"clear_{key}"):
                continue
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str_val),
            )
        conn.commit()
    return get_public_settings(db_path)


def add_sample(
    content: str,
    context_mode: str = "casual_chat",
    source_type: str = "manual",
    weight: float = 1.0,
    db_path: Path | None = None,
) -> int:
    """Insert a single writing sample and return its row ID."""
    init_db(db_path)
    cleaned = content.strip()
    wc = len(cleaned.split())
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO samples (context_mode, source_type, content, word_count, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (context_mode, source_type, cleaned, wc, float(weight), utc_now()),
        )
        conn.commit()
        return int(cur.lastrowid)


def add_samples_bulk(
    items: list[str],
    context_mode: str = "casual_chat",
    source_type: str = "document",
    weight: float = 1.0,
    db_path: Path | None = None,
) -> int:
    """Insert multiple writing samples while skipping exact duplicates in the same mode."""
    init_db(db_path)
    now = utc_now()
    inserted = 0
    with get_connection(db_path) as conn:
        existing = {
            row["content"].strip()
            for row in conn.execute(
                "SELECT content FROM samples WHERE context_mode = ?", (context_mode,)
            ).fetchall()
        }
        rows_to_insert = []
        for item in items:
            cleaned = item.strip()
            if not cleaned or cleaned in existing:
                continue
            existing.add(cleaned)
            rows_to_insert.append(
                (context_mode, source_type, cleaned, len(cleaned.split()), float(weight), now)
            )
        if rows_to_insert:
            conn.executemany(
                """
                INSERT INTO samples (context_mode, source_type, content, word_count, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
            inserted = len(rows_to_insert)
        conn.commit()
    return inserted


def get_samples(
    context_mode: str | None = None,
    limit: int = 500,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch stored samples, newest and highest-weighted first."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if context_mode:
            rows = conn.execute(
                "SELECT * FROM samples WHERE context_mode = ? ORDER BY weight DESC, id DESC LIMIT ?",
                (context_mode, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM samples ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def delete_sample(sample_id: int, db_path: Path | None = None) -> bool:
    """Delete a single sample by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
        conn.commit()
        return cur.rowcount > 0


def clear_samples(context_mode: str | None = None, db_path: Path | None = None) -> int:
    """Delete all samples (or only those for a specific context_mode)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if context_mode:
            cur = conn.execute("DELETE FROM samples WHERE context_mode = ?", (context_mode,))
        else:
            cur = conn.execute("DELETE FROM samples")
        conn.commit()
        return cur.rowcount


def upsert_learned_replacement(
    context_mode: str,
    avoid_text: str,
    prefer_text: str,
    source: str = "auto_diff",
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Insert or reinforce a learned avoid -> prefer replacement rule."""
    init_db(db_path)
    avoid_clean = avoid_text.strip().lower()
    prefer_clean = prefer_text.strip()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO learned_replacements (context_mode, avoid_text, prefer_text, times_seen, source)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(context_mode, avoid_text) DO UPDATE SET
                prefer_text = excluded.prefer_text,
                times_seen = learned_replacements.times_seen + 1,
                source = excluded.source
            """,
            (context_mode, avoid_clean, prefer_clean, source),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM learned_replacements WHERE context_mode = ? AND avoid_text = ?",
            (context_mode, avoid_clean),
        ).fetchone()
        return dict(row)


def get_learned_replacements(
    context_mode: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch learned or custom replacement rules."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if context_mode:
            rows = conn.execute(
                """
                SELECT * FROM learned_replacements
                WHERE context_mode = ?
                ORDER BY times_seen DESC, LENGTH(avoid_text) DESC, id DESC
                """,
                (context_mode,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM learned_replacements ORDER BY times_seen DESC, id DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def delete_learned_replacement(rule_id: int, db_path: Path | None = None) -> bool:
    """Delete a replacement rule by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM learned_replacements WHERE id = ?", (rule_id,))
        conn.commit()
        return cur.rowcount > 0


def add_edit_pair(
    context_mode: str,
    original_input: str,
    ai_draft: str,
    user_final_edit: str,
    extracted_rules: list[str],
    db_path: Path | None = None,
) -> int:
    """Store a user final-edit correction pair for continuous learning."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO edit_pairs (
                context_mode, original_input, ai_draft, user_final_edit, extracted_rules_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                context_mode,
                original_input,
                ai_draft,
                user_final_edit,
                json.dumps(extracted_rules),
                utc_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_edit_pairs(
    context_mode: str | None = None,
    limit: int = 25,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return recent final-edit correction pairs."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if context_mode:
            rows = conn.execute(
                "SELECT * FROM edit_pairs WHERE context_mode = ? ORDER BY id DESC LIMIT ?",
                (context_mode, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM edit_pairs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items = []
        for row in rows:
            d = dict(row)
            try:
                d["extracted_rules"] = json.loads(d.get("extracted_rules_json") or "[]")
            except json.JSONDecodeError:
                d["extracted_rules"] = []
            items.append(d)
        return items


def add_rewrite_history(
    context_mode: str,
    input_text: str,
    output_text: str,
    before_human_score: int,
    after_human_score: int,
    provider_used: str,
    db_path: Path | None = None,
) -> int:
    """Record a rewrite invocation in history."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO rewrite_history (
                context_mode, input_text, output_text, before_human_score,
                after_human_score, provider_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context_mode,
                input_text,
                output_text,
                before_human_score,
                after_human_score,
                provider_used,
                utc_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_rewrite_history(limit: int = 25, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return recent rewrite history entries."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM rewrite_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_pangram_scan(
    scan_type: str,
    title: str,
    input_text: str,
    word_count: int,
    estimated_cost_usd: float,
    headline: str,
    fraction_ai: float,
    fraction_ai_assisted: float,
    fraction_human: float,
    max_humanizer_score: float,
    dashboard_link: str,
    result_payload: dict[str, Any],
    db_path: Path | None = None,
) -> int:
    """Persist a completed Pangram API scan to avoid paying twice for the same check."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO pangram_scans (
                scan_type, title, input_text, word_count, estimated_cost_usd,
                headline, fraction_ai, fraction_ai_assisted, fraction_human,
                max_humanizer_score, dashboard_link, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_type,
                title,
                input_text,
                int(word_count),
                float(estimated_cost_usd),
                headline,
                float(fraction_ai),
                float(fraction_ai_assisted),
                float(fraction_human),
                float(max_humanizer_score),
                dashboard_link,
                json.dumps(result_payload),
                utc_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_pangram_scans(limit: int = 25, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Return recent Pangram inspection records with parsed JSON payloads."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM pangram_scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                d["result"] = json.loads(d.get("result_json") or "{}")
            except json.JSONDecodeError:
                d["result"] = {}
            items.append(d)
        return items


def delete_pangram_scan(scan_id: int, db_path: Path | None = None) -> bool:
    """Delete a stored Pangram scan record by ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM pangram_scans WHERE id = ?", (scan_id,))
        conn.commit()
        return cur.rowcount > 0


def get_cached_ai_detection(cache_key: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Retrieve a SHA-256 cached AI detector report if present."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM ai_detector_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["result_json"])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def set_cached_ai_detection(
    cache_key: str,
    result_payload: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist an AI detector report keyed by SHA-256 content + model config hash."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ai_detector_cache (cache_key, result_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                result_json = excluded.result_json,
                created_at = excluded.created_at
            """,
            (cache_key, json.dumps(result_payload), utc_now()),
        )
        conn.commit()


def seed_demo_samples(db_path: Path | None = None) -> dict[str, int]:
    """Seed realistic starter samples across modes for quick testing."""
    demo_data: dict[str, list[str]] = {
        "casual_chat": [
            "yeah tbh i didn't expect it to take this long either lol. let's just grab coffee after and figure out the rest then",
            "wait really?? i thought we already pushed that fix yesterday. lemme check real quick and ping you back",
            "honestly that sounds way cleaner than what we had before. no rush tho, whenever you get a sec",
            "kinda wiped today ngl -- maybe we punt the big refactor to tomorrow morning when our brains actually work",
            "oh nice! yeah i'm down for 3pm, just drop the link when you're ready",
            "nah don't sweat it, happens to everyone. we'll patch it up in like ten mins",
        ],
        "work_email": [
            "Hi Sam,\n\nQuick update on the rollout -- we hit a small snag with the webhook retries this morning, so I paused the batch.\n\nI'm testing a fix now and will share an update before 4pm. Let me know if anything on your end looks off.\n\nBest,\nAlex",
            "Hey Maya,\n\nThanks for sending those numbers over. Looking at the Q3 drop-off, the onboarding step is where most folks bounce.\n\nCould we trim that form down to two fields this week and see if conversion moves? Happy to jump on a 10-min call if easier.\n\nCheers,",
            "Hi team,\n\nShort version: the migration finished cleanly overnight and latency is down ~30%. No action needed from anyone right now, just keeping you in the loop.",
        ],
        "long_form": [
            "Most productivity tools fail for a boring reason: they assume Tuesday you has the same energy as Sunday night you. You build an elaborate system with tags and priorities when you're rested. Then Wednesday hits, three fires start at once, and the whole thing collapses under its own friction.",
            "I kept noticing the same pattern across every team I worked with. The best docs were never the longest ones. They were written by someone who had just debugged the mess themselves and wrote down the three things they wished someone had told them an hour earlier.",
            "Here's the catch though. Simplicity takes way more editing than complexity does. Anyone can dump twenty caveats into a memo to sound thorough, while cutting it down to what actually matters forces you to take a stand.",
        ],
        "social_post": [
            "hot take: 90% of 'AI strategy' decks could just be a two-paragraph memo answering what actually broke and who owns fixing it",
            "spent 3 hours debugging a race condition only to realize i forgot to save the file. classic friday.",
            "small habit that changed how i write: read it out loud once. if you run out of breath or feel weird saying a phrase to a friend, cut it.",
        ],
    }
    counts: dict[str, int] = {}
    for mode, texts in demo_data.items():
        counts[mode] = add_samples_bulk(
            texts, context_mode=mode, source_type="demo_seed", weight=1.0, db_path=db_path
        )
    upsert_learned_replacement("casual_chat", "however", "tho", source="demo_seed", db_path=db_path)
    upsert_learned_replacement("casual_chat", "approximately", "like", source="demo_seed", db_path=db_path)
    upsert_learned_replacement("work_email", "utilize", "use", source="demo_seed", db_path=db_path)
    return counts
