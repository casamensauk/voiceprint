#!/usr/bin/env python3
"""Comprehensive automated end-to-end tests for VoicePrint backend engine and HTTP server.

Covers:
1. Analyzer tests (AI-tell scanner, burstiness calculation, human vs AI score).
2. Importers tests (WhatsApp, Email, Discord JSON, plain document parsing).
3. Voice DNA extraction tests (quantitative cadence, punctuation, and casing habits).
4. Rewriter & De-AI tests (local mode deterministic cliché stripping and burstiness modulation).
5. Grow With Me final-edit learning loop (user correction diffing and rule application).
6. HTTP Server API smoke tests (GET state, POST rewrite, seed demo, learn-edit, rules).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

# Ensure voiceprint project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.analyzer import (
    analyze_text,
    compute_burstiness,
    compute_contraction_stats,
    scan_ai_tells,
)
from engine.db import (
    add_sample,
    clear_samples,
    get_connection,
    get_learned_replacements,
    get_public_settings,
    get_rewrite_history,
    get_samples,
    get_settings,
    init_db,
    seed_demo_samples,
    update_settings,
    upsert_learned_replacement,
)
from engine.importers import detect_format, parse_imported_content
from engine.rewriter import apply_deterministic_de_ai, rewrite_text
from engine.style_dna import extract_voice_dna, learn_from_user_edit
from server import VoicePrintHandler


class BaseEngineTestCase(unittest.TestCase):
    """Base test case managing an isolated temporary SQLite database for each test."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="voiceprint_test_")
        self.db_path = Path(self.temp_dir) / "test_voiceprint.db"
        self._orig_env = os.environ.get("VOICEPRINT_DB_PATH")
        os.environ["VOICEPRINT_DB_PATH"] = str(self.db_path)
        init_db(self.db_path)
        # Default to local_only provider for offline deterministic tests
        update_settings({"provider": "local_only"}, db_path=self.db_path)

    def tearDown(self) -> None:
        if self._orig_env is not None:
            os.environ["VOICEPRINT_DB_PATH"] = self._orig_env
        else:
            os.environ.pop("VOICEPRINT_DB_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestAnalyzer(unittest.TestCase):
    """Test AI vocabulary detection, burstiness scoring, and human vs AI classification."""

    def test_ai_heavy_text_detection(self) -> None:
        """AI-heavy text with tell words, em-dashes, and uniform sentences scores < 45."""
        ai_text = (
            "We must delve into this intricate tapestry of digital tools to unlock value for teams. "
            "Moreover we can seamlessly leverage each dynamic solution — to foster collaboration across groups. "
            "It is crucial that we embark on this transformative journey — to empower our colleagues. "
            "Furthermore we will meticulously curate the landscape — to optimize every single workflow."
        )
        analysis = analyze_text(ai_text)

        # 1. Human score should be low (< 45) and AI risk high (> 55)
        self.assertLess(
            analysis["human_score"],
            45,
            f"Expected human_score < 45 for AI text, got {analysis['human_score']}",
        )
        self.assertGreater(analysis["ai_risk_score"], 55)

        # 2. Verify exact tell words and em-dashes are flagged
        flagged_phrases_lower = {f["phrase"].lower() for f in analysis["flags"]}
        self.assertIn("delve", flagged_phrases_lower)
        self.assertIn("tapestry", flagged_phrases_lower)
        self.assertIn("moreover", flagged_phrases_lower)

        # Em-dash check
        em_dash_flags = [f for f in analysis["flags"] if "—" in f["phrase"] or "em-dash" in f["phrase"]]
        self.assertTrue(len(em_dash_flags) > 0, "Expected em-dash tell flag to be detected")
        self.assertGreaterEqual(analysis["metrics"]["em_dash_count"], 3)

    def test_natural_bursty_human_text(self) -> None:
        """Natural human text with contractions, varied sentence lengths, and no AI clichés scores > 80."""
        human_text = (
            "yeah tbh i didn't expect it to take this long either lol. "
            "let's just grab coffee after and figure out the rest then. "
            "wait really?? i thought we already pushed that fix yesterday. "
            "lemme check real quick and ping you back. "
            "honestly that sounds way cleaner than what we had before. "
            "no rush tho, whenever you get a sec. "
            "kinda wiped today ngl -- maybe we punt the big refactor to tomorrow morning when our brains actually work."
        )
        analysis = analyze_text(human_text)

        self.assertGreater(
            analysis["human_score"],
            80,
            f"Expected human_score > 80 for natural human text, got {analysis['human_score']}",
        )
        self.assertLess(analysis["ai_risk_score"], 20)
        self.assertEqual(len(analysis["flags"]), 0, f"Expected 0 AI flags, got {analysis['flags']}")
        self.assertGreaterEqual(analysis["burstiness_score"], 65)

    def test_burstiness_metric_variations(self) -> None:
        """Check sentence length variation calculation and coefficient of variation."""
        uniform_text = "This is a sentence. This is a sentence. This is a sentence. This is a sentence."
        uniform_burst = compute_burstiness(uniform_text)
        self.assertEqual(uniform_burst["sentence_count"], 4)
        self.assertAlmostEqual(uniform_burst["coeff_of_variation"], 0.0, places=2)

        varied_text = "Short. This is a moderately longer sentence with extra words. Yes! Here is another very descriptive and elaborate thought that adds length."
        varied_burst = compute_burstiness(varied_text)
        self.assertGreater(varied_burst["coeff_of_variation"], 0.4)
        self.assertGreater(varied_burst["burstiness_score"], uniform_burst["burstiness_score"])

    def test_empty_text_analysis(self) -> None:
        """Empty input returns zeroed metrics without crashing."""
        res = analyze_text("   ")
        self.assertEqual(res["word_count"], 0)
        self.assertEqual(res["human_score"], 0)
        self.assertEqual(res["flags"], [])


class TestImporters(unittest.TestCase):
    """Test importing and parsing WhatsApp, Email, Discord JSON, and Plain Documents."""

    def test_whatsapp_parsing_both_formats_and_filtering(self) -> None:
        """Parse WhatsApp exports with [timestamp] and - timestamp formats, filtering by sender and skipping media."""
        raw_whatsapp = (
            "[12/05/2025, 14:22:10] Alice: Hey Bob how are you doing today?\n"
            "12/05/2025, 14:22 - Alice: Doing well just working on the project now\n"
            "[12/05/2025, 14:23:05] Bob: Doing awesome over here thanks for asking\n"
            "12/05/2025, 14:24 - Alice: <Media omitted>\n"
            "[12/05/2025, 14:25:00] Alice: let us catch up after lunch\n"
        )

        # Detect format
        self.assertEqual(detect_format(raw_whatsapp), "whatsapp")

        # Parse with sender_name filter 'Alice'
        result = parse_imported_content(raw_whatsapp, source_format="auto", sender_name="Alice")
        samples = result["samples"]

        # Alice's 3 real messages should be parsed, Bob filtered out, <Media omitted> skipped
        self.assertEqual(result["detected_format"], "whatsapp")
        self.assertEqual(len(samples), 3)
        self.assertIn("Hey Bob how are you doing today?", samples)
        self.assertIn("Doing well just working on the project now", samples)
        self.assertIn("let us catch up after lunch", samples)
        for s in samples:
            self.assertNotIn("Bob:", s)
            self.assertNotIn("<Media omitted>", s)

    def test_email_parsing_header_and_reply_stripping(self) -> None:
        """Parse email text, stripping From/Subject/Date headers and On ... wrote: reply blocks."""
        raw_email = (
            "From: alice@example.com\n"
            "To: bob@example.com\n"
            "Subject: Q4 Roadmap Progress\n"
            "Date: Mon, 12 May 2025 10:15:00 -0700\n"
            "\n"
            "Hi team,\n"
            "Quick status update on the search latency improvements.\n"
            "Everything is looking solid and we are ready for rollout tomorrow.\n"
            "\n"
            "On May 11, 2025, at 16:30, Bob <bob@example.com> wrote:\n"
            "> Are we still targeting Wednesday for the production push?\n"
            "> Let me know if any blockers pop up.\n"
        )

        result = parse_imported_content(raw_email, source_format="auto")
        self.assertEqual(result["detected_format"], "email")
        self.assertEqual(len(result["samples"]), 1)
        cleaned = result["samples"][0]

        # Verify headers were stripped
        self.assertNotIn("From: alice@example.com", cleaned)
        self.assertNotIn("Subject: Q4 Roadmap Progress", cleaned)
        # Verify reply block was stripped
        self.assertNotIn("On May 11, 2025", cleaned)
        self.assertNotIn("Are we still targeting Wednesday", cleaned)
        # Verify body content preserved
        self.assertIn("Quick status update on the search latency improvements", cleaned)

    def test_discord_json_parsing(self) -> None:
        """Parse Discord JSON export with author object and sender filtering."""
        discord_data = {
            "messages": [
                {"author": {"username": "Alice"}, "content": "hey everyone standup is in five minutes"},
                {"author": {"username": "Bob"}, "content": "sounds good i am joining the link now"},
                {"author": {"username": "Alice"}, "content": "awesome see you there in the main channel"},
            ]
        }
        raw_json = json.dumps(discord_data)
        result = parse_imported_content(raw_json, source_format="auto", sender_name="Alice")

        self.assertEqual(result["detected_format"], "discord_slack")
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(result["samples"][0], "hey everyone standup is in five minutes")
        self.assertEqual(result["samples"][1], "awesome see you there in the main channel")

    def test_plain_document_parsing(self) -> None:
        """Parse plain text documents separated by multi-newlines into clean paragraphs."""
        raw_doc = (
            "Here is the first paragraph detailing the system architecture.\n"
            "It describes the event bus and the retry queues in clean detail.\n\n"
            "Here is the second paragraph analyzing the benchmark results.\n"
            "Latency dropped by over forty percent after the caching pass.\n"
        )
        result = parse_imported_content(raw_doc, source_format="document")
        self.assertEqual(result["detected_format"], "document")
        self.assertEqual(len(result["samples"]), 2)
        self.assertTrue(result["samples"][0].startswith("Here is the first paragraph"))
        self.assertTrue(result["samples"][1].startswith("Here is the second paragraph"))


class TestVoiceDNA(unittest.TestCase):
    """Test quantitative Voice DNA extraction from personal writing samples."""

    def test_casual_voice_dna_extraction(self) -> None:
        """Verify high contraction_rate, lowercase_start_rate, lowercase_i_rate, no_trailing_period_rate, and double_dash."""
        casual_samples = [
            {"content": "yeah i don't think i'm gonna make it -- maybe later tonight", "weight": 1.0},
            {"content": "honestly i'm down to catch up tomorrow if you're free", "weight": 1.0},
            {"content": "nah i don't mind at all -- let me check real quick", "weight": 1.0},
            {"content": "sounds good i'll ping you then", "weight": 1.0},
        ]

        dna = extract_voice_dna(casual_samples)

        # 1. Contraction rate should be high (don't, i'm, you're, i'll)
        self.assertGreaterEqual(dna["contraction_rate"], 0.85)

        # 2. Lowercase start rate (sentences start with yeah, honestly, nah, sounds)
        self.assertGreaterEqual(dna["lowercase_start_rate"], 0.90)

        # 3. Lowercase 'i' rate (only lowercase 'i' used, no standalone 'I')
        self.assertEqual(dna["lowercase_i_rate"], 1.0)

        # 4. No trailing period rate (short casual messages ending with no .!?)
        self.assertEqual(dna["no_trailing_period_rate"], 1.0)

        # 5. Punctuation profile double-dash frequency (> 0)
        self.assertGreater(dna["punctuation_profile"]["double_dash"], 1.0)
        self.assertEqual(dna["punctuation_profile"]["em_dash"], 0.0)

        # 6. Directives summary generated
        summary = " ".join(dna["directives_summary"])
        self.assertIn("lowercase", summary.lower())
        self.assertIn("contractions", summary.lower())
        self.assertIn("double-hyphen", summary.lower())

    def test_empty_samples_fallback_defaults(self) -> None:
        """When no samples are uploaded, extract_voice_dna returns crisp default profile."""
        dna = extract_voice_dna([])
        self.assertEqual(dna["sample_count"], 0)
        self.assertEqual(dna["gold_edit_count"], 0)
        self.assertGreater(dna["avg_sentence_length"], 10.0)
        self.assertTrue(len(dna["directives_summary"]) > 0)


class TestRewriter(BaseEngineTestCase):
    """Test deterministic De-AI engine and personal Voice DNA rewriting in local mode."""

    def test_local_mode_de_ai_rewrite(self) -> None:
        """rewrite_text eliminates AI clichés and em-dashes, improves human score, and applies DNA habits."""
        # 1. Add casual voice samples so the active DNA embodies casual habits
        add_sample("yeah i don't think i'm gonna make it -- maybe later", context_mode="casual_chat", db_path=self.db_path)
        add_sample("nah i don't mind at all -- let me check real quick", context_mode="casual_chat", db_path=self.db_path)
        add_sample("honestly sounds good to me -- ping me whenever", context_mode="casual_chat", db_path=self.db_path)

        # 2. Input text heavy with AI clichés and em-dashes
        ai_input = (
            "We must delve into this intricate tapestry of digital tools to unlock value for teams. "
            "Moreover we can seamlessly leverage each dynamic solution — to foster collaboration across groups. "
            "It is crucial that we embark on this transformative journey — to empower our colleagues."
        )

        res = rewrite_text(
            ai_input,
            context_mode="casual_chat",
            intensity="deep",
            save_history=True,
            db_path=self.db_path,
        )

        rewritten = res["rewritten"]
        before_score = res["before"]["human_score"]
        after_score = res["after"]["human_score"]

        # 3. AI score improvement check
        self.assertGreater(
            after_score,
            before_score + 25,
            f"Expected significant improvement: before={before_score}, after={after_score}",
        )

        # 4. Cliché elimination: delve, tapestry, moreover, leverage, seamless, em-dash
        rewritten_lower = rewritten.lower()
        self.assertNotIn("delve", rewritten_lower)
        self.assertNotIn("tapestry", rewritten_lower)
        self.assertNotIn("moreover", rewritten_lower)
        self.assertNotIn("leverage", rewritten_lower)
        self.assertNotIn("seamless", rewritten_lower)
        self.assertNotIn("—", rewritten)  # Em-dash eliminated

        # 5. Casual DNA adaptation: double-dash '--' replaces em-dash '—'
        self.assertIn("--", rewritten)

        # 6. Flag counts should be empty after de-ai pass
        after_flags = res["after"]["flags"]
        self.assertEqual(len(after_flags), 0, f"Expected 0 AI flags after rewrite, got: {after_flags}")

        # 7. Verify rewrite history was recorded
        history = get_rewrite_history(limit=5, db_path=self.db_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["input_text"], ai_input)
        self.assertEqual(history[0]["output_text"], rewritten)


class TestGrowWithMeLearningLoop(BaseEngineTestCase):
    """Test Grow With Me continuous learning: user final edits teach rules applied to future rewrites."""

    def test_learn_from_user_edit_to_rewrite(self) -> None:
        """User replacing 'perhaps' with 'maybe' and 'unfortunately' with 'sadly' creates rules used in rewrites."""
        original_prompt = "Can we meet tomorrow afternoon?"
        ai_draft = "Perhaps we could meet tomorrow, but unfortunately I have another call scheduled."
        user_edit = "Maybe we could meet tomorrow, but sadly i have another call scheduled."

        # 1. Feed edit correction into learning loop
        learn_res = learn_from_user_edit(
            context_mode="casual_chat",
            original_input=original_prompt,
            ai_draft=ai_draft,
            user_final_edit=user_edit,
            db_path=self.db_path,
        )

        self.assertTrue(learn_res["changed"])
        self.assertGreaterEqual(learn_res["rules_added"], 2)

        # 2. Verify learned_replacements database table
        rules = get_learned_replacements(context_mode="casual_chat", db_path=self.db_path)
        rule_map = {r["avoid_text"].lower(): r["prefer_text"] for r in rules}
        self.assertIn("perhaps", rule_map)
        self.assertEqual(rule_map["perhaps"].lower(), "maybe")
        self.assertIn("unfortunately", rule_map)
        self.assertEqual(rule_map["unfortunately"].lower(), "sadly")

        # 3. Verify sample stored with weight 2.5 and source_type 'learned_edit'
        samples = get_samples(context_mode="casual_chat", db_path=self.db_path)
        gold_samples = [s for s in samples if s["source_type"] == "learned_edit"]
        self.assertEqual(len(gold_samples), 1)
        self.assertEqual(gold_samples[0]["content"], user_edit)
        self.assertEqual(gold_samples[0]["weight"], 2.5)

        # 4. Subsequent rewrite on NEW text containing 'perhaps' should automatically apply 'maybe'
        new_text = "Perhaps this new approach will solve the performance issue."
        rewrite_res = rewrite_text(new_text, context_mode="casual_chat", db_path=self.db_path)
        new_rewritten = rewrite_res["rewritten"]

        self.assertNotIn("perhaps", new_rewritten.lower())
        self.assertIn("maybe", new_rewritten.lower())


class TestHTTPServer(unittest.TestCase):
    """Test HTTP API routes on an ephemeral server instance with isolated temporary database."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp(prefix="voiceprint_server_test_")
        cls.db_path = Path(cls.temp_dir) / "server_test.db"
        cls._orig_env = os.environ.get("VOICEPRINT_DB_PATH")
        os.environ["VOICEPRINT_DB_PATH"] = str(cls.db_path)
        init_db(cls.db_path)
        update_settings({"provider": "local_only"}, db_path=cls.db_path)

        # Spin up ThreadingHTTPServer on an ephemeral port (port 0)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), VoicePrintHandler)
        cls.host, cls.port = cls.httpd.server_address
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join(timeout=3.0)
        if cls._orig_env is not None:
            os.environ["VOICEPRINT_DB_PATH"] = cls._orig_env
        else:
            os.environ.pop("VOICEPRINT_DB_PATH", None)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _get(self, path: str) -> tuple[int, dict[str, str], bytes]:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read()

    def _post(self, path: str, payload: dict | None = None) -> tuple[int, dict[str, str], bytes]:
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read()

    def test_get_root(self) -> None:
        """GET / serves index.html with 200 OK."""
        status, headers, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        self.assertIn(b"VoicePrint", body)

    def test_get_api_state(self) -> None:
        """GET /api/state returns full dashboard state dictionary."""
        status, _, body = self._get("/api/state")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("settings", data)
        self.assertIn("active_context_mode", data)
        self.assertIn("dna_by_mode", data)
        self.assertIn("active_dna", data)
        self.assertIn("maturity", data)

    def test_post_seed_demo(self) -> None:
        """POST /api/samples/seed-demo seeds starter samples across all context modes."""
        status, _, body = self._post("/api/samples/seed-demo", {})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("seeded", data)
        self.assertGreater(data["seeded"].get("casual_chat", 0), 0)
        self.assertGreater(data["seeded"].get("work_email", 0), 0)

    def test_post_rewrite(self) -> None:
        """POST /api/rewrite rewrites text without invoking clipboard copy."""
        payload = {
            "text": "Moreover we must delve into this multifaceted tapestry to elevate productivity.",
            "context_mode": "casual_chat",
            "intensity": "deep",
            "auto_copy": False,
        }
        status, _, body = self._post("/api/rewrite", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("rewritten", data)
        self.assertNotIn("delve", data["rewritten"].lower())
        self.assertNotIn("tapestry", data["rewritten"].lower())

    def test_post_learn_edit(self) -> None:
        """POST /api/learn-edit records final-edit corrections and updates rules."""
        payload = {
            "context_mode": "casual_chat",
            "original_input": "Shall we meet?",
            "ai_draft": "Perhaps we could meet tomorrow.",
            "user_final_edit": "Maybe we could meet tomorrow.",
            "copy": False,
        }
        status, _, body = self._post("/api/learn-edit", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data.get("changed"))
        self.assertGreaterEqual(data.get("rules_added", 0), 1)
        self.assertIn("state", data)

    def test_post_rules(self) -> None:
        """POST /api/rules creates a custom avoid -> prefer replacement rule."""
        payload = {
            "context_mode": "casual_chat",
            "avoid_text": "utilize",
            "prefer_text": "use",
        }
        status, _, body = self._post("/api/rules", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("rule", data)
        self.assertEqual(data["rule"]["avoid_text"], "utilize")
        self.assertEqual(data["rule"]["prefer_text"], "use")


if __name__ == "__main__":
    unittest.main(verbosity=2)
