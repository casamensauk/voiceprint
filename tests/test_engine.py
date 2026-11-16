#!/usr/bin/env python3
"""Comprehensive automated end-to-end tests for VoicePrint backend engine and HTTP server.

Covers:
1. Analyzer tests (AI-tell scanner, burstiness calculation, human vs AI score).
2. Importers tests (WhatsApp, Email, Discord JSON, plain document parsing).
3. Voice DNA extraction tests (quantitative cadence, punctuation, and casing habits).
4. Rewriter & De-AI tests (local mode deterministic cliché stripping and burstiness modulation).
5. Grow With Me final-edit learning loop (user correction diffing and rule application).
6. Pangram Labs client tests (cost estimation, async polling, validation, docx extraction).
7. Academic Coach tests (source citation overlap checking and clarity diagnostics).
8. HTTP Server API smoke tests (GET state, POST rewrite, seed demo, learn-edit, rules, academic & Pangram endpoints).
9. In-House 3-Stage AI Writing Detector (preprocessor offsets, stylometrics, mocked 3-stage ensemble, calibrator asterisk, docx scan).
10. Mac Studio dynamic port discovery and fallback server tests.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure voiceprint project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.academic_coach import (
    check_source_citations,
    coach_academic_draft,
)
from engine.ai_detector import (
    InHouseDetectorError,
    analyze_text_for_ai,
    scan_file_inhouse,
)
from engine.ai_detector.calibrator import calibrate_report
from engine.ai_detector.orchestrator import resolve_openrouter_config
from engine.ai_detector.preprocessor import PreprocessedSegment, segment_document
from engine.ai_detector.stylometrics import (
    detect_unicode_and_homoglyphs,
    score_window_stylometrics,
)
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
    get_pangram_scans,
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
from engine.pangram_client import (
    PangramError,
    estimate_pangram_cost,
    extract_docx_text,
    scan_file_pangram,
    scan_text_pangram,
)
from engine.rewriter import apply_deterministic_de_ai, rewrite_text
from engine.style_dna import extract_voice_dna, learn_from_user_edit
from engine.updater import (
    UpdaterError,
    check_github_update,
    is_newer_version,
    parse_semver,
    stage_and_apply_update,
)
from server import VoicePrintHandler, bind_server


class _MockHTTPResponse:
    """Mock context-manager response for urllib.request.urlopen."""

    def __init__(self, data: dict | list | str, status: int = 200) -> None:
        if isinstance(data, (dict, list)):
            self.raw = json.dumps(data).encode("utf-8")
        elif isinstance(data, str):
            self.raw = data.encode("utf-8")
        else:
            self.raw = data
        self.status = status

    def read(self) -> bytes:
        return self.raw

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


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


class TestPangramClient(BaseEngineTestCase):
    """Test Pangram Labs Pay-As-You-Go client cost estimation, async polling, and docx extraction."""

    def test_cost_estimation_and_min_length(self) -> None:
        """estimate_pangram_cost computes $0.05 per 100 words, 20% bulk discount, and enforces >= 50 chars."""
        # 200 words standard: (200 / 100) * $0.05 = $0.10
        text_200 = "word " * 200
        est_std = estimate_pangram_cost(text_200, is_bulk=False)
        self.assertEqual(est_std["word_count"], 200)
        self.assertEqual(est_std["estimated_cost_usd"], 0.10)
        self.assertTrue(est_std["meets_min_chars"])

        # 200 words bulk: 20% discount -> $0.08
        est_bulk = estimate_pangram_cost(text_200, is_bulk=True)
        self.assertEqual(est_bulk["estimated_cost_usd"], 0.08)

        # < 50 chars sets meets_min_chars=False
        short_text = "short text"
        est_short = estimate_pangram_cost(short_text)
        self.assertFalse(est_short["meets_min_chars"])
        self.assertLess(est_short["char_count"], 50)

    def test_scan_text_async_submit_and_poll(self) -> None:
        """scan_text_pangram submits task, polls stage, verifies headers/model, and parses humanizer flags."""
        resps = [
            _MockHTTPResponse({"task_id": "t-123"}),
            _MockHTTPResponse({"task_id": "t-123", "stage": "STAGE_PREPROCESSING"}),
            _MockHTTPResponse({
                "task_id": "t-123",
                "stage": "STAGE_SUCCESS",
                "fraction_ai": 0.25,
                "fraction_ai_assisted": 0.50,
                "fraction_human": 0.25,
                "dashboard_link": "https://dashboard.pangram.com/t-123",
                "windows": [
                    {
                        "text": "This portion shows signs of automated modification.",
                        "is_humanized": True,
                        "humanizer_score": 0.84,
                    }
                ],
            }),
        ]

        text_to_scan = "This is an academic draft long enough to meet the fifty character threshold easily."
        with patch("urllib.request.urlopen", side_effect=resps) as mock_urlopen:
            res = scan_text_pangram(
                text=text_to_scan,
                api_key="test-api-key-123",
                model="pangram-4",
                poll_s=0.1,
            )

            # Verify POST /task request sent x-api-key header and model "pangram-4"
            first_call_req = mock_urlopen.call_args_list[0][0][0]
            self.assertEqual(first_call_req.get_header("X-api-key"), "test-api-key-123")
            post_payload = json.loads(first_call_req.data.decode("utf-8"))
            self.assertEqual(post_payload.get("model"), "pangram-4")

            # Verify polling succeeded and returned normalized result
            self.assertEqual(res["task_id"], "t-123")
            self.assertEqual(res["fraction_ai"], 0.25)
            self.assertEqual(res["fraction_ai_assisted"], 0.50)
            self.assertEqual(res["fraction_human"], 0.25)
            self.assertEqual(res["dashboard_link"], "https://dashboard.pangram.com/t-123")
            self.assertTrue(res["has_humanized_segments"])
            self.assertEqual(res["max_humanizer_score"], 0.84)
            self.assertEqual(len(res["windows"]), 1)
            self.assertTrue(res["windows"][0]["is_humanized"])
            self.assertEqual(res["windows"][0]["humanizer_score"], 0.84)

    def test_scan_text_validation_and_stage_failed(self) -> None:
        """< 50 chars raises PangramError(400) without network, and STAGE_FAILED raises PangramError(422)."""
        # 1. Validation error (< 50 chars)
        with patch("urllib.request.urlopen") as mock_urlopen:
            with self.assertRaises(PangramError) as cm:
                scan_text_pangram("Short text", api_key="valid-key")
            self.assertEqual(cm.exception.status_code, 400)
            self.assertIn("at least 50 characters", cm.exception.message)
            mock_urlopen.assert_not_called()

        # 2. STAGE_FAILED
        fail_resps = [
            _MockHTTPResponse({"task_id": "t-fail"}),
            _MockHTTPResponse({
                "task_id": "t-fail",
                "stage": "STAGE_FAILED",
                "headline": "Processing failed due to unparseable encoding",
            }),
        ]
        with patch("urllib.request.urlopen", side_effect=fail_resps):
            long_text = "This text easily satisfies the fifty character requirement for testing failure modes."
            with self.assertRaises(PangramError) as cm_fail:
                scan_text_pangram(long_text, api_key="valid-key", poll_s=0.1)
            self.assertEqual(cm_fail.exception.status_code, 422)
            self.assertIn("Pangram scan failed", cm_fail.exception.message)

    def test_docx_extraction_routes_through_pangram_4(self) -> None:
        """scan_file_pangram extracts text from minimal in-memory .docx and routes to pangram-4."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>'
                'This is an extracted academic essay from a docx archive that contains more than sixty characters.'
                '</w:t></w:r></w:p></w:body></w:document>'
            )
            zf.writestr("word/document.xml", xml)
        docx_bytes = bio.getvalue()

        with patch("engine.pangram_client.scan_text_pangram") as mock_scan:
            mock_scan.return_value = {
                "scan_type": "text",
                "task_id": "t-docx",
                "model": "pangram-4",
                "fraction_ai": 0.15,
                "has_humanized_segments": False,
                "max_humanizer_score": 0.0,
            }
            res = scan_file_pangram("essay.docx", docx_bytes, api_key="k", prefer_pangram_4=True)
            self.assertTrue(mock_scan.called)
            called_args, called_kwargs = mock_scan.call_args
            extracted_text = called_args[0]
            self.assertIn("extracted academic essay from a docx archive", extracted_text)
            self.assertEqual(called_kwargs.get("model"), "pangram-4")
            self.assertEqual(called_kwargs.get("api_key"), "k")
            self.assertEqual(res["scan_type"], "file")
            self.assertEqual(res["filename"], "essay.docx")


class TestAcademicCoach(unittest.TestCase):
    """Test author-led citation checking and academic clarity diagnostics."""

    def test_citation_check_flags_unquoted_and_rewards_cited_quotes(self) -> None:
        """Flags verbatim unquoted and close paraphrasing, recognizes cited quotes, and reconstructs draft."""
        source = (
            "Machine learning models require substantial computational resources and massive datasets to achieve state of the art performance across domains. "
            "Deep neural architectures demonstrate remarkable pattern recognition capabilities across diverse domains. "
            "Algorithmic advances enable researchers to explore intricate multidimensional representations."
        )

        draft = (
            "In recent years, computational resources and massive datasets to achieve breakthrough results has become standard. "
            "Neural architectures demonstrate pattern recognition capabilities across domains. "
            'In addition, "algorithmic advances enable researchers to explore intricate multidimensional representations." (Smith, 2024)'
        )

        res = check_source_citations(draft, source)

        # 1. Flag types present
        flag_types = {m["type"] for m in res["matches"]}
        self.assertIn("verbatim_unquoted", flag_types)
        self.assertIn("close_paraphrase", flag_types)

        # 2. Quoted and cited count >= 1
        self.assertGreaterEqual(res["quoted_cited_count"], 1)

        # 3. Concatenating annotated_segments reconstructs the exact original draft
        reconstructed = "".join(seg["text"] for seg in res["annotated_segments"])
        self.assertEqual(reconstructed, draft)

    def test_academic_coach_diagnostics(self) -> None:
        """Detects passive voice, unattributed claims, vague fillers, and repetitive openers."""
        draft = (
            "The experiment was conducted by our laboratory over three months. "
            "Researchers have shown that consistent measurement reduces anomalies. "
            "This result occurred due to the fact that temperature remained stable. "
            "This methodology provides reproducible outcomes across trials."
        )

        res = coach_academic_draft(draft)

        counts = res["counts"]
        self.assertGreaterEqual(counts["passive_voice"], 1)
        self.assertGreaterEqual(counts["unattributed_claim"], 1)
        self.assertGreaterEqual(counts["vague_filler"], 1)
        self.assertGreaterEqual(counts["repetitive_opener"], 1)
        self.assertLess(res["clarity_score"], 90)
        self.assertGreaterEqual(len(res["sentence_feedback"]), 4)


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
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), err.read()

    def _post(self, path: str, payload: dict | None = None) -> tuple[int, dict[str, str], bytes]:
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), err.read()

    def _delete(self, path: str) -> tuple[int, dict[str, str], bytes]:
        req = urllib.request.Request(f"{self.base_url}{path}", method="DELETE")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), err.read()

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

    def test_post_academic_citation_check(self) -> None:
        """POST /api/academic/citation-check returns 200 OK with overall_overlap_pct and annotated_segments."""
        payload = {
            "draft_text": "In recent years, computational resources and massive datasets to achieve breakthrough results has become standard.",
            "source_text": "Machine learning models require substantial computational resources and massive datasets to achieve state of the art.",
        }
        status, _, body = self._post("/api/academic/citation-check", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("overall_overlap_pct", data)
        self.assertIn("annotated_segments", data)
        self.assertIn("matches", data)

    def test_post_academic_coach(self) -> None:
        """POST /api/academic/coach returns 200 OK with clarity_score and sentence_feedback."""
        payload = {
            "draft_text": "The experiment was conducted by our laboratory. Due to the fact that time was limited, researchers have shown that results vary.",
        }
        status, _, body = self._post("/api/academic/coach", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("clarity_score", data)
        self.assertIn("sentence_feedback", data)
        self.assertIn("counts", data)

    def test_post_pangram_scan_without_key_returns_400(self) -> None:
        """POST /api/pangram/scan without an API key returns HTTP 400 with error message."""
        update_settings({"pangram_api_key": "", "clear_pangram_api_key": True}, db_path=self.db_path)
        orig_key = os.environ.pop("PANGRAM_API_KEY", None)
        try:
            payload = {
                "text": "This is a submission that easily exceeds fifty characters in total length for scan.",
            }
            status, _, body = self._post("/api/pangram/scan", payload)
            self.assertEqual(status, 400)
            data = json.loads(body.decode("utf-8"))
            self.assertIn("error", data)
            self.assertIn("API key", data["error"])
        finally:
            if orig_key is not None:
                os.environ["PANGRAM_API_KEY"] = orig_key

    def test_post_pangram_scan_persists_and_deletes(self) -> None:
        """POST /api/pangram/scan with mock returns 200, persists to pangram_scans, and DELETE removes it."""
        update_settings({"pangram_api_key": "test_pangram_key_xyz"}, db_path=self.db_path)

        mock_result = {
            "scan_type": "text",
            "task_id": "test-task-persisted-1",
            "stage": "STAGE_SUCCESS",
            "version": "4.1",
            "model": "pangram-4",
            "text": "This is a sufficiently long text submission for automated scanning.",
            "headline": "Likely AI-generated",
            "prediction": "AI",
            "fraction_ai": 0.85,
            "fraction_ai_assisted": 0.10,
            "fraction_human": 0.05,
            "has_humanized_segments": False,
            "max_humanizer_score": 0.0,
            "dashboard_link": "https://dashboard.pangram.com/test-task-persisted-1",
            "windows": [],
            "estimate": {"word_count": 10, "estimated_cost_usd": 0.005},
        }

        try:
            with patch("server.scan_text_pangram", return_value=mock_result):
                payload = {
                    "text": "This is a sufficiently long text submission for automated scanning.",
                    "title": "Automated Unit Test Submission",
                }
                status, _, body = self._post("/api/pangram/scan", payload)
                self.assertEqual(status, 200)
                data = json.loads(body.decode("utf-8"))
                self.assertIn("result", data)
                scan_id = data["result"].get("scan_id")
                self.assertIsNotNone(scan_id)

                # Verify it is in database
                scans_in_db = get_pangram_scans(db_path=self.db_path)
                self.assertTrue(any(s["id"] == scan_id for s in scans_in_db))

                # DELETE /api/pangram/scans/<id> removes it cleanly
                del_status, _, del_body = self._delete(f"/api/pangram/scans/{scan_id}")
                self.assertEqual(del_status, 200)
                del_data = json.loads(del_body.decode("utf-8"))
                self.assertTrue(del_data.get("deleted"))

                # Verify it is removed from database
                scans_after = get_pangram_scans(db_path=self.db_path)
                self.assertFalse(any(s["id"] == scan_id for s in scans_after))
        finally:
            update_settings({"pangram_api_key": "", "clear_pangram_api_key": True}, db_path=self.db_path)

    def test_api_health_endpoint(self) -> None:
        """GET /api/health returns HTTP 200 with {"app": "voiceprint", "ok": True, ...}."""
        status, _, body = self._get("/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data.get("app"), "voiceprint")
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("port"), self.port)
        self.assertIn("version", data)

    def test_post_ai_detector_scan_offline(self) -> None:
        """POST /api/ai-detector/scan with use_openrouter: False returns 200 and persists in pangram_scans."""
        payload = {
            "text": "This is a comprehensive evaluation submission designed to test the in-house offline detection engine.",
            "title": "Offline In-House AI Scan Test",
            "use_openrouter": False,
        }
        status, _, body = self._post("/api/ai-detector/scan", payload)
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("result", data)
        res = data["result"]
        self.assertIn("fraction_ai", res)
        self.assertIn("fraction_ai_assisted", res)
        self.assertIn("fraction_human", res)
        self.assertIn("windows", res)
        scan_id = res.get("scan_id")
        self.assertIsNotNone(scan_id)

        scans_in_db = get_pangram_scans(db_path=self.db_path)
        self.assertTrue(any(s["id"] == scan_id for s in scans_in_db))


class TestPortDiscovery(unittest.TestCase):
    """Test Mac Studio dynamic port discovery and fallback logic."""

    def test_bind_server_falls_back_when_port_occupied(self) -> None:
        """Bind dummy socket to port, verify bind_server falls back to next free port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        occupied_port = sock.getsockname()[1]
        try:
            httpd = bind_server(host="127.0.0.1", port=occupied_port, auto_port=True)
            try:
                self.assertNotEqual(httpd.server_port, occupied_port)
                self.assertGreater(httpd.server_port, 0)
            finally:
                httpd.server_close()
        finally:
            sock.close()

    def test_api_health_endpoint(self) -> None:
        """GET /api/health returns HTTP 200 with {"app": "voiceprint", "ok": True, ...}."""
        httpd = bind_server(host="127.0.0.1", port=0, auto_port=False)
        port = httpd.server_port
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/health", method="GET")
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data.get("app"), "voiceprint")
                self.assertTrue(data.get("ok"))
                self.assertEqual(data.get("port"), port)
                self.assertIn("version", data)
        finally:
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=2.0)


class TestInHouseAiDetector(BaseEngineTestCase):
    """Test the in-house 3-stage AI writing detector, forensics, and docx extraction."""

    def test_preprocessor_offsets_and_reference_exclusion(self) -> None:
        """Verify segment_document produces exact character offsets and marks references."""
        essay = (
            "Artificial intelligence in modern pedagogy represents a pivotal evolution in educational methods. "
            "Researchers have observed marked differences in student engagement when adaptive learning systems are deployed. "
            "Traditional instructional design often relied on rigid curricular structures that failed to accommodate diverse pacing.\n\n"
            "By contrasting personalized feedback loops against standardized benchmarks, contemporary classrooms foster independent inquiry. "
            "Empirical evaluations suggest measurable gains across critical thinking indicators when guided synthesis tools are incorporated.\n\n"
            "References\n"
            "Smith, J. (2024). Adaptive Frameworks in Digital Learning. Oxford Academic Press."
        )
        segments = segment_document(essay)
        self.assertGreaterEqual(len(segments), 2)
        for seg in segments:
            self.assertEqual(essay[seg.start_index:seg.end_index], seg.text)

        # Bibliography segment must be marked is_excluded_reference=True
        ref_segs = [s for s in segments if s.is_excluded_reference]
        self.assertEqual(len(ref_segs), 1)
        self.assertIn("Smith, J. (2024)", ref_segs[0].text)
        self.assertTrue(segments[-1].is_excluded_reference)

        # Body segment must NOT be marked is_excluded_reference
        body_segs = [s for s in segments if not s.is_excluded_reference]
        self.assertGreaterEqual(len(body_segs), 1)
        self.assertFalse(segments[0].is_excluded_reference)

    def test_stylometrics_detects_ai_and_homoglyph_spoofing(self) -> None:
        """Verify AI-cliché text yields high AI score, and homoglyphs/Unicode yield high humanizer score."""
        ai_text = (
            "We must delve into this intricate tapestry of digital tools to unlock value for teams. "
            "Moreover we can seamlessly leverage each dynamic solution to foster collaboration across groups. "
            "Furthermore we will meticulously curate the landscape to optimize every single workflow. "
            "In addition we must strategically utilize our assets to maximize holistic organizational efficiency."
        )
        res_ai = score_window_stylometrics(ai_text)
        self.assertGreaterEqual(res_ai["stylometric_ai_score"], 0.70)
        self.assertGreaterEqual(res_ai["ai_lexicon_hits"], 2)

        # Cyrillic 'е' in "tеst" (\u0435) and zero-width space (\u200b)
        spoofed_text = "This is a normal paragraph to t\u0435st the evasion detector with invisible\u200b watermarks in the text."
        res_sp = score_window_stylometrics(spoofed_text)
        self.assertGreaterEqual(res_sp["homoglyph_count"], 1)
        self.assertGreaterEqual(res_sp["unicode_anomaly_count"], 1)
        self.assertGreaterEqual(res_sp["stylometric_humanizer_score"], 0.60)

    def test_full_3stage_ensemble_mocked_and_cached(self) -> None:
        """Mock Stage 2 and Stage 3 OpenRouter calls, verify ensemble mode, low word count, and cache hit."""
        draft = (
            "In the modern era of natural language processing, advanced deep learning models have revolutionized "
            "automated text production across various domains. Many educational institutions have begun examining "
            "the authenticity of student work with increased scrutiny. Furthermore, research indicates that stochastic "
            "sequence generation yields identifiable statistical fingerprints. Consequently, linguistic analysts "
            "examine sentence length variance and token probability distributions to distinguish synthetic prose."
        )
        word_count = len(draft.split())
        self.assertLess(word_count, 300)
        self.assertGreaterEqual(word_count, 40)

        words = draft.split()
        split_idx = max(10, int(len(words) * 0.50))
        suffix = " ".join(words[split_idx:])

        def fake_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
            resp_fmt = kwargs.get("response_format")
            if resp_fmt is not None:
                # Stage 3 arbiter structured output
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "overall_summary": "High AI probability detected across windows.",
                                "windows": [{
                                    "window_index": 0,
                                    "ai_score": 0.88,
                                    "is_humanized": False,
                                    "humanizer_score": 0.12,
                                    "reasons": ["Elevated DNA BScore overlap", "Robotic cadence"],
                                }],
                            })
                        }
                    }],
                    "usage": {"total_tokens": 150, "cost": 0.0002}
                }
            # Stage 2 continuation matching suffix
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": suffix,
                    },
                    "logprobs": {"content": [{"logprob": -0.2}]}
                }],
                "usage": {"total_tokens": 80, "cost": 0.0001}
            }

        mock_chat = MagicMock(side_effect=fake_chat)
        with patch("engine.ai_detector.dna_regeneration.openrouter_chat_completion", mock_chat), \
             patch("engine.ai_detector.arbiter.openrouter_chat_completion", mock_chat):
            res = analyze_text_for_ai(draft, override_api_key="sk-or-test", db_path=self.db_path)

            self.assertEqual(res["mode"], "ensemble_3_stage")
            self.assertTrue(res["low_word_count_warning"])
            self.assertFalse(res["cache_hit"])
            self.assertGreaterEqual(len(res["windows"]), 1)
            for win in res["windows"]:
                self.assertEqual(draft[win["start_index"]:win["end_index"]], win["text"])

            call_count_first_run = mock_chat.call_count
            self.assertGreater(call_count_first_run, 0)

            # Second call with identical inputs should hit cache and NOT call openrouter_chat_completion
            res2 = analyze_text_for_ai(draft, override_api_key="sk-or-test", db_path=self.db_path)
            self.assertTrue(res2["cache_hit"])
            self.assertEqual(mock_chat.call_count, call_count_first_run)
            self.assertEqual(res2["mode"], "ensemble_3_stage")
            self.assertEqual(res2["fraction_ai"], res["fraction_ai"])

    def test_calibrator_low_confidence_asterisk_and_offline_fallback(self) -> None:
        """Verify offline mode runs with zero API cost, and 1%-19% AI triggers low_confidence_asterisk."""
        draft = (
            "Here is a sample text submitted for analysis without any external API calls. "
            "It contains enough characters to meet the minimum character threshold easily."
        )
        res_offline = analyze_text_for_ai(draft, use_openrouter=False, db_path=self.db_path)
        self.assertEqual(res_offline["mode"], "forensic_offline")
        self.assertEqual(res_offline["ensemble_breakdown"]["api_cost_usd"], 0.0)
        self.assertEqual(res_offline["ensemble_breakdown"]["api_tokens"], 0)
        self.assertEqual(res_offline["estimate"]["estimated_cost_usd"], 0.0)

        # Verify low_confidence_asterisk in calibrate_report
        # Build 10 segments of 10 words each: 1 AI window (10%), 9 Human windows (90%)
        segments = []
        for i in range(10):
            w_text = f"Window {i} text with enough words to form segment cleanly."
            segments.append(
                PreprocessedSegment(
                    text=w_text,
                    start_index=i * 60,
                    end_index=i * 60 + len(w_text),
                    word_count=10,
                    sentence_count=1,
                    is_excluded_reference=False,
                    sentences=[],
                )
            )
        stylometrics = {0: {"stylometric_ai_score": 0.85, "stylometric_humanizer_score": 0.0}}
        for i in range(1, 10):
            stylometrics[i] = {"stylometric_ai_score": 0.10, "stylometric_humanizer_score": 0.0}

        report_asterisk = calibrate_report(
            text="dummy text " * 100,
            segments=segments,
            stylometrics_by_idx=stylometrics,
            dna_by_idx={},
            arbiter_res={},
            model_name="test-model",
        )
        self.assertAlmostEqual(report_asterisk.fraction_ai + report_asterisk.fraction_ai_assisted, 0.10, places=2)
        self.assertTrue(report_asterisk.low_confidence_asterisk)
        self.assertIn("10%*", report_asterisk.headline)
        self.assertEqual(report_asterisk.prediction_short, "Low-Confidence")

        # Verify outside 1%-19% does NOT trigger asterisk
        stylometrics_all_human = {i: {"stylometric_ai_score": 0.05, "stylometric_humanizer_score": 0.0} for i in range(10)}
        report_human = calibrate_report(
            text="dummy text " * 100,
            segments=segments,
            stylometrics_by_idx=stylometrics_all_human,
            dna_by_idx={},
            arbiter_res={},
            model_name="test-model",
        )
        self.assertFalse(report_human.low_confidence_asterisk)

    def test_docx_inhouse_file_scan(self) -> None:
        """scan_file_inhouse extracts text from in-memory .docx archive and scores offline."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>'
                'This is an extracted academic essay from a docx archive that contains more than sixty characters.'
                '</w:t></w:r></w:p></w:body></w:document>'
            )
            zf.writestr("word/document.xml", xml)
        docx_bytes = bio.getvalue()

        res = scan_file_inhouse("paper.docx", docx_bytes, use_openrouter=False, db_path=self.db_path)
        self.assertEqual(res["filename"], "paper.docx")
        self.assertEqual(res["scan_type"], "inhouse_file")
        self.assertEqual(res["mode"], "forensic_offline")
        self.assertIn("extracted academic essay", res["text"])
        self.assertIn("fraction_ai", res)
        self.assertIn("fraction_human", res)
        self.assertGreaterEqual(len(res["windows"]), 1)

    def test_modern_cliche_free_ai_vs_genuine_human(self) -> None:
        """Modern RLHF AI prose without 2023 clichés scores >=0.80 AI, while human casual & cited academic stay 0.0."""
        modern_ai = (
            "Effective remote collaboration relies on clear communication norms and well-defined project milestones. "
            "When distributed teams establish shared expectations around response times and documentation, daily handoffs become much more predictable. "
            "Instead of relying on constant synchronous meetings, high-performing organizations document key decisions in centralized workspaces where everyone can review context asynchronously.\n\n"
            "At the same time, maintaining team cohesion requires intentional space for informal connection. "
            "Short check-ins, clear ownership of deliverables, and regular retrospectives help surface blockers early before they compound into larger delays. "
            "Ultimately, balancing structured asynchronous workflows with purposeful human interaction enables remote teams to sustain both velocity and morale over the long term."
        )
        res_ai = analyze_text_for_ai(modern_ai, use_openrouter=False, bypass_cache=True, db_path=self.db_path)
        self.assertGreaterEqual(res_ai["fraction_ai"], 0.80)
        self.assertEqual(res_ai["headline"], "Likely AI-Generated")

        casual_human = (
            "Honestly I had no idea the script would take that long to run. Wait, why did it hang on port 8765? "
            "Turns out Slack Dashboard was already sitting there! Let's just switch the default port to 18940 and add a health check so we never open the wrong tab again. Pretty wild catch tbh."
        )
        res_casual = analyze_text_for_ai(casual_human, use_openrouter=False, bypass_cache=True, db_path=self.db_path)
        self.assertEqual(res_casual["fraction_ai"], 0.0)
        self.assertEqual(res_casual["fraction_ai_assisted"], 0.0)
        self.assertEqual(res_casual["headline"], "Likely Human")

        empirical_academic = (
            "We tested 240 undergraduate participants across two sessions in 2024 (Smith & Patel, 2023; n = 240, p < 0.01). "
            "Why did response accuracy drop sharply on trial four? Fatigue explains part of it. "
            "Still, once stimulus latency exceeded 450 ms, error rates climbed by 18.4% while reaction times stayed flat across older cohorts (Jones et al., 2021)."
        )
        res_acad = analyze_text_for_ai(empirical_academic, use_openrouter=False, bypass_cache=True, db_path=self.db_path)
        self.assertEqual(res_acad["fraction_ai"], 0.0)

    def test_multi_provider_detector_fallback(self) -> None:
        """When OpenRouter key is blank, resolve_openrouter_config falls back to configured Gemini/OpenAI/Anthropic."""
        cfg_gemini = resolve_openrouter_config({"gemini_api_key": "gm-test-key", "gemini_model": "gemini-2.5-flash"})
        self.assertEqual(cfg_gemini["provider"], "gemini")
        self.assertEqual(cfg_gemini["api_key"], "gm-test-key")

        cfg_openai = resolve_openrouter_config({"openai_api_key": "sk-openai-test", "openai_model": "gpt-4o-mini"})
        self.assertEqual(cfg_openai["provider"], "openai")
        self.assertEqual(cfg_openai["api_key"], "sk-openai-test")

    def test_lms_coursework_submission_vs_human_student_answer(self) -> None:
        """Collapsed LMS coursework Q&A text scores >=0.90 AI while excluding prompts; genuine human Q&A scores 0.0."""
        lms_submission = (
            "Q1 Outline the main features of four social media channels for business use. (1.1)\n"
            "Your Answer:\n"
            "Meta (Facebook):Business Pages: Gives businesses a dedicated public profile to display opening times, website links, contact details, and location maps so customers can find them easilyCommunity Groups: Allows businesses to build or join local groups to chat directly with customers and answer questions in a community settingMeta Business Suite & Targeted Ads: Tools that let you schedule posts in advance and run adverts targeted at specific age groups, locations, and interestsLinkedIn:Company Profiles & Networking: A professional platform where companies can post industry news, share updates, and network with other businessesJob Postings & Recruitment: Built-in recruitment tools that let businesses advertise job vacancies and search for qualified candidates\n"
            "[Click to edit](https://westcollegescotland.equal-online.com/learning/assessmentplayer?ls=17607056&q=865217)\n"
            "Q2 Give three examples of when each social media platform is most suitable for business use. (1.2)\n"
            "Your Answer:\n"
            "Instagram:Showcasing visual products: When lifestyle brands like clothing shops, florists, or restaurants want to display high-quality photos and videos of their productsInfluencer partnerships: When teaming up with content creators to demo products and build social proofReaching young adults (18\u201324): When marketing to younger audiences who use Instagram heavily for product discoveryLinkedIn:Business-to-Business (B2B) marketing: When selling corporate services, software, or consultancy directly to other company managers and decision-makers\n"
            "[Click to edit](https://westcollegescotland.equal-online.com/learning/assessmentplayer?ls=17607056&q=865222)\n"
            "Q3 Outline the process for a business to publish content on the four social media channels selected in Q1. (1.3)\n"
            "Your Answer:\n"
            "Meta (Facebook):Log into your business page or open Meta Business SuiteClick \"Create Post\" and select whether you want to post to Facebook, Instagram, or bothType out your message, making sure to include a clear call-to-action (like a link to your site)Upload your image or video and tag a product or location if applicableCheck the preview for desktop and mobile, then hit \"Publish\" or schedule it for a later date and time\n"
            "[Click to edit](https://westcollegescotland.equal-online.com/learning/assessmentplayer?ls=17607056&q=865227)\n"
            "Q8 Explain how the following can impact upon a business\u2019 reputation. (3.1)\n"
            "Your Answer:\n"
            "Most customers today expect to look up a business on social media to check that they are legitimate. Having no presence at all makes a company look outdated or untrustworthy, and it gives competitors an easy advantage because customers will just go to the brands they can find.Posting once every few months or leaving accounts dead for weeks makes a business look disorganised or like it might have closed down. It also means customer comments and questions get left unanswered, which makes the company look uncaring.Only replying to praise while ignoring genuine complaints makes a business look dishonest and out of touch. Customers notice when bad reviews are brushed under the carpet, which damages trust and makes people think the company doesn't care about solving real customer issues."
        )
        res_lms = analyze_text_for_ai(lms_submission, use_openrouter=False, bypass_cache=True, db_path=self.db_path)
        self.assertGreaterEqual(res_lms["fraction_ai"], 0.90)
        self.assertEqual(res_lms["headline"], "Likely AI-Generated")
        excluded = [w for w in res_lms["windows"] if w["is_excluded_reference"]]
        self.assertGreaterEqual(len(excluded), 4)
        for w in res_lms["windows"]:
            self.assertEqual(lms_submission[w["start_index"]:w["end_index"]], w["text"])

        human_student = (
            "Meta Business Pages let small shops put up their phone number, address and opening hours. "
            "People can also leave reviews or message them directly. "
            "I think Facebook works best for local cafes or plumbers where older customers already hang out, whereas Instagram is better if you are selling clothes or food because it is all photos. "
            "When I ran our club page last year we just posted reels twice a week and answered DMs after work. "
            "If a business ignores bad comments people stop trusting them, simple as that."
        )
        res_human = analyze_text_for_ai(human_student, use_openrouter=False, bypass_cache=True, db_path=self.db_path)
        self.assertEqual(res_human["fraction_ai"], 0.0)
        self.assertEqual(res_human["headline"], "Likely Human")


class TestAutoUpdater(BaseEngineTestCase):
    """Test GitHub release update checking, semver comparisons, and URL safety."""

    def test_semver_parsing_and_comparison(self) -> None:
        self.assertEqual(parse_semver("v1.2.0"), (1, 2, 0))
        self.assertTrue(is_newer_version("v1.3.0", "1.2.0"))
        self.assertTrue(is_newer_version("v1.10.0", "1.2.0"))
        self.assertFalse(is_newer_version("v1.2.0", "1.2.0"))
        self.assertFalse(is_newer_version("v1.1.9", "1.2.0"))

    def test_check_github_update_mocked_and_http_route(self) -> None:
        fake_release = {
            "tag_name": "v1.3.0",
            "html_url": "https://github.com/casamensauk/voiceprint/releases/tag/v1.3.0",
            "body": "Adds new voice styles.",
            "published_at": "2026-11-16T09:00:00Z",
            "assets": [
                {
                    "name": "VoicePrint-macOS.zip",
                    "browser_download_url": "https://github.com/casamensauk/voiceprint/releases/download/v1.3.0/VoicePrint-macOS.zip",
                }
            ],
        }
        with patch("engine.updater.urllib.request.urlopen", return_value=_MockHTTPResponse(fake_release)):
            info = check_github_update(current_version="1.2.0")
            self.assertTrue(info["ok"])
            self.assertTrue(info["update_available"])
            self.assertEqual(info["latest_version"], "1.3.0")
            self.assertIn("VoicePrint-macOS.zip", info["download_url"])

        with patch("server.check_github_update", return_value=info):
            httpd = bind_server(host="127.0.0.1", port=0, auto_port=False)
            port = httpd.server_port
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/update/check", method="GET")
                with urllib.request.urlopen(req) as resp:
                    self.assertEqual(resp.status, 200)
                    data = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(data["update_available"])
                    self.assertEqual(data["latest_version"], "1.3.0")
            finally:
                httpd.shutdown()
                httpd.server_close()
                t.join(timeout=2.0)

    def test_stage_and_apply_rejects_untrusted_url(self) -> None:
        with self.assertRaises(UpdaterError) as ctx:
            stage_and_apply_update("https://evil.example.com/VoicePrint-macOS.zip", current_port=18940)
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
