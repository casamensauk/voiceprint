#!/usr/bin/env python3
"""Studio rewrite path: imported own writing as examples, no sample cap, caching, placeholder hygiene."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import rewriter  # noqa: E402
from engine.db import (  # noqa: E402
    add_edit_pair,
    add_sample,
    delete_sample,
    get_connection,
    init_db,
    update_settings,
)
from engine.rewriter import (  # noqa: E402
    _build_llm_prompts,
    rewrite_text,
    rewrite_with_model,
    strip_placeholders,
)

PLACEHOLDERS = ["[email]", "[link]", "[phone]", "[postcode]", "[number]"]


class VoicePoolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="voiceprint_pool_")
        self.db_path = Path(self.temp_dir) / "pool.db"
        self._orig_env = os.environ.get("VOICEPRINT_DB_PATH")
        os.environ["VOICEPRINT_DB_PATH"] = str(self.db_path)
        init_db(self.db_path)
        update_settings({"provider": "local_only"}, db_path=self.db_path)
        rewriter.clear_voice_cache()
        self.prompts: list[str] = []

    def tearDown(self) -> None:
        rewriter.clear_voice_cache()
        if self._orig_env is not None:
            os.environ["VOICEPRINT_DB_PATH"] = self._orig_env
        else:
            os.environ.pop("VOICEPRINT_DB_PATH", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _add(self, text: str, mode: str, source: str = "manual", weight: float = 1.0) -> int:
        return add_sample(text, context_mode=mode, source_type=source, weight=weight, db_path=self.db_path)

    def _examples(self, query: str, mode: str) -> list[dict]:
        ctx = rewriter.resolve_voice_context(mode, db_path=self.db_path)
        return rewriter.select_examples(query, mode, ctx["pool"], ctx["index"])

    def _capture(self):
        def fake(provider, settings, system_prompt, user_prompt):
            self.prompts.append(system_prompt)
            return None, "none"

        return patch("engine.rewriter._call_llm_provider", side_effect=fake)


class TestExamplePool(VoicePoolTestCase):
    def test_long_form_uses_own_dna_and_mixes_imported_examples(self) -> None:
        for i in range(5):
            self._add(f"Document number {i} about gardens and long reflective prose, written slowly.", "long_form", "document")
        for i in range(50):
            self._add(f"Hi Sam, quick note {i} about the garden invoice, thanks.", "work_email", "gmail")
        ctx = rewriter.resolve_voice_context("long_form", db_path=self.db_path)
        self.assertEqual(ctx["dna"]["sample_count"], 5)
        ex = rewriter.select_examples("A quick note about the garden invoice", "long_form", ctx["pool"], ctx["index"])
        self.assertEqual(len(ex), 6)
        self.assertEqual(sum(1 for s in ex if s["source_type"] == "document"), 3)
        self.assertEqual(sum(1 for s in ex if s["source_type"] == "gmail"), 3)
        self.assertEqual(len({s["id"] for s in ex}), 6)

    def test_mode_without_own_samples_gets_six_from_pool(self) -> None:
        for i in range(20):
            self._add(f"Morning all, update {i} on the rota.", "work_email", "gmail")
        for i in range(10):
            self._add(f"lol ok see you at {i}", "casual_chat", "google_chat")
        ex = self._examples("update on the rota", "social_post")
        self.assertEqual(len(ex), 6)
        self.assertTrue(all(s["source_type"] in ("gmail", "google_chat") for s in ex))

    def test_two_own_samples_keep_under_three_fallback(self) -> None:
        self._add("First own post, short.", "social_post")
        self._add("Second own post, short.", "social_post")
        for i in range(10):
            self._add(f"Other mode sample {i} here.", "casual_chat")
        ctx = rewriter.resolve_voice_context("social_post", db_path=self.db_path)
        self.assertEqual(ctx["dna"]["sample_count"], 12)
        ex = self._examples("Other mode sample", "social_post")
        own = [s for s in ex if s["context_mode"] == "social_post"]
        self.assertEqual(len(own), 2)
        self.assertEqual(len(ex), 6)

    def test_no_500_cap(self) -> None:
        for i in range(700):
            self._add(f"filler line {i} about nothing much at all", "work_email", "gmail")
        # Oldest sample (outside the newest 500) is the only close match.
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE samples SET content = ? WHERE id = 1",
                ("zebra quartz xylophone marmalade invoice reconciliation",),
            )
            conn.commit()
        ex = self._examples("zebra quartz xylophone marmalade invoice reconciliation", "work_email")
        self.assertIn(1, [s["id"] for s in ex])


class TestCache(VoicePoolTestCase):
    def setUp(self) -> None:
        super().setUp()
        for i in range(6):
            self._add(f"Own sample {i} with some words in it.", "casual_chat")
        for i in range(10):
            self._add(f"Mail {i} regarding the thing.", "work_email", "gmail")

    def _count_calls(self):
        real_dna, real_idx = rewriter.extract_voice_dna, rewriter.build_retrieval_index
        calls = {"dna": 0, "idx": 0}

        def dna(*a, **k):
            calls["dna"] += 1
            return real_dna(*a, **k)

        def idx(*a, **k):
            calls["idx"] += 1
            return real_idx(*a, **k)

        return calls, patch.object(rewriter, "extract_voice_dna", dna), patch.object(rewriter, "build_retrieval_index", idx)

    def test_second_call_hits_cache(self) -> None:
        calls, p1, p2 = self._count_calls()
        with p1, p2:
            rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
            rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
        self.assertEqual(calls, {"dna": 1, "idx": 1})

    def test_add_delete_and_reweight_invalidate(self) -> None:
        calls, p1, p2 = self._count_calls()
        with p1, p2:
            rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
            new_id = self._add("A brand new sample appears.", "casual_chat")
            ctx = rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
            self.assertEqual(calls["dna"], 2)
            self.assertEqual(ctx["dna"]["sample_count"], 7)
            self.assertIn(new_id, [s["id"] for s in ctx["pool"]])

            delete_sample(new_id, db_path=self.db_path)
            ctx = rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
            self.assertEqual(calls["dna"], 3)
            self.assertNotIn(new_id, [s["id"] for s in ctx["pool"]])

            with get_connection(self.db_path) as conn:
                conn.execute("UPDATE samples SET weight = 2.5 WHERE id = 2")
                conn.commit()
            ctx = rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
            self.assertEqual(calls["dna"], 4)
            self.assertEqual(next(s for s in ctx["pool"] if s["id"] == 2)["weight"], 2.5)
            rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
            self.assertEqual(calls["dna"], 4)

    def test_cached_dna_is_not_shared_mutably(self) -> None:
        a = rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
        a["dna"]["sample_count"] = -1
        b = rewriter.resolve_voice_context("casual_chat", db_path=self.db_path)
        self.assertNotEqual(b["dna"]["sample_count"], -1)


class TestPlaceholders(VoicePoolTestCase):
    def test_prompt_contains_no_placeholders(self) -> None:
        samples = [
            {"content": f"Please mail me at {p} or call, it is about the {p} thing (see {p})."} for p in PLACEHOLDERS
        ]
        edits = [{"ai_draft": "Ring [phone] now", "user_final_edit": "Email [email] , later [link]"}]
        sys_prompt, _ = _build_llm_prompts("draft", "work_email", "deep", {}, [], samples, edits)
        for p in PLACEHOLDERS:
            self.assertNotIn(p, sys_prompt)
        self.assertNotIn("()", sys_prompt)
        block = sys_prompt.split("AUTHENTIC WRITING SAMPLES")[1].split("PAST CORRECTIONS")[0]
        self.assertNotIn("  ", block)
        self.assertNotIn(" ,", block)

    def test_excerpt_is_cut_after_cleaning(self) -> None:
        text = "[link] " * 100 + "KEEPME" + "x" * 500
        self.assertNotIn("[link]", strip_placeholders(text))
        sys_prompt, _ = _build_llm_prompts("d", "work_email", "deep", {}, [], [{"content": text}], [])
        self.assertIn("KEEPME", sys_prompt)

    def _run(self, fn, draft: str, reply: str) -> str:
        with patch("engine.rewriter._call_llm_provider", return_value=(reply, "openrouter:test")):
            if fn is rewrite_text:
                return rewrite_text(draft, context_mode="work_email", save_history=False, db_path=self.db_path)["rewritten"]
            return rewrite_with_model(draft, "x/y", context_mode="work_email", db_path=self.db_path)["rewritten"]

    def test_output_placeholders_removed_unless_in_draft(self) -> None:
        for fn in (rewrite_text, rewrite_with_model):
            out = self._run(fn, "Send them the report today please.", "Send them the report at [link] today please.")
            self.assertNotIn("[link]", out)
            self.assertNotIn("  ", out)
            kept = self._run(fn, "Send them [link] today please.", "Send them [link] today please, or [email].")
            self.assertIn("[link]", kept)
            self.assertNotIn("[email]", kept)

    def test_end_to_end_prompt_clean(self) -> None:
        self._add("Call me on [phone] or write to [email] about the [number] invoices.", "work_email", "gmail")
        add_edit_pair("work_email", "a", "Ring [phone]", "Ring [email]", [], db_path=self.db_path)
        with self._capture():
            rewrite_text("Please send the invoices.", context_mode="work_email", save_history=False, db_path=self.db_path)
        self.assertTrue(self.prompts)
        for p in PLACEHOLDERS:
            self.assertNotIn(p, self.prompts[0])


if __name__ == "__main__":
    unittest.main()
