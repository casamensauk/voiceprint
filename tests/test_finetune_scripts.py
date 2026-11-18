#!/usr/bin/env python3
"""Fine-tuning experiment scripts: filters, chunking, notes cache, dataset shape, dry runs, metrics, scan."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "finetune"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common  # noqa: E402
import generate  # noqa: E402
import prepare  # noqa: E402
import scan  # noqa: E402
import train  # noqa: E402
from engine.db import add_sample, get_connection, init_db, update_settings  # noqa: E402

GOOD_NOTES = "- The meeting moved to Thursday\n- Bring the signed forms\n- Lunch is provided"


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def sentence_text(n_sentences: int, tag: str = "a") -> str:
    return " ".join(f"This is sentence {tag}{i} about the garden plan and the budget for spring." for i in range(n_sentences))


def run(mod, argv) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = mod.main(argv)
    return code, buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = self.tmp / "t.db"
        self.work = self.tmp / "work"
        init_db(self.db)

    def add(self, text: str, source: str = "gmail") -> None:
        add_sample(text, source_type=source, db_path=self.db)

    def prep(self, *extra: str) -> tuple[int, str]:
        return run(prepare, ["--db", str(self.db), "--work-dir", str(self.work), *extra])


class FilterTests(Base):
    def counts(self, rows: list[tuple[str, str]]) -> dict:
        _, counts = prepare.build_chunks(rows)
        return counts

    def test_each_filter_is_counted(self) -> None:
        ok = sentence_text(4, "ok")
        rows = [
            ("gmail", "too short to keep"),
            ("gmail", sentence_text(200, "long")),  # > 1500 words
            ("gmail", ok + " Click to unsubscribe here."),
            ("gmail", ok + " Please do not reply to this message."),
            ("gmail", sentence_text(4, "fw") + "\n-----Original Message-----\nFrom: x"),
            ("gmail", sentence_text(4, "dup")),
            ("gmail", sentence_text(4, "dup").upper()),  # near-duplicate
            ("gmail", ok),
        ]
        c = self.counts(rows)
        self.assertEqual(c["dropped_under_20_words"], 1)
        self.assertEqual(c["dropped_over_1500_words"], 1)
        self.assertEqual(c["dropped_automated"], 2)
        self.assertEqual(c["dropped_forwarded_or_original_message"], 1)
        self.assertEqual(c["dropped_near_duplicate"], 1)
        self.assertEqual(c["kept_samples"], 2)

    def test_long_documents_are_not_dropped_but_long_mail_is(self) -> None:
        long = sentence_text(200, "x")
        c_mail = self.counts([("gmail", long)])
        c_doc = self.counts([("document", long)])
        self.assertEqual(c_mail["dropped_over_1500_words"], 1)
        self.assertEqual(c_doc["kept_samples"], 1)
        self.assertGreater(c_doc["chunks"], 1)

    def test_signature_lines_and_placeholders_removed(self) -> None:
        text = sentence_text(3, "sig") + "\nSent from my iPhone\nGet Outlook for iOS\nWrite to [email] today."
        chunks, _ = prepare.build_chunks([("gmail", text)])
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("Sent from", chunks[0]["text"])
        self.assertNotIn("Outlook", chunks[0]["text"])
        self.assertNotIn("[email]", chunks[0]["text"])

    def test_no_documents_flag(self) -> None:
        self.add(sentence_text(4, "m"), "gmail")
        self.add(sentence_text(4, "d"), "document")
        _, out = self.prep()
        self.assertIn("samples_document: 1", out)
        _, out = self.prep("--no-documents")
        self.assertNotIn("samples_document", out)


class ChunkTests(unittest.TestCase):
    def test_chunks_respect_limit_and_sentences(self) -> None:
        paras = "\n\n".join(sentence_text(12, f"p{i}") for i in range(10))
        chunks = common.split_long_text(paras)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(common.word_count(c), 400)
            self.assertTrue(c.rstrip().endswith("."))
            self.assertTrue(c.startswith("This is sentence"))

    def test_big_paragraph_falls_back_to_sentences(self) -> None:
        chunks = common.split_long_text(sentence_text(100, "big"))  # one 1300-word paragraph
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(common.word_count(c), 400)
            self.assertTrue(c.rstrip().endswith("."))
        self.assertEqual(" ".join(chunks).split(), sentence_text(100, "big").split())

    def test_single_huge_sentence_is_the_only_mid_sentence_cut(self) -> None:
        chunks = common.split_long_text(words("w", 900) + ".")
        self.assertEqual([common.word_count(c) for c in chunks], [400, 400, 100])


class NotesTests(Base):
    def setUp(self) -> None:
        super().setUp()
        for i in range(5):
            self.add(sentence_text(4, f"s{i}"))

    def test_resume_and_limit(self) -> None:
        calls = []

        def fake(model, prompt, *a, **k):
            calls.append(prompt)
            return GOOD_NOTES

        with patch.object(common, "ollama_chat", fake):
            code, _ = self.prep("--run", "--limit", "2")
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(common.read_jsonl(self.work / "notes.jsonl")), 2)
            code, out = self.prep("--run")
            self.assertEqual(len(calls), 5)  # only the three uncached chunks were asked
            self.assertEqual(len(common.read_jsonl(self.work / "notes.jsonl")), 5)
            self.assertIn("notes cached: 2, still needed: 3", out)

    def test_invalid_notes_retried_then_skipped(self) -> None:
        calls = []

        def fake(model, prompt, *a, **k):
            calls.append(1)
            return "just one line with no bullets"

        with patch.object(common, "ollama_chat", fake):
            code, out = self.prep("--run", "--limit", "1")
        self.assertEqual(len(calls), 2)  # first try + one retry
        self.assertIn("failed and skipped: 1", out)
        self.assertEqual(common.read_jsonl(self.work / "notes.jsonl"), [])

    def test_retry_recovers(self) -> None:
        replies = iter(["nope", GOOD_NOTES])
        with patch.object(common, "ollama_chat", lambda *a, **k: next(replies)):
            self.assertEqual(len(common.make_notes("text", "m")), 3)

    def test_overlong_notes_rejected(self) -> None:
        self.assertIsNone(common.parse_notes("\n".join(f"- {words('n', 30)}" for _ in range(8))))

    def test_progress_every_25(self) -> None:
        for i in range(30):
            self.add(sentence_text(4, f"extra{i}"))
        with patch.object(common, "ollama_chat", lambda *a, **k: GOOD_NOTES):
            _, out = self.prep("--run")
        self.assertIn("25/35 chunks", out)

    def test_dry_run_writes_nothing_and_calls_nothing(self) -> None:
        with patch.object(common, "ollama_chat", side_effect=AssertionError("HTTP call in dry run")):
            code, out = self.prep()
        self.assertEqual(code, 0)
        self.assertFalse(self.work.exists())
        self.assertIn("DRY RUN", out)
        self.assertIn("still needed: 5", out)
        self.assertIn("5 notes x 8s", out)


class DatasetTests(Base):
    def test_split_is_deterministic_and_dataset_shape(self) -> None:
        for i in range(60):
            self.add(sentence_text(4, f"d{i}"), ("gmail", "google_chat", "document")[i % 3])
        with patch.object(common, "ollama_chat", lambda *a, **k: GOOD_NOTES):
            self.prep("--run")
        first = {n: (self.work / "dataset" / f"{n}.jsonl").read_text() for n in ("train", "valid", "test")}
        shutil.rmtree(self.work / "dataset")
        with patch.object(common, "ollama_chat", side_effect=AssertionError("cached")):
            self.prep("--run")
        for n, text in first.items():
            self.assertEqual((self.work / "dataset" / f"{n}.jsonl").read_text(), text)
        self.assertEqual(prepare.split_for("00000000" + "a" * 56), "train")
        self.assertEqual(prepare.split_for("0000005e" + "a" * 56), "valid")  # 94 % 100
        self.assertEqual(prepare.split_for("0000005f" + "a" * 56), "test")  # 95 % 100
        manifest = json.loads((self.work / "dataset" / "manifest.json").read_text())
        self.assertEqual(sum(manifest["examples"].values()), 60)
        self.assertEqual(manifest["approx_tokens"], int(manifest["total_words"] * 1.4))
        self.assertEqual(set(manifest["per_source"]), {"gmail", "google_chat", "document"})

        kinds = {"an email", "a chat message", "a piece of writing"}
        for n, text in first.items():
            for line in text.splitlines():
                row = json.loads(line)
                self.assertEqual(list(row), ["messages"])
                user, asst = row["messages"]
                self.assertEqual((user["role"], asst["role"]), ("user", "assistant"))
                head, _, rest = user["content"].partition("\n")
                self.assertTrue(any(head.startswith(f"Write {k} of about ") for k in kinds))
                self.assertTrue(head.endswith(" words in my own voice, covering these points:"))
                self.assertTrue(all(b.startswith("- ") for b in rest.splitlines()))
                self.assertNotIn("\n", line)

    def test_word_count_rounded_to_ten(self) -> None:
        chunk = {"source": "gmail", "text": words("q", 87), "hash": "x"}
        user = prepare.example_for(chunk, ["a", "b"])["messages"][0]["content"]
        self.assertTrue(user.startswith("Write an email of about 90 words"))


class TrainTests(Base):
    def dataset(self, n_train: int) -> None:
        (self.work / "dataset").mkdir(parents=True)
        (self.work / "dataset" / "train.jsonl").write_text('{"messages": []}\n')
        (self.work / "dataset" / "manifest.json").write_text(json.dumps({"examples": {"train": n_train}}))

    def test_iteration_arithmetic(self) -> None:
        self.assertEqual(train.compute_iters(1000, 3, 2), 1500)
        self.assertEqual(train.compute_iters(1001, 3, 2), 1502)
        self.assertEqual(train.compute_iters(0, 3, 2), 1)

    def test_dry_run_runs_nothing(self) -> None:
        self.dataset(1000)
        with patch.object(train, "run_command", side_effect=AssertionError("subprocess in dry run")):
            code, out = run(train, ["--work-dir", str(self.work)])
        self.assertEqual(code, 0)
        self.assertIn("mlx-community/gemma-4-12B-it-4bit", out)
        self.assertIn("huggingface.co", out)
        self.assertIn("6.8 GB", out)
        self.assertIn("Iterations: 1500", out)
        self.assertNotIn("--mask-prompt", out)
        self.assertIn("--adapter-path adapter", out)
        self.assertIn("--data dataset-rendered", out)

    def test_refuses_without_venv_or_dataset(self) -> None:
        with patch.object(train, "run_command", side_effect=AssertionError("no")):
            code, out = run(train, ["--work-dir", str(self.work), "--run"])
            self.assertEqual(code, 1)
            self.assertIn("--setup", out)
            (self.work / ".venv" / "bin").mkdir(parents=True)
            (self.work / ".venv" / "bin" / "python").write_text("")
            code, out = run(train, ["--work-dir", str(self.work), "--trial"])
            self.assertEqual(code, 1)
            self.assertIn("prepare.py", out)

    CANNED = (
        "Loading pretrained model\nTrainable parameters: 0.1% (12M/12000M)\n"
        "  Iter    Train loss    Tokens/sec    Trained tokens\n"
        "     5    3.210 ▼      900    2.0k\n"
        "    10    2.900 ▼      1,100    4.4k\n"
        "    15    2.700 ▼      1,300    6.6k\n"
        "Saved final weights to adapter-trial/adapters.safetensors.\n"
    )

    def test_parse_speed_from_canned_log(self) -> None:
        sp = train.parse_speed(self.CANNED)
        self.assertAlmostEqual(sp["tokens_per_sec"], 1200)
        # tokens per iteration is about 440, so ~2.7 iterations/sec
        self.assertAlmostEqual(sp["iters_per_sec"], 1200 / 440, delta=0.15)
        classic = "Iter 10: Train loss 2.1, Learning Rate 1e-5, It/sec 0.5, Tokens/sec 120.0, Trained Tokens 900"
        self.assertEqual(train.parse_speed(classic), {"tokens_per_sec": 120.0, "iters_per_sec": 0.5})
        self.assertEqual(train.parse_speed("nothing useful"), {})

    def test_trial_prints_estimate(self) -> None:
        self.dataset(1000)
        (self.work / ".venv" / "bin").mkdir(parents=True)
        (self.work / ".venv" / "bin" / "python").write_text("")
        seen = []

        def fake(cmd, cwd, log_path=None):
            seen.append(cmd)
            return 0, self.CANNED

        with patch.object(train, "run_command", fake):
            code, out = run(train, ["--work-dir", str(self.work), "--trial"])
        self.assertEqual(code, 0)
        self.assertIn("render_dataset.py", seen[0][1])
        self.assertIn("--iters", seen[1])
        self.assertEqual(seen[1][seen[1].index("--iters") + 1], "40")
        self.assertIn("adapter-trial", seen[1])
        self.assertIn("Estimate for the full run: 1500 iterations", out)

    def test_setup_is_idempotent(self) -> None:
        calls = []

        def fake(cmd, cwd, log_path=None):
            calls.append(cmd[:2])
            if cmd[0] == "uv" and cmd[1] == "venv":
                (self.work / ".venv" / "bin").mkdir(parents=True)
                (self.work / ".venv" / "bin" / "python").write_text("")
            return 0, f"{train.MLX_LM_VERSION}\n"

        with patch.object(train, "run_command", fake):
            run(train, ["--work-dir", str(self.work), "--setup"])
            first = len(calls)
            run(train, ["--work-dir", str(self.work), "--setup"])
        self.assertEqual(len(calls) - first, 1)  # second run only checks the installed version


class GenerateTests(Base):
    def setUp(self) -> None:
        super().setUp()
        (self.work / "dataset").mkdir(parents=True)
        self.copied = " ".join(f"alpha{i}" for i in range(12))
        train_text = f"We agreed to start early. {self.copied} and then we left for the harbour."
        (self.work / "dataset" / "train.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": train_text}]}) + "\n"
        )
        (self.work / ".venv" / "bin").mkdir(parents=True)
        (self.work / ".venv" / "bin" / "python").write_text("")
        (self.work / "adapter").mkdir()
        self.draft = self.tmp / "d.txt"
        self.draft.write_text(
            "Please confirm that Maria will bring 3 laptops to the Leeds office on 14 March. "
            "We also need the invoice from Contoso before Friday."
        )

    def gen(self, outputs: list[str], *extra: str) -> tuple[int, str, list]:
        it = iter(outputs)
        calls = []

        def fake_run(cmd, prompt):
            calls.append((cmd, prompt))
            return "==========\n" + next(it) + "\n==========\nPrompt: 10 tokens\nGeneration: 20 tokens\n"

        with patch.object(common, "ollama_chat", lambda *a, **k: GOOD_NOTES), patch.object(
            generate, "run_mlx_generate", fake_run
        ):
            code, out = run(generate, ["--work-dir", str(self.work), "--file", str(self.draft), *extra])
        return code, out, calls

    def latest(self) -> dict:
        files = sorted((self.work / "outputs").glob("*.json"))
        return json.loads(files[-1].read_text())

    def test_metrics_with_copied_run_and_greeting(self) -> None:
        out_text = f"Hi Sam,\n\nMaria is bringing 3 laptops to Leeds on 14 March. {self.copied} is copied here.\n\nBest,\nAlex"
        code, out, calls = self.gen([out_text, "A plain answer about nothing in particular here."], "--run", "--baseline", "--show", "--max-tries", "1")
        self.assertEqual(code, 0, out)
        data = self.latest()
        m = data["drafts"][0]["adapter"]["metrics"]
        self.assertGreaterEqual(m["longest_copied_run"], 12)
        self.assertTrue(m["email_like"])
        self.assertGreater(m["fact_coverage"], 0.5)
        self.assertIsInstance(m["inhouse_human_score"], int)
        base = data["drafts"][0]["baseline"]["metrics"]
        self.assertFalse(base["email_like"])
        self.assertLess(base["longest_copied_run"], 5)
        self.assertIn("in-house score only", out)
        self.assertIn("--- file d.txt / adapter ---", out)
        # the draft text is never stored
        self.assertNotIn("Contoso", json.dumps(data))
        self.assertNotIn("Contoso", "".join(c[1] for c in calls))

    def test_command_flags(self) -> None:
        _, _, calls = self.gen(["Some answer text for the check, long enough."], "--run", "--temp", "0.5", "--seed", "7")
        cmd, prompt = calls[0]
        self.assertEqual(cmd[1:4], ["-m", "mlx_lm", "generate"])
        self.assertEqual(cmd[cmd.index("--temp") + 1], "0.5")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "7")
        self.assertIn("--adapter-path", cmd)
        self.assertTrue(prompt.startswith("Write a piece of writing of about "))
        words_in = common.word_count(self.draft.read_text())
        self.assertEqual(int(cmd[cmd.index("--max-tokens") + 1]), int(common.round10(words_in) * 2.5))

    def test_dry_run_shows_word_counts_only(self) -> None:
        with patch.object(generate, "run_mlx_generate", side_effect=AssertionError("ran")), patch.object(
            common, "ollama_chat", side_effect=AssertionError("ollama")
        ):
            code, out = run(generate, ["--work-dir", str(self.work), "--file", str(self.draft)])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn("words (file d.txt)", out)
        self.assertNotIn("Contoso", out)
        self.assertFalse((self.work / "outputs").exists())

    def test_history_default_distinct_and_short_skipped(self) -> None:
        with get_connection(self.db) as conn:
            for t in ["short", "x" * 60 + " one", "x" * 60 + " one", "y" * 60 + " two"]:
                conn.execute(
                    "INSERT INTO rewrite_history (context_mode, input_text, output_text, created_at) VALUES ('c', ?, 'o', 'n')",
                    (t,),
                )
            conn.commit()
        drafts = generate.load_drafts([], 5, self.db)
        self.assertEqual(len(drafts), 2)
        code, out = run(generate, ["--work-dir", str(self.work), "--db", str(self.db)])
        self.assertEqual(code, 0)
        self.assertIn("draft 2:", out)

    def test_parse_generated(self) -> None:
        self.assertEqual(generate.parse_generated("==========\nHello there\n==========\nPrompt: 3"), "Hello there")
        self.assertEqual(generate.parse_generated("just text"), "just text")
        self.assertEqual(
            generate.parse_generated("==========\n<|channel>thought\nplan plan\n<channel|>Real answer.\n=========="),
            "Real answer.",
        )
        self.assertEqual(generate.parse_generated("==========\n<|channel>thought\nnever closed\n=========="), "")

    def test_generate_command_switches_thinking_off(self) -> None:
        cmd = generate.build_generate_command(common.Layout(Path("/tmp/x")), "m", None, 100, 0.7, 1)
        self.assertEqual(json.loads(cmd[cmd.index("--chat-template-config") + 1]), {"enable_thinking": False})

    def test_placeholders_and_hidden_chars_cleaned(self) -> None:
        self.assertEqual(generate.clean_output("Hello​ world [link]"), "Hello world")


class ScanTests(Base):
    def setUp(self) -> None:
        super().setUp()
        (self.work / "outputs").mkdir(parents=True)
        self.path = self.work / "outputs" / "20260101T000000000000Z.json"
        self.path.write_text(
            json.dumps(
                {
                    "drafts": [
                        {"source": "file a.txt", "adapter": {"text": words("a", 60)}, "baseline": {"text": words("b", 60)}},
                        {"source": "file b.txt", "adapter": {"text": words("c", 60)}},
                        {"source": "file c.txt", "adapter": {"text": words("d", 60)}},
                    ]
                }
            )
        )

    def test_dry_run_makes_no_call(self) -> None:
        with patch.object(scan, "scan_text_pangram", side_effect=AssertionError("paid call in dry run")):
            code, out = run(scan, ["--db", str(self.db), "--work-dir", str(self.work)])
        self.assertEqual(code, 0)
        self.assertIn("Texts: 3 (180 words)", out)
        self.assertIn("Estimated Pangram cost", out)
        self.assertIn("Pangram key configured: no", out)

    def test_run_stores_rows_prints_summary_and_survives_failure(self) -> None:
        update_settings({"pangram_api_key": "test-key"}, self.db)
        results = {"a": 0.05, "c": 0.9}

        def fake(text, api_key, model="pangram-4", public_dashboard_link=False, **k):
            if text.startswith("c0"):
                raise RuntimeError("boom")
            ai = results["c" if text.startswith("d0") else "a"]
            return {
                "scan_type": "text",
                "headline": "h",
                "fraction_ai": ai,
                "fraction_human": 1 - ai,
                "max_humanizer_score": 0.1,
                "input_text": text,
                "windows": [],
                "estimate": {},
            }

        with patch.object(scan, "scan_text_pangram", fake):
            code, out = run(scan, ["--db", str(self.db), "--work-dir", str(self.work), "--run"])
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)
        self.assertIn("FAIL", out)
        self.assertIn("Pangram error: boom", out)
        self.assertIn("1 of 3 passed", out)
        with get_connection(self.db) as conn:
            rows = conn.execute("SELECT title FROM pangram_scans ORDER BY id").fetchall()
        self.assertEqual([r["title"] for r in rows], ["finetune: file a.txt", "finetune: file c.txt"])
        saved = json.loads(self.path.read_text())
        self.assertTrue(saved["drafts"][0]["adapter"]["pangram"]["passed"])
        self.assertIn("pangram_error", saved["drafts"][1]["adapter"])
        self.assertNotIn("pangram", saved["drafts"][0]["baseline"])  # baseline only with --include-baseline

    def test_include_baseline_counts_more_texts(self) -> None:
        code, out = run(scan, ["--db", str(self.db), "--work-dir", str(self.work), "--include-baseline"])
        self.assertIn("Texts: 4", out)



class GenerateDefectTests(unittest.TestCase):
    def test_clean_text_has_no_defects(self):
        text = "Instagram suits a lifestyle business. People aged 18 to 24 use it to find new products."
        self.assertEqual(generate.find_defects(text, 16), [])

    def test_broken_lines(self):
        text = "ONLINE\nBRAND IDENTITY\nConsists of visual\ncomponents and unique\npersonality traits."
        self.assertIn("broken_lines", generate.find_defects(text, 10))

    def test_lowercase_start_and_unfinished_paragraph(self):
        text = "Use the dashboard to measure your content\n\nuse average watch time to see how well it performs."
        defects = generate.find_defects(text, 15)
        self.assertIn("lowercase_start", defects)
        self.assertIn("unfinished_paragraph", defects)

    def test_repetition(self):
        text = (
            "Instagram is the best platform for me to reach a fitness audience. It is highly visual. "
            "Instagram is the best platform for me as a health and fitness influencer."
        )
        self.assertIn("repetition", generate.find_defects(text, 30))

    def test_too_short(self):
        self.assertIn("too_short", generate.find_defects("A short and complete sentence here.", 100))

    def test_generate_checked_retries_until_clean(self):
        outputs = {5: "bad start\n\nno ending", 6: "A clean sentence that is complete. Another one follows it here."}
        seeds = []

        def make(seed):
            seeds.append(seed)
            return outputs[seed]

        text, defects, tries = generate.generate_checked(make, 10, 5, 4)
        self.assertEqual((seeds, defects, tries), ([5, 6], [], 2))
        self.assertTrue(text.startswith("A clean"))

    def test_generate_checked_keeps_least_faulty(self):
        outputs = {1: "bad start\n\nno ending", 2: "Fine sentence but far too short."}
        text, defects, tries = generate.generate_checked(lambda s: outputs[s], 100, 1, 2)
        self.assertEqual((defects, tries), (["too_short"], 2))


class GenerateSectionsTests(GenerateTests):
    def test_sections_are_written_separately_and_joined(self) -> None:
        heading = "Q1 Outline the features"
        para_a = " ".join(f"alpha{i}" for i in range(30)) + "."
        para_b = " ".join(f"beta{i}" for i in range(30)) + "."
        self.draft.write_text(f"{heading}\n\n{para_a}\n\n{para_b}")
        self.assertEqual(len(generate.split_sections(self.draft.read_text(), 40)), 3)
        out_a = "First section answer that is long enough to count as complete here. It has a second sentence too."
        out_b = "Second section answer that is also long enough to count as complete. It ends properly as well."
        code, out, calls = self.gen([out_a, out_b], "--run", "--sections", "40", "--max-tries", "1")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(calls), 2)  # the heading is kept, not rewritten
        drafts = self.latest()["drafts"]
        self.assertEqual([d["source"].split()[-1] for d in drafts], ["s02", "s03", "joined"])
        self.assertTrue(drafts[-1]["joined"])
        self.assertEqual(drafts[-1]["adapter"]["text"], f"{heading}\n\n{out_a}\n\n{out_b}")


class ScanSkipTests(ScanTests):
    def test_already_scanned_texts_are_not_scanned_again(self) -> None:
        data = json.loads(self.path.read_text())
        data["drafts"][0]["adapter"]["pangram"] = {"passed": True}
        self.path.write_text(json.dumps(data))
        code, out = run(scan, ["--db", str(self.db), "--work-dir", str(self.work), "--include-baseline"])
        self.assertEqual(code, 0)
        self.assertIn("Texts: 3 (180 words)", out)  # b.txt, c.txt and a.txt's baseline; not a.txt's adapter

if __name__ == "__main__":
    unittest.main()


class RenderDatasetTests(unittest.TestCase):
    """render_dataset.py with a stand-in tokenizer shaped like a thinking model's template."""

    class FakeTokenizer:
        bos_token = "<bos>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
            out = "<bos>"
            for m in messages:
                role = "model" if m["role"] == "assistant" else m["role"]
                out += f"<|turn>{role}\n{m['content']}<turn|>\n"
            if add_generation_prompt:
                out += "<|turn>model\n" + ("" if enable_thinking else "<|channel>thought\n<channel|>")
            return out

    def test_render_matches_generation_prompt_and_keeps_one_bos(self) -> None:
        import render_dataset

        msgs = [{"role": "user", "content": "Write it."}, {"role": "assistant", "content": "My answer."}]
        body = "<|turn>user\nWrite it.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>My answer.<turn|>\n"
        # Tokenizer whose encode() adds no BOS: the literal one stays.
        self.assertEqual(render_dataset.render_example(self.FakeTokenizer(), msgs), "<bos>" + body)

        class AddsBos(self.FakeTokenizer):
            bos_token_id = 2

            def encode(self, text):
                return [2, 7]

        # Tokenizer whose encode() adds BOS itself: the literal one is dropped.
        self.assertEqual(render_dataset.render_example(AddsBos(), msgs), body)

    def test_render_dir_writes_single_line_text_rows(self) -> None:
        import render_dataset

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "dataset", Path(tmp) / "dataset-rendered"
            src.mkdir()
            row = {"messages": [{"role": "user", "content": "Line one\nline two"}, {"role": "assistant", "content": "A\nB"}]}
            (src / "train.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            counts = render_dataset.render_dir(self.FakeTokenizer(), src, dst)
            self.assertEqual(counts, {"train": 2})
            lines = (dst / "train.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("A\nB<turn|>", json.loads(lines[0])["text"])

    def test_render_rejects_rows_without_an_answer(self) -> None:
        import render_dataset

        with self.assertRaises(ValueError):
            render_dataset.render_example(self.FakeTokenizer(), [{"role": "user", "content": "x"}])
