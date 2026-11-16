#!/usr/bin/env python3
"""Tests for the review fixes on the Google import: own-text cleaning, scan robustness, atomic commit, request guards."""

from __future__ import annotations

import http.client
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_google_import as base  # noqa: E402
from engine import chat_takeout, db as dbmod, gmail_import, google_auth, pending_import  # noqa: E402
from engine.db import (  # noqa: E402
    clear_samples,
    delete_sample,
    get_imported_ids,
    get_samples,
    init_db,
    update_settings,
)
from engine.own_text import extract_own_text  # noqa: E402
from server import VoicePrintHandler  # noqa: E402

ME = base.ME
GOOD = base.GOOD
make_msg = base.make_msg
b64 = base.b64


# --------------------------------------------------------------------------- own-text table

OWN = "I will send the revised figures over to you by Friday afternoon."

KEEP_WHOLE = [
    ("header-looking lines written by the user",
     "From: London\nTo: Paris\nDate: 3 May\nWe leave at nine, bring the tickets and passports please."),
    ("bare double dash is not a signature delimiter",
     "Intro text here for you.\n--\nI mean this honestly, we should talk about the rest of it."),
    ("on-wrote inside a sentence",
     "On Tuesday the vendor wrote: we cannot deliver. I think we should push back."),
    ("a lone quoted line is kept",
     "As my colleague put it:\n> measure twice, cut once\nand I think that applies to this plan as well."),
    ("long text after dash-dash-space is not a signature",
     "Notes follow.\n-- \n" + "\n".join(f"Point number {i} that I actually want to make today." for i in range(12))),
    ("a name-like final line without a sign-off stays",
     "The meeting notes are attached for everyone to read.\nSam"),
    ("sign-off must not empty the message",
     "Thanks,\nSam"),
    ("apostrophe names are not stripped",
     "The plan is agreed and ready to go ahead today.\nBest,\nO'Neil Smith"),
    ("from line with only a place and one more header, no blank line",
     "From: the office in Leeds\nSent: last week by courier and it arrived today safely."),
    ("on-wrote text mid message with no quote after it",
     "Alex wrote:\nwe cannot deliver this week, which is a real problem for the launch plan."),
]

CUT = [
    ("english on-wrote",
     f"{OWN}\n\nOn Mon, 3 Mar 2025 at 10:00, Alex <alex@example.com> wrote:\n> Can you send them?\n> Thanks"),
    ("top-posted outlook reply with a real header block",
     f"{OWN}\n\nFrom: Alex Example <alex@example.com>\nSent: Monday, 3 March 2025 10:00\nTo: Sam Example\nSubject: Figures\n\nCan you send the figures?"),
    ("outlook header block with no addresses but two header lines and a blank",
     f"{OWN}\n\nFrom: Alex Example\nSent: Monday, 3 March 2025 10:00\n\nCan you send the figures?"),
    ("french",
     f"{OWN}\n\nLe lun. 3 mars 2025 à 10:00, Alex <alex@example.com> a écrit :\n> Peux-tu envoyer les chiffres ?"),
    ("french header block",
     f"{OWN}\n\nDe : Alex Example\nEnvoyé : lundi 3 mars 2025 10:00\nÀ : Sam Example\nObjet : Chiffres\n\nPeux-tu envoyer les chiffres ?"),
    ("german",
     f"{OWN}\n\nAm 03.03.2025 um 10:00 schrieb Alex <alex@example.com>:\n> Kannst du die Zahlen schicken?"),
    ("german header block",
     f"{OWN}\n\nVon: Alex Beispiel\nGesendet: Montag, 3. März 2025 10:00\nAn: Sam Beispiel\nBetreff: Zahlen\n\nKannst du die Zahlen schicken?"),
    ("spanish",
     f"{OWN}\n\nEl lun, 3 mar 2025 a las 10:00, Alex <alex@example.com> escribió:\n> ¿Puedes enviar las cifras?"),
    ("spanish header block",
     f"{OWN}\n\nDe: Alex Ejemplo\nEnviado: lunes, 3 de marzo de 2025 10:00\nPara: Sam Ejemplo\nAsunto: Cifras\n\n¿Puedes enviar las cifras?"),
    ("italian",
     f"{OWN}\n\nIl giorno lun 3 mar 2025 alle ore 10:00 Alex <alex@example.com> ha scritto:\n> Puoi mandare le cifre?"),
    ("italian header block",
     f"{OWN}\n\nDa: Alex Esempio\nInviato: lunedì 3 marzo 2025 10:00\nA: Sam Esempio\nOggetto: Cifre\n\nPuoi mandare le cifre?"),
    ("dutch",
     f"{OWN}\n\nOp ma 3 mrt 2025 om 10:00 schreef Alex <alex@example.com>:\n> Kun je de cijfers sturen?"),
    ("dutch header block",
     f"{OWN}\n\nVan: Alex Voorbeeld\nVerzonden: maandag 3 maart 2025 10:00\nAan: Sam Voorbeeld\nOnderwerp: Cijfers\n\nKun je de cijfers sturen?"),
    ("portuguese",
     f"{OWN}\n\nEm seg., 3 de mar. de 2025 às 10:00, Alex <alex@example.com> escreveu:\n> Pode enviar os números?"),
    ("portuguese header block",
     f"{OWN}\n\nDe: Alex Exemplo\nEnviado: segunda-feira, 3 de março de 2025 10:00\nPara: Sam Exemplo\nAssunto: Números\n\nPode enviar os números?"),
    ("on-wrote marker as the last line",
     f"{OWN}\n\nOn Mon, 3 Mar 2025, Alex <alex@example.com> wrote:"),
    ("quoted run without any marker",
     f"{OWN}\n> Can you send them?\n> Thanks a lot\n> Alex"),
    ("signature delimiter with a short signature",
     f"{OWN}\n-- \nSam Example\nExample Ltd\n07700 900123"),
    ("original message marker in german",
     f"{OWN}\n\n-----Ursprüngliche Nachricht-----\nVon: Alex\nGesendet: Montag\nAlte Nachricht hier"),
    ("sign-off and name",
     f"Hi Alex,\n\n{OWN}\n\nKind regards,\nSam Example"),
]


class TestOwnTextTable(unittest.TestCase):
    def test_table_has_enough_cases(self) -> None:
        self.assertGreaterEqual(len(KEEP_WHOLE) + len(CUT), 20)

    def test_user_words_kept_whole(self) -> None:
        for name, body in KEEP_WHOLE:
            with self.subTest(name):
                out = extract_own_text(body)
                # every word the user typed survives (the greeting/sign-off rules never apply to these)
                for word in ("honestly", "passports", "push", "colleague", "Point number 11", "attached", "agreed", "arrived", "launch"):
                    if word in body:
                        self.assertIn(word, out)
                self.assertTrue(out, name)

    def test_whole_message_kept_exactly(self) -> None:
        cases = {
            KEEP_WHOLE[0][1]: KEEP_WHOLE[0][1],
            KEEP_WHOLE[1][1]: KEEP_WHOLE[1][1],
            KEEP_WHOLE[2][1]: KEEP_WHOLE[2][1],
            KEEP_WHOLE[6][1]: "Thanks,\nSam",
        }
        for body, expected in cases.items():
            self.assertEqual(extract_own_text(body), expected)

    def test_other_peoples_words_removed(self) -> None:
        for name, body in CUT:
            with self.subTest(name):
                out = extract_own_text(body)
                self.assertEqual(out, OWN)

    def test_no_quoted_text_leaks(self) -> None:
        for name, body in CUT:
            with self.subTest(name):
                out = extract_own_text(body)
                for leaked in ("alex@example.com", "Alex Example", "figures?", "Zahlen", "chiffres", "cifras", "cifre", "cijfers", "números", "Subject", "Example Ltd"):
                    self.assertNotIn(leaked, out)

    def test_signoff_strips_only_one_or_two_capitalised_words(self) -> None:
        self.assertEqual(extract_own_text("The plan is agreed and ready to go ahead.\n\nThanks,\nMary Jane"), "The plan is agreed and ready to go ahead.")
        self.assertIn("Thanks", extract_own_text("The plan is agreed and ready to go ahead.\n\nThanks,\nMary Jane Anne Smith"))

    def test_single_quote_line_dropped_after_marker(self) -> None:
        body = f"{OWN}\n\nOn Mon, 3 Mar 2025, Alex <alex@example.com> wrote:\n> only one line"
        self.assertEqual(extract_own_text(body), OWN)


# --------------------------------------------------------------------------- html vs plain


class TestShorterOfHtmlAndPlain(unittest.TestCase):
    def payload(self, plain: str | None, html: str | None) -> dict:
        parts = []
        if plain is not None:
            parts.append({"mimeType": "text/plain", "body": {"data": b64(plain)}})
        if html is not None:
            parts.append({"mimeType": "text/html", "body": {"data": b64(html)}})
        return {"mimeType": "multipart/alternative", "headers": [], "parts": parts}

    def test_html_without_quote_beats_plain_with_unrecognised_quote(self) -> None:
        plain = f"{OWN}\n\nLe lundi, Alex a dit comme suit\nPeux-tu envoyer les chiffres s'il te plait"
        html = f"<div>{OWN}</div><blockquote><div>Peux-tu envoyer les chiffres</div></blockquote>"
        self.assertEqual(gmail_import.own_text_from_payload(self.payload(plain, html)), OWN)

    def test_plain_wins_when_shorter(self) -> None:
        plain = OWN
        html = f"<div>{OWN}</div><div>Something extra that is not marked as a quote at all.</div>"
        self.assertEqual(gmail_import.own_text_from_payload(self.payload(plain, html)), OWN)

    def test_empty_candidate_is_ignored(self) -> None:
        html = f"<div>{OWN}</div>"
        self.assertEqual(gmail_import.own_text_from_payload(self.payload("> quoted only\n> more quoted", html)), OWN)

    def test_single_part_works(self) -> None:
        self.assertEqual(gmail_import.own_text_from_payload(self.payload(OWN, None)), OWN)
        self.assertEqual(gmail_import.own_text_from_payload(self.payload(None, f"<p>{OWN}</p>")), OWN)


# --------------------------------------------------------------------------- http retries


class TestRetries(base.AuthTestBase):
    def run_get(self, seq):
        items = list(seq)

        def fake(url, token, timeout=30.0):
            item = items.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        sleeps: list[float] = []
        with patch.object(google_auth, "get_access_token", return_value=base.ACCESS), patch.object(
            gmail_import, "_http_get", side_effect=fake
        ) as http, patch.object(gmail_import, "_sleep", side_effect=sleeps.append):
            try:
                result = gmail_import.get_message("abc", self.db)
            except Exception as err:  # noqa: BLE001
                result = err
        return result, sleeps, http.call_count

    def test_four_attempts_with_exponential_backoff(self) -> None:
        err = gmail_import.GmailError("x", 503)
        result, sleeps, calls = self.run_get([err, err, err, err, {"id": "never"}])
        self.assertIsInstance(result, gmail_import.GmailError)
        self.assertEqual((calls, sleeps), (4, [1.0, 2.0, 4.0]))

    def test_succeeds_on_third_attempt(self) -> None:
        err = gmail_import.GmailError("x", 500)
        result, sleeps, calls = self.run_get([err, err, {"id": "ok"}])
        self.assertEqual((result, calls, sleeps), ({"id": "ok"}, 3, [1.0, 2.0]))

    def test_retry_after_is_honoured_and_capped(self) -> None:
        _, sleeps, _ = self.run_get([gmail_import.GmailError("x", 429, retry_after=7), {"id": "ok"}])
        self.assertEqual(sleeps, [7.0])
        _, sleeps, _ = self.run_get([gmail_import.GmailError("x", 429, retry_after=500), {"id": "ok"}])
        self.assertEqual(sleeps, [30.0])

    def test_rate_limit_403_is_retried_but_other_403_is_not(self) -> None:
        for reason in ("rateLimitExceeded", "userRateLimitExceeded"):
            result, sleeps, calls = self.run_get([gmail_import.GmailError("x", 403, reason=reason), {"id": "ok"}])
            self.assertEqual((result, calls), ({"id": "ok"}, 2), reason)
        result, sleeps, calls = self.run_get([gmail_import.GmailError("x", 403, reason="forbidden"), {"id": "ok"}])
        self.assertIsInstance(result, gmail_import.GmailError)
        self.assertEqual((calls, sleeps), (1, []))

    def test_404_not_retried(self) -> None:
        _, sleeps, calls = self.run_get([gmail_import.GmailError("x", 404)])
        self.assertEqual((calls, sleeps), (1, []))

    def test_401_refreshes_once_then_reports_expired_sign_in(self) -> None:
        with patch.object(google_auth, "get_access_token", return_value=base.ACCESS), patch.object(
            google_auth, "invalidate_access_token"
        ) as inval, patch.object(gmail_import, "_http_get", side_effect=gmail_import.GmailError("x", 401)) as http:
            with self.assertRaises(google_auth.GoogleAuthError) as ctx:
                gmail_import.get_message("abc", self.db)
        self.assertEqual((http.call_count, inval.call_count), (2, 1))
        self.assertIn("sign-in expired", ctx.exception.message)

    def test_http_error_reason_is_read_from_body_without_message_text(self) -> None:
        body = json.dumps({"error": {"code": 403, "message": "PRIVATE TEXT", "errors": [{"reason": "userRateLimitExceeded"}]}}).encode()
        self.assertEqual(gmail_import._error_reason(body), "userRateLimitExceeded")
        self.assertEqual(gmail_import._error_reason(b"not json"), "")

    def test_repeated_page_token_stops_listing(self) -> None:
        calls = []

        def fake(url, token, timeout=30.0):
            calls.append(url)
            return {"messages": [{"id": f"m{len(calls)}"}], "nextPageToken": "SAME"}

        with patch.object(google_auth, "get_access_token", return_value=base.ACCESS), patch.object(gmail_import, "_http_get", side_effect=fake):
            ids = gmail_import.list_sent_ids(self.db)
        self.assertEqual(len(calls), 2)
        self.assertEqual(ids, ["m1", "m2"])

    def test_page_cap(self) -> None:
        n = {"i": 0}

        def fake(url, token, timeout=30.0):
            n["i"] += 1
            return {"messages": [{"id": f"m{n['i']}"}], "nextPageToken": f"T{n['i']}"}

        with patch.object(google_auth, "get_access_token", return_value=base.ACCESS), patch.object(gmail_import, "_http_get", side_effect=fake):
            gmail_import.list_sent_ids(self.db)
        self.assertEqual(n["i"], 2000)


# --------------------------------------------------------------------------- scan


class TestScanRobustness(base.ScanTestBase):
    def run_with(self, messages: dict, listed: list[str] | None = None):
        def fake_get(mid, db_path=None):
            item = messages[mid]
            if callable(item):
                return item()
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(gmail_import, "get_profile", return_value=ME), patch.object(
            gmail_import, "list_sent_ids", return_value=listed or list(messages)
        ), patch.object(gmail_import, "get_message", side_effect=fake_get):
            gmail_import.scan_sent_mail(self.db, pending_import.PROGRESS)

    def test_one_malformed_message_does_not_stop_the_scan(self) -> None:
        bad = make_msg("bad", GOOD)
        bad["payload"]["parts"][0]["body"]["data"] = "!!!not-base64!!!"
        weird = {"id": "weird", "labelIds": ["SENT"], "payload": 5}
        msgs = {"bad": bad, "weird": weird, "boom": RuntimeError("x"), "ok": make_msg("ok", GOOD)}
        self.run_with(msgs)
        s = pending_import.get_summary()
        self.assertEqual(s["counts"]["failed"], 3)
        self.assertEqual(s["counts"]["kept"], 1)
        self.assertFalse(s["partial"])

    def test_auth_failure_midway_parks_partial_batch(self) -> None:
        def expire():
            raise google_auth.GoogleAuthError("sign-in expired", 401)

        msgs = {"a": make_msg("a", GOOD), "b": expire, "c": make_msg("c", GOOD + " again and again")}
        self.run_with(msgs)
        s = pending_import.get_summary()
        self.assertTrue(s["has_pending"])
        self.assertTrue(s["partial"])
        self.assertIn("sign-in expired", s["partial_reason"])
        self.assertEqual(s["counts"]["kept"], 1)
        self.assertEqual(pending_import.PROGRESS.snapshot()["stage"], "review")

    def test_auth_failure_with_nothing_kept_reports_plain_error(self) -> None:
        def expire():
            raise google_auth.GoogleAuthError("sign-in expired", 401)

        self.run_with({"a": expire})
        snap = pending_import.PROGRESS.snapshot()
        self.assertEqual(snap["stage"], "error")
        self.assertEqual(snap["error"], "Your Google sign-in expired. Connect again to keep going.")
        self.assertFalse(pending_import.get_summary()["has_pending"])

    def test_cancel_parks_partial_batch(self) -> None:
        def cancel_then_ok():
            pending_import.PROGRESS.request_cancel()
            return make_msg("a", GOOD)

        msgs = {"a": cancel_then_ok, "b": make_msg("b", GOOD + " but different words here")}
        pending_import.PROGRESS.update(running=True)
        self.run_with(msgs)
        s = pending_import.get_summary()
        self.assertTrue(s["partial"])
        self.assertEqual(s["counts"]["kept"], 1)
        self.assertIn("cancelled", s["partial_reason"])

    def test_cancel_with_nothing_kept_goes_idle(self) -> None:
        pending_import.PROGRESS.update(running=True)
        pending_import.PROGRESS.request_cancel()
        self.run_with({"a": make_msg("a", GOOD)})
        self.assertFalse(pending_import.get_summary()["has_pending"])
        self.assertEqual(pending_import.PROGRESS.snapshot()["stage"], "idle")

    def test_cancel_request_ignored_when_idle(self) -> None:
        self.assertFalse(pending_import.PROGRESS.request_cancel())

    def test_every_processed_id_is_recorded_with_its_kind_and_failed_are_not(self) -> None:
        msgs = {
            "kept": make_msg("kept", GOOD),
            "other": make_msg("other", GOOD, sender="X <x@example.com>"),
            "short": make_msg("short", "Ok thanks"),
            "broken": gmail_import.GmailError("x", 500),
        }
        self.run_with(msgs)
        pending_import.commit_pending(self.db)
        self.assertEqual(get_imported_ids("gmail", self.db), {"kept", "other", "short"})
        with sqlite3.connect(self.db) as conn:
            kinds = dict(conn.execute("SELECT external_id, kind FROM imported_messages").fetchall())
        self.assertEqual(kinds, {"kept": "kept", "other": "skipped", "short": "skipped"})

    def test_dedup_lookup_does_not_load_whole_rows(self) -> None:
        dbmod.add_sample(GOOD, "work_email", "manual", db_path=self.db)
        with patch.object(gmail_import, "get_samples", create=True, side_effect=AssertionError("loaded whole rows")):
            self.run_with({"dup": make_msg("dup", GOOD)})
        self.assertEqual(pending_import.get_summary()["counts"]["duplicates"], 1)

    def test_discard_during_scan_is_not_undone(self) -> None:
        def discard_midway():
            pending_import.discard_pending(self.db)
            return make_msg("a", GOOD)

        self.run_with({"a": discard_midway})
        self.assertFalse(pending_import.get_summary()["has_pending"])

    def test_newer_batch_is_not_overwritten_by_a_late_scan(self) -> None:
        def newer_batch():
            pending_import.set_pending("google_chat", "casual_chat", ["newer batch text"], [], {})
            return make_msg("a", GOOD)

        self.run_with({"a": newer_batch})
        s = pending_import.get_summary()
        self.assertEqual(s["source_type"], "google_chat")

    def test_discard_with_nothing_kept_still_remembers_what_was_read(self) -> None:
        self.run_with({"short": make_msg("short", "Ok thanks")})
        self.assertEqual(pending_import.get_summary()["counts"]["kept"], 0)
        pending_import.discard_pending(self.db)
        self.assertEqual(get_imported_ids("gmail", self.db), {"short"})

    def test_discard_with_kept_texts_writes_nothing_at_all(self) -> None:
        self.run_with({"a": make_msg("a", GOOD)})
        pending_import.discard_pending(self.db)
        self.assertEqual(get_imported_ids("gmail", self.db), set())
        self.assertEqual(get_samples(db_path=self.db), [])


class TestClaims(unittest.TestCase):
    def setUp(self) -> None:
        pending_import.discard_pending()
        pending_import.PROGRESS.reset()
        self.addCleanup(pending_import.discard_pending)
        self.addCleanup(pending_import.PROGRESS.reset)

    def test_one_at_a_time(self) -> None:
        self.assertEqual(pending_import.claim("scan"), "")
        self.assertIn("scan", pending_import.claim("scan"))
        self.assertIn("scan", pending_import.claim("upload"))
        pending_import.release("scan")
        self.assertEqual(pending_import.claim("upload"), "")
        self.assertIn("file", pending_import.claim("upload"))
        self.assertIn("file", pending_import.claim("scan"))
        pending_import.release("upload")

    def test_rejected_while_a_list_awaits_review(self) -> None:
        pending_import.set_pending("gmail", "work_email", ["some text"], [], {})
        self.assertIn("discard", pending_import.claim("scan"))
        self.assertIn("discard", pending_import.claim("upload"))
        pending_import.discard_pending()
        self.assertEqual(pending_import.claim("scan"), "")
        pending_import.release("scan")

    def test_generation_blocks_stale_set(self) -> None:
        gen = pending_import.current_generation()
        pending_import.discard_pending()  # nothing was pending or running: generation unchanged
        self.assertTrue(pending_import.set_pending("gmail", "work_email", ["a"], [], {}, expected_generation=gen))
        self.assertFalse(pending_import.set_pending("gmail", "work_email", ["b"], [], {}, expected_generation=gen))
        self.assertEqual(pending_import._batch["texts"], ["a"])


# --------------------------------------------------------------------------- commit


class TestAtomicCommit(base.ScanTestBase):
    def test_failure_keeps_batch_and_writes_nothing_then_retry_works(self) -> None:
        pending_import.set_pending("gmail", "work_email", [GOOD], [("gmail", "m1", "kept"), ("gmail", "m2", "skipped")], {})
        with patch.object(pending_import, "commit_import_batch", side_effect=sqlite3.OperationalError("database is locked")):
            res = pending_import.commit_pending(self.db)
        self.assertFalse(res["committed"])
        self.assertIn("error", res)
        self.assertTrue(pending_import.get_summary()["has_pending"])
        self.assertEqual(get_samples(db_path=self.db), [])
        res = pending_import.commit_pending(self.db)
        self.assertEqual((res["committed"], res["inserted"]), (True, 1))
        self.assertEqual(get_imported_ids("gmail", self.db), {"m1", "m2"})

    def test_id_insert_failure_rolls_back_the_samples_too(self) -> None:
        with self.assertRaises(Exception):
            dbmod.commit_import_batch([GOOD], [("gmail",)], "work_email", "gmail", db_path=self.db)
        self.assertEqual(get_samples(db_path=self.db), [])

    def test_second_commit_is_a_no_op(self) -> None:
        pending_import.set_pending("gmail", "work_email", [GOOD], [("gmail", "m1", "kept")], {})
        self.assertEqual(pending_import.commit_pending(self.db)["inserted"], 1)
        again = pending_import.commit_pending(self.db)
        self.assertEqual((again["inserted"], again["committed"]), (0, False))
        self.assertEqual(len(get_samples(db_path=self.db)), 1)

    def test_concurrent_commits_insert_once(self) -> None:
        pending_import.set_pending("gmail", "work_email", [GOOD], [("gmail", "m1", "kept")], {})
        results: list[int] = []
        threads = [threading.Thread(target=lambda: results.append(pending_import.commit_pending(self.db)["inserted"])) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results), [0, 0, 0, 1])


# --------------------------------------------------------------------------- db


class TestImportedIdsLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="vp_ids_")
        self.db = Path(self.tmp) / "t.db"
        init_db(self.db)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def seed(self) -> None:
        dbmod.commit_import_batch([GOOD], [("gmail", "g1", "kept"), ("gmail", "g2", "skipped")], "work_email", "gmail", db_path=self.db)
        dbmod.commit_import_batch(["Chat words that are plenty long enough for the sample."], [("google_chat", "c1", "kept")], "casual_chat", "google_chat", db_path=self.db)

    def test_clear_by_mode_forgets_only_that_sources_ids(self) -> None:
        self.seed()
        clear_samples("work_email", self.db)
        self.assertEqual(get_imported_ids("gmail", self.db), set())
        self.assertEqual(get_imported_ids("google_chat", self.db), {"c1"})

    def test_clear_all_forgets_every_id(self) -> None:
        self.seed()
        clear_samples(None, self.db)
        self.assertEqual(get_imported_ids("gmail", self.db), set())
        self.assertEqual(get_imported_ids("google_chat", self.db), set())

    def test_deleting_one_sample_keeps_the_ids(self) -> None:
        self.seed()
        sample = get_samples("work_email", db_path=self.db)[0]
        self.assertTrue(delete_sample(sample["id"], self.db))
        self.assertEqual(get_imported_ids("gmail", self.db), {"g1", "g2"})

    def test_migration_adds_kind_column_to_an_old_database(self) -> None:
        old = Path(self.tmp) / "old.db"
        with sqlite3.connect(old) as conn:
            conn.execute("CREATE TABLE imported_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, external_id TEXT NOT NULL, UNIQUE(source, external_id))")
            conn.execute("INSERT INTO imported_messages (source, external_id) VALUES ('gmail', 'old1')")
        init_db(old)
        init_db(old)  # idempotent
        with sqlite3.connect(old) as conn:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(imported_messages)")]
            kind = conn.execute("SELECT kind FROM imported_messages WHERE external_id = 'old1'").fetchone()[0]
        self.assertIn("kind", cols)
        self.assertEqual(kind, "kept")

    def test_get_sample_contents_returns_text_only(self) -> None:
        self.seed()
        self.assertEqual(sorted(dbmod.get_sample_contents(self.db))[0][:4], "Chat")
        self.assertTrue(all(isinstance(x, str) for x in dbmod.get_sample_contents(self.db)))


# --------------------------------------------------------------------------- chat


class TestChatImprovements(base.ScanTestBase):
    ADDR = {ME}

    def doc(self, *msgs):
        return {"messages": [{"creator": {"email": e}, "text": t, **({"message_id": i} if i else {})} for e, t, i in msgs]}

    def test_dedups_against_stored_samples(self) -> None:
        text = "We should really get the whole team together for lunch on Friday"
        dbmod.add_sample(text, "work_email", "manual", db_path=self.db)
        chat_takeout.build_batch([self.doc((ME, text, None))], self.ADDR, self.db)
        c = pending_import.get_summary()["counts"]
        self.assertEqual((c["duplicates"], c["kept"]), (1, 0))

    def test_records_message_ids_and_skips_known_ones_next_time(self) -> None:
        d = self.doc(
            (ME, "We should really get the whole team together for lunch on Friday", "spaces/a/messages/1"),
            (ME, "ok", "spaces/a/messages/2"),
        )
        chat_takeout.build_batch([d], self.ADDR, self.db)
        pending_import.commit_pending(self.db)
        self.assertEqual(get_imported_ids("google_chat", self.db), {"spaces/a/messages/1", "spaces/a/messages/2"})
        chat_takeout.build_batch([d], self.ADDR, self.db)
        c = pending_import.get_summary()["counts"]
        self.assertEqual((c["already_imported"], c["kept"]), (2, 0))

    def test_export_without_ids_still_works(self) -> None:
        chat_takeout.build_batch([self.doc((ME, "We should really get the whole team together for lunch on Friday", None))], self.ADDR, self.db)
        self.assertEqual(pending_import.get_summary()["counts"]["kept"], 1)

    def test_merged_runs_are_split_to_400_words(self) -> None:
        msgs = [(ME, " ".join(f"word{m}x{i}" for i in range(90)), None) for m in range(10)]
        chat_takeout.build_batch([self.doc(*msgs)], self.ADDR, self.db)
        texts = pending_import._batch["texts"]
        self.assertGreater(len(texts), 1)
        self.assertTrue(all(len(t.split()) <= 400 for t in texts))
        self.assertEqual(sum(len(t.split()) for t in texts), 900)

    def test_one_huge_message_is_split(self) -> None:
        chat_takeout.build_batch([self.doc((ME, " ".join(f"w{i}" for i in range(1000)), None))], self.ADDR, self.db)
        texts = pending_import._batch["texts"]
        self.assertTrue(all(len(t.split()) <= 400 for t in texts))
        self.assertEqual(sum(len(t.split()) for t in texts), 1000)

    def test_bare_messages_json_over_cap_rejected(self) -> None:
        with patch.object(chat_takeout, "MAX_MESSAGES_JSON_BYTES", 50):
            with self.assertRaises(chat_takeout.ChatTakeoutError) as ctx:
                chat_takeout.read_message_files(json.dumps(self.doc((ME, "x" * 100, None))).encode())
        self.assertEqual(ctx.exception.status_code, 413)

    def test_zip_member_over_cap_rejected_before_reading(self) -> None:
        z = base.TestChatTakeout.make_zip({"Google Chat/Groups/g/messages.json": json.dumps(self.doc((ME, "x" * 500, None))).encode()})
        with patch.object(chat_takeout, "MAX_MESSAGES_JSON_BYTES", 100):
            with self.assertRaises(chat_takeout.ChatTakeoutError) as ctx:
                chat_takeout.read_message_files(z)
        self.assertEqual(ctx.exception.status_code, 413)


# --------------------------------------------------------------------------- server


class ServerBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="vp_guard_")
        cls.db = Path(cls.tmp) / "g.db"
        cls._orig = os.environ.get("VOICEPRINT_DB_PATH")
        os.environ["VOICEPRINT_DB_PATH"] = str(cls.db)
        init_db(cls.db)
        update_settings({"provider": "local_only", "google_account_email": ME}, db_path=cls.db)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), VoicePrintHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=3)
        if cls._orig is not None:
            os.environ["VOICEPRINT_DB_PATH"] = cls._orig
        else:
            os.environ.pop("VOICEPRINT_DB_PATH", None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self) -> None:
        pending_import.discard_pending()
        pending_import.PROGRESS.reset()
        self.addCleanup(pending_import.discard_pending)

    def req(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None, host: str | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host if host is not None else f"127.0.0.1:{self.port}")
        for k, v in (headers or {}).items():
            conn.putheader(k, v)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    JSON = {"Content-Type": "application/json"}


class TestRequestGuards(ServerBase):
    def test_every_route_checks_the_host_header(self) -> None:
        for method, path in (
            ("GET", "/"),
            ("GET", "/static/app.js"),
            ("HEAD", "/"),
            ("GET", "/api/state"),
            ("GET", "/api/health"),
            ("GET", "/api/clipboard"),
            ("POST", "/api/settings"),
            ("POST", "/api/samples/clear"),
            ("DELETE", "/api/samples/1"),
        ):
            for host in ("evil.example", f"evil.example:{self.port}", "127.0.0.1", f"127.0.0.1:{self.port + 1}"):
                body = b"{}" if method == "POST" else None
                status, _ = self.req(method, path, body, self.JSON if body else None, host=host)
                self.assertEqual(status, 403, (method, path, host))

    def test_every_route_rejects_a_foreign_origin(self) -> None:
        for method, path in (("GET", "/"), ("GET", "/api/state"), ("POST", "/api/settings"), ("DELETE", "/api/samples/1")):
            for origin in ("https://evil.example", "null", f"http://127.0.0.1:{self.port + 1}"):
                body = b"{}" if method == "POST" else None
                headers = {"Origin": origin, **(self.JSON if body else {})}
                status, _ = self.req(method, path, body, headers)
                self.assertEqual(status, 403, (method, path, origin))

    def test_own_origin_and_localhost_work_everywhere(self) -> None:
        own = {"Origin": f"http://127.0.0.1:{self.port}"}
        self.assertEqual(self.req("GET", "/", headers=own)[0], 200)
        self.assertEqual(self.req("GET", "/api/state", headers=own)[0], 200)
        self.assertEqual(self.req("GET", "/api/state", host=f"localhost:{self.port}")[0], 200)
        self.assertEqual(self.req("GET", "/static/app.js")[0], 200)
        self.assertEqual(self.req("POST", "/api/settings", b"{}", {**own, **self.JSON})[0], 200)

    def test_json_posts_need_the_json_content_type(self) -> None:
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            status, _ = self.req("POST", "/api/settings", b"{}", {"Content-Type": ctype})
            self.assertEqual(status, 415, ctype)
        status, _ = self.req("POST", "/api/settings", b"{}", {"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(status, 200)
        status, _ = self.req("POST", "/api/google/pending/discard", None)  # no body: nothing to parse
        self.assertEqual(status, 200)

    def test_existing_callers_keep_working(self) -> None:
        # urllib sends Host 127.0.0.1:port and a JSON content type, exactly as macos/voiceprint_cli.py does
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/health",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(json.loads(resp.read())["app"], "voiceprint")
        payload = json.dumps({"text": "Some words to check over", "auto_copy": False}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/analyze", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)

    def test_cancel_route(self) -> None:
        status, data = self.req("POST", "/api/google/gmail/cancel", b"{}", self.JSON)
        self.assertEqual((status, json.loads(data)), (200, {"cancelled": False}))
        pending_import.PROGRESS.update(running=True)
        self.addCleanup(pending_import.PROGRESS.reset)
        status, data = self.req("POST", "/api/google/gmail/cancel", b"{}", self.JSON)
        self.assertEqual(json.loads(data), {"cancelled": True})
        self.assertTrue(pending_import.PROGRESS.cancelled())


class TestChatUploadRoute(ServerBase):
    OCTET = {"Content-Type": "application/octet-stream"}

    def chat_bytes(self, text: str = "This is a long enough chat message that I wrote myself earlier today") -> bytes:
        return json.dumps({"messages": [{"creator": {"email": ME}, "text": text}]}).encode()

    def test_raw_body_upload_works(self) -> None:
        status, data = self.req("POST", "/api/google/chat/upload", self.chat_bytes(), self.OCTET)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["counts"]["kept"], 1)
        self.assertEqual(pending_import.active(), "")

    def test_content_type_must_be_octet_stream(self) -> None:
        for headers in ({}, {"Content-Type": "application/json"}, {"Content-Type": "text/plain"}):
            status, _ = self.req("POST", "/api/google/chat/upload", self.chat_bytes(), headers)
            self.assertEqual(status, 415, headers)

    def test_oversize_content_length_rejected_before_reading(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/api/google/chat/upload", skip_host=True)
        conn.putheader("Host", f"127.0.0.1:{self.port}")
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Content-Length", str(211 * 1024 * 1024))
        conn.endheaders()  # the body is never sent
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 413)

    def test_empty_body_rejected(self) -> None:
        status, _ = self.req("POST", "/api/google/chat/upload", b"", self.OCTET)
        self.assertEqual(status, 400)

    def test_conflict_while_a_list_awaits_review(self) -> None:
        pending_import.set_pending("gmail", "work_email", ["text"], [], {})
        status, _ = self.req("POST", "/api/google/chat/upload", self.chat_bytes(), self.OCTET)
        self.assertEqual(status, 409)

    def test_conflict_while_a_scan_or_another_upload_is_running(self) -> None:
        for kind in ("scan", "upload"):
            self.assertEqual(pending_import.claim(kind), "")
            status, _ = self.req("POST", "/api/google/chat/upload", self.chat_bytes(), self.OCTET)
            self.assertEqual(status, 409, kind)
            pending_import.release(kind)

    def test_commit_and_scan_conflict_while_scan_running(self) -> None:
        self.assertEqual(pending_import.claim("scan"), "")
        self.addCleanup(pending_import.release, "scan")
        self.assertEqual(self.req("POST", "/api/google/pending/commit", b"{}", self.JSON)[0], 409)
        with patch.object(google_auth, "status", return_value={"configured": True, "connected": True, "email": ME, "note": ""}):
            self.assertEqual(self.req("POST", "/api/google/gmail/scan", b"{}", self.JSON)[0], 409)

    def test_scan_conflicts_while_a_list_awaits_review(self) -> None:
        pending_import.set_pending("gmail", "work_email", ["text"], [], {})
        with patch.object(google_auth, "status", return_value={"configured": True, "connected": True, "email": ME, "note": ""}):
            status, data = self.req("POST", "/api/google/gmail/scan", b"{}", self.JSON)
        self.assertEqual(status, 409)
        self.assertIn("discard", json.loads(data)["error"])

    def test_commit_failure_returns_error_and_keeps_the_list(self) -> None:
        pending_import.set_pending("gmail", "work_email", [GOOD], [], {})
        with patch.object(pending_import, "commit_import_batch", side_effect=sqlite3.OperationalError("database is locked")):
            status, data = self.req("POST", "/api/google/pending/commit", b"{}", self.JSON)
        self.assertEqual(status, 500)
        self.assertNotIn("locked", json.loads(data)["error"])
        self.assertTrue(pending_import.get_summary()["has_pending"])

    def test_state_reports_sample_words(self) -> None:
        status, data = self.req("GET", "/api/state")
        self.assertIn("sample_words", json.loads(data))
        self.assertNotIn("own_writing_words", json.loads(data))


if __name__ == "__main__":
    unittest.main()
