#!/usr/bin/env python3
"""Tests for the Google sign-in and own-writing import (all network, Keychain and Google calls are faked)."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import chat_takeout, gmail_import, google_auth, pending_import  # noqa: E402
from engine.db import (  # noqa: E402
    get_public_settings,
    get_samples,
    get_settings,
    init_db,
    total_sample_words,
    update_settings,
)
from engine.own_text import extract_own_text, is_substantive, mask_private_details  # noqa: E402
from server import VoicePrintHandler  # noqa: E402

SECRET = "SECRET-VALUE-do-not-leak"
REFRESH = "REFRESH-TOKEN-do-not-leak"
ACCESS = "ACCESS-TOKEN-do-not-leak"
CODE = "AUTH-CODE-do-not-leak"
ME = "me@example.com"


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


# --------------------------------------------------------------------------- own_text


class TestExtractOwnText(unittest.TestCase):
    def test_cuts_on_wrote_marker(self) -> None:
        body = "Sounds good, let's meet on Friday afternoon then.\n\nOn Mon, 3 Mar 2025 at 10:00, Alex <alex@example.com> wrote:\n> Can we meet?"
        self.assertEqual(extract_own_text(body), "Sounds good, let's meet on Friday afternoon then.")

    def test_cuts_on_wrote_marker_wrapped_over_two_lines(self) -> None:
        body = "I will send the draft over tomorrow morning.\n\nOn Mon, 3 Mar 2025 at 10:00, Alex Example\n<alex@example.com> wrote:\n> Any news?\n"
        self.assertEqual(extract_own_text(body), "I will send the draft over tomorrow morning.")

    def test_cuts_original_message_marker(self) -> None:
        body = "Please see my notes below for the review.\n\n-----Original Message-----\nFrom: Alex\nSent: Monday\nOld text here"
        self.assertEqual(extract_own_text(body), "Please see my notes below for the review.")

    def test_cuts_forwarded_message_marker(self) -> None:
        body = "Passing this along because it matters to you.\n\n---------- Forwarded message ---------\nFrom: x\nBody"
        self.assertEqual(extract_own_text(body), "Passing this along because it matters to you.")

    def test_cuts_begin_forwarded_message(self) -> None:
        body = "Worth a read when you have a minute spare.\n\nBegin forwarded message:\n\nFrom: x\nSubject: y"
        self.assertEqual(extract_own_text(body), "Worth a read when you have a minute spare.")

    def test_cuts_outlook_header_block(self) -> None:
        body = "Agreed, let's go with the second option please.\n\nFrom: Alex Example <alex@example.com>\nSent: Monday, 3 March 2025 10:00\nTo: Sam\nSubject: Plan\n\nOld message"
        self.assertEqual(extract_own_text(body), "Agreed, let's go with the second option please.")

    def test_lone_from_line_is_not_a_marker(self) -> None:
        body = "The letter was sent\nFrom: the office in Leeds last week and arrived today."
        self.assertIn("arrived today", extract_own_text(body))

    def test_drops_quoted_lines(self) -> None:
        body = "First thought here.\n> quoted words\n>> deeper quote\nSecond thought here."
        self.assertEqual(extract_own_text(body), "First thought here.\nSecond thought here.")

    def test_cuts_signature_delimiter(self) -> None:
        body = "This is the real message body text.\n-- \nSam Example\nExample Ltd"
        self.assertEqual(extract_own_text(body), "This is the real message body text.")

    def test_drops_sent_from_device_lines(self) -> None:
        body = "Running late but I will be there soon.\n\nSent from my iPhone"
        self.assertEqual(extract_own_text(body), "Running late but I will be there soon.")

    def test_drops_greeting_and_signoff(self) -> None:
        body = "Hi Alex,\n\nThe report is attached and ready for review.\n\nThanks,\nSam"
        self.assertEqual(extract_own_text(body), "The report is attached and ready for review.")
        self.assertEqual(extract_own_text("Dear Mr Jones,\nI am writing about the invoice we discussed.\nBest regards,\nSam Example"), "I am writing about the invoice we discussed.")

    def test_keeps_sentence_that_starts_like_a_greeting(self) -> None:
        self.assertEqual(extract_own_text("Hey, thanks for sending that over today!"), "Hey, thanks for sending that over today!")

    def test_keeps_standalone_thanks_without_comma(self) -> None:
        self.assertEqual(extract_own_text("Thanks!"), "Thanks!")

    def test_pure_forward_is_empty(self) -> None:
        body = "---------- Forwarded message ---------\nFrom: x\nSubject: y\n\nWhatever somebody else wrote."
        self.assertEqual(extract_own_text(body), "")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(extract_own_text("One   two\t three\n\n\n\n\nfour"), "One two three\n\nfour")

    def test_empty_input(self) -> None:
        self.assertEqual(extract_own_text(""), "")


class TestMaskPrivateDetails(unittest.TestCase):
    def test_email(self) -> None:
        self.assertEqual(mask_private_details("write to sam.x+tag@example.co.uk now"), "write to [email] now")

    def test_url(self) -> None:
        self.assertEqual(mask_private_details("see https://example.com/a?b=1, and www.example.org."), "see [link], and [link].")

    def test_phone(self) -> None:
        for raw in ("+44 7700 900123", "07700 900123", "020 7946 0958", "(555) 010-1234"):
            self.assertEqual(mask_private_details(f"call {raw} today"), "call [phone] today", raw)

    def test_postcode(self) -> None:
        self.assertEqual(mask_private_details("I live at SW1A 1AA, honest"), "I live at [postcode], honest")
        self.assertEqual(mask_private_details("M1 1AE"), "[postcode]")

    def test_long_number(self) -> None:
        self.assertEqual(mask_private_details("card 4111 1111 1111 1111 ok"), "card [number] ok")
        self.assertEqual(mask_private_details("id 1234-5678-9012-3456"), "id [number]")

    def test_plain_text_untouched(self) -> None:
        text = "We met 3 times in 2024 and agreed on a price of 450 pounds."
        self.assertEqual(mask_private_details(text), text)


class TestIsSubstantive(unittest.TestCase):
    def test_threshold(self) -> None:
        self.assertTrue(is_substantive("one two three four five six seven eight"))
        self.assertFalse(is_substantive("one two three four five six seven"))

    def test_placeholders_do_not_count(self) -> None:
        self.assertFalse(is_substantive("see [link] [email] [phone] [number] [postcode] [link] [email] [phone]"))


# --------------------------------------------------------------------------- google_auth


class AuthTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="vp_google_")
        self.db = Path(self.tmp) / "t.db"
        init_db(self.db)
        update_settings({"google_client_id": "client-123.example", "google_client_secret": SECRET}, db_path=self.db)
        google_auth._reset_for_tests()
        self.store = {"token": ""}
        self.fail_write = False
        patches = [
            patch.object(google_auth, "_keychain_write", side_effect=self._kw),
            patch.object(google_auth, "_keychain_read", side_effect=lambda: self.store["token"]),
            patch.object(google_auth, "_keychain_delete", side_effect=self._kd),
        ]
        self.calls: list[tuple[str, dict]] = []
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(google_auth._reset_for_tests)

    def _kw(self, token: str) -> bool:
        if self.fail_write:
            return False
        self.store["token"] = token
        return True

    def _kd(self) -> bool:
        self.store["token"] = ""
        return True

    def fake_post(self, responses):
        def _post(url, data, timeout=20.0):
            self.calls.append((url, dict(data)))
            return responses(url, data) if callable(responses) else responses

        return patch.object(google_auth, "_post_form", side_effect=_post)


class TestGoogleAuth(AuthTestBase):
    def _start(self):
        url = google_auth.start_flow(18940, db_path=self.db)
        parsed = urllib.parse.urlparse(url)
        return url, {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

    def test_start_flow_builds_pkce_url(self) -> None:
        url, q = self._start()
        self.assertTrue(url.startswith("https://accounts.google.com/o/oauth2/v2/auth?"))
        self.assertEqual(q["scope"], "https://www.googleapis.com/auth/gmail.readonly")
        self.assertEqual(q["redirect_uri"], "http://127.0.0.1:18940/api/google/callback")
        self.assertEqual(q["access_type"], "offline")
        self.assertEqual(q["prompt"], "consent")
        self.assertEqual(q["code_challenge_method"], "S256")
        self.assertEqual(q["response_type"], "code")
        verifier = google_auth._pending[q["state"]]["verifier"]
        self.assertEqual(q["code_challenge"], google_auth._challenge(verifier))
        self.assertGreaterEqual(len(verifier), 43)
        self.assertNotIn(SECRET, url)

    def test_pkce_challenge_known_vector(self) -> None:
        # RFC 7636 appendix B
        self.assertEqual(
            google_auth._challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
        )

    def test_states_are_random(self) -> None:
        _, a = self._start()
        _, b = self._start()
        self.assertNotEqual(a["state"], b["state"])

    def test_start_requires_credentials(self) -> None:
        update_settings({"google_client_id": ""}, db_path=self.db)
        with self.assertRaises(google_auth.GoogleAuthError):
            google_auth.start_flow(18940, db_path=self.db)

    def _token_ok(self, url, data):
        return 200, {"access_token": ACCESS, "refresh_token": REFRESH, "expires_in": 3600}

    def test_finish_flow_stores_refresh_token_in_keychain_and_returns_no_token(self) -> None:
        _, q = self._start()
        verifier = google_auth._pending[q["state"]]["verifier"]
        with self.fake_post(self._token_ok), patch.object(gmail_import, "get_profile", return_value=ME):
            result = google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)
        self.assertEqual(self.store["token"], REFRESH)
        self.assertEqual(result, {"configured": True, "connected": True, "email": ME, "note": ""})
        blob = json.dumps(result) + json.dumps(google_auth.status(self.db))
        for secret in (REFRESH, ACCESS, CODE, SECRET):
            self.assertNotIn(secret, blob)
        sent = self.calls[0][1]
        self.assertEqual(sent["grant_type"], "authorization_code")
        self.assertEqual(sent["code"], CODE)
        self.assertEqual(sent["redirect_uri"], "http://127.0.0.1:18940/api/google/callback")
        self.assertEqual(sent["code_verifier"], verifier)
        self.assertEqual(get_settings(self.db)["google_account_email"], ME)

    def test_state_is_single_use(self) -> None:
        _, q = self._start()
        with self.fake_post(self._token_ok), patch.object(gmail_import, "get_profile", return_value=ME):
            google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)
            with self.assertRaises(google_auth.GoogleAuthError):
                google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)

    def test_unknown_state_rejected_without_calling_google(self) -> None:
        with self.fake_post(self._token_ok):
            with self.assertRaises(google_auth.GoogleAuthError):
                google_auth.finish_flow(CODE, "nope", 18940, db_path=self.db)
        self.assertEqual(self.calls, [])

    def test_expired_state_rejected(self) -> None:
        _, q = self._start()
        real = time.time()
        with self.fake_post(self._token_ok), patch.object(google_auth.time, "time", return_value=real + 601):
            with self.assertRaises(google_auth.GoogleAuthError):
                google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)
        self.assertEqual(self.calls, [])

    def test_failed_exchange_error_has_no_secrets(self) -> None:
        _, q = self._start()
        with self.fake_post(lambda u, d: (400, {"error": "invalid_request", "error_description": CODE})):
            with self.assertRaises(google_auth.GoogleAuthError) as ctx:
                google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)
        for secret in (CODE, SECRET, REFRESH):
            self.assertNotIn(secret, ctx.exception.message)

    def test_keychain_failure_connects_for_this_run_only(self) -> None:
        self.fail_write = True
        _, q = self._start()
        with self.fake_post(self._token_ok), patch.object(gmail_import, "get_profile", return_value=ME):
            result = google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)
        self.assertTrue(result["connected"])
        self.assertIn("this session only", result["note"])
        self.assertEqual(self.store["token"], "")
        self.assertNotIn(REFRESH, json.dumps(result))

    def _connect(self):
        _, q = self._start()
        with self.fake_post(self._token_ok), patch.object(gmail_import, "get_profile", return_value=ME):
            google_auth.finish_flow(CODE, q["state"], 18940, db_path=self.db)
        self.calls.clear()

    def test_access_token_cached_while_valid(self) -> None:
        self._connect()
        with self.fake_post((500, {})):
            self.assertEqual(google_auth.get_access_token(self.db), ACCESS)
        self.assertEqual(self.calls, [])

    def test_refreshes_when_expired(self) -> None:
        self._connect()
        google_auth._access["expires_at"] = time.time() - 5
        with self.fake_post((200, {"access_token": "NEW-ACCESS", "expires_in": 3600})):
            self.assertEqual(google_auth.get_access_token(self.db), "NEW-ACCESS")
        url, data = self.calls[0]
        self.assertEqual(url, "https://oauth2.googleapis.com/token")
        self.assertEqual(data["grant_type"], "refresh_token")
        self.assertEqual(data["refresh_token"], REFRESH)

    def test_invalid_grant_clears_stored_token(self) -> None:
        self._connect()
        google_auth._access["expires_at"] = 0
        with self.fake_post((400, {"error": "invalid_grant"})):
            with self.assertRaises(google_auth.GoogleAuthError) as ctx:
                google_auth.get_access_token(self.db)
        self.assertEqual(ctx.exception.message, "sign-in expired")
        self.assertEqual(self.store["token"], "")
        st = google_auth.status(self.db)
        self.assertFalse(st["connected"])
        self.assertEqual(st["note"], "sign-in expired")

    def test_get_access_token_when_not_connected(self) -> None:
        with self.assertRaises(google_auth.GoogleAuthError):
            google_auth.get_access_token(self.db)

    def test_disconnect_revokes_deletes_and_clears_email(self) -> None:
        self._connect()
        with self.fake_post((200, {})):
            st = google_auth.disconnect(self.db)
        self.assertEqual(self.calls[0][0], "https://oauth2.googleapis.com/revoke")
        self.assertEqual(self.calls[0][1], {"token": REFRESH})
        self.assertEqual(self.store["token"], "")
        self.assertFalse(st["connected"])
        self.assertEqual(st["email"], "")
        self.assertEqual(get_settings(self.db)["google_account_email"], "")
        with self.assertRaises(google_auth.GoogleAuthError):
            google_auth.get_access_token(self.db)

    def test_status_configured_flag(self) -> None:
        self.assertEqual(google_auth.status(self.db), {"configured": True, "connected": False, "email": "", "note": ""})
        update_settings({"google_client_secret": "", "clear_google_client_secret": True}, db_path=self.db)
        self.assertFalse(google_auth.status(self.db)["configured"])


class TestSecretSetting(unittest.TestCase):
    def test_client_secret_is_masked_and_kept_when_blank(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        db = Path(tmp) / "s.db"
        update_settings({"google_client_secret": "abcdefghijklmnop"}, db_path=db)
        pub = get_public_settings(db)
        self.assertEqual(pub["google_client_secret"], "")
        self.assertTrue(pub["google_client_secret_configured"])
        update_settings({"google_client_secret": ""}, db_path=db)
        self.assertEqual(get_settings(db)["google_client_secret"], "abcdefghijklmnop")
        update_settings({"google_client_secret": "", "clear_google_client_secret": True}, db_path=db)
        self.assertEqual(get_settings(db)["google_client_secret"], "")


# --------------------------------------------------------------------------- gmail


def make_msg(
    mid: str = "m1",
    body: str = "A fairly long message that the user wrote all by themselves today.",
    sender: str = f"Sam Example <{ME}>",
    labels: tuple[str, ...] = ("SENT",),
    extra_headers: dict[str, str] | None = None,
    html: str | None = None,
    calendar: bool = False,
) -> dict:
    headers = [{"name": "From", "value": sender}]
    for k, v in (extra_headers or {}).items():
        headers.append({"name": k, "value": v})
    parts = []
    if html is None:
        parts.append({"mimeType": "text/plain", "body": {"data": b64(body)}})
    else:
        parts.append({"mimeType": "text/html", "body": {"data": b64(html)}})
    if calendar:
        parts.append({"mimeType": "text/calendar", "body": {"data": b64("BEGIN:VCALENDAR")}})
    return {"id": mid, "labelIds": list(labels), "payload": {"mimeType": "multipart/alternative", "headers": headers, "parts": parts}}


class TestGmailClassify(unittest.TestCase):
    ADDR = {ME, "alias@example.com"}

    def test_ok(self) -> None:
        status, body = gmail_import.classify_message(make_msg(), self.ADDR)
        self.assertEqual(status, "ok")
        self.assertIn("wrote all by themselves", body)

    def test_wrong_from(self) -> None:
        self.assertEqual(gmail_import.classify_message(make_msg(sender="Other <other@example.com>"), self.ADDR)[0], "not_user")

    def test_from_case_insensitive_and_alias(self) -> None:
        self.assertEqual(gmail_import.classify_message(make_msg(sender="X <ALIAS@Example.com>"), self.ADDR)[0], "ok")

    def test_draft_and_missing_sent_label(self) -> None:
        self.assertEqual(gmail_import.classify_message(make_msg(labels=("SENT", "DRAFT")), self.ADDR)[0], "not_user")
        self.assertEqual(gmail_import.classify_message(make_msg(labels=("INBOX",)), self.ADDR)[0], "not_user")

    def test_auto_submitted(self) -> None:
        self.assertEqual(gmail_import.classify_message(make_msg(extra_headers={"Auto-Submitted": "auto-replied"}), self.ADDR)[0], "automatic")
        self.assertEqual(gmail_import.classify_message(make_msg(extra_headers={"Auto-Submitted": "no"}), self.ADDR)[0], "ok")

    def test_calendar_part(self) -> None:
        self.assertEqual(gmail_import.classify_message(make_msg(calendar=True), self.ADDR)[0], "automatic")

    def test_prefers_plain_over_html(self) -> None:
        msg = make_msg()
        msg["payload"]["parts"].append({"mimeType": "text/html", "body": {"data": b64("<p>HTML version</p>")}})
        self.assertIn("wrote all by themselves", gmail_import.extract_body(msg["payload"]))

    def test_html_strips_tags_quote_and_signature(self) -> None:
        html = (
            "<div>First line here.<br>Second &amp; line.</div>"
            "<div class=\"gmail_quote\"><div>On Mon wrote:</div><blockquote><div>nested</div></blockquote></div>"
            "<div class=\"gmail_signature\">Sam Example<br>Example Ltd</div><div>after signature</div>"
        )
        text = gmail_import.extract_body(make_msg(html=html)["payload"])
        self.assertIn("First line here.", text)
        self.assertIn("Second & line.", text)
        for gone in ("On Mon wrote", "nested", "Example Ltd", "after signature"):
            self.assertNotIn(gone, text)

    def test_html_drops_blockquote(self) -> None:
        text = gmail_import.html_to_text("<p>Mine.</p><blockquote><p>Theirs.</p></blockquote><p>Mine again.</p>")
        self.assertIn("Mine again.", text)
        self.assertNotIn("Theirs", text)


class TestGmailHttp(AuthTestBase):
    def test_retry_once_on_429(self) -> None:
        seq = [gmail_import.GmailError("x", 429), {"emailAddress": ME}]

        def fake(url, token, timeout=30.0):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(google_auth, "get_access_token", return_value=ACCESS), patch.object(
            gmail_import, "_http_get", side_effect=fake
        ), patch.object(gmail_import, "_sleep") as slept:
            self.assertEqual(gmail_import.get_profile(self.db), ME)
        slept.assert_called_once()

    def test_no_retry_on_404(self) -> None:
        with patch.object(google_auth, "get_access_token", return_value=ACCESS), patch.object(
            gmail_import, "_http_get", side_effect=gmail_import.GmailError("x", 404)
        ) as http:
            with self.assertRaises(gmail_import.GmailError):
                gmail_import.get_message("abc", self.db)
        self.assertEqual(http.call_count, 1)

    def test_list_sent_ids_pages_and_url(self) -> None:
        pages = {
            "": {"messages": [{"id": "a"}, {"id": "b"}], "nextPageToken": "T2"},
            "T2": {"messages": [{"id": "c"}]},
        }
        urls: list[str] = []

        def fake(url, token, timeout=30.0):
            urls.append(url)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            return pages[(q.get("pageToken") or [""])[0]]

        with patch.object(google_auth, "get_access_token", return_value=ACCESS), patch.object(gmail_import, "_http_get", side_effect=fake):
            self.assertEqual(gmail_import.list_sent_ids(self.db), ["a", "b", "c"])
        self.assertTrue(urls[0].startswith("https://gmail.googleapis.com/gmail/v1/users/me/messages?"))
        self.assertIn("labelIds=SENT", urls[0])
        self.assertIn("maxResults=500", urls[0])

    def test_get_message_requests_full_format(self) -> None:
        with patch.object(google_auth, "get_access_token", return_value=ACCESS), patch.object(
            gmail_import, "_http_get", return_value={"id": "z"}
        ) as http:
            gmail_import.get_message("z", self.db)
        self.assertIn("format=full", http.call_args[0][0])


# --------------------------------------------------------------------------- scan / pending


GOOD = "I think we should move the launch to Thursday because the testing is not finished yet."


class ScanTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="vp_scan_")
        self.db = Path(self.tmp) / "t.db"
        init_db(self.db)
        update_settings({"google_account_email": ME, "google_extra_addresses": "alias@example.com"}, db_path=self.db)
        pending_import.discard_pending()
        pending_import.PROGRESS.reset()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(pending_import.discard_pending)

    def run_scan(self, messages: dict[str, object]) -> None:
        def fake_get(mid, db_path=None):
            item = messages[mid]
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(gmail_import, "get_profile", return_value=ME), patch.object(
            gmail_import, "list_sent_ids", return_value=list(messages)
        ), patch.object(gmail_import, "get_message", side_effect=fake_get):
            gmail_import.scan_sent_mail(self.db, pending_import.PROGRESS)


class TestScan(ScanTestBase):
    def messages(self) -> dict[str, object]:
        quoted = f"{GOOD}\n\nOn Mon, 3 Mar 2025 at 10:00, Alex <alex@example.com> wrote:\n> old"
        return {
            "ok1": make_msg("ok1", GOOD),
            "ok2": make_msg("ok2", "Hi Alex,\n\n" + GOOD.replace("Thursday", "Friday") + "\n\nThanks,\nSam", sender="Sam <alias@example.com>"),
            "dup": make_msg("dup", quoted),
            "other": make_msg("other", GOOD, sender="Other <other@example.com>"),
            "draft": make_msg("draft", GOOD, labels=("SENT", "DRAFT")),
            "auto": make_msg("auto", GOOD, extra_headers={"Auto-Submitted": "auto-generated"}),
            "cal": make_msg("cal", GOOD, calendar=True),
            "fwd": make_msg("fwd", "---------- Forwarded message ---------\nFrom: x\nsome text from someone else entirely"),
            "short": make_msg("short", "Ok thanks see you then."),
            "broken": gmail_import.GmailError("boom", 500),
        }

    def test_stage_counts_and_pending_not_written_until_commit(self) -> None:
        self.run_scan(self.messages())
        summary = pending_import.get_summary()
        c = summary["counts"]
        self.assertEqual(
            (c["listed"], c["already_imported"], c["not_user"], c["automatic"], c["empty"], c["too_short"], c["duplicates"], c["failed"], c["kept"]),
            (10, 0, 2, 2, 1, 1, 1, 1, 2),
        )
        self.assertEqual(summary["total_words"], sum(len(t.split()) for t in pending_import._batch["texts"]))
        self.assertEqual(get_samples(db_path=self.db), [])
        self.assertEqual(pending_import.PROGRESS.snapshot()["stage"], "review")

    def test_summary_has_at_most_five_short_excerpts(self) -> None:
        long_text = "word " * 200
        msgs = {f"m{i}": make_msg(f"m{i}", f"{long_text} number{i}") for i in range(8)}
        self.run_scan(msgs)
        s = pending_import.get_summary()
        self.assertEqual(len(s["samples"]), 5)
        self.assertTrue(all(len(x) <= 300 for x in s["samples"]))

    def test_commit_writes_samples_with_source_and_mode(self) -> None:
        self.run_scan(self.messages())
        res = pending_import.commit_pending(self.db)
        self.assertEqual(res["inserted"], 2)
        rows = get_samples(db_path=self.db)
        self.assertEqual({r["source_type"] for r in rows}, {"gmail"})
        self.assertEqual({r["context_mode"] for r in rows}, {"work_email"})
        self.assertEqual(total_sample_words(self.db), sum(r["word_count"] for r in rows))
        self.assertFalse(pending_import.get_summary()["has_pending"])

    def test_discard_drops_batch_and_writes_nothing(self) -> None:
        self.run_scan(self.messages())
        self.assertTrue(pending_import.discard_pending())
        self.assertFalse(pending_import.get_summary()["has_pending"])
        self.assertEqual(pending_import.commit_pending(self.db)["inserted"], 0)
        self.assertEqual(get_samples(db_path=self.db), [])

    def test_second_scan_skips_already_imported_ids(self) -> None:
        msgs = self.messages()
        self.run_scan(msgs)
        pending_import.commit_pending(self.db)
        fetched: list[str] = []
        real = dict(msgs)

        def fake_get(mid, db_path=None):
            fetched.append(mid)
            item = real[mid]
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(gmail_import, "get_profile", return_value=ME), patch.object(
            gmail_import, "list_sent_ids", return_value=list(real)
        ), patch.object(gmail_import, "get_message", side_effect=fake_get):
            gmail_import.scan_sent_mail(self.db, pending_import.PROGRESS)
        # every successfully read message (kept or skipped) is remembered; only the one that failed is fetched again
        self.assertEqual(fetched, ["broken"])
        c = pending_import.get_summary()["counts"]
        self.assertEqual(c["already_imported"], 9)
        self.assertEqual(c["kept"], 0)

    def test_auth_failure_reported_without_pending(self) -> None:
        with patch.object(gmail_import, "get_profile", side_effect=google_auth.GoogleAuthError("sign-in expired", 401)):
            gmail_import.scan_sent_mail(self.db, pending_import.PROGRESS)
        snap = pending_import.PROGRESS.snapshot()
        self.assertEqual((snap["running"], snap["stage"], snap["error"]), (False, "error", "Your Google sign-in expired. Connect again to keep going."))
        self.assertFalse(pending_import.get_summary()["has_pending"])

    def test_start_scan_refuses_second_concurrent_run(self) -> None:
        gate = threading.Event()

        def slow(db_path, progress):
            gate.wait(3)

        with patch.object(gmail_import, "scan_sent_mail", side_effect=slow):
            self.assertTrue(gmail_import.start_scan(self.db))
            self.assertFalse(gmail_import.start_scan(self.db))
            gate.set()
            for _ in range(50):
                if not pending_import.active():
                    break
                time.sleep(0.05)
        self.assertEqual(pending_import.active(), "")


# --------------------------------------------------------------------------- chat takeout


def chat_doc(*msgs: tuple[str, str]) -> dict:
    return {"messages": [{"creator": {"name": "N", "email": e}, "created_date": "x", "text": t} for e, t in msgs]}


class TestChatTakeout(ScanTestBase):
    ADDR = {ME}

    def test_filters_merges_and_masks(self) -> None:
        doc = chat_doc(
            (ME, "I was thinking we could try the new venue next week"),
            (ME, "and book it before the end of the day, ring me on 07700 900123"),
            ("other@example.com", "sounds good"),
            (ME, "ok"),
            ("other@example.com", "great"),
            (ME, "Let me know what you think of the plan when you have a moment please"),
        )
        chat_takeout.build_batch([doc], self.ADDR, self.db)
        s = pending_import.get_summary()
        texts = pending_import._batch["texts"]
        self.assertEqual(len(texts), 2)
        self.assertIn("venue next week\nand book it", texts[0])
        self.assertIn("[phone]", texts[0])
        self.assertNotIn("07700", texts[0])
        self.assertEqual(s["counts"]["not_user"], 2)
        self.assertEqual(s["counts"]["too_short"], 1)
        self.assertEqual(s["counts"]["kept"], 2)
        self.assertEqual(s["source_type"], "google_chat")

    def test_commit_uses_chat_source_and_casual_mode(self) -> None:
        chat_takeout.build_batch([chat_doc((ME, "We should really get the whole team together for lunch on Friday"))], self.ADDR, self.db)
        pending_import.commit_pending(self.db)
        row = get_samples(db_path=self.db)[0]
        self.assertEqual((row["source_type"], row["context_mode"]), ("google_chat", "casual_chat"))

    def test_tolerates_missing_fields(self) -> None:
        doc = {"messages": [{}, {"creator": None, "text": "x"}, {"creator": {"email": ME}}, "junk", {"creator": {"email": ME.upper()}, "text": "Plenty of words in this one to pass the length check easily"}]}
        chat_takeout.build_batch([doc], self.ADDR, self.db)
        self.assertEqual(pending_import.get_summary()["counts"]["kept"], 1)

    def test_requires_an_address(self) -> None:
        with self.assertRaises(chat_takeout.ChatTakeoutError):
            chat_takeout.build_batch([chat_doc((ME, "x"))], set(), self.db)

    def test_plain_json_bytes(self) -> None:
        raw = json.dumps(chat_doc((ME, "A message of sufficient length to be considered substantive here"))).encode()
        chat_takeout.import_export(raw, self.ADDR, self.db)
        self.assertEqual(pending_import.get_summary()["counts"]["kept"], 1)

    def test_rejects_non_chat_json(self) -> None:
        with self.assertRaises(chat_takeout.ChatTakeoutError):
            chat_takeout.read_message_files(b'{"hello": 1}')
        with self.assertRaises(chat_takeout.ChatTakeoutError):
            chat_takeout.read_message_files(b"not json")

    @staticmethod
    def make_zip(members: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return buf.getvalue()

    def test_zip_reads_group_files_only(self) -> None:
        good = json.dumps(chat_doc((ME, "x"))).encode()
        z = self.make_zip(
            {
                "Takeout/Google Chat/Groups/DM abc/messages.json": good,
                "Takeout/Google Chat/Groups/Space 1/messages.json": good,
                "Takeout/Google Chat/Users/u/user_info.json": b"{}",
                "Takeout/Other/messages.json": b"{}",
            }
        )
        self.assertEqual(len(chat_takeout.read_message_files(z)), 2)

    def test_zip_rejects_parent_and_absolute_paths(self) -> None:
        for name in ("../evil/messages.json", "a/../../b.json", "/abs/messages.json"):
            z = self.make_zip({name: b"{}", "Google Chat/Groups/g/messages.json": b'{"messages": []}'})
            with self.assertRaises(chat_takeout.ChatTakeoutError, msg=name):
                chat_takeout.read_message_files(z)

    def test_zip_rejects_oversize_uncompressed(self) -> None:
        z = self.make_zip({"Google Chat/Groups/g/messages.json": b"x" * 100_000})
        self.assertLess(len(z), 5000)
        with patch.object(chat_takeout, "MAX_UNCOMPRESSED_BYTES", 5000):
            with self.assertRaises(chat_takeout.ChatTakeoutError):
                chat_takeout.read_message_files(z)

    def test_zip_without_chat_files(self) -> None:
        with self.assertRaises(chat_takeout.ChatTakeoutError):
            chat_takeout.read_message_files(self.make_zip({"readme.txt": b"hi"}))


# --------------------------------------------------------------------------- routes


class TestGoogleRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="vp_routes_")
        cls.db = Path(cls.tmp) / "r.db"
        cls._orig = os.environ.get("VOICEPRINT_DB_PATH")
        os.environ["VOICEPRINT_DB_PATH"] = str(cls.db)
        init_db(cls.db)
        update_settings({"google_client_id": "client-123.example", "google_client_secret": SECRET, "provider": "local_only"}, db_path=cls.db)
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
        google_auth._reset_for_tests()
        pending_import.discard_pending()
        self.store = {"token": ""}
        for p in (
            patch.object(google_auth, "_keychain_write", side_effect=lambda t: self.store.update(token=t) or True),
            patch.object(google_auth, "_keychain_read", side_effect=lambda: self.store["token"]),
            patch.object(google_auth, "_keychain_delete", side_effect=lambda: self.store.update(token="") or True),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(google_auth._reset_for_tests)
        self.addCleanup(pending_import.discard_pending)
        self.seen: list[str] = []

    def call(self, method: str, path: str, body=None, headers: dict[str, str] | None = None, host: str | None = None, raw: bytes | None = None):
        if raw is not None:
            data = raw
        else:
            data = json.dumps(body if body is not None else {}).encode() if method == "POST" else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method)
        if method == "POST":
            req.add_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if host is not None:
            req.remove_header("Host")
            req.add_unredirected_header("Host", host)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw, status = resp.read().decode(), resp.status
        except urllib.error.HTTPError as err:
            raw, status = err.read().decode(), err.code
        self.seen.append(raw)
        return status, raw

    def test_foreign_origin_rejected_on_every_route(self) -> None:
        routes = [
            ("GET", "/api/google/status"),
            ("GET", "/api/google/progress"),
            ("GET", "/api/google/pending"),
            ("GET", "/api/google/callback?code=a&state=b"),
            ("POST", "/api/google/auth/start"),
            ("POST", "/api/google/disconnect"),
            ("POST", "/api/google/gmail/cancel"),
            ("POST", "/api/google/gmail/scan"),
            ("POST", "/api/google/pending/commit"),
            ("POST", "/api/google/pending/discard"),
            ("POST", "/api/google/chat/upload"),
        ]
        for method, path in routes:
            for origin in ("https://evil.example", "null", f"http://127.0.0.1:{self.port + 1}"):
                status, _ = self.call(method, path, headers={"Origin": origin})
                self.assertEqual(status, 403, (method, path, origin))

    def test_wrong_host_rejected(self) -> None:
        for host in ("evil.example", "evil.example:80", f"127.0.0.1:{self.port + 1}", "127.0.0.1"):
            status, _ = self.call("GET", "/api/google/status", host=host)
            self.assertEqual(status, 403, host)

    def test_own_origin_and_localhost_host_allowed(self) -> None:
        status, _ = self.call("GET", "/api/google/status", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        status, _ = self.call("GET", "/api/google/status", host=f"localhost:{self.port}")
        self.assertEqual(status, 200)

    def test_callback_requires_pending_flow(self) -> None:
        status, raw = self.call("GET", "/api/google/callback?code=abc&state=def")
        self.assertEqual(status, 400)
        self.assertNotIn("abc", raw)

    def test_full_flow_and_no_secret_in_any_response(self) -> None:
        def post(url, data, timeout=20.0):
            if url.endswith("/token"):
                return 200, {"access_token": ACCESS, "refresh_token": REFRESH, "expires_in": 3600}
            return 200, {}

        with patch.object(google_auth, "_post_form", side_effect=post), patch.object(gmail_import, "get_profile", return_value=ME):
            status, raw = self.call("GET", "/api/google/status")
            self.assertEqual(json.loads(raw), {"configured": True, "connected": False, "email": "", "note": ""})
            status, raw = self.call("POST", "/api/google/auth/start")
            self.assertEqual(status, 200)
            auth_url = json.loads(raw)["auth_url"]
            state = urllib.parse.parse_qs(urllib.parse.urlparse(auth_url).query)["state"][0]
            status, page = self.call("GET", f"/api/google/callback?code={CODE}&state={state}")
            self.assertEqual(status, 200)
            self.assertIn("close this tab", page)
            self.assertEqual(self.store["token"], REFRESH)
            # state is single use
            status, page2 = self.call("GET", f"/api/google/callback?code={CODE}&state={state}")
            self.assertEqual(status, 400)
            status, raw = self.call("GET", "/api/google/status")
            self.assertEqual(json.loads(raw)["email"], ME)
            self.call("GET", "/api/state")
            self.call("POST", "/api/settings", {"google_client_id": "client-123.example"})
            status, _ = self.call("POST", "/api/google/pending/commit")
            self.assertEqual(status, 200)
            status, _ = self.call("POST", "/api/google/pending/discard")
            self.assertEqual(status, 200)
            self.call("GET", "/api/google/pending")
            self.call("GET", "/api/google/progress")
            status, _ = self.call("POST", "/api/google/disconnect")
            self.assertEqual(status, 200)
        blob = "\n".join(self.seen)
        for secret in (SECRET, REFRESH, ACCESS, CODE):
            self.assertNotIn(secret, blob)

    def test_scan_requires_connection(self) -> None:
        status, _ = self.call("POST", "/api/google/gmail/scan")
        self.assertEqual(status, 400)

    def test_scan_409_when_running(self) -> None:
        with patch.object(google_auth, "status", return_value={"configured": True, "connected": True, "email": ME, "note": ""}), patch.object(
            gmail_import, "start_scan", return_value=False
        ):
            status, _ = self.call("POST", "/api/google/gmail/scan")
        self.assertEqual(status, 409)

    def test_scan_starts(self) -> None:
        with patch.object(google_auth, "status", return_value={"configured": True, "connected": True, "email": ME, "note": ""}), patch.object(
            gmail_import, "start_scan", return_value=True
        ):
            status, raw = self.call("POST", "/api/google/gmail/scan")
        self.assertEqual((status, json.loads(raw)), (200, {"started": True}))

    def test_chat_upload_pending_and_commit(self) -> None:
        update_settings({"google_account_email": ME}, db_path=self.db)
        doc = chat_doc((ME, "This is a long enough chat message that I wrote myself earlier today"))
        status, raw = self.call("POST", "/api/google/chat/upload", raw=json.dumps(doc).encode())
        self.assertEqual(status, 200)
        summary = json.loads(raw)
        self.assertEqual(summary["counts"]["kept"], 1)
        self.assertLessEqual(max(len(s) for s in summary["samples"]), 300)
        status, raw = self.call("GET", "/api/google/pending")
        self.assertTrue(json.loads(raw)["has_pending"])
        status, raw = self.call("POST", "/api/google/pending/commit")
        res = json.loads(raw)
        self.assertEqual(res["inserted"], 1)
        self.assertGreater(res["state"]["sample_words"], 0)
        self.assertEqual(get_samples(db_path=self.db)[0]["source_type"], "google_chat")

    def test_chat_upload_bad_input(self) -> None:
        update_settings({"google_account_email": "", "google_extra_addresses": ""}, db_path=self.db)
        status, _ = self.call("POST", "/api/google/chat/upload", raw=b"nope")
        self.assertEqual(status, 400)  # no address known yet
        update_settings({"google_account_email": ME}, db_path=self.db)
        status, _ = self.call("POST", "/api/google/chat/upload", raw=b"nope")
        self.assertEqual(status, 400)
        status, _ = self.call("POST", "/api/google/chat/upload", {"content_base64": "!!!"})
        self.assertEqual(status, 415)  # a JSON body is no longer accepted here

    def test_unknown_google_route_404(self) -> None:
        self.assertEqual(self.call("GET", "/api/google/nothing")[0], 404)
        self.assertEqual(self.call("POST", "/api/google/nothing")[0], 404)


if __name__ == "__main__":
    unittest.main()
