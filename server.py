#!/usr/bin/env python3
"""Threaded HTTP API & Studio UI server for VoicePrint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 18940

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
DIST_DIR = PROJECT_ROOT / "dist"
DOWNLOADABLE_PACKAGES = {
    "/download/VoicePrint-macOS.dmg": "VoicePrint-macOS.dmg",
    "/download/VoicePrint-macOS.zip": "VoicePrint-macOS.zip",
}
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.academic_coach import check_source_citations, coach_academic_draft  # noqa: E402
from engine.ai_detector import InHouseDetectorError, analyze_text_for_ai, scan_file_inhouse  # noqa: E402
from engine.ai_detector.agreement import pangram_agreement  # noqa: E402
from engine.ai_detector.dna_regeneration import OpenRouterError, test_openrouter_key  # noqa: E402
from engine.ai_detector.orchestrator import resolve_openrouter_config  # noqa: E402
from engine.analyzer import analyze_text  # noqa: E402
from engine.db import (  # noqa: E402
    CONTEXT_MODES,
    persist_pangram_result,
    add_samples_bulk,
    clear_samples,
    delete_learned_replacement,
    delete_pangram_scan,
    delete_sample,
    get_edit_pairs,
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
from engine.importers import parse_imported_content  # noqa: E402
from engine.pangram_client import (  # noqa: E402
    PangramError,
    check_web_plagiarism_pangram,
    list_pangram_models,
    resolve_pangram_key,
    scan_file_pangram,
    scan_text_pangram,
)
from engine.rewriter import resolve_voice_corpus, rewrite_text  # noqa: E402
from engine.style_dna import extract_voice_dna, learn_from_user_edit  # noqa: E402
from engine.updater import UpdaterError, check_github_update, stage_and_apply_update  # noqa: E402
from engine.version import APP_VERSION  # noqa: E402
from macos.install_quick_action import install_quick_action  # noqa: E402


_UTF8_ENV = {**os.environ, "LANG": "en_US.UTF-8", "LC_CTYPE": "UTF-8"}


def _read_pbpaste() -> str:
    proc = subprocess.run(
        ["pbpaste"], capture_output=True, text=True, encoding="utf-8", env=_UTF8_ENV, check=False
    )
    return proc.stdout or ""


def _write_pbcopy(text: str) -> bool:
    proc = subprocess.run(
        ["pbcopy"], input=text, text=True, encoding="utf-8", env=_UTF8_ENV, check=False
    )
    return proc.returncode == 0


def _compute_maturity(total_samples: int, gold_edits: int, rules_count: int) -> dict[str, Any]:
    score = min(100, total_samples * 3 + gold_edits * 10 + rules_count * 4)
    if total_samples == 0 and gold_edits == 0:
        label = "Seedling (Upload Samples)"
    elif score < 30:
        label = "Early Voice"
    elif score < 65:
        label = "Growing Twin"
    elif score < 90:
        label = "Tuned Voice"
    else:
        label = "Deep Mirror"
    return {
        "score": score,
        "label": label,
        "total_samples": total_samples,
        "gold_edits": gold_edits,
        "rules_count": rules_count,
    }


def build_state_payload(requested_mode: str | None = None) -> dict[str, Any]:
    """Assemble full dashboard state including all mode DNA profiles."""
    settings = get_public_settings()
    active_mode = requested_mode or settings.get("active_context_mode") or "casual_chat"

    dna_by_mode: dict[str, Any] = {}
    all_samples = get_samples(limit=1000)
    all_rules = get_learned_replacements()

    for mode in CONTEXT_MODES:
        mode_samples = [s for s in all_samples if s["context_mode"] == mode]
        mode_rules = [r for r in all_rules if r["context_mode"] == mode]
        dna_by_mode[mode] = extract_voice_dna(mode_samples, mode_rules)

    active_samples = [s for s in all_samples if s["context_mode"] == active_mode][:100]
    active_rules = [r for r in all_rules if r["context_mode"] == active_mode]
    gold_edits = sum(1 for s in all_samples if float(s.get("weight", 1.0)) > 1.5)

    return {
        "app_version": APP_VERSION,
        "settings": settings,
        "active_context_mode": active_mode,
        "dna_by_mode": dna_by_mode,
        "active_dna": dna_by_mode.get(active_mode, {}),
        "samples": active_samples,
        "learned_replacements": active_rules,
        "edit_pairs": get_edit_pairs(context_mode=active_mode, limit=20),
        "history": get_rewrite_history(limit=20),
        "pangram_scans": get_pangram_scans(limit=20),
        "pangram_agreement": pangram_agreement(),
        "maturity": _compute_maturity(len(all_samples), gold_edits, len(all_rules)),
    }


def _persist_pangram_result(res: dict[str, Any], title: str) -> int:
    """Save a completed Pangram text, file, or plagiarism scan in SQLite."""
    return persist_pangram_result(res, title)


class VoicePrintHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving static studio assets and JSON API routes."""

    server_version = "VoicePrint/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, file_path: Path, head_only: bool = False) -> None:
        if not file_path.is_file() or STATIC_DIR not in file_path.resolve().parents and file_path.resolve() != STATIC_DIR / "index.html":
            self._send_json({"error": "Not found"}, status=404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        size = file_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        if not head_only:
            self.wfile.write(file_path.read_bytes())

    def _send_download(self, filename: str, head_only: bool = False) -> None:
        file_path = (DIST_DIR / filename).resolve()
        if not file_path.is_file() or DIST_DIR not in file_path.parents:
            self._send_json({"error": "Package not built yet"}, status=404)
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        size = file_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            with file_path.open("rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send_file(STATIC_DIR / "index.html", head_only=True)
            return
        if path.startswith("/static/"):
            self._send_file((STATIC_DIR / path[len("/static/") :]).resolve(), head_only=True)
            return
        if path in DOWNLOADABLE_PACKAGES:
            self._send_download(DOWNLOADABLE_PACKAGES[path], head_only=True)
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self._send_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            self._send_file((STATIC_DIR / rel).resolve())
            return
        if path in DOWNLOADABLE_PACKAGES:
            self._send_download(DOWNLOADABLE_PACKAGES[path])
            return

        if path == "/api/health":
            self._send_json(
                {
                    "app": "voiceprint",
                    "version": APP_VERSION,
                    "ok": True,
                    "port": self.server.server_port,
                }
            )
            return

        if path == "/api/update/check":
            try:
                self._send_json(check_github_update(current_version=APP_VERSION))
            except UpdaterError as err:
                self._send_json({"ok": False, "error": err.message}, status=err.status_code)
            return

        if path == "/api/state":
            mode = (qs.get("mode") or [None])[0]
            self._send_json(build_state_payload(mode))
            return

        if path == "/api/ai-detector/test-key":
            settings = get_settings()
            override_key = (qs.get("api_key") or [""])[0]
            override_url = (qs.get("base_url") or [""])[0]
            cfg = resolve_openrouter_config(settings, override_api_key=override_key)
            base_url = override_url.strip() or cfg["base_url"]
            try:
                res = test_openrouter_key(cfg["api_key"], base_url=base_url)
                self._send_json(res)
            except OpenRouterError as err:
                self._send_json({"ok": False, "error": err.message}, status=err.status_code)
            return

        if path == "/api/clipboard":
            settings = get_public_settings()
            mode = (qs.get("mode") or [settings.get("active_context_mode") or "casual_chat"])[0]
            text = _read_pbpaste()
            samples, rules, _ = resolve_voice_corpus(mode)
            dna = extract_voice_dna(samples, rules)
            self._send_json({"text": text, "analysis": analyze_text(text, dna), "context_mode": mode})
            return

        if path == "/api/pangram/models":
            settings = get_settings()
            override = (qs.get("api_key") or [""])[0]
            try:
                key = resolve_pangram_key(settings, override)
                models = list_pangram_models(key)
                self._send_json({"models": models, "ok": True})
            except PangramError as err:
                self._send_json({"error": err.message, "ok": False}, status=err.status_code)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if path == "/api/clipboard":
            text = str(body.get("text") or "")
            ok = _write_pbcopy(text)
            self._send_json({"ok": ok, "length": len(text)})
            return

        if path == "/api/analyze":
            text = str(body.get("text") or "")
            mode = str(body.get("context_mode") or get_public_settings().get("active_context_mode") or "casual_chat")
            samples, rules, _ = resolve_voice_corpus(mode)
            dna = extract_voice_dna(samples, rules)
            self._send_json({"analysis": analyze_text(text, dna), "context_mode": mode})
            return

        if path == "/api/rewrite":
            text = str(body.get("text") or "")
            if body.get("from_clipboard") and not text.strip():
                text = _read_pbpaste()
            mode = body.get("context_mode")
            intensity = body.get("intensity")
            result = rewrite_text(text, context_mode=mode, intensity=intensity, save_history=True)
            if body.get("auto_copy") and result["rewritten"]:
                _write_pbcopy(result["rewritten"])
            self._send_json(result)
            return

        if path == "/api/learn-edit":
            mode = str(body.get("context_mode") or "casual_chat")
            original_input = str(body.get("original_input") or "")
            ai_draft = str(body.get("ai_draft") or "")
            user_final_edit = str(body.get("user_final_edit") or "")
            if body.get("copy", True) and user_final_edit.strip():
                _write_pbcopy(user_final_edit)
            learn_res = learn_from_user_edit(
                context_mode=mode,
                original_input=original_input,
                ai_draft=ai_draft,
                user_final_edit=user_final_edit,
            )
            samples, rules, _ = resolve_voice_corpus(mode)
            dna = extract_voice_dna(samples, rules)
            analysis = analyze_text(user_final_edit, dna)
            self._send_json({
                **learn_res,
                "analysis": analysis,
                "state": build_state_payload(mode),
            })
            return

        if path == "/api/samples/import":
            raw_text = str(body.get("raw_text") or "")
            source_format = str(body.get("source_format") or "auto")
            sender_name = str(body.get("sender_name") or "")
            mode = str(body.get("context_mode") or "casual_chat")
            parsed = parse_imported_content(raw_text, source_format=source_format, sender_name=sender_name)
            items = parsed["samples"]
            inserted = add_samples_bulk(
                items,
                context_mode=mode,
                source_type=str(parsed["detected_format"]),
                weight=1.0,
            )
            self._send_json({
                "detected_format": parsed["detected_format"],
                "parsed_count": len(items),
                "inserted_count": inserted,
                "state": build_state_payload(mode),
            })
            return

        if path == "/api/samples/seed-demo":
            counts = seed_demo_samples()
            mode = str(body.get("context_mode") or "casual_chat")
            self._send_json({"seeded": counts, "state": build_state_payload(mode)})
            return

        if path == "/api/samples/clear":
            mode = body.get("context_mode")
            removed = clear_samples(context_mode=mode)
            self._send_json({"removed": removed, "state": build_state_payload(mode)})
            return

        if path == "/api/rules":
            mode = str(body.get("context_mode") or "casual_chat")
            avoid_text = str(body.get("avoid_text") or "").strip()
            prefer_text = str(body.get("prefer_text") or "").strip()
            if not avoid_text:
                self._send_json({"error": "avoid_text is required"}, status=400)
                return
            rule = upsert_learned_replacement(mode, avoid_text, prefer_text, source="manual")
            self._send_json({"rule": rule, "state": build_state_payload(mode)})
            return

        if path == "/api/settings":
            updated = update_settings(body)
            mode = updated.get("active_context_mode")
            self._send_json({"settings": updated, "state": build_state_payload(mode)})
            return

        if path == "/api/macos/install-hotkey":
            res = install_quick_action()
            self._send_json(res)
            return

        if path == "/api/pangram/scan":
            settings = get_settings()
            text = str(body.get("text") or "")
            model = str(body.get("model") or settings.get("pangram_model") or "pangram-4")
            pub_raw = body.get("public_dashboard_link")
            public_link = (
                bool(pub_raw)
                if pub_raw is not None
                else settings.get("pangram_public_link", "false").lower() == "true"
            )
            title = str(body.get("title") or "").strip() or (text.strip()[:52] + ("..." if len(text.strip()) > 52 else ""))
            try:
                key = resolve_pangram_key(settings, str(body.get("api_key") or ""))
                res = scan_text_pangram(text, api_key=key, model=model, public_dashboard_link=public_link)
                scan_id = _persist_pangram_result(res, title or "Pasted Submission")
                res["scan_id"] = scan_id
                self._send_json({"result": res, "state": build_state_payload(settings.get("active_context_mode"))})
            except PangramError as err:
                self._send_json({"error": err.message}, status=err.status_code)
            return

        if path == "/api/pangram/file":
            settings = get_settings()
            filename = str(body.get("filename") or "submission.txt")
            b64 = str(body.get("content_base64") or "")
            prefer_p4 = bool(body.get("prefer_pangram_4", True))
            pub_raw = body.get("public_dashboard_link")
            public_link = (
                bool(pub_raw)
                if pub_raw is not None
                else settings.get("pangram_public_link", "false").lower() == "true"
            )
            try:
                file_bytes = base64.b64decode(b64, validate=True)
            except Exception:
                self._send_json({"error": "Invalid base64 file content."}, status=400)
                return
            try:
                key = resolve_pangram_key(settings, str(body.get("api_key") or ""))
                res = scan_file_pangram(
                    filename,
                    file_bytes,
                    api_key=key,
                    prefer_pangram_4=prefer_p4,
                    public_dashboard_link=public_link,
                )
                scan_id = _persist_pangram_result(res, Path(filename).name)
                res["scan_id"] = scan_id
                self._send_json({"result": res, "state": build_state_payload(settings.get("active_context_mode"))})
            except PangramError as err:
                self._send_json({"error": err.message}, status=err.status_code)
            return

        if path == "/api/pangram/plagiarism":
            settings = get_settings()
            text = str(body.get("text") or "")
            title = str(body.get("title") or "").strip() or (text.strip()[:52] + ("..." if len(text.strip()) > 52 else ""))
            try:
                key = resolve_pangram_key(settings, str(body.get("api_key") or ""))
                res = check_web_plagiarism_pangram(text, api_key=key)
                scan_id = _persist_pangram_result(res, title or "Plagiarism Check")
                res["scan_id"] = scan_id
                self._send_json({"result": res, "state": build_state_payload(settings.get("active_context_mode"))})
            except PangramError as err:
                self._send_json({"error": err.message}, status=err.status_code)
            return

        if path == "/api/academic/citation-check":
            draft_text = str(body.get("draft_text") or "")
            source_text = str(body.get("source_text") or "")
            self._send_json(check_source_citations(draft_text, source_text))
            return

        if path == "/api/academic/coach":
            draft_text = str(body.get("draft_text") or "")
            self._send_json(coach_academic_draft(draft_text))
            return

        if path == "/api/ai-detector/scan":
            settings = get_settings()
            text = str(body.get("text") or "")
            use_or = bool(body.get("use_openrouter", True))
            bypass_cache = bool(body.get("bypass_cache", False))
            title = str(body.get("title") or "").strip() or (text.strip()[:52] + ("..." if len(text.strip()) > 52 else ""))
            try:
                res = analyze_text_for_ai(
                    text,
                    use_openrouter=use_or,
                    override_api_key=str(body.get("api_key") or ""),
                    bypass_cache=bypass_cache,
                    scan_type="inhouse",
                )
                scan_id = _persist_pangram_result(res, title or "In-House Scan")
                res["scan_id"] = scan_id
                self._send_json({"result": res, "state": build_state_payload(settings.get("active_context_mode"))})
            except InHouseDetectorError as err:
                self._send_json({"error": err.message}, status=err.status_code)
            return

        if path == "/api/ai-detector/file":
            settings = get_settings()
            filename = str(body.get("filename") or "submission.txt")
            b64 = str(body.get("content_base64") or "")
            use_or = bool(body.get("use_openrouter", True))
            try:
                file_bytes = base64.b64decode(b64, validate=True)
            except Exception:
                self._send_json({"error": "Invalid base64 file content."}, status=400)
                return
            try:
                res = scan_file_inhouse(
                    filename,
                    file_bytes,
                    use_openrouter=use_or,
                    override_api_key=str(body.get("api_key") or ""),
                )
                scan_id = _persist_pangram_result(res, Path(filename).name)
                res["scan_id"] = scan_id
                self._send_json({"result": res, "state": build_state_payload(settings.get("active_context_mode"))})
            except InHouseDetectorError as err:
                self._send_json({"error": err.message}, status=err.status_code)
            return

        if path == "/api/update/apply":
            download_url = str(body.get("download_url") or "").strip()
            try:
                if not download_url:
                    info = check_github_update(current_version=APP_VERSION)
                    download_url = str(info.get("download_url") or "").strip()
                    if not download_url:
                        self._send_json({"ok": False, "error": "Latest GitHub release does not contain VoicePrint-macOS.zip."}, status=404)
                        return
                res = stage_and_apply_update(download_url=download_url, current_port=self.server.server_port)
                self._send_json(res)
            except UpdaterError as err:
                self._send_json({"ok": False, "error": err.message}, status=err.status_code)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        mode = (qs.get("mode") or [None])[0]

        if path.startswith("/api/samples/"):
            try:
                sample_id = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._send_json({"error": "Invalid sample ID"}, status=400)
                return
            ok = delete_sample(sample_id)
            self._send_json({"deleted": ok, "state": build_state_payload(mode)})
            return

        if path.startswith("/api/rules/"):
            try:
                rule_id = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._send_json({"error": "Invalid rule ID"}, status=400)
                return
            ok = delete_learned_replacement(rule_id)
            self._send_json({"deleted": ok, "state": build_state_payload(mode)})
            return

        if path.startswith("/api/pangram/scans/"):
            try:
                scan_id = int(path.rsplit("/", 1)[-1])
            except ValueError:
                self._send_json({"error": "Invalid scan ID"}, status=400)
                return
            ok = delete_pangram_scan(scan_id)
            self._send_json({"deleted": ok, "state": build_state_payload(mode)})
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def bind_server(
    host: str = "127.0.0.1",
    port: int | None = None,
    auto_port: bool = False,
) -> ThreadingHTTPServer:
    """Bind ThreadingHTTPServer to the requested port or first free fallback port when auto_port=True."""
    requested = port if port is not None else int(os.environ.get("PORT", str(DEFAULT_PORT)))
    if not auto_port:
        return ThreadingHTTPServer((host, requested), VoicePrintHandler)

    for candidate in range(requested, requested + 50):
        try:
            return ThreadingHTTPServer((host, candidate), VoicePrintHandler)
        except OSError:
            continue
    return ThreadingHTTPServer((host, 0), VoicePrintHandler)


def run_server(
    host: str = "127.0.0.1",
    port: int | None = None,
    auto_port: bool = False,
) -> None:
    """Start the threaded HTTP server and optionally record the bound port to VOICEPRINT_PORT_FILE."""
    init_db()
    httpd = bind_server(host=host, port=port, auto_port=auto_port)
    actual_port = httpd.server_port
    port_file = os.environ.get("VOICEPRINT_PORT_FILE", "").strip()
    if port_file:
        p = Path(port_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(actual_port), encoding="utf-8")

    print(f"VoicePrint Studio running at http://{host}:{actual_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down VoicePrint server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoicePrint Studio HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--auto-port", action="store_true", help="Automatically fall back to a free localhost port if occupied")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, auto_port=args.auto_port)
