"""In-app GitHub Release update checker and self-updater for VoicePrint.app and source."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .version import APP_VERSION, GITHUB_REPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class UpdaterError(Exception):
    """Structured error raised during update check or staging."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def parse_semver(version_str: str) -> tuple[int, ...]:
    """Parse a tag like 'v1.2.0' or '1.2.0' into a comparable integer tuple."""
    cleaned = (version_str or "").strip().lstrip("vV")
    parts = [int(m) for m in re.findall(r"\d+", cleaned)]
    return tuple(parts) if parts else (0,)


def is_newer_version(latest: str, current: str) -> bool:
    """Return True if latest semver strictly exceeds current semver."""
    a = parse_semver(latest)
    b = parse_semver(current)
    max_len = max(len(a), len(b))
    a_pad = a + (0,) * (max_len - len(a))
    b_pad = b + (0,) * (max_len - len(b))
    return a_pad > b_pad


def detect_install_target() -> dict[str, str]:
    """Detect whether VoicePrint is running inside a .app bundle or source directory."""
    parts = PROJECT_ROOT.parts
    if len(parts) >= 4 and parts[-3:] == ("Contents", "Resources", "app"):
        bundle_path = PROJECT_ROOT.parent.parent.parent
        if bundle_path.suffix == ".app":
            return {
                "mode": "app_bundle",
                "target_path": str(bundle_path),
                "app_root": str(PROJECT_ROOT),
            }
    return {
        "mode": "source_dir",
        "target_path": str(PROJECT_ROOT),
        "app_root": str(PROJECT_ROOT),
    }


def check_github_update(
    current_version: str = APP_VERSION,
    repo: str = GITHUB_REPO,
) -> dict[str, Any]:
    """Check GitHub Releases /latest for a newer VoicePrint release."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"VoicePrint-Updater/{current_version}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = str(err)
        raise UpdaterError(f"GitHub update check failed ({err.code}): {detail}", int(err.code or 502)) from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise UpdaterError(f"Could not contact GitHub Releases: {err}", 502) from err

    tag = str(data.get("tag_name") or "").strip()
    latest_clean = tag.lstrip("vV") or current_version
    zip_url = ""
    for asset in data.get("assets") or []:
        if isinstance(asset, dict) and str(asset.get("name") or "") == "VoicePrint-macOS.zip":
            zip_url = str(asset.get("browser_download_url") or "")
            break

    install_info = detect_install_target()
    return {
        "ok": True,
        "current_version": current_version.lstrip("vV"),
        "latest_version": latest_clean,
        "latest_tag": tag,
        "update_available": is_newer_version(latest_clean, current_version),
        "release_url": str(data.get("html_url") or f"https://github.com/{repo}/releases"),
        "download_url": zip_url,
        "release_notes": str(data.get("body") or ""),
        "published_at": str(data.get("published_at") or ""),
        "install_mode": install_info["mode"],
        "target_path": install_info["target_path"],
    }


def stage_and_apply_update(download_url: str, current_port: int) -> dict[str, Any]:
    """Download VoicePrint-macOS.zip, verify bundle, and apply + restart cleanly in background."""
    allowed_prefix = f"https://github.com/{GITHUB_REPO}/releases/download/"
    if not download_url or not download_url.startswith(allowed_prefix):
        raise UpdaterError(
            f"Refusing untrusted download URL; must start with {allowed_prefix}",
            400,
        )

    staging_dir = Path(tempfile.mkdtemp(prefix="voiceprint_update_"))
    zip_path = staging_dir / "VoicePrint-macOS.zip"
    extract_dir = staging_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": f"VoicePrint-Updater/{APP_VERSION}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            with zip_path.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise UpdaterError(f"Failed to download update package: {err}", 502) from err

    try:
        subprocess.run(
            ["/usr/bin/ditto", "-x", "-k", str(zip_path), str(extract_dir)],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise UpdaterError(f"Failed to extract update archive: {err}", 500) from err

    extracted_app = extract_dir / "VoicePrint.app"
    extracted_server = extracted_app / "Contents" / "Resources" / "app" / "server.py"
    if not extracted_server.is_file():
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise UpdaterError("Update archive is missing VoicePrint.app/Contents/Resources/app/server.py.", 422)

    subprocess.run(["/usr/bin/xattr", "-cr", str(extracted_app)], capture_output=True, check=False)
    subprocess.run(
        ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(extracted_app)],
        capture_output=True,
        check=False,
    )

    install_info = detect_install_target()
    mode = install_info["mode"]
    target_path = Path(install_info["target_path"])
    old_pid = os.getpid()
    python_bin = sys.executable or "/usr/bin/python3"
    log_file = Path.home() / "Library" / "Logs" / "VoicePrint.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if mode == "app_bundle":
        new_server_py = target_path / "Contents" / "Resources" / "app" / "server.py"
        copy_block = (
            f"rm -rf {shlex.quote(str(target_path))}\n"
            f"/usr/bin/ditto {shlex.quote(str(extracted_app))} {shlex.quote(str(target_path))}\n"
            f"/usr/bin/codesign --force --deep --sign - {shlex.quote(str(target_path))} >/dev/null 2>&1 || true\n"
        )
    else:
        new_server_py = target_path / "server.py"
        ext_app_root = extracted_app / "Contents" / "Resources" / "app"
        copy_lines = [
            f"cp {shlex.quote(str(ext_app_root / 'server.py'))} {shlex.quote(str(target_path / 'server.py'))}",
        ]
        for sub in ("engine", "static", "macos"):
            copy_lines.append(
                f"rm -rf {shlex.quote(str(target_path / sub))} && /usr/bin/ditto {shlex.quote(str(ext_app_root / sub))} {shlex.quote(str(target_path / sub))}"
            )
        apps_bundle = Path("/Applications/VoicePrint.app")
        if apps_bundle.exists() and os.access(apps_bundle.parent, os.W_OK):
            copy_lines.append(
                f"rm -rf {shlex.quote(str(apps_bundle))} && /usr/bin/ditto {shlex.quote(str(extracted_app))} {shlex.quote(str(apps_bundle))}"
            )
        copy_block = "\n".join(copy_lines) + "\n"

    script_path = staging_dir / "run_update.sh"
    script_body = f"""#!/bin/bash
set -u
sleep 0.8
{copy_block}
kill {old_pid} 2>/dev/null || true
for _ in $(seq 1 25); do
    if ! kill -0 {old_pid} 2>/dev/null; then
        break
    fi
    sleep 0.1
done
nohup {shlex.quote(python_bin)} {shlex.quote(str(new_server_py))} --port {int(current_port)} --auto-port >>{shlex.quote(str(log_file))} 2>&1 &
rm -rf {shlex.quote(str(staging_dir))}
"""
    script_path.write_text(script_body, encoding="utf-8")
    os.chmod(script_path, 0o755)

    subprocess.Popen(
        ["/bin/bash", str(script_path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "ok": True,
        "restarting": True,
        "port": int(current_port),
        "install_mode": mode,
        "target_path": str(target_path),
    }
