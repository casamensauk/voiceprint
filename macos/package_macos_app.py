#!/usr/bin/env python3
"""Package VoicePrint into a standalone VoicePrint.app, .dmg, and .zip for macOS."""

from __future__ import annotations

import math
import os
import plistlib
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
APP_NAME = "VoicePrint.app"
DMG_NAME = "VoicePrint-macOS.dmg"
ZIP_NAME = "VoicePrint-macOS.zip"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _generate_app_icon_png(output_png: Path, size: int = 1024) -> None:
    """Render a crisp 1024x1024 RGBA icon using pure-Python stdlib (dark squircle + waveform)."""
    half = size / 2.0
    corner_r = size * 0.22
    pad = size * 0.04
    box_limit = half - pad - corner_r
    circle_r = size * 0.30
    bar_specs = [
        (-0.15, 0.16),
        (-0.075, 0.30),
        (0.0, 0.40),
        (0.075, 0.28),
        (0.15, 0.14),
    ]
    bar_half_w = size * 0.022

    rows = bytearray()
    for y in range(size):
        rows.append(0)  # PNG filter type 0
        dy = abs(y - half)
        row_px = bytearray()
        for x in range(size):
            dx = abs(x - half)
            # Signed distance to rounded squircle
            qx = max(dx - box_limit, 0.0)
            qy = max(dy - box_limit, 0.0)
            dist_sq = math.hypot(qx, qy) - corner_r
            if dist_sq > 1.0:
                row_px.extend((0, 0, 0, 0))
                continue

            alpha = 255 if dist_sq <= 0.0 else int(max(0, min(255, (1.0 - dist_sq) * 255)))
            # Background dark slate gradient (#131822 -> #0b0e14)
            t_bg = (x + y) / (2.0 * size)
            r = int(19 * (1 - t_bg) + 11 * t_bg)
            g = int(24 * (1 - t_bg) + 14 * t_bg)
            b = int(34 * (1 - t_bg) + 20 * t_bg)

            # Subtle border ring
            if -2.5 <= dist_sq <= 0.0:
                r = int(r * 0.5 + 124 * 0.5)
                g = int(g * 0.5 + 108 * 0.5)
                b = int(b * 0.5 + 255 * 0.5)

            # Central glowing gradient circle (#7c6cff -> #22c55e)
            dist_center = math.hypot(x - half, y - half)
            if dist_center <= circle_r:
                tg = (x - (half - circle_r)) / (2.0 * circle_r)
                tg = max(0.0, min(1.0, tg))
                cr = int(124 * (1 - tg) + 34 * tg)
                cg = int(108 * (1 - tg) + 197 * tg)
                cb = int(255 * (1 - tg) + 94 * tg)
                edge_blend = max(0.0, min(1.0, circle_r - dist_center))
                r = int(r * (1 - edge_blend) + cr * edge_blend)
                g = int(g * (1 - edge_blend) + cg * edge_blend)
                b = int(b * (1 - edge_blend) + cb * edge_blend)

                # White voiceprint equalizer bars inside circle
                nx = (x - half) / size
                ny = abs(y - half) / size
                for bx, bh in bar_specs:
                    if abs(nx - bx) <= (bar_half_w / size) and ny <= bh * 0.5:
                        r, g, b = 255, 255, 255
                        break

            row_px.extend((r, g, b, alpha))
        rows.extend(row_px)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=6)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )
    output_png.write_bytes(png_bytes)


def _build_icns(resources_dir: Path, work_dir: Path) -> Path:
    base_png = work_dir / "icon_1024.png"
    _generate_app_icon_png(base_png, size=1024)

    iconset_dir = work_dir / "AppIcon.iconset"
    iconset_dir.mkdir(parents=True, exist_ok=True)
    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    for px, filename in sizes:
        dest = iconset_dir / filename
        if px == 1024:
            shutil.copy2(base_png, dest)
        else:
            subprocess.run(
                ["sips", "-z", str(px), str(px), str(base_png), "--out", str(dest)],
                capture_output=True,
                check=True,
            )

    icns_path = resources_dir / "AppIcon.icns"
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)],
        capture_output=True,
        check=True,
    )
    return icns_path


def _copy_app_sources(app_root: Path) -> None:
    app_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "server.py", app_root / "server.py")
    ignore_fn = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    for subdir in ("engine", "static", "macos"):
        src = PROJECT_ROOT / subdir
        dst = app_root / subdir
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_fn)


def _write_launcher_script(macos_dir: Path) -> Path:
    launcher = macos_dir / "VoicePrint"
    script = """#!/bin/bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
export LANG="${LANG:-en_US.UTF-8}"
export LC_CTYPE="${LC_CTYPE:-UTF-8}"

APP_ROOT="$(cd "$(dirname "$0")/../Resources/app" && pwd)"
APP_SUPPORT="$HOME/Library/Application Support/VoicePrint"
LOG_DIR="$HOME/Library/Logs"
PORT_FILE="$APP_SUPPORT/server_port.txt"
LOG_FILE="$LOG_DIR/VoicePrint.log"
mkdir -p "$APP_SUPPORT" "$LOG_DIR"

export VOICEPRINT_DB_PATH="$APP_SUPPORT/voiceprint.db"
export VOICEPRINT_PORT_FILE="$PORT_FILE"

PYTHON_BIN="$(command -v python3 || echo /usr/bin/python3)"
"$PYTHON_BIN" "$APP_ROOT/macos/install_quick_action.py" >/dev/null 2>&1 || true

is_voiceprint_port() {
    local port="$1"
    curl -sf --max-time 1 "http://127.0.0.1:${port}/api/health" 2>/dev/null | grep -q '"voiceprint"'
}

# Reuse existing verified VoicePrint daemon if already running
if [ -f "$PORT_FILE" ]; then
    SAVED_PORT="$(tr -dc '0-9' < "$PORT_FILE" || true)"
    if [ -n "$SAVED_PORT" ] && is_voiceprint_port "$SAVED_PORT"; then
        open "http://127.0.0.1:${SAVED_PORT}"
        exit 0
    fi
fi

# Also check default port 18940 in case started manually
if is_voiceprint_port 18940; then
    echo "18940" > "$PORT_FILE"
    open "http://127.0.0.1:18940"
    exit 0
fi

# Pick a guaranteed free localhost port starting at 18940
FREE_PORT="$("$PYTHON_BIN" -c '
import socket
for p in range(18940, 18995):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", p))
            print(p)
            raise SystemExit(0)
        except OSError:
            continue
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
')"

nohup "$PYTHON_BIN" "$APP_ROOT/server.py" --port "$FREE_PORT" --auto-port >>"$LOG_FILE" 2>&1 &

READY=0
for _ in $(seq 1 50); do
    if [ -f "$PORT_FILE" ]; then
        ACTIVE_PORT="$(tr -dc '0-9' < "$PORT_FILE" || true)"
        if [ -n "$ACTIVE_PORT" ] && is_voiceprint_port "$ACTIVE_PORT"; then
            READY=1
            FREE_PORT="$ACTIVE_PORT"
            break
        fi
    elif is_voiceprint_port "$FREE_PORT"; then
        READY=1
        echo "$FREE_PORT" > "$PORT_FILE"
        break
    fi
    sleep 0.1
done

if [ "$READY" -eq 1 ]; then
    open "http://127.0.0.1:${FREE_PORT}"
    exit 0
fi

ERR_TAIL="$(tail -n 6 "$LOG_FILE" 2>/dev/null | tr '"' "'" | tr '\n' ' ')"
osascript -e "display alert \"VoicePrint could not start\" message \"Check $LOG_FILE. Details: $ERR_TAIL\" as critical" >/dev/null 2>&1 || true
exit 1
"""
    launcher.write_text(script, encoding="utf-8")
    os.chmod(launcher, 0o755)
    return launcher


def _codesign_bundle(app_bundle: Path) -> None:
    """Ad-hoc sign the app bundle so macOS Apple Silicon Gatekeeper validates it cleanly."""
    subprocess.run(
        ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app_bundle)],
        capture_output=True,
        check=False,
    )


def package_macos() -> dict[str, str]:
    """Build VoicePrint.app, VoicePrint-macOS.zip, VoicePrint-macOS.dmg, and install locally."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    app_bundle = DIST_DIR / APP_NAME
    if app_bundle.exists():
        shutil.rmtree(app_bundle)

    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    app_root = resources_dir / "app"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    info_plist = {
        "CFBundleName": "VoicePrint",
        "CFBundleDisplayName": "VoicePrint",
        "CFBundleIdentifier": "com.voiceprint.studio",
        "CFBundleVersion": "1.2.1",
        "CFBundleShortVersionString": "1.2.1",
        "CFBundleExecutable": "VoicePrint",
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    }
    with (contents_dir / "Info.plist").open("wb") as f:
        plistlib.dump(info_plist, f)

    with tempfile.TemporaryDirectory(prefix="voiceprint_pkg_") as tmp:
        work_dir = Path(tmp)
        _build_icns(resources_dir, work_dir)
        _copy_app_sources(app_root)
        _write_launcher_script(macos_dir)
        _codesign_bundle(app_bundle)

        zip_path = DIST_DIR / ZIP_NAME
        if zip_path.exists():
            zip_path.unlink()
        subprocess.run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app_bundle), str(zip_path)],
            check=True,
        )

        dmg_path = DIST_DIR / DMG_NAME
        if dmg_path.exists():
            dmg_path.unlink()
        dmg_staging = work_dir / "dmg_staging"
        dmg_staging.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ditto", str(app_bundle), str(dmg_staging / APP_NAME)], check=True)
        os.symlink("/Applications", dmg_staging / "Applications")
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                "VoicePrint",
                "-srcfolder",
                str(dmg_staging),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            capture_output=True,
            check=True,
        )

    # Install to /Applications (or ~/Applications if /Applications is not writable)
    install_target = Path("/Applications") / APP_NAME
    try:
        if install_target.exists():
            shutil.rmtree(install_target)
        subprocess.run(["ditto", str(app_bundle), str(install_target)], check=True)
        installed_app = str(install_target)
    except Exception:
        fallback_dir = Path.home() / "Applications"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_target = fallback_dir / APP_NAME
        if fallback_target.exists():
            shutil.rmtree(fallback_target)
        subprocess.run(["ditto", str(app_bundle), str(fallback_target)], check=True)
        installed_app = str(fallback_target)

    return {
        "app_bundle": str(app_bundle),
        "zip_path": str(zip_path),
        "dmg_path": str(dmg_path),
        "installed_app": installed_app,
    }


if __name__ == "__main__":
    result = package_macos()
    for k, v in result.items():
        print(f"{k}: {v}")
