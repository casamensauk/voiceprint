#!/usr/bin/env python3
"""Installs a macOS Automator Quick Action (.workflow) in ~/Library/Services for global hotkeys."""

from __future__ import annotations

import os
import plistlib
import shlex
import sys
import uuid
from pathlib import Path

WORKFLOW_NAME = "VoicePrint Rewrite Clipboard.workflow"
CLI_PATH = Path(__file__).resolve().parent / "voiceprint_cli.py"


def install_quick_action(services_dir: Path | None = None) -> dict[str, str]:
    """Generate a valid Automator Quick Action bundle so macOS Keyboard Shortcuts can bind a hotkey."""
    target_dir = services_dir or (Path.home() / "Library" / "Services")
    workflow_dir = target_dir / WORKFLOW_NAME
    contents_dir = workflow_dir / "Contents"
    contents_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable or "/usr/bin/python3"
    command_str = f"{shlex.quote(python_bin)} {shlex.quote(str(CLI_PATH))}"
    db_override = os.environ.get("VOICEPRINT_DB_PATH", "").strip()
    if db_override:
        command_str = f"VOICEPRINT_DB_PATH={shlex.quote(db_override)} {command_str}"

    info_plist = {
        "NSServices": [
            {
                "NSMenuItem": {"default": "VoicePrint: Rewrite Clipboard in My Voice"},
                "NSMessage": "runWorkflowAsService",
            }
        ]
    }
    with (contents_dir / "Info.plist").open("wb") as f:
        plistlib.dump(info_plist, f)

    action_uuid = str(uuid.uuid4()).upper()
    in_uuid = str(uuid.uuid4()).upper()
    out_uuid = str(uuid.uuid4()).upper()

    document_wflow = {
        "AMApplicationBuild": "523",
        "AMApplicationVersion": "2.10",
        "AMDocumentVersion": "2",
        "actions": [
            {
                "action": {
                    "AMAccepts": {"Container": "List", "Optional": True, "Types": ["com.apple.cocoa.string"]},
                    "AMActionVersion": "2.0.3",
                    "AMApplication": ["Automator"],
                    "AMParameterProperties": {
                        "COMMAND_STRING": {},
                        "CheckedForUserDefaultShell": {},
                        "inputMethod": {},
                        "shell": {},
                        "source": {},
                    },
                    "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
                    "ActionBundlePath": "/System/Library/Automator/Run Shell Script.action",
                    "ActionName": "Run Shell Script",
                    "ActionParameters": {
                        "COMMAND_STRING": command_str,
                        "CheckedForUserDefaultShell": True,
                        "inputMethod": 0,
                        "shell": "/bin/zsh",
                        "source": "",
                    },
                    "BundleIdentifier": "com.apple.RunShellScript",
                    "CFBundleVersion": "2.0.3",
                    "CanShowSelectedItemsWhenRun": False,
                    "CanShowWhenRun": True,
                    "Category": ["AMCategoryUtilities"],
                    "Class Name": "RunShellScriptAction",
                    "InputUUID": in_uuid,
                    "Keywords": ["Shell", "Script", "Command", "Run", "Unix"],
                    "OutputUUID": out_uuid,
                    "UUID": action_uuid,
                    "UnlocalizedApplications": ["Automator"],
                    "arguments": {
                        "0": {"default value": 0, "name": "inputMethod", "required": "0", "type": "0", "uuid": "0"},
                        "1": {"default value": False, "name": "CheckedForUserDefaultShell", "required": "0", "type": "0", "uuid": "1"},
                        "2": {"default value": "", "name": "source", "required": "0", "type": "0", "uuid": "2"},
                        "3": {"default value": "", "name": "COMMAND_STRING", "required": "0", "type": "0", "uuid": "3"},
                        "4": {"default value": "/bin/sh", "name": "shell", "required": "0", "type": "0", "uuid": "4"},
                    },
                    "isViewVisible": 1,
                },
                "isViewVisible": 1,
            }
        ],
        "connectors": {},
        "workflowMetaData": {
            "applicationBundleIDsByPath": {},
            "applicationPaths": [],
            "inputTypeIdentifier": "com.apple.Automator.nothing",
            "outputTypeIdentifier": "com.apple.Automator.nothing",
            "presentationMode": 11,
            "processesInput": 0,
            "serviceInputTypeIdentifier": "com.apple.Automator.nothing",
            "serviceOutputTypeIdentifier": "com.apple.Automator.nothing",
            "serviceProcessesInput": 0,
            "systemImageName": "NSActionTemplate",
            "useAutomaticInputType": 0,
            "workflowTypeIdentifier": "com.apple.Automator.servicesMenu",
        },
    }
    with (contents_dir / "document.wflow").open("wb") as f:
        plistlib.dump(document_wflow, f)

    return {
        "installed_path": str(workflow_dir),
        "command": command_str,
        "instructions": (
            "Open System Settings -> Keyboard -> Keyboard Shortcuts -> Services -> General, "
            "find 'VoicePrint: Rewrite Clipboard in My Voice', and assign a hotkey (e.g. Cmd+Shift+V)."
        ),
    }


if __name__ == "__main__":
    res = install_quick_action()
    print(f"Installed Quick Action at: {res['installed_path']}")
    print(res["instructions"])
