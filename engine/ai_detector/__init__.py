"""In-house 3-stage multi-signal AI authorship & humanizer detection engine."""

from __future__ import annotations

from .orchestrator import InHouseDetectorError, analyze_text_for_ai, scan_file_inhouse

__all__ = [
    "InHouseDetectorError",
    "analyze_text_for_ai",
    "scan_file_inhouse",
]
