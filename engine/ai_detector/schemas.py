"""Schemas and Pangram-compatible payload dataclasses for the in-house AI detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WindowSignals:
    """Per-window forensic, DNA-GPT regeneration, and arbiter sub-scores."""

    stylometric_ai_score: float = 0.0
    stylometric_humanizer_score: float = 0.0
    dna_bscore: float | None = None
    dna_entropy: float | None = None
    arbiter_ai_score: float | None = None
    arbiter_humanizer_score: float | None = None
    burstiness_cv: float = 0.0
    contraction_rate: float = 0.5
    mattr: float = 0.0
    ai_lexicon_hits: int = 0
    unicode_anomaly_count: int = 0
    homoglyph_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectorWindow:
    """Window entry matching Pangram v4.1 contract plus forensic trigger explanations."""

    text: str
    label: str
    ai_assistance_score: float
    confidence: str
    is_humanized: bool
    humanizer_score: float
    start_index: int
    end_index: int
    word_count: int
    token_length: int
    is_excluded_reference: bool = False
    reasons: list[str] = field(default_factory=list)
    signals: WindowSignals = field(default_factory=WindowSignals)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class DetectorReport:
    """Top-level report compatible with Pangram v4.1 UI & API expectations."""

    scan_type: str
    engine: str
    mode: str
    version: str
    model: str
    text: str
    headline: str
    prediction: str
    prediction_short: str
    fraction_ai: float
    fraction_ai_assisted: float
    fraction_human: float
    num_ai_segments: int
    num_ai_assisted_segments: int
    num_human_segments: int
    has_humanized_segments: bool
    max_humanizer_score: float
    low_word_count_warning: bool
    low_confidence_asterisk: bool
    cache_hit: bool
    ensemble_breakdown: dict[str, Any]
    windows: list[DetectorWindow]
    estimate: dict[str, Any]
    dashboard_link: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_type": self.scan_type,
            "engine": self.engine,
            "mode": self.mode,
            "version": self.version,
            "model": self.model,
            "text": self.text,
            "headline": self.headline,
            "prediction": self.prediction,
            "prediction_short": self.prediction_short,
            "fraction_ai": self.fraction_ai,
            "fraction_ai_assisted": self.fraction_ai_assisted,
            "fraction_human": self.fraction_human,
            "num_ai_segments": self.num_ai_segments,
            "num_ai_assisted_segments": self.num_ai_assisted_segments,
            "num_human_segments": self.num_human_segments,
            "has_humanized_segments": self.has_humanized_segments,
            "max_humanizer_score": self.max_humanizer_score,
            "low_word_count_warning": self.low_word_count_warning,
            "low_confidence_asterisk": self.low_confidence_asterisk,
            "cache_hit": self.cache_hit,
            "ensemble_breakdown": self.ensemble_breakdown,
            "windows": [w.to_dict() for w in self.windows],
            "estimate": self.estimate,
            "dashboard_link": self.dashboard_link,
        }
