"""Open Detector: a calibrated ensemble for AI-generated text detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import DetectorConfig
    from .detector import Detector, Report, SentenceScore

__all__ = ["Detector", "DetectorConfig", "Report", "SentenceScore"]


def __getattr__(name: str) -> Any:
    """Load detector classes only when requested, keeping utility submodules lightweight."""
    if name == "DetectorConfig":
        from .config import DetectorConfig
        return DetectorConfig
    if name in {"Detector", "Report", "SentenceScore"}:
        from . import detector
        return getattr(detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
