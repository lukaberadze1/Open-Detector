"""Open Detector: a calibrated ensemble for AI-generated text detection."""

from .config import DetectorConfig
from .detector import Detector, Report, SentenceScore

__all__ = ["Detector", "DetectorConfig", "Report", "SentenceScore"]
