"""The detection pipeline: normalize, window, score with every engine, combine, decide."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from .combiner import Combiner
from .config import DetectorConfig
from .normalize import NormalizationReport, normalize
from .segment import Sentence, Window, count_words, sentence_windows, split_sentences

# Used only when no combiner has been fitted yet: a plain average of the classifiers'
# probabilities, with conservative cut-offs. The app labels these results uncalibrated.
FALLBACK_THRESHOLDS = {"ai": 0.9, "review": 0.6}

VERDICTS = ("insufficient_text", "likely_human", "needs_review", "likely_ai")


class WindowClassifier(Protocol):
    name: str

    def window_logits(self, texts: list[str]) -> np.ndarray: ...


class SpanScorer(Protocol):
    name: str

    def span_scores(self, text: str, spans: list[tuple[int, int]]) -> np.ndarray: ...


@dataclass
class SentenceScore:
    start: int
    end: int
    text: str
    ai_probability: float


@dataclass
class Report:
    text: str  # the normalized text that offsets refer to
    words: int
    verdict: str
    ai_probability: float | None
    calibrated: bool
    thresholds: dict[str, float]
    features: dict[str, float] = field(default_factory=dict)
    engines: dict[str, float] = field(default_factory=dict)
    sentences: list[SentenceScore] = field(default_factory=list)
    normalization: NormalizationReport = field(default_factory=NormalizationReport)

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Detector:
    def __init__(
        self,
        classifiers: list[WindowClassifier],
        binoculars: SpanScorer | None = None,
        combiner: Combiner | None = None,
        config: DetectorConfig | None = None,
    ):
        self.classifiers = list(classifiers)
        self.binoculars = binoculars
        self.combiner = combiner
        self.config = config or DetectorConfig()

    @classmethod
    def from_config(cls, config: DetectorConfig | None = None, load_combiner: bool = True) -> Detector:
        from .engines import build_engines  # heavy imports only when real models are needed

        config = config or DetectorConfig()
        classifiers, binoculars = build_engines(config)
        combiner = None
        if load_combiner and config.combiner_path.exists():
            combiner = Combiner.load(config.combiner_path)
        return cls(classifiers, binoculars, combiner, config)

    @property
    def feature_names(self) -> list[str]:
        names = [f"{c.name}_{stat}" for c in self.classifiers for stat in ("mean", "max")]
        if self.binoculars is not None:
            names += ["bino_doc", "bino_min"]
        return names

    @property
    def missing_features(self) -> list[str]:
        """Features the combiner was fitted on that the loaded engines cannot produce."""
        return [f for f in self.combiner.features if f not in self.feature_names] if self.combiner else []

    # -- scoring --------------------------------------------------------------------

    def _engine_scores(self, text: str, windows: list[Window]) -> tuple[dict[str, np.ndarray], float]:
        """Raw score of every engine on every window, plus Binoculars on the whole text."""
        chunks = [text[w.start : w.end] for w in windows]
        scores = {c.name: np.asarray(c.window_logits(chunks), dtype=float) for c in self.classifiers}
        bino_doc = math.nan
        if self.binoculars is not None:
            spans = [(0, len(text))] + [(w.start, w.end) for w in windows]
            bino = np.asarray(self.binoculars.span_scores(text, spans), dtype=float)
            bino_doc, scores["bino"] = float(bino[0]), bino[1:]
        return scores, bino_doc

    def _doc_features(self, scores: dict[str, np.ndarray], windows: list[Window], bino_doc: float) -> dict[str, float]:
        weights = np.array([w.words for w in windows], dtype=float)
        feats = {}
        for c in self.classifiers:
            logits = scores[c.name]
            feats[f"{c.name}_mean"] = float(np.average(logits, weights=weights))
            feats[f"{c.name}_max"] = float(logits.max())
        if self.binoculars is not None:
            bino = scores["bino"]
            feats["bino_doc"] = bino_doc
            feats["bino_min"] = float(np.nanmin(bino)) if not np.isnan(bino).all() else bino_doc
        return feats

    def _window_features(self, scores: dict[str, np.ndarray], k: int) -> dict[str, float]:
        """Treat one local window as a tiny document."""
        feats = {}
        for c in self.classifiers:
            feats[f"{c.name}_mean"] = feats[f"{c.name}_max"] = float(scores[c.name][k])
        if self.binoculars is not None:
            feats["bino_doc"] = feats["bino_min"] = float(scores["bino"][k])
        return feats

    def _probabilities(self, rows: list[dict[str, float]]) -> np.ndarray:
        if self.combiner is not None:
            return self.combiner.predict_proba(rows)
        if not self.classifiers:
            return np.full(len(rows), np.nan)
        return np.array([np.mean([_sigmoid(r[f"{c.name}_mean"]) for c in self.classifiers]) for r in rows])

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self.combiner.thresholds) if self.combiner else dict(FALLBACK_THRESHOLDS)

    def verdict(self, p: float) -> str:
        t = self.thresholds
        if math.isnan(p):
            return "insufficient_text"
        if p > t["ai"]:
            return "likely_ai"
        if p > t["review"]:
            return "needs_review"
        return "likely_human"

    # -- public API -----------------------------------------------------------------

    def _prepare(self, text: str) -> tuple[str, NormalizationReport, list[Sentence]]:
        clean, report = normalize(text)
        return clean, report, split_sentences(clean)

    def features(self, text: str) -> dict[str, float]:
        """Document-level features only (what the combiner is fitted on)."""
        clean, _, sentences = self._prepare(text)
        if not sentences:
            return {name: math.nan for name in self.feature_names}
        cfg = self.config
        windows = sentence_windows(sentences, cfg.doc_window_words, cfg.doc_stride_words)
        scores, bino_doc = self._engine_scores(clean, windows)
        return self._doc_features(scores, windows, bino_doc)

    def score(self, text: str) -> float:
        return float(self._probabilities([self.features(text)])[0])

    def analyze(self, text: str, sentences: bool = True) -> Report:
        cfg = self.config
        clean, norm, sents = self._prepare(text)
        words = count_words(clean)
        base = dict(text=clean, words=words, calibrated=self.combiner is not None,
                    thresholds=self.thresholds, normalization=norm)
        if words < cfg.min_words or not sents:
            return Report(verdict="insufficient_text", ai_probability=None, **base)

        doc_windows = sentence_windows(sents, cfg.doc_window_words, cfg.doc_stride_words)
        local_windows = sentence_windows(sents, cfg.local_window_words, cfg.local_stride_words) if sentences else []
        scores, bino_doc = self._engine_scores(clean, doc_windows + local_windows)
        nd = len(doc_windows)
        doc_scores = {k: v[:nd] for k, v in scores.items()}
        local_scores = {k: v[nd:] for k, v in scores.items()}

        feats = self._doc_features(doc_scores, doc_windows, bino_doc)
        p = float(self._probabilities([feats])[0])

        engines = {c.name: _sigmoid(feats[f"{c.name}_mean"]) for c in self.classifiers}
        if self.binoculars is not None:
            engines["bino"] = feats["bino_doc"]

        sentence_scores = []
        if local_windows:
            window_p = self._probabilities([self._window_features(local_scores, k) for k in range(len(local_windows))])
            totals, counts = np.zeros(len(sents)), np.zeros(len(sents))
            for w, wp in zip(local_windows, window_p, strict=True):
                totals[w.first : w.last + 1] += wp
                counts[w.first : w.last + 1] += 1
            for s, total, count in zip(sents, totals, counts, strict=True):
                sentence_scores.append(SentenceScore(s.start, s.end, clean[s.start : s.end], float(total / count)))

        return Report(verdict=self.verdict(p), ai_probability=p, features=feats, engines=engines,
                      sentences=sentence_scores, **base)
