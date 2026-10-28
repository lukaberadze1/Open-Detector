"""The Detect page's highlights and charts, built from a hand-made sentence list (no models)."""

import re

from opendetector.detector import SentenceScore
from ui.components import highlighted_text, score_bars, sentence_chart

THRESHOLDS = {"review": 0.35, "ai": 0.82}
TEXT = "One human line here. Two review line here.\n\nThree AI line here. Four AI line too."


def sentences():
    spans, start = [], 0
    for text, p in (("One human line here.", 0.05), ("Two review line here.", 0.5), ("Three AI line here.", 0.9),
                    ("Four AI line too.", 0.99)):
        start = TEXT.index(text, start)
        spans.append(SentenceScore(start, start + len(text), text, p))
        start += len(text)
    return spans


BANDS = ["likely_human", "needs_review", "likely_ai", "likely_ai"]


def test_every_sentence_gets_its_band_and_score():
    page = highlighted_text(TEXT, sentences(), BANDS, THRESHOLDS)
    assert re.findall(r'class="od-hl (\w+)"', page) == ["human", "review", "ai", "ai"]
    assert 'data-tip="99% AI likelihood\nLikely AI · sentence 4"' in page
    assert "<br><br>" in page  # the paragraph break survives
    # Deeper into the AI band shades stronger.
    shades = [float(a) for a in re.findall(r"--a:(\d+)%", page)]
    assert shades[3] > shades[2]
    # The legend gives each band's share of the words: 4 human, 4 review, 8 AI.
    assert "Likely AI <b>50%</b>" in page and "Likely human <b>25%</b>" in page
    # The map merges the two AI sentences into one run that jumps to the first.
    assert len(re.findall(r'class="od-run ', page)) == 3
    assert re.search(r'data-jump="od-\w+-s3"', page)


def test_flagged_only_leaves_human_sentences_plain():
    page = highlighted_text(TEXT, sentences(), BANDS, THRESHOLDS, flagged_only=True)
    assert re.findall(r'class="od-hl (\w+)"', page) == ["review", "ai", "ai"]
    assert 'class="od-run plain"' in page


def test_sentence_text_is_escaped():
    text = "A <b>bold</b> claim & more."
    page = highlighted_text(text, [SentenceScore(0, len(text), text, 0.9)], ["likely_ai"], THRESHOLDS)
    assert "<b>bold</b>" not in page and "&lt;b&gt;bold&lt;/b&gt;" in page


def test_charts_build_and_skip_when_nothing_was_scored():
    assert sentence_chart(sentences(), BANDS, THRESHOLDS).to_dict()
    rows = [{"label": "Slide 1", "probability": 0.9, "verdict": "likely_ai"},
            {"label": "Slide 1", "probability": None, "verdict": None}]
    spec = score_bars(rows, THRESHOLDS).to_dict()
    assert [row["key"] for row in next(iter(spec["datasets"].values()))] == ["Slide 1", "Slide 1 (2)"]
    assert score_bars([{"label": "Slide 1", "probability": None, "verdict": None}], THRESHOLDS) is None
