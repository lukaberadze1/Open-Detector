"""Measure the built-in human style profile, styles/human_default_profile.json.

    python -m eval.build_human_profile

The default profile is no one writer's: it describes how people write, measured from MAGE's human texts
(Li et al., 2024), which the Hugging Face cache already holds for the evaluation. Its finished-work
register comes from news, science abstracts, encyclopedia articles and short stories; its quick-notes
register from informal writing people type for others online: argument posts, explanations and reviews.
Each source counts equally (SAMPLE texts of 150 words or more).

It writes numbers only, never the texts, into styles/human_default/:
- measurements.json: each register measured as a whole (update_style_profile.summarize), and how far
  writers differ (the 25th, 50th and 75th percentiles of each text's own measurements), for the
  profile's prose, which is written from them like any other profile's.
- shapes.json: each text's sentence shapes (lengths and punctuation) for "Writer's sentences".
- yardstick.json: what the style match ranks a sample against, for passages of 50 to 600 words.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "styles/human_default"
PROFILE = ROOT / "styles/human_default_profile.json"
SOURCES = {"finalized": ("xsum", "sci_gen", "squad", "wp"),  # news, science abstracts, Wikipedia, stories
           "notes": ("cmv", "eli5", "yelp")}  # argument posts, explanations, reviews
SAMPLE = 250
MIN_WORDS = 150
LONGEST = 120
SIZES = range(50, 601, 50)
SPREAD = (("mean sentence length", ("sentence_length", "mean")),
          ("coefficient of variation", ("sentence_length", "coefficient_of_variation")),
          ("sentences under 6 words %", ("sentence_length", "distribution_pct", "under_6")),
          ("sentences 30+ words %", ("sentence_length", "distribution_pct", "30_plus")),
          ("commas per sentence", ("commas_per_sentence",)), ("MATTR-100", ("vocabulary", "mattr_100")),
          ("mean word length", ("vocabulary", "mean_word_length")),
          ("unknown words %", ("spelling", "unknown_lowercase_word_pct")),
          ("contractions without apostrophe %", ("mechanics", "contractions_without_apostrophe_pct")),
          *((f"{name} per 1000", ("per_1000_words", name)) for name in (
              "em_dash", "en_dash", "parenthesis", "semicolon", "colon", "question_mark", "exclamation_mark",
              "however_initial", "pronoun_I", "we_us_our", "you_your", "contraction_with_apostrophe")))


def human_texts(split: str = "test", seed: int = 0) -> dict[str, list[str]]:
    """SAMPLE human texts of MIN_WORDS words or more from each source, by register."""
    from datasets import load_dataset

    from opendetector.segment import split_sentences

    df = load_dataset("yaful/MAGE", split=split).to_pandas()
    rng = random.Random(seed)
    texts = {}
    for register, sources in SOURCES.items():
        texts[register] = []
        for source in sources:
            # A "sentence" past LONGEST words is a text with its punctuation lost (lists, formulas), not prose.
            pool = [text for text in df.loc[df["src"] == f"{source}_human", "text"]
                    if len(text.split()) >= MIN_WORDS
                    and max((s.words for s in split_sentences(text)), default=0) <= LONGEST]
            texts[register] += rng.sample(pool, min(SAMPLE, len(pool)))
    return texts


def pieces(texts: list[str], words: int) -> list[str]:
    """The texts cut at sentence ends into runs of about `words` words, so each size of yardstick
    compares passages of that size (MAGE keeps no paragraph breaks)."""
    from opendetector.segment import split_sentences

    out = []
    for text in texts:
        start, count = 0, 0
        for sentence in split_sentences(text):
            count += sentence.words
            if count >= words:
                out.append(text[start:sentence.end].strip())
                start, count = sentence.end, 0
    return out


def _value(summary: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        summary = summary.get(key) if isinstance(summary, dict) else None
    return float(summary) if isinstance(summary, int | float) else None


def spread(summaries: list[dict]) -> dict[str, list[float]]:
    """The 25th, 50th and 75th percentiles across writers of each measure in SPREAD."""
    out = {}
    for label, path in SPREAD:
        values = [value for summary in summaries if (value := _value(summary, path)) is not None]
        if len(values) >= 4:
            out[label] = [round(value, 2) for value in statistics.quantiles(values, n=4)]
    return out


def main(argv: list[str] | None = None) -> int:
    import update_style_profile as usp
    from eval.redteam_batch import paragraph_shapes
    from eval.tune_generation import StyleYardstick

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    banned = usp.banned_patterns(json.loads(PROFILE.read_text(encoding="utf-8"))) if PROFILE.exists() else {}
    texts = human_texts(args.split)
    OUT.mkdir(parents=True, exist_ok=True)
    measurements, shapes, yardsticks = {}, {}, {}
    for register, register_texts in texts.items():
        each = [summary for text in register_texts if (summary := usp.summarize(usp.measure([text], banned)))]
        whole = usp.summarize(usp.measure(register_texts, banned))
        measurements[register] = {"sources": list(SOURCES[register]), "texts": len(register_texts),
                                  "whole": whole, "across_writers": spread(each)}
        shapes[register] = paragraph_shapes(register_texts)
        yardsticks[register] = {}
        for words in SIZES:
            if yardstick := StyleYardstick.from_paragraphs(pieces(register_texts, words), words, banned, whole):
                yardsticks[register][str(words)] = {"spread": yardstick.spread, "passages": yardstick.passages}
        yardsticks[register]["target"] = whole
        print(f"{register}: {len(register_texts)} texts, {whole['words']:,} words, mean sentence "
              f"{whole['sentence_length']['mean']} words")
    (OUT / "measurements.json").write_text(json.dumps(measurements, indent=2, ensure_ascii=False) + "\n")
    (OUT / "shapes.json").write_text(json.dumps(shapes, separators=(",", ":")) + "\n")
    (OUT / "yardstick.json").write_text(json.dumps(yardsticks, separators=(",", ":")) + "\n")
    print(f"Wrote {OUT}/measurements.json, shapes.json and yardstick.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
