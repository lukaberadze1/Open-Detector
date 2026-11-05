"""Takes the slim detector apart: which level of a text Desklib reads, which words really move it, and what raw score
the combiner needs. Each change is applied to a whole document window and scored on AI and human MAGE texts, so the
gap it closes shows whether the signal lives in words, order, punctuation or length. No API. Resumes from --out.

    python -m eval.detector_anatomy --texts 30
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from eval.bench_math import LOGITS, POOL, bench_texts
from eval.math_attack import CachedScorer
from opendetector.normalize import normalize
from opendetector.segment import split_sentences

ROOT = Path(__file__).resolve().parents[1]
_FUNCTION_TEXT = """
a about also am an and are as at be been being but by can could did do does for from had has have he
her here him his how i if in into is it its just may me might must my no not of on onto or our over
she should so than that the their them then there these they this those to under us very was we were
what when where which who whom whose why will with without would you your
"""
FUNCTION_WORDS = frozenset(_FUNCTION_TEXT.split())
MASKED_WORDS = 40  # words masked per AI text: each costs one model run
WORD = re.compile(r"[A-Za-z']+")


def _sentences(text: str) -> list[str]:
    return [text[s.start:s.end] for s in split_sentences(text)]


def _masked(text: str, keep_function: bool) -> str:
    """Every word of one class becomes 'thing', keeping the other class and all punctuation."""
    def swap(match: re.Match) -> str:
        return match[0] if (match[0].lower() in FUNCTION_WORDS) == keep_function else "thing"
    return WORD.sub(swap, text)


def _shuffle_inside(text: str, rng: random.Random) -> str:
    out = []
    for sentence in _sentences(text):
        words = sentence.split()
        rng.shuffle(words)
        out.append(" ".join(words))
    return " ".join(out)


def _shuffled(items: list[str], rng: random.Random) -> list[str]:
    items = list(items)
    rng.shuffle(items)
    return items


def _first_words(text: str, count: int) -> str:
    return " ".join(text.split()[:count])


CHANGES = {
    "unchanged": lambda t, r: t,
    "lowercase": lambda t, r: t.lower(),
    "no_punctuation": lambda t, r: re.sub(r"[^\w\s']", "", t),
    "one_paragraph": lambda t, r: " ".join(t.split()),
    "sentences_shuffled": lambda t, r: " ".join(_shuffled(_sentences(t), r)),
    "words_shuffled_in_sentence": _shuffle_inside,
    "all_words_shuffled": lambda t, r: " ".join(_shuffled(t.split(), r)),
    "content_words_masked": lambda t, r: _masked(t, True),
    "function_words_masked": lambda t, r: _masked(t, False),
    "first_50_words": lambda t, r: _first_words(t, 50),
    "first_100_words": lambda t, r: _first_words(t, 100),
    "first_200_words": lambda t, r: _first_words(t, 200),
}


def required_raw_score(combiner, target: float, gap: float = 0.0) -> float:
    """The window mean a text needs for `target` likelihood when its hottest window sits `gap` above the mean."""
    from math import log

    mean_i, max_i = combiner.features.index("desklib_mean"), combiner.features.index("desklib_max")
    a, b = combiner.coef[mean_i] / combiner.scale[mean_i], combiner.coef[max_i] / combiner.scale[max_i]
    offset = combiner.intercept - a * combiner.mean[mean_i] - b * combiner.mean[max_i]
    return (log(target / (1 - target)) - offset - b * gap) / (a + b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=int, default=30, help="AI texts, and as many human ones")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/cache/anatomy.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "eval/reports/anatomy.md")
    args = parser.parse_args()
    from eval.datasets import load_mage
    from opendetector import Detector, DetectorConfig
    from opendetector.segment import count_words

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    engine = next(c for c in detector.classifiers if hasattr(c, "embedded_log_odds"))
    frame = load_mage()
    humans = frame[(frame.label == 0) & (frame.text.map(count_words) >= 150)].sample(args.texts, random_state=7).text
    # Half saturated, half borderline: the math script's edits only ever moved borderline text, so that is where what
    # the detector reads matters most.
    scorer = CachedScorer(detector, path=LOGITS)
    pool = bench_texts(POOL)
    scores = {text: scorer(text) for text in pool}
    borderline = [t for t in pool if 0.6 <= scores[t] <= 0.95][:args.texts // 2]
    saturated = [t for t in pool if scores[t] > 0.95][:args.texts - len(borderline)]
    texts = ([("ai", t) for t in saturated] + [("ai_borderline", t) for t in borderline]
             + [("human", t) for t in humans])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.out.open()] if args.out.exists() else []
    done = {(r["label"], r["number"]) for r in rows}
    with args.out.open("a", encoding="utf-8") as sink:
        for number, (label, text) in enumerate(texts):
            if (label, number) in done:
                continue
            chunk = " ".join(normalize(text)[0].split(" ")[:detector.config.doc_window_words])
            rng = random.Random(number)
            changed = {name: change(chunk, rng) for name, change in CHANGES.items()}
            logits = engine.window_logits(list(changed.values()))
            row = {"label": label, "number": number, "logits": dict(zip(changed, map(float, logits), strict=True))}
            if label != "human":  # real masking: gradient times input tracked it no better than chance
                words = list(WORD.finditer(chunk))
                chosen = random.Random(number).sample(words, min(MASKED_WORDS, len(words)))
                masked = engine.window_logits([f"{chunk[:m.start()]}[MASK]{chunk[m.end():]}" for m in chosen])
                row["pushes"] = [[round(row["logits"]["unchanged"] - float(score), 4), m[0].lower(),
                                  round(m.start() / len(chunk), 3)] for m, score in zip(chosen, masked, strict=True)]
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            rows.append(row)
            print(f"{label} {number + 1}/{len(texts)}", flush=True)
    args.report.write_text(report(rows, detector.combiner), encoding="utf-8")
    print(args.report)


def report(rows: list[dict], combiner) -> str:
    def typical(label: str, name: str) -> str:
        values = [r["logits"][name] for r in rows if r["label"] == label]
        return f"{median(values):.2f}" if values else "-"

    lines = ["# What the slim detector reads", "", f"{len(rows)} MAGE texts, first document window. Written by "
             "`python -m eval.detector_anatomy`. Scores are Desklib's raw log-odds, medians. A change that closes the "
             "gap between AI and human removes what the detector reads.", "",
             "| Change | AI, saturated | AI, borderline | Human |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {typical('ai', name)} | {typical('ai_borderline', name)} | {typical('human', name)} |"
              for name in CHANGES]
    lines += ["", "## What masking one word does (AI texts)", ""]
    pushes = [p for r in rows if r["label"] != "human" for p in r["pushes"]]
    positive = [p for p in pushes if p[0] > 0]
    total = sum(p[0] for p in positive) or 1
    function = sum(p[0] for p in positive if p[1] in FUNCTION_WORDS)
    top = sorted((p[0] for p in positive), reverse=True)
    lines += [f"- Masking a word lowers the score {len(positive) / max(1, len(pushes)):.0%} of the time, by "
              f"{sum(p[0] for p in positive) / max(1, len(positive)):.2f} on average",
              f"- Function words' share of the push toward AI: {function / total:.0%}",
              f"- Top 5% of words' share: {sum(top[:max(1, len(top) // 20)]) / total:.0%}", "",
              "| Part of the window | Share of the push |", "|---|---:|"]
    parts = defaultdict(float)
    for p in positive:
        parts[min(int(p[2] * 5), 4)] += p[0]
    lines += [f"| {part * 20}-{part * 20 + 20}% | {parts[part] / total:.0%} |" for part in range(5)]
    seen = defaultdict(list)
    for p in pushes:
        seen[p[1]].append(p[0])
    words = {word: sum(values) / len(values) for word, values in seen.items() if len(values) >= 3}
    lines += ["", "Words seen 3+ times whose masking lowered the score: "
              + ", ".join(w for w, _ in sorted(words.items(), key=lambda i: -i[1])[:40]), "",
              "## The raw score the combiner needs", "",
              "| Target | All windows equal | Hottest window 2 above the mean | 4 above |", "|---|---:|---:|---:|"]
    thresholds = combiner.thresholds
    for name, target in (("Leave Likely AI", thresholds["ai"]), ("Likely human", thresholds["review"])):
        cells = " | ".join(f"{required_raw_score(combiner, target, gap):.2f}" for gap in (0, 2, 4))
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
