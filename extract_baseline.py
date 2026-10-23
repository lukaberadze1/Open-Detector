"""Extract a stylometric profile from a Markdown (or plain-text) writing sample.

    python extract_baseline.py sample.md                # writes style_profile.json
    python extract_baseline.py sample.md --out profiles/alice.json

Metrics:
- sentences: length distribution in words (mean, std, variance, percentiles, histogram)
  and burstiness, both as the coefficient of variation (std/mean) and as Goh-Barabási
  burstiness (std - mean)/(std + mean), which runs from -1 (metronomic) to 1 (bursty)
- vocabulary: type-token ratio, plus length-robust variants (moving-average TTR over
  50-word windows, Guiraud's root TTR) and the hapax ratio, since plain TTR falls as a
  text gets longer
- structure: paragraph lengths, punctuation per 1,000 words, transition words, and the
  words sentences most often start with
- markdown: headings, list items, quotes, emphasis, code blocks, tables

Sentences are split with the same splitter the detector uses (opendetector.segment).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from opendetector.normalize import normalize
from opendetector.segment import split_sentences

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
MATTR_WINDOW = 50
LENGTH_BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 40), (41, None)]

TRANSITIONS = [
    "additionally", "also", "as a result", "besides", "but", "consequently", "conversely",
    "finally", "first", "firstly", "for example", "for instance", "furthermore", "hence",
    "however", "importantly", "in addition", "in conclusion", "in contrast", "in fact",
    "in other words", "in summary", "indeed", "instead", "likewise", "meanwhile", "moreover",
    "nevertheless", "nonetheless", "notably", "on the other hand", "overall", "second",
    "secondly", "similarly", "so", "still", "then", "therefore", "thus", "to summarize",
    "ultimately", "yet",
]
_TRANSITION_RES = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in TRANSITIONS}
PUNCTUATION = {
    "comma": ",", "semicolon": ";", "colon": ":", "em_dash": "—", "en_dash": "–",
    "spaced_hyphen": " - ", "parenthesis": "(", "question_mark": "?", "exclamation_mark": "!",
    "ellipsis": "...", "double_quote": '"', "curly_quote": "“",
}


# -- Markdown --------------------------------------------------------------------------

def strip_markdown(markdown: str) -> tuple[str, dict[str, int]]:
    """Plain prose with paragraph breaks kept, plus counts of the Markdown structure removed."""
    counts = Counter()
    code_blocks = re.findall(r"^(```|~~~).*?^\1", markdown, flags=re.MULTILINE | re.DOTALL)
    counts["code_blocks"] = len(code_blocks)
    text = re.sub(r"^(```|~~~).*?^\1[^\n]*", "", markdown, flags=re.MULTILINE | re.DOTALL)
    counts["images"] = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text))
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    counts["links"] = len(re.findall(r"\[([^\]]+)\]\([^)]*\)", text))
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>\n]+>", "", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    counts["bold"] = len(re.findall(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", text))
    text = re.sub(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", r"\2", text)
    counts["italic"] = len(re.findall(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])", text))
    text = re.sub(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])", r"\2", text)

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"([-*_]\s*){3,}", stripped):
            continue  # horizontal rule
        if stripped.startswith("|") or re.fullmatch(r"\|?\s*:?-{3,}.*", stripped):
            counts["table_rows"] += 1
            continue
        if re.match(r"#{1,6}\s", stripped):
            counts["headings"] += 1
            lines.append("")  # a heading ends the paragraph before it
            continue
        if stripped.startswith(">"):
            counts["blockquote_lines"] += 1
            stripped = stripped.lstrip("> ").strip()
        if re.match(r"([-*+]|\d+[.)])\s+", stripped):
            counts["list_items"] += 1
            stripped = re.sub(r"^([-*+]|\d+[.)])\s+", "", stripped)
        lines.append(stripped)
    return "\n".join(lines), dict(counts)


# -- Metrics ---------------------------------------------------------------------------

def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def distribution(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)

    def pct(p: float) -> float:
        return float(ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))])

    return {"count": len(values), "mean": round(mean, 2), "std": round(std, 2),
            "variance": round(std * std, 2), "min": ordered[0], "p10": pct(0.1),
            "median": statistics.median(values), "p90": pct(0.9), "max": ordered[-1]}


def sentence_metrics(lengths: list[int]) -> dict:
    stats = distribution(lengths)
    if not lengths:
        return stats
    mean, std = stats["mean"], stats["std"]
    stats["burstiness"] = {
        "coefficient_of_variation": round(std / mean, 3) if mean else None,
        "goh_barabasi": round((std - mean) / (std + mean), 3) if std + mean else None,
    }
    histogram = {}
    for lo, hi in LENGTH_BUCKETS:
        n = sum(lo <= x and (hi is None or x <= hi) for x in lengths)
        histogram[f"{lo}+" if hi is None else f"{lo}-{hi}"] = round(n / len(lengths), 3)
    stats["length_histogram"] = histogram
    stats["share_short_le_8"] = round(sum(x <= 8 for x in lengths) / len(lengths), 3)
    stats["share_long_ge_30"] = round(sum(x >= 30 for x in lengths) / len(lengths), 3)
    return stats


def vocabulary_metrics(tokens: list[str]) -> dict:
    if not tokens:
        return {"tokens": 0}
    lowered = [t.lower() for t in tokens]
    counts = Counter(lowered)
    types = len(counts)
    if len(lowered) >= MATTR_WINDOW:
        windows = [len(set(lowered[i : i + MATTR_WINDOW])) / MATTR_WINDOW
                   for i in range(len(lowered) - MATTR_WINDOW + 1)]
        mattr = round(statistics.fmean(windows), 4)
    else:
        mattr = None
    return {
        "tokens": len(lowered),
        "types": types,
        "ttr": round(types / len(lowered), 4),
        f"mattr_{MATTR_WINDOW}": mattr,
        "guiraud_root_ttr": round(types / len(lowered) ** 0.5, 3),
        "hapax_ratio": round(sum(c == 1 for c in counts.values()) / types, 4),
        "mean_word_length": round(statistics.fmean(len(t) for t in lowered), 2),
    }


def profile(markdown: str, source: str = "") -> dict:
    prose, markdown_counts = strip_markdown(markdown)
    prose, norm = normalize(prose)
    paragraphs = [p for p in re.split(r"\n\s*\n", prose) if words(p)]
    sentences = [prose[s.start : s.end] for s in split_sentences(prose)]
    sentences = [s for s in sentences if words(s)]
    tokens = words(prose)
    per_1000 = 1000 / len(tokens) if tokens else 0.0

    transitions = {t: len(rx.findall(prose)) for t, rx in _TRANSITION_RES.items()}
    transitions = {t: n for t, n in sorted(transitions.items(), key=lambda kv: -kv[1]) if n}
    openers = Counter(words(s)[0].lower() for s in sentences)
    initial_transitions = Counter()
    for s in sentences:
        head = s.lower().lstrip("\"'“‘(")
        for t in TRANSITIONS:
            if re.match(rf"{re.escape(t)}\b", head):
                initial_transitions[t] += 1
                break

    return {
        "source": source,
        "extracted": date.today().isoformat(),
        "words": len(tokens),
        "sentences": sentence_metrics([len(words(s)) for s in sentences]),
        "vocabulary": vocabulary_metrics(tokens),
        "paragraphs": {
            "words": distribution([len(words(p)) for p in paragraphs]),
            "sentences": distribution([sum(1 for s in split_sentences(p) if words(p[s.start:s.end]))
                                       for p in paragraphs]),
        },
        "punctuation_per_1000_words": {name: round(prose.count(mark) * per_1000, 2)
                                       for name, mark in PUNCTUATION.items()},
        "sentence_endings": {
            "question": round(sum(s.rstrip("\"'”’)").endswith("?") for s in sentences) / max(len(sentences), 1), 3),
            "exclamation": round(sum(s.rstrip("\"'”’)").endswith("!") for s in sentences) / max(len(sentences), 1), 3),
        },
        "transitions": {
            "per_1000_words": round(sum(transitions.values()) * per_1000, 2),
            "counts": transitions,
            "sentence_initial": dict(initial_transitions.most_common()),
            "share_of_sentences_opening_with_transition":
                round(sum(initial_transitions.values()) / max(len(sentences), 1), 3),
        },
        "sentence_openers": dict(openers.most_common(15)),
        "markdown": markdown_counts,
        "hidden_characters_removed": {"invisible": norm.invisible_chars, "homoglyphs": norm.homoglyphs},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", type=Path, help="Markdown or plain-text writing sample")
    parser.add_argument("--out", type=Path, default=Path("style_profile.json"))
    args = parser.parse_args()

    if not args.sample.is_file():
        parser.error(f"{args.sample} is not a file")
    result = profile(args.sample.read_text(encoding="utf-8"), source=str(args.sample))
    if result["words"] == 0:
        print(f"No prose found in {args.sample}", file=sys.stderr)
        return 1
    if result["words"] < 300:
        print(f"Warning: only {result['words']} words; these metrics are noisy below a few hundred.",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    s, v = result["sentences"], result["vocabulary"]
    print(f"{result['words']} words, {s['count']} sentences, {result['paragraphs']['words']['count']} paragraphs")
    print(f"sentence length {s['mean']} ± {s['std']} words "
          f"(burstiness CV {s['burstiness']['coefficient_of_variation']}), "
          f"TTR {v['ttr']}, MATTR-{MATTR_WINDOW} {v[f'mattr_{MATTR_WINDOW}']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
