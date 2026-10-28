"""Print a short reading of `python -m opendetector` JSON reports for the /ai-detect skill.

Usage: summarize.py REPORT.json [REPORT.json ...]"""

import json
import sys
from pathlib import Path

VERDICTS = {
    "likely_ai": "Likely AI",
    "needs_review": "Needs review",
    "likely_human": "Likely human",
    "insufficient_text": "No verdict: under the minimum word count",
    "not_prose": "No verdict: not prose (lists, tables of short cells, poetry)",
}
TOP_SENTENCES = 8
MIN_SENTENCE_WORDS = 6


def show(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"== {path.name}")
    line = f"Verdict: {VERDICTS.get(report['verdict'], report['verdict'])} · {report['words']} words"
    if report["ai_probability"] is not None:
        line += f" · AI likelihood {report['ai_probability']:.0%}"
        line += "" if report["calibrated"] else " (uncalibrated)"
    print(line)
    if report["ai_probability"] is not None:
        cuts = report["thresholds"]
        print(f"Cut-offs: needs review above {cuts['review']:.0%}, likely AI above {cuts['ai']:.0%}")
        engines = [f"{name} {value:.2f}" if name == "bino" else f"{name} {value:.0%}"
                   for name, value in report["engines"].items()]
        print(f"Engines: {', '.join(engines)}  (Binoculars: lower is more AI-like)")
    norm = report["normalization"]
    if norm["invisible_chars"] or norm["homoglyphs"]:
        print(f"Removed before scoring: {norm['invisible_chars']} hidden characters, "
              f"{norm['homoglyphs']} look-alike letters")
    if "sections" in report:
        print(f"Sections: {report['sections_extracted']} extracted, "
              f"{len(report['sections'])} long enough to score alone")
        for section in report["sections"]:
            p = section["ai_probability"]
            score = f"{p:.0%}" if p is not None else "no score"
            print(f"  {section['label']} ({section['words']} words): "
                  f"{VERDICTS.get(section['verdict'], section['verdict'])}, {score}")
    # Repeated cells (a calendar of "Notion entry: ...") and short fragments without a full stop
    # dominate the highlights in worksheets while saying nothing about the prose, so list each text
    # once and keep only real sentences.
    best: dict[str, float] = {}
    for sentence in report.get("sentences", []):
        text = " ".join(sentence["text"].split())
        if len(text.split()) >= MIN_SENTENCE_WORDS and text[-1:] in ".!?\"'\u201d\u2019":
            best[text] = max(best.get(text, 0.0), sentence["ai_probability"])
    if best:
        print("Most AI-like sentences (highlights are not calibrated):")
        for text, p in sorted(best.items(), key=lambda item: item[1], reverse=True)[:TOP_SENTENCES]:
            print(f"  {p:.0%}  {text[:160]}{'…' if len(text) > 160 else ''}")
    print()


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.exists() and path.stat().st_size:
            show(path)
        else:
            print(f"== {path.name}: no report (the run failed or was skipped)\n")
