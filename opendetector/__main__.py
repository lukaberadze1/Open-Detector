"""Command line: python -m opendetector essay.txt  (or pipe text on stdin).

Markdown, Word and PowerPoint files are read like uploads in the app: the whole document gets a
verdict, and every section long enough for one is also scored on its own."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import Detector, DetectorConfig
from .documents import extract_answer_slots, extract_document
from .segment import count_words

DOCUMENT_TYPES = (".md", ".docx", ".pptx")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a text or document for AI generation.")
    parser.add_argument("path", nargs="?", help="text, .md, .docx or .pptx file to score; reads stdin if omitted")
    parser.add_argument("--no-sentences", action="store_true", help="skip sentence-level scores")
    parser.add_argument("--profile", choices=("full", "slim"), default="full",
                        help="full: every engine (about 3.5 GB); slim: Desklib alone on the CPU")
    parser.add_argument("--answers-only", action="store_true",
                        help="score only the answer cells of a worksheet's question/answer tables, "
                             "leaving out the template's own prompts and instructions")
    parser.add_argument("--device", help="override the profile's device: auto, cpu, mps or cuda")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else None
    sections = None
    if path and path.suffix.lower() in DOCUMENT_TYPES:
        data = path.read_bytes()
        try:
            if args.answers_only:
                slots = extract_answer_slots(path.name, data)
                sections = [slot.target for slot in slots if slot.previous_answer]
            else:
                sections = extract_document(path.name, data)
        except ValueError as exc:
            parser.error(str(exc))
        if not sections:
            parser.error(f"found no {'filled-in answer cells' if args.answers_only else 'text'} in {path.name}")
        text = "\n\n".join(section.text for section in sections)
    elif args.answers_only:
        parser.error("--answers-only needs a .md, .docx or .pptx file")
    else:
        text = path.read_text(encoding="utf-8") if path else sys.stdin.read()

    config = DetectorConfig.for_profile(args.profile)
    if args.device:
        config = dataclasses.replace(config, device=args.device)
    detector = Detector.from_config(config)
    report = detector.analyze(text, sentences=not args.no_sentences).to_dict()
    report.pop("text")
    if sections is not None:
        report["sections_extracted"] = len(sections)
        report["sections"] = []
        for section in sections:
            words = count_words(section.text)
            if words >= detector.config.min_words:
                result = detector.analyze(section.text, sentences=False)
                report["sections"].append({"label": section.label, "words": words, "verdict": result.verdict,
                                           "ai_probability": result.ai_probability})
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
