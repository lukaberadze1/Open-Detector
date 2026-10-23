"""Command line: python -m opendetector essay.txt  (or pipe text on stdin)."""

from __future__ import annotations

import argparse
import json
import sys

from . import Detector, DetectorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a text for AI generation.")
    parser.add_argument("path", nargs="?", help="text file to score; reads stdin if omitted")
    parser.add_argument("--no-sentences", action="store_true", help="skip sentence-level scores")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    text = open(args.path, encoding="utf-8").read() if args.path else sys.stdin.read()
    detector = Detector.from_config(DetectorConfig(device=args.device))
    report = detector.analyze(text, sentences=not args.no_sentences).to_dict()
    report.pop("text")
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
