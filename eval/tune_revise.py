"""Measures revising a whole-document rewrite against the detector, with the page's prompts and settings.

Each repeat writes one rewrite of a document's opening blocks (tune_generation.run_trial), then revises
that same rewrite twice, up to `--rounds` rounds each (redteam_batch.revise_blocks): once choosing blocks by
sentence scores and once by the hottest document window, so the two compare on the same start. Every text
is logged as labelled AI data. The minimum over repeats, as the report's best of three, costs nothing to
compute: any three independent repeats are three tries. Resumable (`--out`). Spends Gemini quota.

    python -m eval.tune_revise data/essays/sources/claude_salamis.md --repeats 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

from eval import generate_essays
from eval.generate_essays import (GEMINI_API_URL, APIError, QuotaExhausted, load_style_profile, style_instructions,
                                  styled_clean)
from eval.redteam_batch import (GROUP_SCHEMA, REVISE, REVISE_KEEPING, WIN_BELOW, document_windows, revise_blocks,
                                style_guard)
from eval.tune_generation import StyleYardstick, _log_labelled, run_trial, sample_jobs, writer_shapes
from opendetector.env import load_env
from opendetector.segment import count_words

ROOT = Path(__file__).resolve().parents[1]
GROUP = 8
METHODS = ("sentences", "windows")
BUSY_WAITS = (60, 120, 300, 300)  # seconds before writing a repeat again after Gemini is overloaded or times out
# Instructions tried on top of the style profile, by name (--note); each is labelled in the attack.
NOTES = {
    "active": "Prefer verbs to abstract nouns: write 'they decided' over 'they made a decision', and use the "
              "active voice wherever the sentence allows it.",
    "asymmetric": "Avoid even, balanced writing: don't give every point the same weight or length, skip the "
                  "introduction-and-summary shape, and let some points get one clause and others several sentences.",
    "punctuation": "Vary your punctuation unevenly: some sentences with no commas at all, some with several, an "
                   "occasional dash or colon, and straight or curly quotes as they come.",
    "parenthetical": "Now and then put a brief aside of your own in brackets, a passing reaction to the point being "
                     "made. Add no facts that aren't in the original.",
    "interjection": "Once or twice a passage, break in with a short rhetorical question or a blunt remark of a few "
                    "words, as someone thinking aloud would. Add no facts that aren't in the original.",
    "idiom": "Use an everyday idiom or turn of phrase where it fits naturally, the kind people say out loud, and "
             "avoid stock literary ones.",
    "digression": "Let a sentence wander a little now and then: a small qualifier, a self-correction such as 'or "
                  "rather' or 'at least in most cases', or a minor side remark. Add no facts that aren't in the "
                  "original.",
}


def _summary(rows: list[dict]) -> str:
    starts = [row["start"] for row in rows]
    lines = [f"{len(rows)} repeats, bar {WIN_BELOW:.0%}.",
             f"written: mean {statistics.mean(starts):.3f}, range {min(starts):.3f}-{max(starts):.3f}, "
             f"under the bar {sum(p < WIN_BELOW for p in starts)}"]
    for method in METHODS:
        ends = [row[method]["history"][-1] for row in rows if method in row]
        if ends:
            length = statistics.mean(row[method]["words"] / row["words"] for row in rows if method in row)
            lines.append(f"revised by {method}: mean {statistics.mean(ends):.3f}, "
                         f"range {min(ends):.3f}-{max(ends):.3f}, "
                         f"under the bar {sum(p < WIN_BELOW for p in ends)} of {len(ends)}, length x{length:.2f}")
            if len(ends) >= 3:
                best = [min(trio) for trio in combinations(ends, 3)]
                lines.append(f"  best of three, every trio: mean {statistics.mean(best):.3f}, "
                             f"under the bar {sum(p < WIN_BELOW for p in best)} of {len(best)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("document", type=Path)
    parser.add_argument("--style", type=Path, default=ROOT / "data/styles/alexandria_profile.json")
    parser.add_argument("--register", default="finalized")
    parser.add_argument("--structure", default="writer")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--words", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--methods", default=",".join(METHODS), help="how to choose blocks to revise")
    parser.add_argument("--note", choices=list(NOTES), help="an instruction added to the style profile")
    parser.add_argument("--as-page", action="store_true",
                        help="revise as the page does: keep names and numbers, and stay close to the style profile")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--out", type=Path, default=ROOT / "eval/cache/tune_revise.jsonl")
    args = parser.parse_args(argv)

    load_env()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"${args.api_key_env} is not set")
    data = args.document.read_bytes()
    source = hashlib.sha256(data).hexdigest()[:16]
    jobs, plan = sample_jobs(args.document.name, data, args.words, min_blocks=2 * GROUP, structure=args.structure)
    shapes = writer_shapes(args.style, args.register)
    plan.shapes = shapes
    plan.clean = styled_clean(args.style, args.register)
    system = style_instructions(load_style_profile(args.style), args.register)
    style = f"{args.style.stem}:{args.register}+{args.structure}"
    if args.note:
        system += f"\n\n{NOTES[args.note]}"
        style += f"+{args.note}"
    style += f"+t{args.temperature:g}" + ("+as-page" if args.as_page else "")
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    rows = [json.loads(line) for line in args.out.open(encoding="utf-8")] if args.out.exists() else []
    rows = [row for row in rows if row["source"] == source and row["style"] == style and row["model"] == args.model]
    done = {row["repeat"] for row in rows}

    def ask(prompt: str, pending: list[dict]) -> str:
        return generate_essays.chat(GEMINI_API_URL, api_key, args.model, prompt, args.temperature, retries=2,
                                    system=system, thinking=args.thinking, json_schema=GROUP_SCHEMA,
                                    timeout=60 + 15 * len(pending))

    def write_repeat(trial: dict) -> tuple[dict, list[tuple[dict, str]]]:
        """One repeat's scores, and its texts to log, kept back until it is whole so a retry logs nothing twice."""
        result = run_trial(trial, jobs, plan, system, api_key, args.temperature)
        logs = [(result, style)]
        parts = {f"{number:03d}": part for number, part in enumerate(result["text"].split("\n\n"), 1)}
        row = {"source": source, "style": style, "model": args.model, "repeat": trial["repeat"],
               "start": detector.analyze(result["text"], sentences=False).ai_probability,
               "words": count_words(result["text"]), "requests": result["requests"],
               "copied": result["copied"], "words_kept": result["words_kept"]}
        for method in methods:
            windows = document_windows(detector) if method == "windows" else None
            fits = None
            if args.as_page and (yardstick := StyleYardstick.build(args.style, row["words"], args.register)):
                fits = style_guard(yardstick.text_distance, "\n\n".join(parts.values()))
            changed, history = revise_blocks(parts, detector.analyze, ask, plan.clean, shapes, args.rounds,
                                             windows=windows, fits=fits,
                                             prompt=REVISE_KEEPING if args.as_page else REVISE)
            text = "\n\n".join(changed.get(key, part) for key, part in parts.items())
            row[method] = {"history": history, "changed": len(changed), "words": count_words(text)}
            logs.append(({**result, "key": f"{trial['key']}-{method}", "text": text},
                         f"{style}+revise{args.rounds}-{method}"))
        return row, logs

    for repeat in range(1, args.repeats + 1):
        if repeat in done:
            continue
        variant = (args.note or "base") + ("-as-page" if args.as_page else "")
        trial = {"key": f"revise-{variant}-{args.model}-{args.thinking}-{GROUP}-{args.temperature:g}-{repeat}",
                 "model": args.model, "thinking": args.thinking, "group_size": GROUP,
                 "temperature": args.temperature, "repeat": repeat}
        waits = iter(BUSY_WAITS)
        while True:
            try:
                row, logs = write_repeat(trial)
                break
            except QuotaExhausted as exc:
                print(f"Stopped at repeat {repeat}: {exc}", file=sys.stderr)
                row = None
                break
            except APIError as exc:
                wait = next(waits, None)
                if wait is None:
                    print(f"Stopped at repeat {repeat}: {exc}", file=sys.stderr)
                    row = None
                    break
                print(f"Repeat {repeat} failed, writing it again in {wait} s: {str(exc)[:80]}", flush=True)
                time.sleep(wait)
        if row is None:
            break
        for logged, label in logs:
            _log_labelled(logged, args.document.name, source, label)
        rows.append(row)
        with args.out.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")
        print(f"repeat {repeat}: start {row['start']:.3f}, " + ", ".join(
            f"{method} {' > '.join(f'{p:.3f}' for p in row[method]['history'])} ({row[method]['changed']} blocks)"
            for method in methods), flush=True)
    if rows:
        print(_summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
