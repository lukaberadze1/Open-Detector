"""What the local translation hop does to the detector: the math script's fixed AI texts, through each language and back
on this computer, scored by the slim detector before and after. No API and no quota. It answers two questions: does the
hop move saturated text at all, and does it bring text into the range where the math script works (`--math`)?

Words kept and names kept say what it cost; read some of the saved texts too (`--out`), since neither measures meaning.
Every text saved is labelled AI. Resumes from `--out`.

    python -m eval.bench_hop --texts 12 --out eval/cache/hop.jsonl
    python -m eval.bench_hop --texts 12 --pivots German --math --independent --out eval/cache/hop.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from eval.bench_math import LOGITS, bench_texts, independent_scores
from eval.math_attack import CachedScorer, guided_attack
from eval.redteam_batch import rewrite_measures
from eval.text_attacks import LOCAL_PIVOTS, local_attack_name, local_round_trip
from eval.tune_revise import facts_kept

MATH_BAND = (0.15, 0.95)  # where the math script has been seen to move text


def saved_rows(path: Path | None) -> dict[tuple[str, str], dict]:
    rows = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:  # a half-written last line, as a crash can leave
                continue
            rows[row["source_text"], row["attack"]] = row
    return rows


def hop_row(source: str, hopped: str, attack: str, before: float, after: float) -> dict:
    return {"text": hopped, "label": 1, "model": "local-mt", "attack": attack, "source_text": source,
            "score_before": before, "score_after": after, "facts": facts_kept(source, hopped),
            **rewrite_measures(source, hopped)}


def summary(rows: list[dict], threshold: float) -> str:
    before = [row["score_before"] for row in rows]
    after = [row["score_after"] for row in rows]
    facts = [row["facts"] for row in rows if row["facts"] is not None]
    left = sum(a <= threshold < b for a, b in zip(after, before, strict=True))
    kept = statistics.mean(row["words_kept"] for row in rows)
    line = (f"{rows[0]['attack']}: {len(rows)} texts, mean {statistics.mean(before):.3f} -> "
            f"{statistics.mean(after):.3f}, median {statistics.median(after):.3f}, left Likely AI {left}, "
            f"under 0.5 {sum(a < 0.5 for a in after)}, words kept {kept:.0f}%")
    if facts:
        line += f", names and numbers kept {statistics.mean(facts):.0f}%"
    if math_rows := [row for row in rows if "score_math" in row]:
        line += (f"; math on {len(math_rows)} in range: mean "
                 f"{statistics.mean(row['score_after'] for row in math_rows):.3f} -> "
                 f"{statistics.mean(row['score_math'] for row in math_rows):.3f}")
    if held := [row for row in rows if "independent_after" in row]:
        line += (f"; independent detector {statistics.mean(row['independent_before'] for row in held):.3f} -> "
                 f"{statistics.mean(row['independent_after'] for row in held):.3f}")
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--texts", type=int, default=12)
    parser.add_argument("--pivots", nargs="+", default=["German", "Finnish"], choices=LOCAL_PIVOTS)
    parser.add_argument("--math", action="store_true", help="then run the math script on text left in its range")
    parser.add_argument("--budget", type=int, default=150, help="detector passes per text with --math")
    parser.add_argument("--independent", action="store_true", help="also score with a detector the hop never saw")
    parser.add_argument("--out", type=Path, help="save each text before and after, labelled AI, as .jsonl")
    parser.add_argument("--report", type=Path, help="also write the summary lines here")
    args = parser.parse_args()
    from opendetector import Detector, DetectorConfig

    detector = Detector.from_config(DetectorConfig.for_profile("slim"))
    scorer = CachedScorer(detector, path=LOGITS)
    texts = bench_texts(args.texts)
    saved = saved_rows(args.out)
    lines = []
    for pivot in args.pivots:
        attack = local_attack_name(pivot)
        waiting = [text for text in texts if (text, attack) not in saved]
        rows = [saved[text, attack] for text in texts if (text, attack) in saved]
        began = time.time()
        # All of a language's texts in one pass, so each model loads once. Paragraphs stay in their texts.
        pieces = [text.split("\n\n") for text in waiting]
        flat, _ = local_round_trip([piece for parts in pieces for piece in parts], pivot) if waiting else ([], 0)
        print(f"{pivot}: {len(waiting)} texts there and back in {time.time() - began:.0f}s", flush=True)
        at = 0
        for text, parts in zip(waiting, pieces, strict=True):
            hopped = "\n\n".join(flat[at:at + len(parts)])
            at += len(parts)
            started = time.time()
            row = hop_row(text, hopped, attack, scorer(text), scorer(hopped))
            if args.math and MATH_BAND[0] <= row["score_after"] <= MATH_BAND[1]:
                edited, _, history = guided_attack(hopped, scorer, budget=args.budget)
                row |= {"text": edited, "hopped": hopped, "score_math": history[-1]}
            if args.independent:
                scores = independent_scores([text, row["text"]])
                row |= {"independent_before": scores[0], "independent_after": scores[1]}
            rows.append(row)
            then = f" -> {row['score_math']:.3f}" if "score_math" in row else ""
            print(f"  {row['score_before']:.3f} -> {row['score_after']:.3f}{then}  words kept "
                  f"{row['words_kept']:.0f}%  {time.time() - started:.0f}s", flush=True)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        lines.append(summary(rows, detector.thresholds["ai"]))
    print("\n".join(lines))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("".join(f"- {line}\n" for line in lines), encoding="utf-8")


if __name__ == "__main__":
    main()
