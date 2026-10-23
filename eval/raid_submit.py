"""Score the hidden RAID test set and write predictions.json for the public leaderboard.

    python -m eval.raid_submit --out leaderboard/predictions.json

RAID-test has ~672k texts: plan on a GPU machine, not a laptop. The run resumes from
<out>.partial.jsonl if interrupted. Then follow github.com/liamdugan/raid#leaderboard to
open a pull request. If your combiner was fitted on RAID-train, declare that in the
submission metadata: the leaderboard lists such detectors separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from opendetector import Detector, DetectorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="leaderboard/predictions.json")
    parser.add_argument("--limit", type=int, help="score only the first N texts (smoke tests)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    detector = Detector.from_config(DetectorConfig(device=args.device))
    if detector.combiner is None:
        raise SystemExit("Fit a combiner first (python -m eval.fit_combiner ...).")

    from datasets import load_dataset

    test = load_dataset("liamdugan/raid", "raid_test", split="test", streaming=True)
    out = Path(args.out)
    partial = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in partial.read_text().splitlines()} if partial.exists() else set()

    with partial.open("a") as f:
        for i, row in enumerate(tqdm(test, total=args.limit)):
            if args.limit and i >= args.limit:
                break
            if row["id"] in done:
                continue
            f.write(json.dumps({"id": row["id"], "score": detector.score(row["generation"])}) + "\n")
            f.flush()

    predictions = [json.loads(line) for line in partial.read_text().splitlines()]
    out.write_text(json.dumps(predictions))
    print(f"Wrote {len(predictions)} predictions to {out}")


if __name__ == "__main__":
    main()
