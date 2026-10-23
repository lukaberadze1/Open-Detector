"""Run every engine over a labelled dataset and cache document features as JSONL.

    python -m eval.featurize raid --out eval/cache/raid.jsonl
    python -m eval.featurize mage --out eval/cache/mage.jsonl --n 2000
    python -m eval.featurize persuade --path persuade_2.0.csv --out eval/cache/persuade.jsonl --n 1000
    python -m eval.featurize local --path data/essays/mirrors.jsonl --out eval/cache/mirrors.jsonl

Re-running with the same --out resumes where it stopped. Texts aren't stored in the cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from opendetector import Detector, DetectorConfig
from opendetector.segment import count_words

from . import datasets


def load(args: argparse.Namespace):
    if args.dataset == "raid":
        return datasets.load_raid(split=args.split or "train", ai_per_bucket=args.ai_per_bucket,
                                  human_per_bucket=args.human_per_bucket, chunks=args.raid_chunks,
                                  csv_path=args.path, seed=args.seed)
    if args.dataset == "mage":
        return datasets.load_mage(split=args.split or "test", n=args.n or 2000, seed=args.seed)
    if args.dataset == "persuade":
        return datasets.load_persuade(args.path, n=args.n, seed=args.seed)
    return datasets.load_local(args.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["raid", "mage", "persuade", "local"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--path", help="CSV/JSONL for persuade/local, or a raid-bench CSV for raid")
    parser.add_argument("--split")
    parser.add_argument("--n", type=int, help="sample size (mage, persuade)")
    parser.add_argument("--ai-per-bucket", type=int, default=3)
    parser.add_argument("--human-per-bucket", type=int, default=30)
    parser.add_argument("--raid-chunks", type=int, default=300, help="random 1 MB slices of RAID to read")
    parser.add_argument("--limit", type=int, help="stop after this many texts (smoke tests)")
    parser.add_argument("--min-words", type=int, default=DetectorConfig().min_words,
                        help="skip shorter texts; the detector abstains on them anyway")
    parser.add_argument("--engines", default="glyph,desklib,bino",
                        help="comma-separated engines to run (e.g. desklib for the slim profile)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="auto, float32 or bfloat16; match the deployment's")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = sorted(set(engines) - {"glyph", "desklib", "bino"})
    if unknown:
        parser.error(f"unknown engines: {', '.join(unknown)} (choose from glyph, desklib, bino)")

    df = load(args)
    long_enough = df["text"].map(lambda t: count_words(t) if isinstance(t, str) else 0) >= args.min_words
    print(f"Skipping {int((~long_enough).sum())} texts under {args.min_words} words")
    df = df[long_enough]
    if args.limit:
        df = df.head(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    todo = df[~df["id"].isin(done)]
    print(f"{len(df)} texts ({int(df.label.sum())} AI), {len(done)} already cached, {len(todo)} to score")
    if todo.empty:
        return

    from opendetector.engines import ENGINE_NAMES, build_engine, free_memory

    config = DetectorConfig(device=args.device, dtype=args.dtype)
    texts = {row.id: row.text if isinstance(row.text, str) else "" for row in todo.itertuples(index=False)}
    features: dict[str, dict] = {i: {} for i in texts}
    # One engine at a time, so peak memory is a single model; each pass resumes on its own.
    for name in (n for n in ENGINE_NAMES if n in engines):
        part = out.with_name(f"{out.name}.{name}.partial")
        cached = read_jsonl(part)
        remaining = [i for i in texts if i not in cached]
        if remaining:
            engine = build_engine(name, config)
            if engine is None:
                continue
            detector = Detector([] if name == "bino" else [engine], engine if name == "bino" else None, None, config)
            with part.open("a") as f:
                for i in tqdm(remaining, desc=name):
                    f.write(json.dumps({"id": i, **detector.features(texts[i])}) + "\n")
                    f.flush()
            del detector, engine
            free_memory()
            cached = read_jsonl(part)
        for i in texts:
            features[i].update(cached.get(i, {}))

    with out.open("a") as f:
        for row in todo.itertuples(index=False):
            record = {c: str(getattr(row, c)) for c in datasets.COLUMNS if c not in ("text", "label")}
            record["label"] = int(row.label)
            record["words"] = count_words(texts[row.id])
            record.update(features[row.id])
            f.write(json.dumps(record) + "\n")
    for name in ENGINE_NAMES:
        out.with_name(f"{out.name}.{name}.partial").unlink(missing_ok=True)


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r.pop("id"): r for r in rows}


if __name__ == "__main__":
    main()
