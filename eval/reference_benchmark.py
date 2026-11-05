"""The reference standard: every detector graded against text whose origin is known, never against another detector.

    python -m eval.reference_benchmark                         # slim, full and RoBERTa, about 300 MAGE texts
    python -m eval.reference_benchmark --detectors slim --mage 50    # a quick look
    python -m eval.reference_benchmark --export zerogpt.csv --export-count 30
    python -m eval.reference_benchmark --external zerogpt=zerogpt_scored.csv

Where the labels come from:

- MAGE, only the side of its split `eval/fit_combiner.py` keeps out of fitting (genre: its domain)
- DAIGT V2 if it is in data/pairs/ (genre: student essay)
- HC3 if it is in the Hugging Face cache (genre: its source; most answers are too short for the detector)
- data/reference/human/<genre>/*.txt and data/reference/ai/<genre>/*.txt: your own held-back texts, which is where fresh
  human writing goes
- the saved red-team samples: AI by how they were made. Those that used the detector while being written are marked
  guided, and they can't be a fair test of the detector that guided them, which is what the transfer gap measures.

An outside detector is one more test subject: --export writes public texts to a CSV (never saved samples or anything
in data/reference/, which may hold private writing), you add an `ai_probability` column by hand, and --external reads
it back. RoBERTa has no calibrated cut-off here, so its verdict is 0.5 and its rates aren't comparable to our builds'.

Scores are cached by detector, combiner and text, so a stopped run resumes. Read-only otherwise: writes
eval/reports/reference.md and the cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from eval.redteam_report import SAMPLES, load_samples
from opendetector import DetectorConfig
from opendetector.config import DATA_ROOT
from opendetector.metrics import wilson_interval
from opendetector.segment import count_words, looks_like_prose

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval/cache/reference_scores.jsonl"
REPORT = ROOT / "eval/reports/reference.md"
FOLDER = DATA_ROOT / "data/reference"
PAIRS = DATA_ROOT / "data/pairs"
EXTRA_SAMPLES = (DATA_ROOT / "data/essays/contextual_redteam.jsonl",)
DETECTORS = ("slim", "full", "roberta")
SPLIT_SEED = 0
FEW = 30  # rows: a group smaller than this is marked, since its rate says little
_GUIDED = ("revise", "math", "contextual", "challenge", "gradient")
_PUBLIC = ("mage", "daigt", "hc3")


def is_guided(attack: str | None) -> bool:
    """Whether a sample's attack used the detector's scores while writing or editing it."""
    return any(word in (attack or "") for word in _GUIDED)


def row(text: str, label: int, source: str, genre: str, attack: str = "none", group: str | None = None) -> dict:
    return {"text": text, "label": label, "source": source, "genre": genre, "attack": attack or "none",
            "guided": is_guided(attack),
            "group_key": group or hashlib.sha256(text.encode()).hexdigest()[:16]}


def readable(rows: list[dict]) -> list[dict]:
    """What the detector reads, once each: it abstains on under 150 words and on non-prose."""
    seen: dict[str, dict] = {}
    for item in rows:
        if count_words(item["text"]) >= DetectorConfig().min_words and looks_like_prose(item["text"]):
            seen.setdefault(item["text"], item)
    return list(seen.values())


def load_folder(folder: Path = FOLDER) -> list[dict]:
    rows = []
    for label, name in ((0, "human"), (1, "ai")):
        for path in sorted((folder / name).glob("*/*.txt")):
            rows.append(row(path.read_text(encoding="utf-8"), label, "reference", path.parent.name))
    return rows


def load_saved(paths: list[Path] | None = None) -> list[dict]:
    """Saved samples. Human originals count only as human, and copies of originals are already left out."""
    found = load_samples(paths if paths is not None else [*SAMPLES, *EXTRA_SAMPLES])
    return [row(item["text"], int(item.get("label", 1)), "saved", "red-team", item.get("attack"),
                item.get("group_key")) for item in found]


def load_mage_side(count: int) -> list[dict]:
    """MAGE's texts on the test side of the combiner's split. Asks for more than `count`, since most of the dev side
    and the short texts go."""
    from eval.datasets import load_mage
    from eval.fit_combiner import assign_split

    frame = load_mage(n=count * 10)
    frame = frame[assign_split(frame.group_key, SPLIT_SEED) == "test"]
    frame = frame[frame.text.map(count_words) >= DetectorConfig().min_words].head(count)
    return [row(item.text, int(item.label), "mage", item.domain, item.attack, item.group_key)
            for item in frame.itertuples()]


def load_daigt(count: int, folder: Path = PAIRS) -> list[dict]:
    """DAIGT V2's `train_v2_drcat` CSV, if it's there: `text` and `label` (1 is AI)."""
    import pandas as pd

    paths = [p for pattern in ("*daigt*.csv", "train_v2*.csv") for p in folder.glob(pattern)]
    if not paths or not count:
        return []
    frame = pd.read_csv(paths[0])
    frame = frame.groupby("label", group_keys=False).apply(lambda part: part.sample(min(len(part), count // 2),
                                                                                    random_state=0))
    return [row(item.text, int(item.label), "daigt", "student essay") for item in frame.itertuples()]


def load_hc3(count: int) -> list[dict]:
    """HC3's human and ChatGPT answers of 150+ words, up to half of `count` each. One question's answers share a key."""
    if not count:
        return []
    from huggingface_hub import snapshot_download

    try:
        root = Path(snapshot_download("Hello-SimpleAI/HC3", repo_type="dataset", local_files_only=True))
    except Exception:  # not downloaded
        return []
    found: dict[int, list[dict]] = {0: [], 1: []}
    for line in (root / "all.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = hashlib.sha256(item["question"].encode()).hexdigest()[:16]
        for label, field in ((0, "human_answers"), (1, "chatgpt_answers")):
            for text in item.get(field) or []:
                if len(found[label]) < count // 2 and count_words(text) >= DetectorConfig().min_words:
                    found[label].append(row(text, label, "hc3", f"hc3 {item.get('source', '')}".strip(), group=key))
    return [*found[0], *found[1]]


def cache_key(detector: str, version: str, text: str) -> str:
    return hashlib.sha256(f"{detector}\n{version}\n{text}".encode()).hexdigest()[:24]


def read_cache(path: Path) -> dict[str, dict]:
    known = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:  # a line a crash cut short
                continue
            known[entry["key"]] = entry
    return known


def combiner_version(profile: str) -> str:
    path = DetectorConfig.for_profile(profile).combiner_path
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def build_scorer(name: str) -> tuple[Callable[[str], tuple[float, str]], Callable[[], None]]:
    """A detector as text to (AI chance, verdict), and how to free it: the 8 GB Mac holds one at a time."""
    from opendetector.engines import free_memory

    if name == "roberta":
        from eval.bench_math import _independent, independent_scores

        def score_roberta(text: str) -> tuple[float, str]:
            value = independent_scores([text])[0]
            return value, "likely_ai" if value >= 0.5 else "likely_human"

        return score_roberta, lambda: (_independent.cache_clear(), free_memory())
    from opendetector import Detector

    detector = Detector.from_config(DetectorConfig.for_profile(name))

    def score(text: str) -> tuple[float, str]:
        report = detector.analyze(text, sentences=False)
        return report.ai_probability, report.verdict

    return score, free_memory


def score_rows(rows: list[dict], name: str, version: str = "", cache: Path = CACHE,
               build: Callable = build_scorer, cached_only: bool = False) -> dict[str, dict]:
    """This detector's score for each text, by text, saving each as it comes so a stop loses at most one. With
    `cached_only`, just the texts already scored, to report on a run that didn't finish."""
    known = read_cache(cache)
    todo = [item for item in rows if cache_key(name, version, item["text"]) not in known]
    if todo and not cached_only:
        score, free = build(name)
        cache.parent.mkdir(parents=True, exist_ok=True)
        began = time.time()
        with cache.open("a", encoding="utf-8") as output:
            for number, item in enumerate(todo, 1):
                value, verdict = score(item["text"])
                entry = {"key": cache_key(name, version, item["text"]), "p": value, "verdict": verdict}
                known[entry["key"]] = entry
                output.write(json.dumps(entry) + "\n")
                output.flush()
                left = (time.time() - began) / number * (len(todo) - number) / 60
                filled = 20 * number // len(todo)
                print(f"\r{name:8} [{'#' * filled}{'-' * (20 - filled)}] {number}/{len(todo)} ~{left:.0f}m left   ",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        del score
        free()
    return {item["text"]: known[key] for item in rows if (key := cache_key(name, version, item["text"])) in known}


def read_external(path: Path) -> dict[str, dict]:
    """A CSV with `text` and `ai_probability` (0 to 1, or a percentage); blank rows weren't checked."""
    scores = {}
    with path.open(encoding="utf-8", newline="") as file:
        for item in csv.DictReader(file):
            raw = (item.get("ai_probability") or "").strip().rstrip("%")
            if raw:
                value = float(raw)
                value = value / 100 if value > 1 else value
                scores[item["text"]] = {"p": value, "verdict": "likely_ai" if value >= 0.5 else "likely_human"}
    return scores


def export_public(rows: list[dict], path: Path, count: int) -> int:
    """Public texts only, half each label, for checking by hand on an outside site."""
    chosen = []
    for label in (0, 1):
        chosen += [item for item in rows if item["source"] in _PUBLIC and item["label"] == label][:count // 2]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ["label", "genre", "text", "ai_probability"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chosen)
    return len(chosen)


def _rate(hits: int, total: int) -> str:
    low, high = wilson_interval(hits, total)
    return f"{hits / total:.0%} ({low:.0%} to {high:.0%}){' *' if total < FEW else ''}"


def _group_table(title: str, column: str, rows: list[dict], scores: dict[str, dict], field: str, verdict: str,
                 heading: str) -> list[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        if item["text"] in scores:
            groups[item[field]].append(item)
    if not groups:
        return []
    lines = [f"#### {title}", "", f"| {column} | Texts | {heading} | Mean AI likelihood |", "|---|---:|---:|---:|"]
    for name, group in sorted(groups.items(), key=lambda pair: -len(pair[1])):
        hits = sum(scores[item["text"]]["verdict"] == verdict for item in group)
        mean = statistics.fmean(scores[item["text"]]["p"] for item in group)
        lines.append(f"| {name} | {len(group)} | {_rate(hits, len(group))} | {mean:.0%} |")
    return [*lines, ""]


def transfer_gap(rows: list[dict], first: dict[str, dict], second: dict[str, dict]) -> tuple[int, float] | None:
    """On guided AI samples both scored: how much lower the first's mean is than the second's. Large means the
    edits fool the first, not AI detection in general."""
    both = [item["text"] for item in rows if item["label"] == 1 and item["guided"]
            and item["text"] in first and item["text"] in second]
    if not both:
        return None
    return len(both), statistics.fmean(second[t]["p"] - first[t]["p"] for t in both)


def report(rows: list[dict], results: dict[str, dict[str, dict]]) -> str:
    human = [item for item in rows if item["label"] == 0]
    ai = [item for item in rows if item["label"] == 1]
    lines = ["# Reference benchmark", "",
             f"{len(ai)} AI and {len(human)} human texts of known origin, 150 words or more. Written by "
             "`python -m eval.reference_benchmark`. Brackets are Wilson 95% intervals, which treat texts as "
             f"independent, so versions of one source make them too narrow. * marks under {FEW} texts. Saved samples "
             "include text tuned against this detector, so they aren't a held-back test.", "",
             "## Every detector", "",
             "| Detector | AI texts | Missed | Human texts | Wrongly called AI | Needs review (human) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, scores in results.items():
        got_ai = [i for i in ai if i["text"] in scores]
        got_human = [i for i in human if i["text"] in scores]
        count = lambda group, verdict, known=scores: sum(known[i["text"]]["verdict"] == verdict for i in group)  # noqa: E731
        lines.append(f"| {name} | {len(got_ai)} | "
                     f"{_rate(count(got_ai, 'likely_human'), len(got_ai)) if got_ai else 'n/a'} | {len(got_human)} | "
                     f"{_rate(count(got_human, 'likely_ai'), len(got_human)) if got_human else 'n/a'} | "
                     f"{_rate(count(got_human, 'needs_review'), len(got_human)) if got_human else 'n/a'} |")
    lines.append("")
    names = list(results)
    gaps = [(a, b, gap) for a in names for b in names if a != b and (gap := transfer_gap(rows, results[a], results[b]))]
    if gaps:
        lines += ["## Transfer gap on guided samples", "",
                  "How much higher the second detector scores text tuned against our own, in points of AI likelihood. "
                  "A big gap means the tuning fools the first detector, not AI detection in general.", "",
                  "| Lower detector | Compared with | Texts | Mean gap |", "|---|---|---:|---:|"]
        lines += [f"| {a} | {b} | {gap[0]} | {gap[1]:+.0%} |" for a, b, gap in gaps]
        lines.append("")
    for name, scores in results.items():
        lines += [f"## {name}", ""]
        lines += _group_table("False accusations by genre", "Genre", human, scores, "genre", "likely_ai",
                              "Wrongly called AI")
        lines += _group_table("Misses by genre", "Genre", ai, scores, "genre", "likely_human", "Missed")
        lines += _group_table("Misses by attack", "Attack", ai, scores, "attack", "likely_human", "Missed")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", default=",".join(DETECTORS), help=f"any of {','.join(DETECTORS)}")
    parser.add_argument("--mage", type=int, default=300, help="MAGE texts from the side kept out of fitting")
    parser.add_argument("--daigt", type=int, default=200, help="DAIGT V2 texts, half each label, if it's there")
    parser.add_argument("--hc3", type=int, default=100, help="HC3 answers, half each label, if it's there")
    parser.add_argument("--no-saved", action="store_true", help="leave out the saved red-team samples")
    parser.add_argument("--export", type=Path, help="write public texts to this CSV for checking by hand, and stop")
    parser.add_argument("--export-count", type=int, default=30)
    parser.add_argument("--external", action="append", default=[], metavar="NAME=CSV",
                        help="an outside detector's scores")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--cached-only", action="store_true", help="score nothing: report on the texts already scored")
    args = parser.parse_args(argv)
    rows = readable([*load_folder(), *([] if args.no_saved else load_saved()),
                     *(load_mage_side(args.mage) if args.mage else []), *load_daigt(args.daigt), *load_hc3(args.hc3)])
    if args.export:
        print(f"Wrote {export_public(rows, args.export, args.export_count)} public texts to {args.export}")
        return 0
    if not rows:
        print("No labelled text of 150+ words found.")
        return 1
    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    if unknown := set(names) - set(DETECTORS):
        parser.error(f"unknown detectors: {sorted(unknown)}")
    print(f"{len(rows)} texts: {sum(r['label'] == 1 for r in rows)} AI, {sum(r['label'] == 0 for r in rows)} human")
    results = {name: score_rows(rows, name, combiner_version(name) if name != "roberta" else "",
                                cached_only=args.cached_only) for name in names}
    for item in args.external:
        name, _, path = item.partition("=")
        results[name] = read_external(Path(path))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(rows, results), encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
