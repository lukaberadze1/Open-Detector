"""Labelled evaluation data. Every loader returns a DataFrame with the columns in COLUMNS:

    id, text, label (0 = human, 1 = AI), dataset, domain, model, attack, group, group_key

`group` is a fairness slice (e.g. "ell" for English-language learners). `group_key` ties
together texts that must land in the same split — a RAID source document and all of its
generations and attacked copies, or a human essay and its AI mirror — so near-duplicates
never leak from fitting data into test data.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

COLUMNS = ["id", "text", "label", "dataset", "domain", "model", "attack", "group", "group_key"]


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    for col, default in [("domain", "unknown"), ("model", "unknown"), ("attack", "none"), ("group", "all")]:
        df[col] = df[col].fillna(default).astype(str) if col in df else default
    if "group_key" not in df:
        df["group_key"] = df["id"]
    df["id"] = df["id"].astype(str)
    df["group_key"] = df["group_key"].astype(str)
    df["label"] = df["label"].astype(int)
    return df[COLUMNS].reset_index(drop=True)


# -- RAID ------------------------------------------------------------------------------

RAID_URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/{split}.csv"
# RAID rows start with a UUID id; finding one after a newline resyncs mid-file.
_ROW_START = re.compile(rb"\n(?=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12},)")


def _get(url: str, start: int, end: int, retries: int = 5) -> tuple[bytes, int]:
    """One byte range of `url`, retried with backoff: a sample takes hundreds of requests, and
    a single network blip (DNS failure, TLS timeout, 5xx) shouldn't throw the whole run away."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers["Content-Range"].rsplit("/", 1)[1])
                return resp.read(), total
        except urllib.error.HTTPError as exc:
            if (exc.code < 500 and exc.code != 429) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 2 ** attempt * 3))
    raise AssertionError("unreachable")


def _iter_raid_random(split: str, seed: int, chunks: int, chunk_bytes: int = 1 << 20) -> Iterator[dict]:
    """Random sample of the RAID CSV without downloading it: the file (11.8 GB for train)
    is sorted by domain and attack, so reading it from the top gives a skewed sample.
    Instead fetch `chunks` random byte ranges and parse the complete rows inside each."""
    url = RAID_URL.format(split=split)
    head, size = _get(url, 0, 1 << 16)
    columns = next(csv.reader(io.StringIO(head.split(b"\n", 1)[0].decode())))
    rng = random.Random(seed)
    for offset in sorted(rng.randrange(1 << 16, size - chunk_bytes) for _ in range(chunks)):
        data, _ = _get(url, offset, offset + chunk_bytes)
        starts = [m.end() for m in _ROW_START.finditer(data)]
        if len(starts) < 2:
            continue
        body = data[starts[0] : starts[-1]].decode("utf-8", errors="replace")
        yield from csv.DictReader(io.StringIO(body), fieldnames=columns)


def _iter_raid(split: str, csv_path: str | None, seed: int, chunks: int) -> Iterator[dict]:
    if csv_path:
        for chunk in pd.read_csv(csv_path, chunksize=50_000):
            yield from chunk.sample(frac=1, random_state=seed).to_dict("records")
    else:
        yield from _iter_raid_random(split, seed, chunks)


def load_raid(
    split: str = "train",
    ai_per_bucket: int = 3,
    human_per_bucket: int = 30,
    chunks: int = 300,
    include_adversarial: bool = True,
    csv_path: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Stratified sample of RAID: up to `ai_per_bucket` texts per (model, domain, attack)
    and `human_per_bucket` per (domain, attack). Reads `chunks` random 1 MB slices of the
    Hugging Face copy (~400 rows each), or a local CSV downloaded with raid-bench. GLYPH saw
    RAID's non-adversarial training data, so its RAID scores are optimistic."""
    counts: Counter = Counter()
    rows = []
    for r in _iter_raid(split, csv_path, seed, chunks):
        attack = str(r.get("attack") or "none")
        if attack == "nan":
            attack = "none"
        if not include_adversarial and attack != "none":
            continue
        human = r["model"] == "human"
        key = ("human", r["domain"], attack) if human else (r["model"], r["domain"], attack)
        if counts[key] >= (human_per_bucket if human else ai_per_bucket):
            continue
        counts[key] += 1
        candidates = (r.get("source_id"), r.get("adv_source_id"), r["id"])
        source = next((v for v in candidates if isinstance(v, str) and v), r["id"])
        rows.append({
            "id": r["id"], "text": r["generation"], "label": 0 if human else 1, "dataset": "raid",
            "domain": r["domain"], "model": r["model"], "attack": attack, "group_key": source,
        })
    return _finish(pd.DataFrame(rows))


# -- MAGE ------------------------------------------------------------------------------

def load_mage(split: str = "test", n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """MAGE (Li et al., 2024): 10 domains, 27 generators. Note MAGE's own `label` uses
    1 = human, the reverse of ours, so we derive labels from `src` and check they agree."""
    from datasets import load_dataset

    df = load_dataset("yaful/MAGE", split=split).to_pandas()
    df = df.sample(n=min(n, len(df)), random_state=seed)
    human = df["src"].str.contains("human")
    agreement = ((df["label"] == 1) == human).mean()
    if agreement < 0.99:
        raise ValueError(f"MAGE labels disagree with src for {1 - agreement:.1%} of rows; check the dataset card")
    out = pd.DataFrame({
        "id": "mage-" + df.index.astype(str),
        "text": df["text"],
        "label": (~human).astype(int),
        "dataset": "mage",
        "domain": df["src"].str.split("_").str[0],
        "model": df["src"].where(~human, "human"),
    })
    return _finish(out)


# -- Student essays --------------------------------------------------------------------

def load_persuade(csv_path: str, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """PERSUADE 2.0 human essays (grades 6-12, with English-language-learner status).
    Download the CSV from github.com/scrosseye/persuade_corpus_2.0 (CC BY-NC-SA 4.0)."""
    df = pd.read_csv(csv_path)
    text_col = next((c for c in ("full_text", "text") if c in df), None)
    id_col = next((c for c in ("essay_id_comp", "essay_id", "id") if c in df), None)
    if text_col is None or id_col is None:
        raise ValueError(f"Unrecognised PERSUADE columns: {list(df.columns)}")
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    group = "all"
    if "ell_status" in df:
        group = df["ell_status"].map(lambda v: "ell" if str(v).strip().lower() in ("yes", "1", "true") else "non-ell")
    out = pd.DataFrame({
        "id": df[id_col].astype(str),
        "text": df[text_col],
        "label": 0,
        "dataset": "persuade",
        "domain": df.get("prompt_name", "essay"),
        "model": "human",
        "group": group,
    })
    return _finish(out)


def load_local(path: str, dataset: str | None = None) -> pd.DataFrame:
    """Your own data as JSONL or CSV. Needs `text` and `label` (0/1 or "human"/"ai");
    optional: id, domain, model, attack, group, group_key."""
    p = Path(path)
    if p.suffix == ".jsonl":
        df = pd.DataFrame([json.loads(line) for line in p.read_text().splitlines() if line.strip()])
    else:
        df = pd.read_csv(p)
    if "text" not in df or "label" not in df:
        raise ValueError(f"{path} needs 'text' and 'label' columns")
    df["label"] = df["label"].map(lambda v: {"human": 0, "ai": 1}.get(str(v).lower(), v)).astype(int)
    if "id" not in df:
        df["id"] = [f"{p.stem}-{i}" for i in range(len(df))]
    df["dataset"] = dataset or df.get("dataset", p.stem)
    return _finish(df)
