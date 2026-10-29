"""Pick an adversarial challenge-generation setting from four measured priorities.

eval/tune_generation.py measures settings (a model, its thinking level, and how many blocks go in one
request) on a document and writes what it found to models/generation_tuning.json. Each setting has
four measures:

- detector: challenge strength, measured by the generated text's mean AI likelihood on this detector
- style: similarity to the selected style profile, measured against the writer's own passages, of the
  same length, that are further from the profile than it is
- speed: seconds of writing per block
- quota: requests and tokens per block, and how much of the model's daily allowance of requests a
  run takes (the table's `daily_requests`, from the limits Gemini's quota errors state)

The writer spreads POINTS points over the four, at most MAX_POINTS on one, and the setting with the
highest points-weighted sum of its four scores is picked. Each score runs from 0 to 1. The detector
and style scores are already percentages, so they keep their own scale: 1 minus the AI likelihood,
and the style match over 100. Speed and quota have no natural top, so they are a ratio to the best
setting in the table: half as fast scores 0.5. Quota is the mean of that ratio for requests, the
same for tokens, and the share of the model's daily requests the document at hand would leave: a
99-block batch at 8 blocks per request takes about 12 of a 20-a-day model's requests and scores 0.4
there, while a single sample barely touches any allowance. A model whose limit isn't known loses nothing
on that part. Scaling each measure from its worst setting to its best instead would stretch a gap of a
few points, smaller than the noise between two draws, over the whole range, and would let one bad
setting in the table reorder the others. With points on every measure, a setting that another one
beats on all four is never picked. Points spread evenly give the default. Speed points also set how
many requests a batch sends at once, which costs no quota.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MEASURES = ("detector", "style", "speed", "quota")
POINTS, MAX_POINTS = 12, 6
EVEN = dict.fromkeys(MEASURES, POINTS // len(MEASURES))
TABLE = Path(__file__).resolve().parents[1] / "models/generation_tuning.json"
REFERENCE_BLOCKS = 100  # the document size a batch's pick is made for when none is given
PROMPT_CHARS = 2000  # a request's own instructions, beside the style profile and document it carries


@dataclass(frozen=True)
class Setting:
    """A setting from the table, and how many requests a batch sends at once with it."""

    model: str
    thinking: str
    group_size: int
    parallel: int
    ai: float | None  # the detector's mean AI likelihood, 0 to 1
    style: float | None  # style match, 0 to 100
    seconds_per_block: float
    requests_per_block: float
    tokens_per_block: float
    measured: bool  # False when predicted from the settings around it
    input_tokens_per_block: float | None = None  # the part of tokens_per_block the requests carried in


def load_table(path: Path = TABLE) -> dict | None:
    """The measured table, or None before the benchmark has run."""
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return table if isinstance(table, dict) and table.get("rows") else None


def parallel_for(points: dict[str, int]) -> int:
    """Requests at once: one, and one more for every two speed points, up to four."""
    return min(4, 1 + points.get("speed", 0) // 2)


def _ratio(values: list[float]) -> list[float]:
    """The best (lowest) value over each value: 1 for the best, 0.5 for one that costs twice as much."""
    best = min(values)
    return [best / value if value > 0 else 1.0 for value in values]


def daily_share(row: dict, blocks: int, limits: dict[str, int]) -> float | None:
    """The share of its model's daily requests `blocks` blocks would take, or None when the limit
    isn't known."""
    limit = limits.get(row["model"])
    return blocks * row["requests_per_block"] / limit if limit else None


def scores(rows: list[dict], blocks: int = REFERENCE_BLOCKS, limits: dict[str, int] | None = None
           ) -> list[dict[str, float]]:
    """Each row's four scores, 0 to 1: the detector and style on their own scales (an unmeasured one
    counts 0.5 for every row), speed against the best row, and quota against the best row's requests
    and tokens and by what is left of the model's daily allowance after `blocks` blocks."""
    speed = _ratio([row["seconds_per_block"] for row in rows])
    requests = _ratio([row["requests_per_block"] for row in rows])
    tokens = _ratio([row["tokens_per_block"] for row in rows])
    left = [1 - min(1.0, share) if (share := daily_share(row, blocks, limits or {})) is not None else 1.0
            for row in rows]
    return [{"detector": 0.5 if row["ai"] is None else 1 - row["ai"],
             "style": 0.5 if row["style"] is None else row["style"] / 100,
             "speed": v, "quota": (r + t + d) / 3}
            for row, v, r, t, d in zip(rows, speed, requests, tokens, left, strict=True)]


def choose(table: dict, points: dict[str, int], models: list[str] | None = None, batch: bool = True,
           blocks: int | None = None) -> Setting | None:
    """The table's best setting for these points among `models` (all of them when None), for a batch of
    `blocks` blocks (REFERENCE_BLOCKS when None). A single sample is one block, so it is picked from
    the settings that write one block per request."""
    rows = [row for row in table["rows"]
            if (models is None or row["model"] in models) and (batch or row["group_size"] == 1)]
    if not rows:
        return None
    spent = sum(points.get(measure, 0) for measure in MEASURES)
    weights = {measure: points.get(measure, 0) / spent if spent else 1 / len(MEASURES) for measure in MEASURES}
    scored = scores(rows, (blocks or REFERENCE_BLOCKS) if batch else 1, table.get("daily_requests"))

    def value(index: int) -> tuple[float, float]:  # ties go to the setting that sends fewer requests
        return sum(weights[m] * scored[index][m] for m in MEASURES), -rows[index]["requests_per_block"]

    row = rows[max(range(len(rows)), key=value)]
    return Setting(row["model"], row["thinking"], row["group_size"], parallel_for(points) if batch else 1,
                   row["ai"], row["style"], row["seconds_per_block"], row["requests_per_block"],
                   row["tokens_per_block"], row["measured"], row.get("input_tokens_per_block"))


def estimate(setting: Setting, blocks: int, table: dict, context_chars: int | None = None) -> dict:
    """About how long, how many requests and how many tokens `blocks` blocks take with `setting`, and
    the model's daily limit when known. Most of a request's input is the style profile and the
    document it carries (`context_chars` characters, never less than the prompt's own instructions),
    so the input tokens are scaled by that against what the table's requests carried; what the model
    writes and thinks is not."""
    tokens = setting.tokens_per_block
    if context_chars is not None and table.get("context_chars"):
        scale = max(context_chars, PROMPT_CHARS) / table["context_chars"]
        carried = setting.input_tokens_per_block if setting.input_tokens_per_block is not None else tokens
        tokens = carried * scale + tokens - carried
    return {"seconds": blocks * setting.seconds_per_block / setting.parallel,
            "requests": blocks * setting.requests_per_block,
            "tokens": blocks * tokens,
            "daily_limit": (table.get("daily_requests") or {}).get(setting.model)}
