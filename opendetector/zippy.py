"""Compression readings, ported from Thinkst's ZipPy (MIT, see ZIPPY_LICENSE).

The compressor is seeded with AI text (zippy_seed.txt): a sample that adds little to the compressed size of the seed
resembles it. Python's own LZMA and zlib only, so it needs no model and runs on every build. A positive reading is
more AI-like, in percentage points of compression ratio, weighted by chunk length as ZipPy does.
"""

from __future__ import annotations

import lzma
import math
import re
import zlib
from functools import lru_cache
from pathlib import Path

SEED_FILE = Path(__file__).with_name("zippy_seed.txt")
CHUNK_CHARS = 1500  # keeps a long text from overwhelming the seed
LZMA_PRESET = 4
ZLIB_LEVEL = 9
ZLIB_WINDOW = 2**15


def clean_text(text: str) -> str:
    """Drop formatting and anything outside ZipPy's alphabet, which would skew the ratios."""
    text = re.sub(" +", " ", text).replace("\t", "")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\n | \n", "\n", text)
    return re.sub(r"[^0-9A-Za-z,\.\(\) \n]", "", text)


def _lzma_ratio(text: str) -> float:
    data = text.encode()
    return len(lzma.compress(data, preset=LZMA_PRESET)) / len(data)


def _zlib_ratio(text: str) -> float:
    data = text.encode()
    compressor = zlib.compressobj(level=ZLIB_LEVEL, wbits=-15, memLevel=9)
    return len(compressor.compress(data) + compressor.flush()) / len(data)


@lru_cache(maxsize=1)
def _seed() -> tuple[str, float, tuple[str, ...], float]:
    """The seed for LZMA with its ratio, and for zlib split to fit its 32 KB window, with the mean ratio."""
    seed = clean_text(SEED_FILE.read_text(encoding="utf-8"))
    lines = seed.split("\n")
    parts = math.ceil(len(seed) / ZLIB_WINDOW)
    size, extra = divmod(len(lines), parts)
    pieces, start = [], 0
    for k in range(parts):
        end = start + size + (k < extra)
        pieces.append("\n".join(lines[start:end]))
        start = end
    return seed, _lzma_ratio(seed), tuple(pieces), sum(map(_zlib_ratio, pieces)) / parts


def _chunks(text: str) -> list[str]:
    chunks, start = [], 0
    while start + CHUNK_CHARS < len(text):
        end = text.rfind(" ", start, start + CHUNK_CHARS + 1)
        if end <= start:
            break
        chunks.append(text[start:end])
        start = end + 1
    chunks.append(text[start:])
    return chunks


def compression_readings(text: str) -> dict[str, float]:
    """`zippy_lzma` and `zippy_zlib` for a text; NaN when nothing is left of it after cleaning."""
    clean = clean_text(text)
    if not clean.strip():
        return {"zippy_lzma": math.nan, "zippy_zlib": math.nan}
    seed, seed_lzma, pieces, seed_zlib = _seed()
    lzma_total = zlib_total = 0.0
    for chunk in _chunks(clean):
        share = len(chunk) / len(clean)
        lzma_total += share * (seed_lzma - _lzma_ratio(seed + chunk))
        zlib_total += share * (seed_zlib - sum(_zlib_ratio(p + chunk) for p in pieces) / len(pieces))
    return {"zippy_lzma": 100 * lzma_total, "zippy_zlib": 100 * zlib_total}
