"""The compression readings, against ZipPy's own seed text."""

import math
import random
import string

from opendetector.zippy import SEED_FILE, _chunks, clean_text, compression_readings


def test_clean_text_keeps_zippy_alphabet():
    assert clean_text("a  b\t!\n\n c?") == "a b\nc"


def test_seed_text_reads_more_ai_like_than_unrelated_text():
    seed = SEED_FILE.read_text(encoding="utf-8")[:3000]
    rng = random.Random(0)
    other = " ".join("".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(430))
    copied, unrelated = compression_readings(seed), compression_readings(other)
    assert copied["zippy_lzma"] > unrelated["zippy_lzma"]
    assert copied["zippy_zlib"] > unrelated["zippy_zlib"]


def test_chunks_cover_long_text_and_survive_no_spaces():
    text = "word " * 1000
    assert " ".join(_chunks(text)) == text
    assert _chunks("x" * 4000) == ["x" * 4000]


def test_nothing_left_after_cleaning_is_nan():
    assert all(math.isnan(v) for v in compression_readings("!!! ???").values())
