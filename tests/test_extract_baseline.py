import json
import subprocess
import sys
from pathlib import Path

import pytest

from extract_baseline import profile, strip_markdown

SAMPLE = """# My trip

We went to the lake. It rained all weekend, which was **not** what we planned!

However, the diner was great. I had two slices of [pie](https://example.com) and a coffee.

```python
print("this code is not prose")
```

- First, we packed.
- Then we drove home; it took *four* hours.

> Moreover, the dog was sick.
"""


def test_strip_markdown_keeps_prose_and_counts_structure():
    prose, counts = strip_markdown(SAMPLE)
    assert "My trip" not in prose and "print(" not in prose and "https://" not in prose
    assert "not what we planned" in prose and "slices of pie and" in prose and "four hours" in prose
    assert counts == {"code_blocks": 1, "images": 0, "links": 1, "bold": 1, "italic": 1,
                      "headings": 1, "list_items": 2, "blockquote_lines": 1}


def test_sentence_and_vocabulary_metrics():
    p = profile(SAMPLE)
    s = p["sentences"]
    # We went to the lake (5) | It rained ... planned! (10) | However ... great (5)
    # I had two slices ... coffee (9) | First, we packed (3) | Then ... hours (8) | Moreover ... sick (5)
    assert s["count"] == 7 and p["words"] == 45
    assert s["mean"] == pytest.approx(45 / 7, abs=0.01)
    assert s["min"] == 3 and s["max"] == 10 and s["median"] == 5
    assert -1 < s["burstiness"]["goh_barabasi"] < 0
    assert sum(s["length_histogram"].values()) == pytest.approx(1.0, abs=0.01)
    v = p["vocabulary"]
    assert v["tokens"] == 45 and 0 < v["ttr"] <= 1 and v["mattr_50"] is None  # too short for a window


def test_structure_metrics():
    p = profile(SAMPLE)
    assert p["paragraphs"]["words"]["count"] == 4
    assert p["transitions"]["counts"]["however"] == 1 and p["transitions"]["counts"]["moreover"] == 1
    assert p["transitions"]["sentence_initial"] == {"however": 1, "first": 1, "then": 1, "moreover": 1}
    assert p["sentence_endings"]["exclamation"] == pytest.approx(1 / 7, abs=0.001)
    assert p["punctuation_per_1000_words"]["semicolon"] == pytest.approx(1000 / 45, abs=0.01)
    assert p["sentence_openers"]["we"] == 1


def test_cli_writes_json(tmp_path):
    sample = tmp_path / "sample.md"
    sample.write_text(SAMPLE)
    out = tmp_path / "style_profile.json"
    root = Path(__file__).resolve().parent.parent
    run = subprocess.run([sys.executable, str(root / "extract_baseline.py"), str(sample), "--out", str(out)],
                         capture_output=True, text=True, cwd=root)
    assert run.returncode == 0, run.stderr
    data = json.loads(out.read_text())
    assert data["words"] == 45 and data["source"] == str(sample)
