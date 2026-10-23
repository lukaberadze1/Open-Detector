"""featurize's model-free style pass and --refresh, on a tiny local dataset (no models load)."""

import json
import sys

from eval import featurize

ESSAY = "The river rose all week. We moved the chairs upstairs, then the books, then the dog. "


def write_source(path, n):
    rows = [{"id": f"t{i}", "text": ESSAY * (i + 2), "label": i % 2} for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["featurize", *args, "--min-words", "5"])
    featurize.main()


def cache(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def test_style_pass_then_refresh_keeps_other_fields(tmp_path, monkeypatch):
    src, out = tmp_path / "src.jsonl", tmp_path / "out.jsonl"
    write_source(src, 2)
    run(monkeypatch, "local", "--path", str(src), "--out", str(out), "--engines", "style")
    first = cache(out)
    assert set(first) == {"t0", "t1"} and "style_len_cv" in first["t0"]

    # Pretend the style features changed, and that a model pass left its own field behind.
    rows = cache(out)
    for r in rows.values():
        r["style_mattr"], r["desklib_mean"] = -1.0, 3.5
    out.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    write_source(src, 3)  # and a new text arrived
    run(monkeypatch, "local", "--path", str(src), "--out", str(out), "--engines", "style", "--refresh", "style")

    after = cache(out)
    assert list(after) == ["t0", "t1", "t2"]
    assert after["t0"]["style_mattr"] == first["t0"]["style_mattr"], "refreshed"
    assert after["t0"]["desklib_mean"] == 3.5, "fields of other engines are kept"
    assert "style_aiisms" in after["t2"] and after["t2"]["words"] > 0
    assert not list(tmp_path.glob("*.partial")) and not list(tmp_path.glob("*.tmp"))
