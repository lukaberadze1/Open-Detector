"""The demo grading: stand-in scores, no models."""

import json

from eval.score_demo import AI_SOURCE, HUMAN, demo_groups, report

LONG = " ".join(["word"] * 160)


def _folder(tmp_path):
    seed = {"messages": [{}, {"content": f"source {LONG}"}], "desirable_response": f"human {LONG}"}
    short = {"messages": [{}, {"content": "too short"}], "desirable_response": "too short"}
    candidate = {"results": [{"temperature": 0.3, "rewrite": f"cool {LONG}"},
                             {"temperature": 1.0, "rewrite": f"hot {LONG}"},
                             {"temperature": 1.0, "rewrite": f"hot again {LONG}"}]}
    (tmp_path / "kto_seed_demo.jsonl").write_text("\n".join(map(json.dumps, (seed, short))), encoding="utf-8")
    (tmp_path / "kto_candidates_demo.jsonl").write_text(json.dumps(candidate), encoding="utf-8")
    return tmp_path


def test_the_groups_hold_what_the_detector_would_read_with_rewrites_by_temperature(tmp_path):
    groups = demo_groups(_folder(tmp_path))
    assert [len(groups[name]) for name in (AI_SOURCE, HUMAN)] == [1, 1]
    assert len(groups["Their model, temperature 0.3"]) == 1 and len(groups["Their model, temperature 1.0"]) == 2


def test_the_report_counts_verdicts_at_the_detectors_thresholds(tmp_path):
    groups = {"Texts": ["a", "b", "c"]}
    scores = {"a": 0.99, "b": 0.4, "c": 0.05}
    text = report(groups, scores.__getitem__, {"ai": 0.9, "review": 0.15})
    assert "| Texts | 3 | 0.480 | 0.400 | 1 | 1 | 2 |" in text
