import json

import pytest

from eval import recipe


def run(style, model, end, facts, seconds=300, requests=10, history=None, method="sentences"):
    return {"document": "d.md", "style": style, "model": model, "seconds": seconds, "requests": requests,
            method: {"history": history or [0.9, end], "facts": facts, "distance": 1.0}}


BASE = "profile:finalized+writer+t0.8+as-page+best3"


def test_features_come_from_the_label():
    assert recipe.features_of(BASE) == ()
    label = BASE.replace("+best3", "+masked+best3+roundtrip-german")
    assert recipe.features_of(label) == ("masked", "roundtrip-german")
    assert recipe.features_of(BASE + "+drafts3+anchored") == ("drafts3", "anchored")
    assert recipe.describe("roundtrip-german") == "Round trip through German"


def test_a_feature_is_measured_against_its_own_models_baseline():
    rows = ([run(BASE, "a", 0.5, 40) for _ in range(4)] + [run(BASE, "b", 0.3, 40) for _ in range(4)]
            + [run(BASE + "+masked", "a", 0.2, 80) for _ in range(4)])
    masked = next(s for s in recipe.build(rows)["setups"] if s["features"] == ["masked"])
    assert masked["end"] == pytest.approx(0.4 - 0.3), "model a's own baseline, not the pooled one, is what it beat"
    assert masked["facts"] == 80 and masked["runs"] == 4
    rows = ([run(BASE, "a", 0.5, 40) for _ in range(4)] + [run(BASE, "b", 0.1, 40) for _ in range(4)]
            + [run(BASE + "+masked", "a", 0.1, 80) for _ in range(4)])
    masked = next(s for s in recipe.build(rows)["setups"] if s["features"] == ["masked"])
    assert masked["under"] == 1.0, "half under, pooled, and a gain of every run is still at most every run"


def test_meaning_is_a_floor_points_cant_buy_off():
    rows = ([run(BASE, "a", 0.05, 30, seconds=60, requests=5) for _ in range(4)]
            + [run(BASE + "+masked", "a", 0.10, 80, seconds=600, requests=20) for _ in range(4)])
    table = recipe.build(rows)
    for points in ({"speed": 6, "quota": 6}, {"detector": 12}):
        picked = recipe.choose(table, points)
        assert picked.features == ("masked",) and not picked.below_floor, points


def test_points_choose_among_setups_that_keep_the_meaning():
    rows = ([run(BASE + "+masked", "a", 0.30, 80, seconds=60) for _ in range(4)]
            + [run(BASE + "+masked+roundtrip-german", "a", 0.05, 75, seconds=600) for _ in range(4)])
    table = recipe.build(rows)
    assert recipe.choose(table, {"detector": 12}).features == ("masked", "roundtrip-german")
    assert recipe.choose(table, {"speed": 12}).features == ("masked",)


def test_only_what_the_page_can_run_and_enough_runs_are_offered():
    rows = ([run(BASE + "+drafts3", "a", 0.0, 90) for _ in range(9)] + [run(BASE + "+masked", "a", 0.1, 80)])
    assert recipe.choose(recipe.build(rows), {"detector": 12}) is None, "multi-start isn't on the page; one run is few"
    rows += [run(BASE + "+masked", "a", 0.1, 80) for _ in range(3)]
    assert recipe.choose(recipe.build(rows), {"detector": 12}).features == ("masked",)


def test_when_nothing_keeps_the_floor_the_one_keeping_most_is_picked_and_said():
    rows = [run(BASE, "a", 0.05, 30) for _ in range(4)] + [run(BASE + "+masked", "a", 0.2, 60) for _ in range(4)]
    picked = recipe.choose(recipe.build(rows), {"detector": 12})
    assert picked.features == ("masked",) and picked.below_floor


def test_the_anchored_loop_is_read_and_counts_its_rounds(tmp_path):
    rows = [run(BASE + "+anchored", "a", 0.05, 80, history=[0.9, 0.5, 0.05], method="anchored") for _ in range(4)]
    table = recipe.build(rows)
    setup = table["setups"][0]
    assert setup["loop"] == "anchored" and setup["rounds"] == 2
    picked = recipe.choose(table, {})
    assert picked.features == () and picked.measures["loop"] == "anchored"


def test_main_writes_the_table_and_load_reads_it(tmp_path, monkeypatch):
    runs = tmp_path / "revise.jsonl"
    runs.write_text("".join(json.dumps(run(BASE, "a", 0.1, 80)) + "\n" for _ in range(4)) + '{"half', "utf-8")
    out = tmp_path / "recipe.json"
    assert recipe.main([str(runs), "--out", str(out)]) == 0
    monkeypatch.setattr(recipe, "TABLE", out)
    assert recipe.load()["setups"][0]["runs"] == 4


def test_runs_on_ai_text_decide_and_human_sources_are_kept_apart():
    human = BASE + "+masked+anchored+human-sources"
    rows = ([run(BASE + "+masked", "a", 0.02, 90) for _ in range(4)]
            + [run(human, "a", 0.10, 95, history=[0.5, 0.1], method="anchored") for _ in range(4)])
    table = recipe.build(rows)
    assert {setup["source"] for setup in table["setups"]} == {"ai", "human"}
    picked = recipe.choose(table, {"detector": 12})
    assert picked.measures["source"] == "ai", "a human original is no test: the AI-text runs decide"
