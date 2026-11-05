"""The reference benchmark, with stand-in detectors. No models or downloads."""

import csv

from eval import reference_benchmark as rb

PROSE = "This is a plain sentence that runs along for a while without saying anything at all. " * 12


def item(text, label, **more):
    return {**rb.row(text, label, more.pop("source", "mage"), more.pop("genre", "news"), more.pop("attack", "none")),
            **more}


def test_guided_attacks_are_told_apart():
    assert rb.is_guided("prompt+alexandria_profile+revise3") and rb.is_guided("math+contextual")
    assert not rb.is_guided("prompt+human_default_profile+restyle") and not rb.is_guided(None)


def test_short_text_and_repeats_are_left_out():
    rows = [item(PROSE, 1), item(PROSE, 1), item("Too short.", 0)]
    assert len(rb.readable(rows)) == 1


def test_a_folder_gives_labels_and_genres(tmp_path):
    (tmp_path / "human/story").mkdir(parents=True)
    (tmp_path / "ai/story").mkdir(parents=True)
    (tmp_path / "human/story/a.txt").write_text("mine", encoding="utf-8")
    (tmp_path / "ai/story/b.txt").write_text("theirs", encoding="utf-8")
    assert sorted((r["label"], r["genre"]) for r in rb.load_folder(tmp_path)) == [(0, "story"), (1, "story")]


def test_a_stopped_run_resumes_from_the_cache(tmp_path):
    cache = tmp_path / "scores.jsonl"
    rows = [item("first", 1), item("second", 0)]
    built = []

    def build(name):
        built.append(name)
        return (lambda text: (0.9, "likely_ai")), (lambda: None)

    scores = rb.score_rows(rows, "slim", "v1", cache, build)
    assert scores["first"]["p"] == 0.9 and built == ["slim"]
    rb.score_rows(rows, "slim", "v1", cache, build)
    assert built == ["slim"], "everything was cached, so no model loads"
    rb.score_rows(rows, "slim", "v2", cache, build)
    assert len(built) == 2, "a refitted combiner scores again"
    with cache.open("a", encoding="utf-8") as file:
        file.write('{"key": "cut off')
    assert rb.read_cache(cache)


def test_the_export_holds_only_public_text(tmp_path):
    path = tmp_path / "out.csv"
    rows = [item("public human", 0), item("public ai", 1), item("my story", 0, source="reference"),
            item("my sample", 1, source="saved")]
    assert rb.export_public(rows, path, 10) == 2
    text = path.read_text(encoding="utf-8")
    assert "my story" not in text and "my sample" not in text


def test_outside_scores_read_percentages_and_skip_blanks(tmp_path):
    path = tmp_path / "scored.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([["text", "ai_probability"], ["a", "67%"], ["b", "0.2"], ["c", ""]])
    scores = rb.read_external(path)
    assert scores["a"] == {"p": 0.67, "verdict": "likely_ai"} and scores["b"]["verdict"] == "likely_human"
    assert "c" not in scores


def test_the_report_gives_rates_and_the_transfer_gap():
    rows = [item("tuned", 1, attack="revise3"), item("kind", 0, genre="story")]
    ours = {"tuned": {"p": 0.1, "verdict": "likely_human"}, "kind": {"p": 0.9, "verdict": "likely_ai"}}
    theirs = {"tuned": {"p": 0.7, "verdict": "likely_ai"}, "kind": {"p": 0.2, "verdict": "likely_human"}}
    assert rb.transfer_gap(rows, ours, theirs) == (1, 0.6)
    text = rb.report(rows, {"slim": ours, "roberta": theirs})
    assert "False accusations by genre" in text and "story" in text and "+60%" in text
