"""The revision benchmark's own arithmetic: no models."""

from eval.tune_revise import fact_terms, facts_kept, report


def test_facts_are_names_and_numbers_and_their_share_is_what_survives():
    source = "However, Xerxes sent 300 ships past Salamis. The fleet burned."
    assert fact_terms(source) == {"Xerxes", "300", "Salamis"}
    assert facts_kept(source, "Xerxes lost at the strait.") == 33.3
    assert facts_kept("nothing to keep here", "anything") is None


def _row(document, end, start=0.9):
    return {"document": document, "start": start, "words": 100, "distance": 1.5,
            "sentences": {"history": [start, end], "words": 90, "distance": 1.4, "facts": 80.0}}


def test_the_report_counts_runs_and_documents_under_the_bar():
    text = report([_row("a.md", 0.05), _row("a.md", 0.07), _row("b.md", 0.5)], "sentences")
    assert "3 runs on 2 AI-written documents" in text
    assert "2 of 3 runs" in text and "**Documents whose mean is under the bar:** 1 of 2" in text
    assert "| a.md | 2 | 90 | 6 | 5-7 | 2 | - | 1.50 > 1.40 | 80 | x0.90 |" in text


def test_the_human_profile_kit_holds_none_of_the_owners_writing():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "deploy/remote_test/build_package.py"
    spec = importlib.util.spec_from_file_location("build_package", path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    parts = {part for name in build.project_files(build.HUMAN_EXTRA) for part in name.relative_to(build.ROOT).parts[:2]}
    assert "styles" in parts, "the built-in profile goes along"
    names = [name.relative_to(build.ROOT).as_posix() for name in build.project_files(build.HUMAN_EXTRA)]
    assert not [name for name in names if name.startswith(("data/styles", "data/essays/revise_samples", ".env"))
                and name != ".env.example"]


def test_first_drafts_are_ranked_lowest_first_against_the_base():
    from eval.tune_drafts import SETUPS, table

    assert "top-p 0.8" in SETUPS and "digression" in SETUPS
    rows = {"base": [{"start": 0.9, "facts": 70.0, "words_kept": 50.0}], "idiom": [{"start": 0.5}]}
    lines = table(rows).splitlines()
    assert lines[6].startswith("| idiom | 1 | 50 | 50-50 | 1 | -40 |") and lines[7].startswith("| base | 1 | 90 |")


def test_human_swaps_replace_stock_phrases_and_are_stable():
    from eval.human_swaps import human_swaps

    text = "Furthermore, it is important to note that the fleet did not utilize Salamis in order to win in 480 BC."
    swapped = human_swaps(text)
    assert "urthermore" not in swapped and "utilize" not in swapped.lower() and "in order to" not in swapped
    assert "Salamis" in swapped and "480" in swapped and swapped[0].isupper()
    assert swapped == human_swaps(text), "a text picks its swaps the same way every time"
    assert human_swaps("") == ""


def test_only_the_built_in_profile_gets_word_swaps(tmp_path):
    import json

    from eval.generate_essays import styled_clean

    reply = "Furthermore, the fleet went north in the spring of that year."
    for stem, swapped in (("human_default_profile", True), ("mine", False)):
        path = tmp_path / f"{stem}.json"
        path.write_text(json.dumps({"tone": "plain"}))
        assert ("urthermore" not in styled_clean(path, swaps=True)(reply)) is swapped
        assert "Furthermore" in styled_clean(path, swaps=False)(reply), "off unless asked for, as on a public deploy"


def test_pairs_become_two_labelled_rows_sharing_a_group(tmp_path):
    import json

    from eval.datasets import load_local, pairs_to_local

    source, out = tmp_path / "pairs.jsonl", tmp_path / "local.jsonl"
    rows = [{"id": "a", "ai": "Machine text here.", "human": "My own text.", "gen_model": "m"},
            {"id": "b", "ai": "", "human": "No machine side."}]
    source.write_text("\n".join(json.dumps(row) for row in rows))
    assert pairs_to_local(source, out) == 2
    frame = load_local(str(out))
    assert sorted(frame["label"]) == [0, 1] and set(frame["group_key"]) == {"a"}


def test_the_free_round_trip_goes_through_every_hop_and_keeps_what_comes_back_wrong():
    from eval.text_attacks import FREE_HOPS, free_round_trip

    hops = []

    def translate(text, source, target):
        hops.append((source, target))
        return text.replace("the", "that") if target == "en" else text

    parts = ["In 480 the fleet met the enemy at the strait near the shore.", ""]
    result, changed = free_round_trip(parts, translate, pause=0)
    assert changed == 1 and "that fleet" in result[0] and result[1] == ""
    assert hops == [("en", "de"), ("de", "fr"), ("fr", "en")] and FREE_HOPS == ("en", "de", "fr", "en")
    lost, changed = free_round_trip(parts, lambda text, s, t: text.replace("480", "many"), pause=0)
    assert lost == parts and changed == 0, "a paragraph that loses a number stays as it was"


def test_the_free_round_trip_stops_quietly_when_the_endpoint_refuses():
    from eval.text_attacks import free_round_trip

    def refuse(text, source, target):
        raise OSError("429")

    parts = ["The first paragraph of this text goes here today.", "The second paragraph of this text goes here."]
    assert free_round_trip(parts, refuse, pause=0) == (parts, 0)


def test_only_the_built_in_profile_is_offered_the_free_round_trip():
    import inspect

    from ui import redteam

    source = inspect.getsource(redteam._attacks)
    assert "BUILTIN_STYLE_DIR" in source and "FREE_TRIP" in source
