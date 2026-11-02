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
