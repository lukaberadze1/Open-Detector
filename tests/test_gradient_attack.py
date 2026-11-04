"""The word search's candidates and word choice, with no model."""

from eval import gradient_attack as ga


def test_only_listed_synonyms_that_fit_both_neighbours(monkeypatch):
    monkeypatch.setattr(ga, "_pair_index", lambda: ({"the": {"song": 9, "tune": 9, "melody": 9}},
                                                    {"was": {"song": 9, "tune": 9, "season": 9}}))
    monkeypatch.setattr(ga, "SAFE_SYNONYMS", {"song": ["tune", "melody", "season"]})
    # 'season' fits only the right, 'melody' only the left, and neither neighbour makes a word a synonym.
    assert ga.fitting_words("song", "the", "was") == ["tune"]


def test_a_word_outside_the_table_has_no_replacement(monkeypatch):
    monkeypatch.setattr(ga, "SAFE_SYNONYMS", {})
    assert ga.fitting_words("uncle", "her", "was") == []


def test_names_function_words_negations_and_math_are_locked():
    text = "Then Maria said the plan would never work with $value$ inside"
    tokens = [(0, len(text), 1.0)]
    words = {text[start:end] for _, start, end in ga.word_pushes(text, tokens)}
    assert "plan" in words and "work" in words
    assert not words & {"Maria", "the", "never", "value"}


def test_a_word_pushing_toward_human_is_left_alone():
    assert ga.word_pushes("plan", [(0, 4, -1.0)]) == []


def test_no_gradient_engine_hands_the_text_back():
    class Plain:
        classifiers = ()

    assert ga.gradient_search("Some text.", Plain()) == ("Some text.", 0)
