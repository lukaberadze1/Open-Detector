"""The model-backed word search, with stand-ins for its models. No downloads."""

import pytest

from eval import contextual_attack as ca

TEXT = ("The quick method helps people finish large projects every single day. "
        "Nobody can deny that a big team will finish sooner than expected.")


@pytest.fixture(autouse=True)
def any_word_is_close(monkeypatch):
    monkeypatch.setattr(ca, "close_word", lambda old, new: True)


class Pass:
    def keeps(self, original, options, **_):
        return [True] * len(options)


class Refuse:
    def keeps(self, original, options, **_):
        return [False] * len(options)


def score_by(word):
    return lambda text: 0.2 if word in text else 0.99


def test_tokens_keep_their_places_and_articles_follow_the_word():
    tokens = ca.Tokens(TEXT)
    place = tokens.parts.index("big")
    new = ca.swapped(tokens.parts, place, "enormous")
    assert len(new) == len(tokens.parts) and "an enormous team" in ca.join(new)


def test_contractions_and_hyphens_are_not_editable():
    tokens = ca.Tokens("We don't use well-known tools here today at all, ever.")
    assert not tokens.editable(tokens.parts.index("don"))
    assert not tokens.editable(tokens.parts.index("well"))
    assert tokens.editable(tokens.parts.index("tools"))


def test_allowed_refuses_function_words_negations_and_other_forms():
    assert ca.allowed("helps", "assists", "it helps")
    assert not ca.allowed("helps", "helped", "it helps")
    assert not ca.allowed("big", "not", "a big team")
    assert not ca.allowed("big", "the", "a big team")
    assert not ca.allowed("big", "team", "a big team")


def test_structural_words_and_wrong_tenses_are_refused():
    assert not ca.allowed("after", "later", "soon after that")
    assert not ca.allowed("told", "tell", "she told him")
    assert ca.allowed("humidity", "moisture", "the humidity rose")


def run(monkeypatch, guard, score):
    monkeypatch.setattr(ca, "_fit", lambda *args: 100)
    place = ca.Tokens(TEXT).parts.index("large")
    return ca.beam_search(TEXT, [place], score, guard, proposer=lambda *args: ["massive", "huge"])


def test_the_search_keeps_the_lowest_scoring_swap(monkeypatch):
    result = run(monkeypatch, Pass(), score_by("huge"))
    assert "huge projects" in result.text and result.swaps == [("large", "huge")] and result.history[-1] < 0.3


def test_a_swap_the_guard_refuses_never_lands(monkeypatch):
    assert run(monkeypatch, Refuse(), score_by("huge")).text == TEXT


def test_the_guard_compares_with_the_original_sentence(monkeypatch):
    seen = []

    class Watch(Pass):
        def keeps(self, original, options, **_):
            seen.append(original)
            return [True] * len(options)

    monkeypatch.setattr(ca, "_fit", lambda *args: 100)
    parts = ca.Tokens(TEXT).parts
    places = [parts.index("large"), parts.index("projects")]
    ca.beam_search(TEXT, places, lambda text: 0.5 if "huge" in text else 0.9, Watch(),
                   proposer=lambda parts, i, bounds: ["huge"] if i == places[0] else ["tasks"])
    assert len(set(seen)) == 1 and "large projects" in seen[0]


def test_the_budget_stops_the_search(monkeypatch):
    monkeypatch.setattr(ca, "_fit", lambda *args: 100)
    calls = []

    def score(text):
        calls.append(text)
        return 0.9

    places = list(range(1, len(ca.Tokens(TEXT).parts), 2))
    ca.beam_search(TEXT, places, score, Pass(), proposer=lambda *args: ["massive", "huge"], budget=3)
    assert len(calls) <= 3


def test_dipper_gets_its_own_prompt():
    assert ca.paraphrase_prompt("dipper", "Hi.", "Before.") == "lexical = 40, order = 40 Before. <sent> Hi. </sent>"
    assert ca.paraphrase_prompt("t5", "Hi.") == "paraphrase: Hi."


def test_paraphrase_keeps_only_what_lowers_the_score_and_keeps_facts():
    text = "The plan cost 300 dollars across the whole region last year. The second part went very well for everybody."

    def rewrite(sentences, contexts):
        return [["The plan cost 500 dollars across the region last year.",
                 "Last year the plan cost 300 dollars across the region."]]

    result = ca.paraphrase_stage(text, lambda t: 0.5 if "Last year" in t else 0.9, Pass(), rewrite)
    assert "Last year the plan cost 300" in result.text and "500" not in result.text


def test_rewrite_quality_refuses_new_names_and_unfounded_words(monkeypatch):
    class Word:
        def __init__(self, text, pos):
            self.text, self.pos_, self.is_stop = text, pos, False

    docs = {"a": [Word("Paris", "PROPN"), Word("film", "NOUN")], "b": [Word("Rome", "PROPN"), Word("film", "NOUN")],
            "c": [Word("Paris", "PROPN"), Word("movie", "NOUN")], "d": [Word("Paris", "PROPN"), Word("read", "NOUN")]}
    monkeypatch.setattr(ca, "_spacy", lambda: docs.__getitem__)
    monkeypatch.setattr(ca, "close_word", lambda old, new: {old, new} == {"film", "movie"})
    quality = ca.Quality()
    assert not quality.rewrite_ok("a", "b")
    assert quality.rewrite_ok("a", "c")
    assert not quality.rewrite_ok("a", "d")


def test_locked_words_stay():
    assert not ca.allowed("days", "times", "seven days of mourning")
    assert not ca.allowed("keep", "remain", "we keep on working")
