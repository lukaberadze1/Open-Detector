"""The attacks on finished text: stand-in models only."""

import pytest

from eval.text_attacks import (
    CLAUSES,
    attack_label,
    attack_name,
    reorder_clauses,
    round_trip,
    run_attacks,
    turned_around,
)


def _translator(changed=lambda text: f"{text} again"):
    """A stand-in model: odd requests go there and come back unchanged, even ones back through `changed`."""
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        tagged = prompt.split("\n\n", 1)[1]
        if len(prompts) % 2:
            return tagged
        import re

        return "\n\n".join(f"<p{number}>{changed(text)}</p{number}>"
                           for number, text in re.findall(r"<p(\d+)>(.*?)</p\1>", tagged, re.S))

    ask.prompts = prompts
    return ask


@pytest.mark.parametrize(("sentence", "turned"), [
    ("Because the river rose quickly, the farmers left their land.",
     "The farmers left their land because the river rose quickly."),
    ("The farmers left their land because the river rose quickly.",
     "Because the river rose quickly, the farmers left their land."),
    ("I stayed at home although the storm had already passed.",
     "Although the storm had already passed, I stayed at home."),
])
def test_a_subordinate_clause_moves_to_the_other_side(sentence, turned):
    assert turned_around(sentence) == turned


@pytest.mark.parametrize("sentence", [
    "Marcus left the city because the war had begun.",  # its first word may be a name, so its capital can't go
    "They kept on working even if the rain came down.",  # "even if" is one phrase
    "After dinner, the whole family went to bed.",  # a phrase, not a clause
    'She said "stay here" because the road was closed.',  # quotations stay as they are
    "The farmers left, and they sold the land, because it failed.",  # commas elsewhere
    "The farmers stayed on their land.",  # nothing to move
    "I really wonder if it will rain today.",  # the clause is what is wondered, not when
    "The officers asked them if the road was open.",
    "That was the year when the old king died.",  # the clause says which year
])
def test_a_sentence_that_would_not_survive_stays_as_it_is(sentence):
    assert turned_around(sentence) is None


TURNABLE = ("The farmers left their land because the river rose quickly. It rained for days.\n\n"
            "Although the storm had passed, the roads stayed closed. Nobody came.")


def test_reordering_changes_no_words_and_keeps_paragraphs():
    text, changed = reorder_clauses(TURNABLE, share=1.0)
    assert changed == 2
    assert text == ("Because the river rose quickly, the farmers left their land. It rained for days.\n\n"
                    "The roads stayed closed although the storm had passed. Nobody came.")


def test_reordering_is_the_same_each_time_and_its_share_can_be_none():
    assert reorder_clauses(TURNABLE, seed="a") == reorder_clauses(TURNABLE, seed="a")
    assert reorder_clauses(TURNABLE, share=0.0) == (TURNABLE, 0)


def test_a_round_trip_asks_there_and_back_and_keeps_each_paragraph_in_its_place():
    ask = _translator()
    result, changed = round_trip(["The first of 3 paragraphs.", "", "The last paragraph here."], ask, "German")
    assert result == ["The first of 3 paragraphs. again", "", "The last paragraph here. again"]
    assert changed == 2
    assert "into German" in ask.prompts[0] and "into English" in ask.prompts[1]


def test_a_paragraph_that_loses_a_number_or_its_length_stays_as_it_was():
    lost = _translator(lambda text: text.replace("300", "many"))
    fleet = ["Xerxes sent 300 ships past the strait."]
    assert round_trip(fleet, lost) == (fleet, 0)
    short = _translator(lambda text: "Ships.")
    assert round_trip(["Xerxes sent his ships past the strait."], short)[1] == 0


def test_a_paragraph_a_reply_leaves_out_stays_as_it_was():
    assert round_trip(["Xerxes sent his ships past."], lambda prompt: "Sorry, I can't.") == (
        ["Xerxes sent his ships past."], 0)


def test_long_texts_go_in_several_requests():
    ask = _translator()
    paragraphs = [" ".join(["word"] * 300) for _ in range(3)]
    assert round_trip(paragraphs, ask, words=400)[1] == 3
    assert len(ask.prompts) == 6


def test_attacks_run_in_turn_and_count_what_each_changed():
    pieces = ["The farmers left their land because the river rose quickly.", "Nothing moves here at all."]
    attacks = [attack_name("German"), CLAUSES]
    result, changes = run_attacks(pieces, attacks, _translator())
    assert changes[attack_name("German")] == 2 and set(changes) == set(attacks)
    assert len(result) == 2
    assert attack_label(attacks) == "+roundtrip-german+clauses"


def test_an_attack_that_needs_a_model_says_so_and_an_unknown_one_is_refused():
    with pytest.raises(ValueError, match="needs a model"):
        run_attacks(["text"], [attack_name("German")])
    with pytest.raises(ValueError, match="unknown attack"):
        run_attacks(["text"], ["shuffle"])
