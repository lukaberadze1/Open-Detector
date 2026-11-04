"""The no-model attack: repeatable, keeps facts and protected spans, and moves sentence lengths."""

from eval.math_attack import (
    _cv,
    compression_change,
    contract,
    drop_markers,
    fitted_swaps,
    guided_attack,
    math_attack,
    plain_markers,
    sanitize,
    vary_lengths,
)
from opendetector.segment import count_words, split_sentences

EVEN = ("Furthermore, the committee reviewed the 2024 budget in great detail last week, and the members did not find "
        "any errors in it. The plan is short. It is clear. Moreover, the staff prepared a long summary for the board, "
        "and they sent it out to everyone before the meeting began on Monday.")


def _spread(text):
    return _cv([count_words(text[s.start:s.end]) for s in split_sentences(text)])


def test_same_text_gives_the_same_result():
    assert math_attack(EVEN) == math_attack(EVEN)


def test_sentence_opening_markers_are_rotated():
    new, applied = math_attack(EVEN, ("rules",))
    assert "Additionally, the committee" in new and "Furthermore, the staff" in new
    assert applied == ["rules"]


def test_lengths_spread_and_paragraphs_stay():
    text = f"{EVEN}\n\n{EVEN}"
    new = vary_lengths(text, 5.0)
    assert new.count("\n\n") == 1
    assert _spread(new) > _spread(text)


def test_numbers_negations_and_protected_spans_survive():
    text = EVEN + r" We use $x_1=2$ and \cite{work2024} here."
    new, _ = math_attack(text)
    assert "2024" in new and r"$x_1=2$" in new and r"\cite{work2024}" in new
    assert "not" in new or "n't" in new


def test_guided_keeps_only_edits_that_lower_the_score():
    new, kept, history = guided_attack(EVEN, lambda text: 0.9 - 0.1 * ("Furthermore" not in text), budget=20)
    assert "Furthermore" not in new and kept[0] == "plain"
    assert history == sorted(history, reverse=True)
    unchanged, kept, _ = guided_attack(EVEN, lambda text: 0.9 + 0.01 * (text != EVEN), budget=20)
    assert unchanged == EVEN and kept == []


def test_guided_stops_at_the_budget_and_can_leave_swaps_out():
    calls = []
    guided_attack(EVEN, lambda text: calls.append(text) or 0.9, budget=3)
    assert len(calls) == 3
    _, kept, _ = guided_attack(EVEN, lambda text: 1 - len(calls.__class__(text)) / 1e4, budget=50, swaps=False)
    assert "swaps" not in kept


def test_drop_markers_and_contract():
    assert drop_markers("Furthermore, the plan works. It is fine.") == "The plan works. It is fine."
    assert contract("It is what they do not want.") == "It's what they don't want."


def test_fitted_swaps_refuse_what_human_writing_never_says():
    text = "Also, make sure you stay until the end of the day."
    assert "create sure" not in fitted_swaps(text)
    assert "the conclude of" not in fitted_swaps(text)


def test_plain_markers_replace_rather_than_rotate():
    assert plain_markers("However, it works. Moreover, it is fast.") == "But it works. Also, it is fast."


def test_one_harmful_sentence_does_not_veto_the_others():
    text = "It is late. They do not know. We are here. It is cold."
    new, _, _ = guided_attack(text, lambda t: 0.9 - 0.01 * ("It's cold" in t) + 0.05 * ("don't" in t), budget=30)
    assert "It's cold" in new and "don't" not in new


def test_sanitize_strips_hidden_characters_and_keeps_layout():
    assert sanitize("a​b﻿  c\n\nd⁠") == "ab  c\n\nd"
    assert sanitize("é") == "é"


def test_guided_cleans_its_input_even_with_no_edit_kept():
    new, kept, _ = guided_attack("Plain​ words.", lambda text: 0.5, budget=5)
    assert new == "Plain words." and kept == []


def test_mined_rules_and_new_markers_apply():
    new, _ = math_attack("We use it as well as this. It is important to note that this can lead to errors.", ("rules",))
    assert "and this" in new and "can cause" in new and "Note that" in new


def test_compression_change_reports_both_sides():
    assert set(compression_change("some text here", "some other text")) >= {"zippy_lzma_before", "zippy_lzma_after"}


def test_a_list_block_is_left_alone():
    text = "- one item\n- another item"
    assert vary_lengths(text, 5.0) == text
