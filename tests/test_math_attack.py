"""The no-model attack: repeatable, keeps facts and protected spans, and moves sentence lengths."""

import numpy as np
import pytest

from eval.math_attack import (
    CachedScorer,
    _cv,
    changed_passages,
    compression_change,
    contract,
    documents,
    drop_markers,
    fix_articles,
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
    assert "and this" in new and "This can cause" in new


def test_compression_change_reports_both_sides():
    assert set(compression_change("some text here", "some other text")) >= {"zippy_lzma_before", "zippy_lzma_after"}


def test_every_paragraph_keeps_its_own_text():
    first, second = "One short line here. Two short words.", "Third line is different. Fourth one too."
    new = vary_lengths(f"{first}\n\n{second}", 5.0)
    assert "Third" in new and "Fourth" in new and new.count("One") == 1


def test_the_whole_pipeline_keeps_every_paragraph():
    new, _ = math_attack("First sentence here.  Second sentence here.\n\nThird one stays.")
    assert "Third one stays." in new and new.count("First sentence") == 1


def test_code_urls_and_tags_stay():
    text = ("Use `in order to` here. However, see https://example.com/in-order-to now. <a href='in order to'>x</a>\n\n"
            "    return x  # in order to\n\nMore text.")
    new, _ = math_attack(text)
    assert "`in order to`" in new and "/in-order-to" in new and "'in order to'" in new
    assert "    return x  # in order to" in new


def test_contractions_only_where_they_read():
    assert contract("I know what it is. Yes it is.") == "I know what it is. Yes it is."
    assert contract("It will not work, but it is fine.") == "It won't work, but it's fine."
    assert contract("THIS IS LOUD, IT IS.") == "THIS IS LOUD, IT IS."


def test_articles_follow_the_swap():
    assert fix_articles("an clearly good and a extremely old") == "a clearly good and an extremely old"
    assert fix_articles("a unique plan, an hour, a one") == "a unique plan, an hour, a one"


def test_lexicon_keeps_the_word_form():
    assert "shows" in fitted_swaps("The film showcases love.")
    assert "used" in fitted_swaps("They leveraged it.")


def test_unsafe_senses_are_gone():
    assert "earlier" not in fitted_swaps("He is 70 years old.")
    assert "despite that" not in fitted_swaps("It works despite the fact that it failed.").lower()
    assert "goal" not in fitted_swaps("We are at a point in time.")


def test_an_abbreviation_is_not_a_sentence_start():
    new = fitted_swaps("Moreover, Dr. Smith, e.g. the expert, left.")
    assert "e.g. the expert" in new and new.startswith("Dr.")


def test_dropped_opener_leaves_a_capital():
    new = fitted_swaps('She said, "Yes." Moreover, he agreed. "Furthermore, we go," he said.')
    assert '"Yes." He agreed' in new and '"We go,"' in new


def test_a_note_phrase_mid_sentence_is_not_mangled():
    new, _ = math_attack("The expert said it is important to note that costs rose.", ("rules",))
    assert "said note that" not in new


def test_sanitize_keeps_emoji_whole_and_plain_line_ends():
    family = "\U0001f468\u200d\U0001f469"
    assert sanitize(f"a{family}b\r\nc\u200bd") == f"a{family}b\ncd"


def test_loud_markers_stay():
    assert plain_markers("HOWEVER, THIS IS LOUD.") == "HOWEVER, THIS IS LOUD."


def test_cached_scorer_matches_the_detector_and_only_runs_new_windows():
    from opendetector.combiner import Combiner
    from opendetector.config import DetectorConfig
    from opendetector.detector import Detector

    class Stub:
        name = "desklib"
        calls = 0

        def window_logits(self, texts):
            Stub.calls += len(texts)
            return np.array([float(len(text) % 7) for text in texts])

    combiner = Combiner(features=["desklib_mean", "desklib_max"], medians=[0.0, 0.0], mean=[0.0, 0.0], scale=[1.0, 1.0],
                        coef=[1.0, 1.0], intercept=0.0)
    detector = Detector([Stub()], combiner=combiner, config=DetectorConfig.for_profile("slim"))
    paragraph = " ".join(f"Sentence number {word} has a few plain words in it." for word in "abcdefghij" * 12)
    scorer = CachedScorer(detector)
    assert scorer(paragraph) == pytest.approx(detector.score(paragraph))
    first = scorer.runs
    assert first > 2
    assert scorer(paragraph) == pytest.approx(detector.score(paragraph)) and scorer.runs == first
    changed = paragraph.replace("number a", "number z", 1)
    assert scorer(changed) == pytest.approx(detector.score(changed)) and scorer.runs - first < first


def test_documents_join_a_groups_ai_blocks_only():
    rows = [{"id": "a", "text": "One.", "label": 1, "group_key": "k"},
            {"id": "b", "text": "Human.", "label": 0, "group_key": "k"},
            {"id": "c", "text": "Two.", "label": 1, "group_key": "k"},
            {"id": "c2", "text": "Two.", "label": 1, "group_key": "k"},
            {"id": "x-draft2", "text": "Other version.", "label": 1, "group_key": "k"},
            {"id": "d", "text": "Alone.", "label": 1}]
    assert [row["text"] for row in documents(rows)] == ["One.\n\nTwo.", "Alone."]


def test_changed_passages_show_the_change_in_context():
    assert changed_passages("a b c d e", "a b X d e", context=1) == [("b c d", "b X d")]
    assert changed_passages("same", "same") == []


def test_a_list_block_is_left_alone():
    text = "- one item\n- another item"
    assert vary_lengths(text, 5.0) == text
