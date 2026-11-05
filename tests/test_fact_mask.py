from types import SimpleNamespace

from eval.fact_mask import FactMask, article_for
from eval.redteam_batch import keeps_facts, meaning_kept, rewrite_group

PASSAGE = ("Although Themistocles was outnumbered, the Greek fleet beat Xerxes at Salamis in 480 BC, sinking 200 "
           "ships. Themistocles had spent 40% of the budget on triremes (Herodotus, 2003).")


def test_hide_holds_names_and_numbers_and_restore_gives_them_back():
    mask = FactMask(PASSAGE)
    hidden = mask.hide(PASSAGE)
    for value in ("Themistocles", "Greek", "Xerxes", "Salamis", "480 BC", "200", "40%", "(Herodotus, 2003)"):
        assert value in mask.values, value
        assert value not in hidden
    assert "Although" in hidden, "a capital at a sentence's start says nothing about a name"
    assert mask.restore(hidden) == PASSAGE


def test_a_name_at_a_sentence_start_is_held_when_the_document_shows_it_is_one():
    text = "Xerxes watched from the shore. The fleet of Xerxes burned."
    assert FactMask(text).hide(text).startswith("[F1] watched")
    alone = "Xerxes watched from the shore."
    assert FactMask(alone).hide(alone) == alone, "with nothing to go on it stays, as keeps_facts reads it"


def test_one_value_has_one_placeholder_across_passages():
    mask = FactMask("Met Salamis.", "At Salamis again.")
    assert mask.hide("Near Salamis it rained.") == mask.hide("Far from Salamis it rained.").replace("Far from", "Near")
    assert mask.values == ["Salamis"]


def test_things_that_only_look_like_facts_stay_and_other_kinds_are_held():
    mask = FactMask("x")
    assert mask.hide("It cost $5 and then $6 more.") == "It cost [F1] and then [F2] more."
    assert mask.values == ["$5", "$6"]
    assert "https://a.io/b" in (m := FactMask("x")).restore(m.hide("See https://a.io/b.")) and m.values == ["https://a.io/b"]
    assert (m := FactMask("x")).hide("Set `x = 1` and $a+b$ see [3].") == "Set [F1] and [F2] see [F3]."
    source = "A literal [F1] and 7 more."
    assert (m := FactMask(source)).restore(m.hide(source)) == source


def test_restore_fits_the_article_to_what_went_back():
    mask = FactMask("x")
    mask.hide("in the Greek fleet by the Athenian and 18 more at Oxford University")
    assert article_for("Athenian") == "an" and article_for("Greek") == "a"
    assert article_for("18") == "an" and article_for("University") == "a" and article_for("FBI") == "an"
    athenian = mask.values.index("Athenian") + 1
    greek = mask.values.index("Greek") + 1
    assert mask.restore(f"He was a [F{athenian}], not an [F{greek}].") == "He was an Athenian, not a Greek."
    assert mask.restore(f"An [F{greek}] won.") == "A Greek won."


def test_lost_reports_dropped_and_invented_placeholders_and_accepts_the_value_written_out():
    mask = FactMask("x")
    hidden = mask.hide("In 480 the fleet met Xerxes there.")
    assert mask.lost(hidden, "In [F1] the ships met [F2].") == []
    assert mask.lost(hidden, "In 480 the ships met [F2].") == [], "writing the value out keeps it just the same"
    assert mask.lost(hidden, "The ships met [F2].") == ["480"]
    assert mask.lost(hidden, "In [F1] the ships met [F2] and [F9].") == ["[F9]"]
    assert mask.restore("met \\[F2\\] and [ f1 ]") == "met Xerxes and 480"


def _plan(**fields) -> SimpleNamespace:
    return SimpleNamespace(prompt="", context_text=PASSAGE, blocks=[], answer_slots=[], match_existing_lengths=True,
                           answer_words=100, structure="free", shapes=[], **fields)


def _job(key: str = "a") -> dict:
    return {"kind": "text_block", "label": key, "question": "", "previous": PASSAGE, "key": key}


def _block_reply(text: str) -> str:
    import json

    return json.dumps({"blocks": [{"id": "B1", "text": text}]})


def test_rewrite_group_sends_placeholders_and_gives_back_the_facts():
    prompts = []

    def ask(prompt, jobs):
        prompts.append((prompt, jobs))
        mask = FactMask(PASSAGE)
        hidden = mask.hide(PASSAGE)
        # A reply using each placeholder of the block, in new words around them.
        import re
        tokens = list(dict.fromkeys(re.findall(r"\[F\d+\]", hidden)))
        return _block_reply("Quite a story, this one: " + " then ".join(tokens) + " and the rest of it went on.")

    job = _job()
    written, alone = rewrite_group(_plan(mask_facts=True), [job], ask, str)
    block_text = prompts[0][0].split("Existing text to regenerate:")[1]
    assert "Xerxes" not in block_text.split("They stand for:")[0], "the block itself goes out masked"
    assert "= Xerxes" in prompts[0][0], "the legend says what each stands for"
    assert "[F" in prompts[0][1][0]["previous"]
    assert job["previous"] == PASSAGE, "the caller's job isn't changed"
    assert alone == []
    for value in ("Themistocles", "Xerxes", "480 BC", "40%"):
        assert value in written["a"]
    assert "[F" not in written["a"]
    assert keeps_facts(PASSAGE, written["a"])


def test_rewrite_group_asks_again_for_a_lost_placeholder_then_leaves_the_block_to_be_written_alone():
    asked = []

    def ask(prompt, jobs):
        asked.append(prompt)
        return _block_reply("A rewrite that has dropped all of the facts it was given to keep in place, sadly.")

    job = _job()
    written, alone = rewrite_group(_plan(mask_facts=True), [job], ask, str)
    assert len(asked) == 2 and "left placeholders out" in asked[1]
    assert written == {}
    assert alone == [job], "the original job, with its facts in it, not the masked one"


def test_rewrite_group_keeps_a_first_reply_when_only_the_second_loses_a_placeholder():
    import re

    hidden = FactMask(PASSAGE).hide(PASSAGE)
    tokens = list(dict.fromkeys(re.findall(r"\[F\d+\]", hidden)))
    replies = iter([_block_reply(hidden), _block_reply("Now nothing is left of any of it, which is no use at all.")])
    written, alone = rewrite_group(_plan(mask_facts=True), [_job()], lambda prompt, jobs: next(replies), str)
    assert alone == [] and tokens and written["a"] == PASSAGE, "a copy is asked again; its text stands if that fails"


def test_without_masking_nothing_changes():
    seen = []

    def ask(prompt, jobs):
        seen.append(prompt)
        return _block_reply("Themistocles and the allies won at Salamis, though in other words than these.")

    written, _ = rewrite_group(_plan(), [_job()], ask, str)
    assert "[F1]" not in seen[0] and "They stand for" not in seen[0]
    assert written["a"].startswith("Themistocles")


def test_a_restored_revision_passes_the_meaning_check():
    mask = FactMask(PASSAGE)
    hidden = mask.hide(PASSAGE)
    assert meaning_kept(PASSAGE, mask.restore(hidden))
