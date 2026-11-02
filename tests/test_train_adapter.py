"""The adapter trainer's data and arithmetic, on a tiny stand-in model: no downloads."""

import json
from types import SimpleNamespace

import pytest
import torch

from eval.train_adapter import (
    IGNORE,
    REWRITE,
    adapter_row,
    answer_logprobs,
    batched,
    collate,
    dpo_loss,
    dpo_step,
    encode,
    evaluate,
    held_out,
    load_pairs,
    lora_step,
    modelfile,
    passages,
    run_training,
    training_rows,
    usable_paraphrase,
    with_references,
)


class Tokens:
    """A tokenizer with one id for each character, and a chat template that only marks where the answer starts."""

    eos_token = "$"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"{messages[0]['content']}>"

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(letter) % 32 for letter in text]}


class Tiny(torch.nn.Module):
    """A bigram model: enough to show a loss going down."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.table = torch.nn.Embedding(32, 32)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.table(input_ids))


def _entry(note, words, **more):
    return {"note": note, "text": " ".join(["word"] * words), **more}


def test_passages_join_neighbours_of_one_note_up_to_the_limit():
    found = passages([_entry("a", 30), _entry("a", 30), _entry("a", 150), _entry("b", 50), _entry("c", 10)])
    assert [(passage["note"], len(passage["text"].split())) for passage in found] == [("a", 60), ("a", 150), ("b", 50)]
    assert "\n\n" in found[0]["text"]


def test_passages_leave_out_what_was_superseded_and_keep_a_long_paragraph_whole():
    assert passages([_entry("a", 100, superseded="2026-W30")]) == []
    assert len(passages([_entry("a", 400)])[0]["text"].split()) == 400


def test_a_notes_passages_stay_on_one_side_of_the_split():
    assert held_out("note") == held_out("note")
    shares = [held_out(f"note {number}") for number in range(2000)]
    assert 0.07 < sum(shares) / len(shares) < 0.13


def test_a_paraphrase_must_change_the_passage_and_keep_its_length():
    never, always = (lambda original, rewrite: False), (lambda original, rewrite: True)
    original = " ".join(["word"] * 100)
    assert usable_paraphrase(original, " ".join(["other"] * 100), never)
    assert not usable_paraphrase(original, " ".join(["other"] * 100), always)
    assert not usable_paraphrase(original, "too short", never)
    assert not usable_paraphrase(original, " ".join(["other"] * 200), never)
    assert not usable_paraphrase(original, "", never)


def test_only_the_answer_is_counted_and_a_pair_too_long_is_left_out():
    row = encode(Tokens(), "ab", "cd")
    assert len(row["input_ids"]) == 6  # a, b, >, c, d, $
    assert row["labels"][:3] == [IGNORE] * 3 and row["labels"][3:] == row["input_ids"][3:]
    assert encode(Tokens(), "ab", "cd", max_tokens=5) is None


def test_rows_are_padded_and_padding_is_not_counted():
    batch = collate([{"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3]}, {"input_ids": [4], "labels": [4]}], pad=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE, 2, 3], [4, IGNORE, IGNORE]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_an_answers_log_probability_adds_up_only_its_tokens():
    logits = torch.zeros(1, 3, 4)  # every next token equally likely
    total, tokens = answer_logprobs(logits, torch.tensor([[IGNORE, IGNORE, 2]]))
    assert tokens.item() == 1
    assert total.item() == pytest.approx(torch.log(torch.tensor(0.25)).item())


def test_dpo_is_pleased_only_by_moving_toward_the_chosen_from_where_it_started():
    start = torch.tensor([-10.0])
    level = dpo_loss(start, start, start, start, beta=1.0)
    assert level.item() == pytest.approx(0.6931, abs=1e-3)  # no change: log 2
    assert dpo_loss(start + 5, start, start, start, beta=1.0) < level
    assert dpo_loss(start, start + 5, start, start, beta=1.0) > level
    # Already preferring the chosen side earns nothing: only the movement counts.
    assert dpo_loss(start + 5, start, start + 5, start, beta=1.0).item() == pytest.approx(level.item())


def test_pairs_become_what_each_method_trains_on():
    pairs = [{"neutral": "n", "text": "human", "rejected": "ai"}, {"neutral": "m", "text": "writer"}]
    seen = []

    def encoded(prompt, answer):
        seen.append((prompt, answer))
        return {"input_ids": [1, 2], "labels": [IGNORE, 2]}

    assert len(training_rows(pairs, "lora", encoded)) == 2
    assert seen[0] == (REWRITE.format(text="n"), "human")
    assert len(training_rows(pairs, "dpo", encoded)) == 1  # the second has nothing to reject


def test_batches_cover_every_row_and_shuffle_only_with_a_seed():
    assert batched([1, 2, 3], 2) == [[1, 2], [3]]
    assert sorted(row for batch in batched(list(range(9)), 4, seed=1) for row in batch) == list(range(9))
    assert batched(list(range(9)), 4, seed=1) == batched(list(range(9)), 4, seed=1)


def _rows(text="abcabcabc"):
    return [encode(Tokens(), "a", text) for _ in range(4)]


def test_lora_training_lowers_the_loss_and_saves_the_best():
    model = Tiny()
    step = lambda net, batch: lora_step(net, batch, 0, "cpu")  # noqa: E731
    before = evaluate(model, step, _rows(), 2)
    saves = []
    history = run_training(model, step, _rows(), _rows(), epochs=5, size=2, accumulate=1, learning_rate=0.05,
                           save=lambda: saves.append(1))
    assert history[-1]["held_out"] < before
    assert len(saves) >= 1 and [row["epoch"] for row in history] == [1, 2, 3, 4, 5]


def test_dpo_training_moves_the_model_toward_the_chosen_side():
    model = Tiny()
    rows = [{"chosen": encode(Tokens(), "a", "bcbc"), "rejected": encode(Tokens(), "a", "dede")} for _ in range(4)]
    scored = with_references(model, rows, 0, "cpu", 2)
    step = lambda net, batch: dpo_step(net, batch, 0, "cpu")  # noqa: E731
    before = evaluate(model, step, scored, 2)
    assert before == pytest.approx(0.6931, abs=1e-3)  # nothing has moved yet
    run_training(model, step, scored, [], epochs=5, size=2, accumulate=1, learning_rate=0.05, save=lambda: None)
    assert evaluate(model, step, scored, 2) < before


def test_what_an_adapter_writes_is_labelled_ai():
    row = adapter_row("new text", "the source", "voice", {"base": "base/model", "method": "lora"}, 40.0)
    assert row["label"] == 1 and row["group"] == "ai" and row["attack"] == "adapter+lora"
    assert row["model"] == "base/model+voice" and row["words_kept"] == 40.0
    again = adapter_row("other", "the source", "voice", {"base": "b", "method": "dpo"})
    assert row["group_key"] == again["group_key"]


def test_the_adapter_stays_out_of_the_app():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    build = (root / "deploy/macos/build_app.py").read_text(encoding="utf-8")
    assert "train_adapter" not in build and "data/adapters" not in build
    assert not any("train_adapter" in path.read_text(encoding="utf-8") for path in (root / "ui").glob("*.py"))
    assert "train_adapter" not in (root / "app.py").read_text(encoding="utf-8")


def test_pairs_load_from_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    assert load_pairs(path) == []
    path.write_text(json.dumps({"id": "a"}) + "\n\n", encoding="utf-8")
    assert load_pairs(path) == [{"id": "a"}]


def test_the_modelfile_builds_from_its_own_folder():
    assert modelfile(0.8) == "FROM .\nPARAMETER temperature 0.8\n"
