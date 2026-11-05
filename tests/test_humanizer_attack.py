import json
import random

import pandas as pd

from eval import humanizer_attack as attack
from eval.redteam_batch import copied_row
from opendetector.segment import count_words

TEXT = ("Furthermore, the committee's report demonstrates that the new policy can't work without funding. It is "
        "important to note that 120 towns utilize the scheme. However, the data show a robust trend. ") * 8


def test_academic_control_does_what_the_app_does_bugs_included():
    out = attack.academic_control("The committee's plan can't work. It won't last.", random.Random(1))
    assert "committee is plan" in out, "its possessive bug is part of what people get"
    assert "cannot work" in out and "will not last" in out


def test_stealth_rules_change_the_words_and_decide_by_hash():
    out = attack.stealth_rules(TEXT, random.Random(1))
    assert out == attack.stealth_rules(TEXT, random.Random(99))
    assert "demonstrates" not in out and "utilize" not in out and "Furthermore" not in out
    assert "120" in out


def test_chunks_keep_whole_sentences_near_the_size():
    pieces = attack.chunks(TEXT, 30)
    assert len(pieces) > 2 and all(count_words(piece) <= 45 for piece in pieces)
    assert " ".join(pieces).split() == TEXT.split()


def test_tidied_repairs_a_sampled_output():
    assert attack.tidied("the the plan worksWell , really,, ok") == "the plan works Well, really, ok."


def test_best_by_detector_trades_score_against_fidelity_and_drops_what_loses_facts():
    source = "The fleet of 300 ships sailed from Athens in spring with the allies aboard."
    faithful_high = "The fleet of 300 ships sailed from Athens in the spring with allies aboard."
    faithful_low = "A fleet of 300 ships left Athens that spring carrying the allies on board."
    wrong = "A wholly different tale of nothing at all, and with no ships in it."
    scores = {faithful_high: 0.9, faithful_low: 0.2, wrong: 0.0}
    assert attack.best_by_detector(source, [faithful_high, faithful_low, wrong],
                                   lambda texts: [scores[text] for text in texts]) == faithful_low


def test_humanize_bart_saves_the_first_sample_and_the_winner_chunk_by_chunk():
    def write(chunk):
        return [f"{chunk} first", f"{chunk} second"]

    out = attack.humanize_bart(TEXT, write, lambda texts: [0.9 if text.endswith("first") else 0.1 for text in texts])
    assert out["sampled"].count("first") == len(attack.chunks(TEXT)) and "second" not in out["sampled"]
    assert "first" not in out[f"best{attack.BART_SAMPLES}"]


def test_humanize_gemma_goes_up_the_ladder_and_ranks_by_style():
    temperatures = []

    def ask(text, temperature):
        temperatures.append(temperature)
        if temperature == 0.85:
            return "Furthermore, moreover, additionally the plan goes on and on the same way it did."
        return "The plan goes on. And on the same way it did before, as it always has done here, on and on, it's so."

    out = attack.humanize_gemma("The plan goes on and on the same way it did.", ask, lambda texts: [0.5] * len(texts))
    assert temperatures == list(attack.GEMMA_TEMPERATURES)
    assert out["single"].startswith("Furthermore")
    assert out["best4"].startswith("The plan goes on.")


def _frame() -> pd.DataFrame:
    body = "A plain sentence about the topic at hand goes here for a while. " * 20
    rows = [{"id": f"{domain}-{number}", "text": f"{domain} {number} {body}", "label": 1, "attack": "none",
             "domain": domain, "model": "gpt", "group_key": f"{domain}-{number}"}
            for domain in ("news", "wiki") for number in range(5)]
    base = {"domain": "news", "model": "p"}
    rows += [{**base, "id": "human", "text": body, "label": 0, "attack": "none", "group_key": "h"},
             {**base, "id": "para", "text": body, "label": 1, "attack": "paraphrase", "group_key": "p"},
             {**base, "id": "short", "text": "Too short.", "label": 1, "attack": "none", "group_key": "s"}]
    return pd.DataFrame(rows)


def test_pick_sources_takes_ai_text_no_combiner_saw_from_each_domain_in_turn():
    chosen = attack.pick_sources(_frame(), 4, used={"news-0"})
    ids = {row["id"] for row in chosen}
    assert not ids & {"human", "para", "short", "news-0"}
    assert sorted(row["domain"] for row in chosen) == ["news", "news", "wiki", "wiki"]
    assert attack.pick_sources(_frame(), 4, used=set()) == attack.pick_sources(_frame(), 4, used=set())


def _source() -> dict:
    return {"id": "mage-1", "text": TEXT, "domain": "news", "model": "gpt", "group_key": "g1"}


def test_a_saved_row_stays_ai_however_little_it_changed():
    row = attack.sample_row(_source(), "stealth-bart", "best8", TEXT, "m")
    assert row["label"] == 1 and row["group_key"] == "g1" and row["attack"] == "humanizer+stealth-bart+best8"
    assert row["words_retained"] == 100 and "words_kept" not in row
    assert not copied_row(row), "its source is AI text, not a writer's own"


def test_generate_saves_each_output_resumes_and_passes_over_what_cant_start(tmp_path, capsys, monkeypatch):
    out = tmp_path / "out.jsonl"
    args = type("Args", (), {"seed": 1})()
    calls = []

    def writer_for(name, _):
        if name == "stealth-bart":
            raise OSError("no such model")

        def write(text, rng):
            calls.append(name)
            return {"": text.replace("the", "a")}
        return write, "m", lambda: None

    monkeypatch.setattr(attack, "writer_for", writer_for)
    sources = [_source(), {**_source(), "id": "mage-2"}]
    assert attack.generate(sources, ["stealth-bart", "stealth-rules"], out, args) == 1
    assert attack.generate(sources, ["stealth-rules"], out, args) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert calls == ["stealth-rules"] * 2, "a second run has nothing left to do"
    assert [row["source_id"] for row in rows] == ["mage-1", "mage-2"]
    assert "stealth-bart: can't start" in capsys.readouterr().err


def test_report_compares_each_humanizer_with_the_same_texts_before():
    source = {**_source(), "ai_probability": 0.9, "verdict": "likely_ai"}
    rows = [{**attack.sample_row(_source(), "cive-bart", "best8", TEXT, "m"), "ai_probability": 0.2,
             "verdict": "likely_human"},
            {**attack.sample_row(_source(), "academic-control", "", TEXT, "m"), "ai_probability": 0.99,
             "verdict": "likely_ai"}]
    text = attack.report(rows, [source], "slim")
    lines = text.splitlines()
    first = next(line for line in lines if line.startswith("| cive-bart+best8"))
    assert "| 90 | 20 |" in first and "100% (" in first
    assert lines.index(first) < next(i for i, line in enumerate(lines) if line.startswith("| academic-control"))


def test_write_report_scores_the_outputs_and_their_sources_together(tmp_path, monkeypatch):
    from eval import redteam_report

    out, sources, path = tmp_path / "out.jsonl", tmp_path / "sources.jsonl", tmp_path / "report.md"
    assert attack.write_report(out, sources, path, "slim", tmp_path / "c.jsonl").startswith("No output")
    out.write_text(json.dumps(attack.sample_row(_source(), "stealth-rules", "", TEXT, "m")) + "\n")
    sources.write_text(json.dumps(_source()) + "\n")

    def score(rows, profile, cache):
        assert profile == "slim" and cache == tmp_path / "c.jsonl"
        return [{**row, "verdict": "likely_ai", "ai_probability": 0.25 if "humanizer" in row["id"] else 0.75}
                for row in rows]

    monkeypatch.setattr(redteam_report, "score", score)
    assert attack.write_report(out, sources, path, "slim", tmp_path / "c.jsonl") == f"Wrote {path}."
    assert "| stealth-rules | 1 | 75 | 25 |" in path.read_text()


def test_bart_samples_without_beams_since_these_models_default_to_four(monkeypatch):
    import torch
    import transformers

    asked = {}

    class Tokens(dict):
        def __init__(self):
            super().__init__(input_ids=torch.zeros((1, 5), dtype=torch.long))

    class Tokenizer:
        def __call__(self, *_, **__):
            return Tokens()

        def batch_decode(self, output, **_):
            return ["A sentence."] * len(output)

    class Model:
        def eval(self):
            return self

        def generate(self, **options):
            asked.update(options)
            return [0] * options["num_return_sequences"]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_, **__: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", lambda *_, **__: Model())
    assert len(attack.bart_writer("m", samples=8)("text")) == 8
    assert asked["num_beams"] == 1 and asked["num_return_sequences"] == 8


def test_bart_writer_samples_even_when_the_models_config_asks_for_beam_search(monkeypatch):
    import torch
    import transformers
    from transformers import BartConfig, BartForConditionalGeneration

    class Tokenizer:
        def __call__(self, text, return_tensors, truncation, max_length):
            ids = torch.tensor([[1, *range(4, 4 + len(text.split())), 2]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def batch_decode(self, output, skip_special_tokens):
            return [" ".join(f"word{int(token)}" for token in row if int(token) > 3) for row in output]

    model = BartForConditionalGeneration(BartConfig(
        vocab_size=60, d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=16, decoder_ffn_dim=16, max_position_embeddings=128, bos_token_id=1,
        eos_token_id=2, pad_token_id=0, decoder_start_token_id=2))
    model.generation_config.num_beams = 4  # as both humanizers' generation_config.json do
    loaded = {}

    def from_pretrained(_, use_safetensors=None):
        loaded["safetensors"] = use_safetensors
        return model

    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _: Tokenizer())
    assert len(attack.bart_writer("tiny", samples=8)("The fleet sailed from Athens in spring.")) == 8
    assert loaded["safetensors"] is True
