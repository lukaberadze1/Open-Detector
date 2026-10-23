"""Binoculars chunking, checked with tiny context-free language models: their logits depend
only on the current token, so scoring in overlapping chunks must equal one long pass."""

import re
from types import SimpleNamespace

import numpy as np
import torch

from opendetector.engines.binoculars import BinocularsEngine

VOCAB = 50


class FakeTokenizer:
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        words = list(re.finditer(r"\S+", text))
        return {"input_ids": [sum(map(ord, w.group())) % VOCAB for w in words],
                "offset_mapping": [(w.start(), w.end()) for w in words]}


class FakeLM(torch.nn.Module):
    """Hidden state = one-hot of the current token; the output head maps token i to table[i]."""

    def __init__(self, seed):
        super().__init__()
        self.table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(seed))
        self.head = torch.nn.Linear(VOCAB, VOCAB, bias=False)
        self.head.weight.data = self.table.T.clone()
        self.config = SimpleNamespace(vocab_size=VOCAB)

    def base_model(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, VOCAB).float())

    def get_output_embeddings(self):
        return self.head


def engine(max_tokens, stride):
    eng = BinocularsEngine.__new__(BinocularsEngine)
    eng.device, eng.max_tokens, eng.stride, eng.vocab = "cpu", max_tokens, stride, VOCAB
    eng.tokenizer, eng.observer, eng.performer = FakeTokenizer(), FakeLM(1), FakeLM(2)
    return eng


TEXT = " ".join(f"word{i % 37}" for i in range(700))


def test_chunked_scoring_matches_single_pass():
    _, nll_a, xent_a = engine(10_000, 5_000).token_stats(TEXT)
    _, nll_b, xent_b = engine(64, 32).token_stats(TEXT)
    assert np.isnan(nll_a[0]) and np.isnan(nll_b[0])
    assert np.isfinite(nll_b[1:]).all() and np.isfinite(xent_b[1:]).all()
    np.testing.assert_allclose(nll_a, nll_b, rtol=1e-5)
    np.testing.assert_allclose(xent_a, xent_b, rtol=1e-5)


def test_token_stats_match_definitions():
    eng = engine(64, 32)
    _, nll, xent = eng.token_stats(TEXT)
    ids = FakeTokenizer()(TEXT)["input_ids"]
    prev, cur = ids[4], ids[5]
    log_p = torch.log_softmax(eng.performer.table[prev], -1)
    p_obs = torch.softmax(eng.observer.table[prev], -1)
    assert abs(nll[5] + log_p[cur].item()) < 1e-5
    assert abs(xent[5] + (p_obs * log_p).sum().item()) < 1e-5


def test_span_scores_are_ratios_over_tokens_in_span():
    eng = engine(64, 32)
    offsets, nll, xent = eng.token_stats(TEXT)
    a, b = 100, 2000
    mids = offsets.mean(1)
    sel = (mids >= a) & (mids < b) & ~np.isnan(nll)
    expected = nll[sel].sum() / xent[sel].sum()
    scores = eng.span_scores(TEXT, [(a, b), (0, 5)])
    assert scores[0] == np.float64(expected)
    assert np.isnan(scores[1]), "spans with too few tokens get NaN"
