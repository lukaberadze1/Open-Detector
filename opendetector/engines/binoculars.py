"""Binoculars zero-shot detector (Hans et al., 2024).

score = log-perplexity of the text under the performer model
        / cross-entropy between the observer's and performer's next-token distributions

Machine text is unsurprising to the performer *relative to* how unsurprising the observer
expects any text to be, so it scores low. Dividing by the cross-entropy is what keeps this
from flagging plain, simple prose the way raw perplexity does (the failure Liang et al.
found on non-native essays).

Token statistics are computed once per document with overlapping chunks, then summed over
any character span, so sentence windows get scores with full left context.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MIN_TOKENS = 8  # spans with fewer scored tokens get NaN
# Positions per output-head block. Full logits for a 512-token chunk with a ~150k vocabulary
# are ~300 MB per model; running the head on small blocks keeps Binoculars inside 8 GB.
_BLOCK = 32


class BinocularsEngine:
    name = "bino"

    def __init__(self, observer_id: str, performer_id: str, device: str, dtype: torch.dtype = torch.float32,
                 max_tokens: int = 512, stride: int = 256):
        self.device = device
        self.max_tokens = max_tokens
        self.stride = stride
        self.tokenizer = AutoTokenizer.from_pretrained(observer_id)
        if self.tokenizer.get_vocab() != AutoTokenizer.from_pretrained(performer_id).get_vocab():
            raise ValueError("Binoculars observer and performer must share a tokenizer")
        self.observer = AutoModelForCausalLM.from_pretrained(observer_id, dtype=dtype).to(device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer_id, dtype=dtype).to(device).eval()
        self.vocab = min(self.observer.config.vocab_size, self.performer.config.vocab_size)

    @torch.inference_mode()
    def token_stats(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-token (char offsets, performer NLL, observer→performer cross-entropy).
        The first token has no prediction and stays NaN."""
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)
        n = len(ids)
        nll = np.full(n, np.nan)
        xent = np.full(n, np.nan)

        obs_head = self.observer.get_output_embeddings()
        perf_head = self.performer.get_output_embeddings()
        context = self.max_tokens - self.stride
        start, scored = 0, 1
        while scored < n:
            end = min(start + self.max_tokens, n)
            chunk = torch.tensor([ids[start:end]], device=self.device)
            obs = self.observer.base_model(input_ids=chunk, use_cache=False).last_hidden_state[0, :-1]
            perf = self.performer.base_model(input_ids=chunk, use_cache=False).last_hidden_state[0, :-1]
            targets = chunk[0, 1:]
            # Prediction k is for position start + 1 + k; only fill positions not yet scored.
            for a in range(scored - start - 1, end - start - 1, _BLOCK):
                b = min(a + _BLOCK, end - start - 1)
                log_p = torch.log_softmax(perf_head(perf[a:b])[:, : self.vocab].float(), dim=-1)
                p_obs = torch.softmax(obs_head(obs[a:b])[:, : self.vocab].float(), dim=-1)
                nll[start + 1 + a : start + 1 + b] = (-log_p.gather(-1, targets[a:b, None]).squeeze(-1)).cpu().numpy()
                xent[start + 1 + a : start + 1 + b] = (-(p_obs * log_p).sum(-1)).cpu().numpy()
            scored = end
            start = max(0, end - context)
        return offsets, nll, xent

    def span_scores(self, text: str, spans: Sequence[tuple[int, int]]) -> np.ndarray:
        offsets, nll, xent = self.token_stats(text)
        mids = offsets.mean(axis=1)
        valid = ~np.isnan(nll)
        scores = np.full(len(spans), np.nan)
        for i, (a, b) in enumerate(spans):
            sel = valid & (mids >= a) & (mids < b)
            if sel.sum() >= MIN_TOKENS:
                scores[i] = nll[sel].sum() / xent[sel].sum()
        return scores
