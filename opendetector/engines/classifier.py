"""Fine-tuned classifiers (GLYPH, Desklib) that score whole windows of text."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)


class ClassifierEngine:
    """Wraps a sequence classifier so that it returns one AI log-odds value per window."""

    def __init__(
        self,
        name: str,
        tokenizer,
        model: nn.Module,
        log_odds: Callable[[nn.Module, dict], torch.Tensor],
        device: str,
        max_length: int,
        batch_size: int,
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self._log_odds = log_odds
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def window_logits(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(
                list(texts[i : i + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out.append(self._log_odds(self.model, enc).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def load_glyph(model_id: str, device: str, dtype: torch.dtype, batch_size: int) -> ClassifierEngine:
    # The model card warns the fast DeBERTa-v3 tokenizer crashes on load in transformers>=4.47.
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype)

    def log_odds(m, enc):
        logits = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        return logits[:, 1] - logits[:, 0]  # LABEL_1 = AI

    return ClassifierEngine("glyph", tokenizer, model, log_odds, device, 512, batch_size)


class DesklibAIDetectionModel(PreTrainedModel):
    """Architecture from the desklib/ai-text-detector-v1.01 model card: DeBERTa-v3-large,
    attention-masked mean pooling, one linear logit."""

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled).squeeze(-1)


def load_desklib(model_id: str, device: str, dtype: torch.dtype, batch_size: int) -> ClassifierEngine:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = DesklibAIDetectionModel.from_pretrained(model_id, dtype=dtype)

    def log_odds(m, enc):
        return m(enc["input_ids"], enc["attention_mask"])

    return ClassifierEngine("desklib", tokenizer, model, log_odds, device, 768, batch_size)
