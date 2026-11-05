"""Translation on this computer with MarianMT (Helsinki-NLP's opus-mt models, about 570 MB on disk each direction,
fetched from Hugging Face the first time): no key, no quota, no bot check, and the text stays here. These aren't LLMs,
so what comes back through them is a translation engine's English, the idea taken from lynote-ai/humanize-text (MIT).
Each direction is loaded for its one pass and let go, since two of them and the detector share 8 GB. torch comes in
only inside `translate`, so eval/text_attacks.py stays importable without it.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence

CODES = {"English": "en", "German": "de", "Finnish": "fi", "Russian": "ru", "Chinese": "zh"}
BATCH = 8
MAX_TOKENS = 512  # what these models read; a longer sentence comes back cut short, and the length check then drops it


def model_id(source: str, target: str) -> str:
    return f"Helsinki-NLP/opus-mt-{CODES[source]}-{CODES[target]}"


def translate(texts: Sequence[str], source: str, target: str, batch: int = BATCH) -> list[str]:
    """`texts` from `source` to `target`, in their order. On the CPU in float32, like the slim detector. Texts of a
    similar length go together, since a batch runs at the pace of its longest."""
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    name = model_id(source, target)
    tokenizer = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name).to("cpu").eval()
    order = sorted(range(len(texts)), key=lambda number: len(texts[number]))
    results = [""] * len(texts)
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            chosen = order[start:start + batch]
            encoded = tokenizer([texts[number] for number in chosen], return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_TOKENS)
            written = tokenizer.batch_decode(model.generate(**encoded, max_new_tokens=MAX_TOKENS),
                                             skip_special_tokens=True)
            for number, text in zip(chosen, written, strict=True):
                results[number] = text.strip()
    del model
    gc.collect()
    return results
