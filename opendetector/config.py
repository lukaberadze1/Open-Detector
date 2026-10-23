from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DetectorConfig:
    # Engines; set any model id to None to disable that engine.
    glyph_model: str | None = "ogmatrixllm/glyph-v1.1"
    desklib_model: str | None = "desklib/ai-text-detector-v1.01"
    # Binoculars needs a base/instruct pair sharing one tokenizer. The paper uses
    # tiiuae/falcon-7b + falcon-7b-instruct (~28 GB); the 0.5B Qwen pair fits an 8 GB
    # laptop. Swap in Qwen/Qwen2.5-1.5B(-Instruct) or larger if you have the memory.
    binoculars_observer: str | None = "Qwen/Qwen2.5-0.5B"
    binoculars_performer: str | None = "Qwen/Qwen2.5-0.5B-Instruct"

    device: str = "auto"  # auto | cpu | mps | cuda
    dtype: str = "auto"  # auto = bfloat16 on GPU (halves memory), float32 on CPU
    batch_size: int = 8

    # Document windows feed the calibrated score; local windows drive sentence highlights.
    doc_window_words: int = 350
    doc_stride_words: int = 175
    local_window_words: int = 100
    local_stride_words: int = 50

    # Below this many words we return no verdict. Turnitin's reported minimum is ~300.
    min_words: int = 150

    combiner_path: Path = ROOT / "models" / "combiner.json"
