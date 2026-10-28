from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Where the app keeps what it writes, its data/ folder and .env: the project folder, or OPENDETECTOR_HOME
# when that is set. The downloadable Mac app sets it to ~/Library/Application Support/Open Detector,
# because an app must not change its own signed bundle.
DATA_ROOT = Path(os.environ["OPENDETECTOR_HOME"]).expanduser() if os.environ.get("OPENDETECTOR_HOME") else ROOT


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
    # Local windows stay inside paragraphs; shorter paragraphs merge with a neighbour first.
    local_min_paragraph_words: int = 60

    # Below this many words we return no verdict. Turnitin's reported minimum is ~300.
    min_words: int = 150

    combiner_path: Path = ROOT / "models" / "combiner.json"

    @classmethod
    def for_profile(cls, profile: str) -> DetectorConfig:
        """'full': every engine on the best available hardware. 'slim': Desklib alone, float32 on
        the CPU, sized for Streamlit Community Cloud's free tier (about 1.5 GB peak).

        Measured alternatives for slim: bfloat16 is emulated on most CPUs (minutes per essay);
        int8 dropped Desklib's AUROC from 0.998 to 0.934; a bf16 embedding table used more memory
        than float32, which transformers maps straight from the model file."""
        if profile == "full":
            return cls()
        if profile == "slim":
            return cls(glyph_model=None, binoculars_observer=None, binoculars_performer=None,
                       device="cpu", dtype="float32", batch_size=2,
                       combiner_path=ROOT / "models" / "combiner_slim.json")
        raise ValueError(f"unknown profile {profile!r}; use 'full' or 'slim'")
