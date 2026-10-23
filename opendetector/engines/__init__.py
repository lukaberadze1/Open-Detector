"""Model-backed scoring engines. Importing this package loads torch and transformers."""

from __future__ import annotations

import gc

import torch

from ..config import DetectorConfig
from .binoculars import BinocularsEngine
from .classifier import ClassifierEngine, load_desklib, load_glyph


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(preference: str, device: str) -> torch.dtype:
    if preference != "auto":
        return getattr(torch, preference)
    return torch.bfloat16 if device in ("mps", "cuda") else torch.float32


ENGINE_NAMES = ("glyph", "desklib", "bino")


def build_engine(name: str, config: DetectorConfig) -> ClassifierEngine | BinocularsEngine | None:
    """Load one engine, or return None if the config disables it."""
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    if name == "glyph":
        return load_glyph(config.glyph_model, device, dtype, config.batch_size) if config.glyph_model else None
    if name == "desklib":
        return load_desklib(config.desklib_model, device, dtype, config.batch_size) if config.desklib_model else None
    if name == "bino":
        if config.binoculars_observer and config.binoculars_performer:
            return BinocularsEngine(config.binoculars_observer, config.binoculars_performer, device, dtype)
        return None
    raise ValueError(f"unknown engine {name!r}")


def free_memory() -> None:
    """Return memory from engines the caller has already dropped (`del`) to the system."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_engines(config: DetectorConfig) -> tuple[list[ClassifierEngine], BinocularsEngine | None]:
    engines = {name: build_engine(name, config) for name in ENGINE_NAMES}
    classifiers = [engines[n] for n in ("glyph", "desklib") if engines[n] is not None]
    return classifiers, engines["bino"]


__all__ = ["ENGINE_NAMES", "BinocularsEngine", "ClassifierEngine", "build_engine", "build_engines", "free_memory",
           "resolve_device", "resolve_dtype"]
