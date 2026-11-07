"""The detector models the app has loaded, shared by both pages, every session and batch workers."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable

import streamlit as st

from opendetector import Detector, DetectorConfig
from opendetector.combiner import Combiner

PROFILES = ("full", "slim")
PROFILE_LABELS = {"full": "Full", "slim": "Slim"}
ENGINES = {
    "glyph": ("GLYPH", "DeBERTa-v3-base classifier"),
    "desklib": ("Desklib", "DeBERTa-v3-large classifier"),
    "bino": ("Binoculars", "zero-shot, two small language models"),
}
DOWNLOAD_GB = {"glyph": 0.7, "desklib": 1.7, "bino": 2.0}  # each engine's model files, fetched on first use


def _ram_gb() -> float:
    """This computer's memory. Windows has no os.sysconf, so it asks the system directly."""
    if hasattr(os, "sysconf"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                    ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                    ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

    status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 16.0  # unknown: assume enough for the full build
    return status.total / 2**30


RAM_GB = _ram_gb()
# A host can pin the profile: the free Streamlit deploy runs slim. Otherwise machines with 8 GB or
# less start on slim, because the full build (about 3.5 GB) next to a local text model means heavy
# swapping there. On the held-out test, slim scores as well as full (eval/reports/slim.md).
PINNED_PROFILE = os.environ.get("OPENDETECTOR_PROFILE") or None
DEFAULT_PROFILE = PINNED_PROFILE or ("slim" if RAM_GB <= 8.5 else "full")


def engines_of(profile: str) -> list[str]:
    """The engines a profile runs, in feature order."""
    config = DetectorConfig.for_profile(profile)
    return [name for name, model in (("glyph", config.glyph_model), ("desklib", config.desklib_model),
                                     ("bino", config.binoculars_observer)) if model]


def download_gb(profile: str, names: Iterable[str] | None = None) -> float:
    """How much the first run of `profile` downloads, for the engines in `names` (by default all of them)."""
    return sum(DOWNLOAD_GB[name] for name in engines_of(profile) if names is None or name in names)


def active_profile() -> str:
    """The build chosen on the Settings page: this session's pick, else the one saved on this computer."""
    if PINNED_PROFILE:
        return PINNED_PROFILE
    if st.session_state.get("profile") not in PROFILES:
        from . import prefs, style

        saved = None if style.public() else prefs.load().get("profile")
        st.session_state["profile"] = saved if saved in PROFILES else DEFAULT_PROFILE
    return st.session_state["profile"]


def remember_profile(profile: str) -> None:
    """Keeps the build picked on the Settings page for next time, on this computer."""
    from . import prefs, style

    if not style.public() and profile in PROFILES:
        prefs.update(profile=profile)


@st.cache_resource
def _shared() -> dict:
    """Kept by Streamlit across reruns, sessions and edits to this file. Holds one profile's engines
    at a time: two profiles would not fit next to a local text model on an 8 GB machine."""
    return {"lock": threading.Lock(), "profile": None, "engines": {}}


_LOADED = _shared()


def is_loaded(profile: str, names: Iterable[str]) -> bool:
    return _LOADED["profile"] == profile and all(name in _LOADED["engines"] for name in names)


def get_detector(profile: str, names: Iterable[str] | None = None) -> Detector:
    """A detector for `profile` with the engines in `names` (by default all the profile has). Loads
    any engine not loaded yet, dropping another profile's engines first. It makes no Streamlit calls,
    so a batch worker's thread can use it."""
    from opendetector import engines  # imports torch

    config = DetectorConfig.for_profile(profile)
    wanted = [name for name in engines_of(profile) if names is None or name in names]
    with _LOADED["lock"]:
        if _LOADED["profile"] != profile:
            _LOADED["engines"].clear()
            _LOADED["profile"] = profile
            engines.free_memory()
        for name in wanted:
            if name not in _LOADED["engines"]:
                _LOADED["engines"][name] = engines.build_engine(name, config)
        loaded = dict(_LOADED["engines"])
    combiner = Combiner.load(config.combiner_path) if config.combiner_path.exists() else None
    classifiers = [loaded[name] for name in ("glyph", "desklib") if name in wanted]
    return Detector(classifiers, loaded["bino"] if "bino" in wanted else None, combiner, config)


def release() -> None:
    """Drop every loaded engine, so a local text model has the memory to itself."""
    from opendetector.engines import free_memory

    with _LOADED["lock"]:
        _LOADED["engines"].clear()
        _LOADED["profile"] = None
    free_memory()
