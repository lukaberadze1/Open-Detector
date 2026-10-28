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
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
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
    """The profile chosen in the sidebar."""
    return PINNED_PROFILE or st.session_state.get("profile") or DEFAULT_PROFILE


def profile_picker() -> None:
    """The sidebar's detector choice, shared by both pages."""
    st.subheader("Detector")
    if PINNED_PROFILE:
        st.caption("Slim build: Desklib alone, the strongest single engine, on the CPU to fit this host."
                   if PINNED_PROFILE == "slim" else "Full build: every engine.")
        return
    st.session_state.setdefault("profile", DEFAULT_PROFILE)
    profile = st.segmented_control("Build", PROFILES, format_func=PROFILE_LABELS.get, key="profile",
                                   required=True, label_visibility="collapsed")
    if profile == "slim":
        st.caption("Desklib alone on the CPU, about 1.5 GB. Scores as well as the full build on the "
                   "held-out test.")
    else:
        st.caption(f"GLYPH, Desklib and Binoculars, about 3.5 GB. This computer has {RAM_GB:.0f} GB"
                   + (", so Slim leaves more room for a local text model." if RAM_GB <= 8.5 else "."))


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
