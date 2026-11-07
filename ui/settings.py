"""The Settings page, "Behind the counter": what is set once and used every time. The look and Bistro's finish,
the detector build, and, off a public deploy, the text models and keys the Lab writes with and the local model
used after every hosted one fails. Detect and the Lab read these (style, models.active_profile,
redteam._connection) and link here."""

from __future__ import annotations

import streamlit as st

from . import kitchen, models, prefs, style

LOOK_PREVIEWS = {  # trusted HTML from the code, on top of each look's card
    "bistro": '<span style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;'
              "background-color:#F3EADB;background-image:radial-gradient(rgba(120,90,60,.14) 1px,transparent 1.3px);"
              "background-size:6px 6px;border:1px solid #D9C7AA;border-radius:10px;font-family:'JetBrains Mono',"
              'monospace;font-size:16px;font-weight:800;color:#2A1F17">open<span style="color:#C2410C">/</span>'
              "detector</span>",
    "glass": '<span style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;'
             "background:radial-gradient(10rem 6rem at 20% 0%,#9EB3FF,transparent 70%),radial-gradient(10rem 6rem "
             "at 90% 100%,#C4B5FD,transparent 70%),#EEF1FA;border-radius:10px;font-family:Geist,sans-serif;"
             'font-size:16px;font-weight:700;color:#0F172A">Open Detector</span>',
}
LOOK_NOTES = {"bistro": "Paper, espresso, paprika, Cloche", "glass": "Frosted panels over an aurora"}


def _look() -> None:
    before = style.current_look()
    chosen = kitchen.cards([(look, style.LOOK_LABELS[look], LOOK_NOTES[look]) for look in style.LOOKS], style.KEY,
                           "Look", check=True, previews=LOOK_PREVIEWS)
    if chosen != before:  # the page was drawn in the old look: save the new one and draw it again
        if not style.public():
            prefs.update(look=chosen)
        if chosen == "bistro":
            kitchen.cue("happy", "Welcome back to the bistro.", ms=2600)
        st.rerun()
    if style.bistro():
        with st.container(horizontal=True, vertical_alignment="center", key="od-finish"):
            note = ("Paper grain, kraft sidebar, cards with depth." if style.finish() == "textured" else
                    "Flat cream paper, same layout.")
            st.markdown(f"**Finish**  \n:gray[{note}]", width="stretch")
            st.segmented_control("Finish", style.FINISHES, format_func=style.FINISH_LABELS.get, key=style.FINISH_KEY,
                                 required=True, label_visibility="collapsed", on_change=style._save_finish)
        if style.finish() == "textured":
            style.wall()
            with st.container(horizontal=True, vertical_alignment="center", key="od-wall"):
                note = ("White ceramic kitchen tiles, glazed and grouted." if style.wall() == "tiles" else
                        "Grained paper.")
                st.markdown(f"**Wall**  \n:gray[{note}]", width="stretch")
                st.segmented_control("Wall", style.WALLS, format_func=style.WALL_LABELS.get, key=style.WALL_KEY,
                                     required=True, label_visibility="collapsed", on_change=style._save_wall)
    st.caption("Light or dark: the **⋮** menu at the top right, then **Settings**."
               + (" This site doesn't remember the look between visits." if style.public() else ""))


def _detector() -> None:
    if models.PINNED_PROFILE:
        st.caption(("Slim build: Desklib alone, the strongest single engine, on the CPU to fit this host."
                    if models.PINNED_PROFILE == "slim" else "Full build: every engine.")
                   + " This site is set to it.")
        return
    before = models.active_profile()
    chosen = kitchen.cards(
        [("full", "Full build", "GLYPH, Desklib and Binoculars · about 3.5 GB"),
         ("slim", "Slim build", "Desklib alone · about 1.5 GB · scores as well on the held-out test")],
        "profile", "Detector build", check=True)
    if chosen != before:
        models.remember_profile(chosen)
        if chosen == "full":
            kitchen.cue("happy", "All three critics. Bring napkins.", ms=2600)
    st.caption(f"Detect starts on this build, and the Lab scores with it. This computer has {models.RAM_GB:.0f} GB"
               + (", so Slim leaves room for a local model." if models.RAM_GB <= 8.5 else ".")
               + " Engines can still be switched off for one check on the Detect page.")


def render() -> None:
    style.page_header("Settings", "Set these once. Detect and the Lab use them every time.", page="settings")
    look, detector = st.columns(2, gap="medium")
    with look, style.group("Look"):
        _look()
    with detector, style.group("Detector", kitchen="the critic"):
        _detector()
    if style.public():
        with style.group("Keys and models", kitchen="the cooks"):
            st.caption("On this site you bring your own Gemini key: paste it in the Adversarial Lab, under "
                       "**Model & connection**. It is never written down here.")
        _wall_end()
        return
    from . import redteam  # the Lab's own settings, drawn here

    redteam.load_env()
    cooks, computer = st.columns([1.35, 1], gap="medium")
    with cooks, style.group("Keys and models", kitchen="the cooks"):
        redteam.connection_settings()
    with computer, style.group("This computer", kitchen="the back-up cook"):
        st.caption("Used after every hosted model fails. Runs on this computer, with no key and no quota.")
        redteam.local_settings()
    _wall_end()


def _wall_end() -> None:
    if style.tiled():  # the easter egg: the tiles stop at the foot of the page
        style.wall_end()
