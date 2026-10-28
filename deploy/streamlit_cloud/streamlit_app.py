"""Streamlit Community Cloud entrypoint: the slim detector (Desklib alone, float32 on the CPU) and
the Red team page in public mode, where each visitor brings their own Gemini key and style profile
and sees only their own samples.

At share.streamlit.io, create an app from this repository with:
    branch:          feat/phase1-ensemble (or main once merged)
    main file path:  deploy/streamlit_cloud/streamlit_app.py
    Python version:  3.12 (under Advanced settings)
Dependencies come from the requirements.txt next to this file, which Community Cloud prefers
over the repository's root requirements.txt.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["OPENDETECTOR_PROFILE"] = "slim"
# Set, not left to the default: an app process that ran the earlier entrypoint, which set it to
# "off", keeps that value when Community Cloud pulls new code without restarting.
os.environ["OPENDETECTOR_REDTEAM"] = "on"
os.environ["OPENDETECTOR_PUBLIC"] = "on"
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
