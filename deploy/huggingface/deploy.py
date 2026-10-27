"""Deploy the detector app to a Hugging Face Space (Docker, CPU).

    .venv/bin/hf auth login                                # once, with a write token
    .venv/bin/python deploy/huggingface/deploy.py          # private Space <you>/open-detector
    .venv/bin/python deploy/huggingface/deploy.py --public --space you/name

Uploads only what the app needs: app.py, its ui package and theme without the Red team page, the
opendetector package, the fitted combiner, and the Dockerfile, README and requirements in this
folder. No .env or API keys are uploaded; the app doesn't use any.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalTokenNotFoundError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def stage(dest: Path) -> None:
    """Copy exactly the files the Space runs into `dest`."""
    for name in ("Dockerfile", "README.md", "requirements.txt"):
        shutil.copy(HERE / name, dest / name)
    shutil.copy(ROOT / "app.py", dest / "app.py")
    shutil.copytree(ROOT / "ui", dest / "ui", ignore=shutil.ignore_patterns("__pycache__", "redteam.py"))
    shutil.copytree(ROOT / ".streamlit", dest / ".streamlit", ignore=shutil.ignore_patterns("secrets.toml"))
    shutil.copytree(ROOT / "opendetector", dest / "opendetector", ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "models").mkdir()
    shutil.copy(ROOT / "models" / "combiner.json", dest / "models" / "combiner.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", help="owner/name of the Space (default: <your username>/open-detector)")
    parser.add_argument("--public", action="store_true", help="make a new Space public (default: private)")
    args = parser.parse_args()

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except LocalTokenNotFoundError:
        print("Not logged in to Hugging Face. Run `.venv/bin/hf auth login` with a write token from "
              "https://huggingface.co/settings/tokens, then run this again.", file=sys.stderr)
        return 1
    space = args.space or f"{user}/open-detector"

    api.create_repo(space, repo_type="space", space_sdk="docker", private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space", commit_message="Deploy Open Detector")
    print(f"Deployed. The Space builds in a few minutes: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
