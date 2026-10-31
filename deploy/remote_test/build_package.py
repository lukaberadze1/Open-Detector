"""Pack the project for a test run on someone else's computer: dist/open-detector-remote-test.zip.

    python deploy/remote_test/build_package.py

Holds the code as it is in this checkout (uncommitted changes included), the style profiles and writing
in data/styles/, the samples to score in data/essays/revise_samples/, and START_HERE.md with the run
scripts at the top. Never the .env, the virtual environment, caches or other saved samples; the build
stops if any packed file holds a key from .env.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "dist/open-detector-remote-test.zip"
TOP = "open-detector-test"
EXTRA = ("data/styles", "data/essays/revise_samples")  # gitignored, but the tests need them
AT_TOP = ("START_HERE.md", "run_windows.ps1", "run_mac_linux.sh")
SKIP_PARTS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".agents", "Open-Detector", "dist", "cache"}


def project_files() -> set[Path]:
    """Tracked and new files of the project, as they are on disk."""
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split("\0")
    files = {ROOT / name for name in listed if name}
    for folder in EXTRA:
        files |= {path for path in (ROOT / folder).rglob("*") if path.is_file()}
    return {path for path in files if path.is_file() and not SKIP_PARTS & set(path.relative_to(ROOT).parts)
            and path.name not in (".env", ".DS_Store") and not path.name.endswith(".pyc")}


def secrets() -> list[str]:
    """The values in .env, to make sure none is packed."""
    env = ROOT / ".env"
    if not env.exists():
        return []
    values = (line.split("=", 1)[1].strip().strip("'\"") for line in env.read_text().splitlines()
              if "=" in line and not line.lstrip().startswith("#"))
    return [value for value in values if len(value) >= 12]


def main() -> int:
    samples = list((ROOT / "data/essays/revise_samples").glob("*.jsonl"))
    if not samples:
        sys.exit("No samples in data/essays/revise_samples/: copy the revised samples' .jsonl files there first.")
    files, keys = sorted(project_files()), secrets()
    for path in files:
        data = path.read_bytes()
        if any(key.encode() in data for key in keys):
            sys.exit(f"Stopped: {path.relative_to(ROOT)} holds a value from .env.")
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{TOP}/{path.relative_to(ROOT).as_posix()}")
        for name in AT_TOP:
            archive.write(HERE / name, f"{TOP}/{name}")
    print(f"{OUT.relative_to(ROOT)}: {len(files)} files, {len(samples)} sample files, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
