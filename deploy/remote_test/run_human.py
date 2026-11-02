"""Test the built-in human profile on another computer, and pack the results to send back.

    python deploy/remote_test/run_human.py            # needs GEMINI_API_KEY in .env
    python deploy/remote_test/run_human.py --no-full  # under 16 GB of memory: the slim detector only

Each AI-written document in data/essays/sources/ is rewritten once in each setup, hardest-pushing first, so a
spent daily quota costs the plainest comparison last. Everything written is labelled AI data. A run that
stops starts where it stopped. The one file to send back is human_test_results.zip; it never holds the key.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "styles/human_default_profile.json"
STACK = ("asymmetric", "punctuation", "parenthetical", "interjection", "idiom", "digression", "active")
SETUPS = {  # name: what it adds to revising as the page does
    "max": ["--no-guard", "--rounds", "8", *(part for note in STACK for part in ("--note", note))],
    "no_guard": ["--no-guard"],
    "page": [],
}
OUT = ROOT / "human_test_results"


def documents() -> list[str]:
    sources = ROOT / "data/essays/sources"
    return [str(path) for path in [sources / "claude_salamis.md", *sorted((sources / "revise").glob("*.md"))]
            if path.exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-full", action="store_true", help="skip the full detector (about 3.5 GB of memory)")
    parser.add_argument("--documents", type=int, help="only the first N documents, to try the run out")
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    out = OUT / "runs.jsonl"
    docs = documents()[:args.documents]
    base = [sys.executable, "-m", "eval.tune_revise", "--style", str(PROFILE), "--as-page", "--model", args.model,
            "--out", str(out)]
    failed = False
    for name, extra in SETUPS.items():
        print(f"\n=== Setup {name}: {len(docs)} documents ===", flush=True)
        report = OUT / f"{name}.md"
        code = subprocess.run([*base, *extra, "--report", str(report), "--repeats", str(args.repeats), *docs],
                              cwd=ROOT, check=False).returncode
        failed = failed or code != 0
        if not args.no_full:
            subprocess.run([*base, *extra, "--report", str(report), "--report-only", "--full"], cwd=ROOT, check=False)
    labelled = ROOT / "data/essays/tuning_redteam.jsonl"
    with zipfile.ZipFile(ROOT / "human_test_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [*OUT.iterdir(), labelled]:
            if path.is_file():
                archive.write(path, path.name)
    print(f"\nDone{' for now: run it again to finish' if failed else ''}. Send back human_test_results.zip.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
