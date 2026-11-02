#!/usr/bin/env bash
# Sets up Open Detector and tests the built-in human profile on macOS or Linux. Run from the unzipped folder:
#   bash run_human_mac_linux.sh            (add --no-full under 16 GB of memory)
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  echo "uv is not installed. Install it first (see START_HUMAN.md), open a new terminal, then run this again:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python deploy/remote_test/run_human.py "$@"
