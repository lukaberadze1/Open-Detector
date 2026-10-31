#!/usr/bin/env bash
# Sets up Open Detector and runs the remote tests on macOS or Linux. Run from the unzipped folder:
#   bash run_mac_linux.sh            (add --no-generate to skip step 4, which needs a Gemini key in .env)
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  echo "uv is not installed. Install it first (see START_HERE.md), open a new terminal, then run this again:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python deploy/remote_test/run_tests.py "$@"
