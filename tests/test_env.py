"""Keys in the .env file of the app's data folder (opendetector/env.py), and where that folder is."""

import os
import subprocess
import sys
from pathlib import Path

from opendetector.env import load_env, save_env


def test_save_env_replaces_one_key_keeps_the_rest_and_can_remove_it(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("# keys\nOPENAI_API_KEY=abc\nexport GEMINI_API_KEY=old\n")
    save_env("GEMINI_API_KEY", "new", path)
    assert path.read_text() == "# keys\nOPENAI_API_KEY=abc\nGEMINI_API_KEY=new\n"
    assert os.environ["GEMINI_API_KEY"] == "new"
    if os.name != "nt":  # Windows has no Unix file permissions
        assert path.stat().st_mode & 0o777 == 0o600
    save_env("GEMINI_API_KEY", "", path)
    assert path.read_text() == "# keys\nOPENAI_API_KEY=abc\n"
    assert "GEMINI_API_KEY" not in os.environ


def test_a_saved_key_is_loaded_next_time(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path = tmp_path / "new folder" / ".env"
    save_env("GEMINI_API_KEY", "kept", path)
    monkeypatch.delenv("GEMINI_API_KEY")
    load_env(path)
    assert os.environ["GEMINI_API_KEY"] == "kept"


def test_opendetector_home_moves_the_data_folder():
    root = Path(__file__).resolve().parent.parent

    def data_root(**env) -> str:
        env = {key: value for key, value in os.environ.items() if key != "OPENDETECTOR_HOME"} | env
        return subprocess.run([sys.executable, "-c", "from opendetector.config import DATA_ROOT; print(DATA_ROOT)"],
                              cwd=root, env=env, capture_output=True, text=True, check=True).stdout.strip()

    assert data_root() == str(root)
    assert data_root(OPENDETECTOR_HOME="~/Somewhere") == str(Path.home() / "Somewhere")
