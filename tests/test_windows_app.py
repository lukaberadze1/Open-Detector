"""The Windows app (deploy/windows) and the launcher's Windows side. The build itself runs on GitHub's Windows
machines (.github/workflows/windows-app.yml), which also run these tests."""

import importlib.metadata
import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deploy.macos import launcher

ROOT = Path(__file__).resolve().parents[1]
on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

spec = importlib.util.spec_from_file_location("windows_build_app", ROOT / "deploy/windows/build_app.py")
build_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_app)


def pinned() -> dict[str, str]:
    lines = build_app.CONSTRAINTS.read_text().splitlines()
    return dict(line.lower().split("==", 1) for line in lines if "==" in line)


def test_constraints_pin_every_package_the_app_installs():
    required = (ROOT / "deploy/macos/requirements.txt").read_text().splitlines()
    names = [re.split(r"[<>=!~; ]", line)[0].lower() for line in required if line and not line.startswith("#")]
    assert names and set(names) <= set(pinned())


def test_constraints_match_the_versions_the_tests_run_on():
    for name, version in pinned().items():
        if name in ("torch", "transformers", "streamlit", "numpy"):
            assert importlib.metadata.version(name) == version, "run: deploy/windows/build_app.py --pin"


def test_the_executable_runs_the_launcher_the_app_carries():
    assert "deploy/macos/launcher.py" in build_app.mac.APP_FILES
    assert "\\\\app\\\\deploy\\\\macos\\\\launcher.py" in build_app.EXECUTABLE_SOURCE
    assert "python$short.dll" in build_app.EXECUTABLE_SOURCE


def test_the_installer_is_named_as_the_readme_links_it():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"releases/latest/download/{build_app.INSTALLER_NAME}.exe" in readme


def test_pruning_leaves_what_python_needs(tmp_path):
    packages = tmp_path / "Lib/site-packages"
    for name in ("torch/lib/dnnl.lib", "torch/lib/torch_cpu.dll", "torch/include/a.h", "numpy/tests/t.py",
                 "numpy/core.py"):
        (packages / name).parent.mkdir(parents=True, exist_ok=True)
        (packages / name).touch()
    (tmp_path / "Lib/test").mkdir()
    (tmp_path / "Lib/os.py").touch()
    build_app.prune_python(tmp_path)
    assert sorted(path.name for path in packages.rglob("*") if path.is_file()) == ["core.py", "torch_cpu.dll"]
    assert (tmp_path / "Lib/os.py").exists() and not (tmp_path / "Lib/test").exists()


def test_the_page_and_the_launcher_agree_on_the_batch_mutex():
    source = (ROOT / "ui/redteam.py").read_text(encoding="utf-8")
    assert f'BATCH_MUTEX = "{launcher.BATCH_MUTEX}"'.replace("\\", "\\\\") in source


@on_windows
def test_a_process_holding_the_mutex_means_a_batch_is_running():
    from ui import redteam

    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert not launcher.batch_running(server)
        awake = redteam._keep_awake()
        assert awake is not None
        deadline = time.monotonic() + 10
        while not launcher.batch_running(server) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert launcher.batch_running(server)
        awake.terminate()
        awake.wait()
        # In a .venv python.exe is a launcher, and the Python that holds the mutex ends just after it.
        deadline = time.monotonic() + 10
        while launcher.batch_running(server) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not launcher.batch_running(server)
    finally:
        server.kill()
        server.wait()


@on_windows
def test_waiting_for_a_process_returns_when_it_ends():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    started = time.monotonic()
    launcher.wait_for_exit(process.pid)
    assert time.monotonic() - started < 20
    process.wait()


@on_windows
def test_dark_mode_answers_on_windows():
    assert launcher.dark_mode() in (True, False)


@pytest.mark.skipif(sys.platform != "win32" or not shutil.which("cl"), reason="needs Windows and the MSVC tools")
def test_the_executable_compiles(tmp_path):
    build_app.compile_executable(tmp_path / "Open Detector.exe", "312")
    assert (tmp_path / "Open Detector.exe").stat().st_size > 10_000


def test_optional_attack_packages_stay_out_of_the_base_requirements():
    """gemini_webapi once broke the Windows build: it can't be installed with the pinned anthropic."""
    base = (ROOT / "requirements.txt").read_text().lower()
    for name in ("gemini_webapi", "gemini-webapi", "browser-cookie3", "spacy", "lemminflect"):
        assert name not in base
