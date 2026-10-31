"""The Mac app (deploy/macos): the launcher's server handling and the bundle build_app.py writes. No
window opens; the build needs macOS with the Xcode command line tools and the project's .venv."""

import json
import os
import plistlib
import re
import shutil
import socket
import subprocess
import sys
import time

import pytest

from deploy.macos import build_app, launcher

on_mac = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
can_build = pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("clang") or not (build_app.ROOT / ".venv/pyvenv.cfg").exists(),
    reason="needs macOS, the Xcode command line tools and the project's .venv")


def test_the_server_listens_on_this_computer_only():
    args = launcher.streamlit_args(8517)
    assert args[:2] == ["run", str(launcher.ROOT / "app.py")]
    assert "--server.address=127.0.0.1" in args and "--server.port=8517" in args


def test_pick_port_keeps_its_port_unless_something_holds_it():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert launcher.pick_port(port) != port
    assert launcher.pick_port(port) == port


@on_mac
def test_a_caffeinate_child_of_the_server_means_a_batch_is_running():
    idle = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    busy = subprocess.Popen([sys.executable, "-c", "import subprocess, time; "
                             "subprocess.Popen(['caffeinate', '-t', '30']); time.sleep(30)"])
    try:
        deadline = time.monotonic() + 5
        while not launcher.batch_running(busy) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert launcher.batch_running(busy)
        assert not launcher.batch_running(idle)
    finally:
        subprocess.run(["pkill", "-P", str(busy.pid)], check=False)
        for process in (idle, busy):
            process.kill()
            process.wait()
    assert not launcher.batch_running(busy)


def test_the_error_page_escapes_the_log(tmp_path, monkeypatch):
    log = tmp_path / "server.log"
    log.write_text("starting\n<script>alert(1)</script>\n")
    monkeypatch.setattr(launcher, "LOG_PATH", log)
    page = launcher.error_page("It stopped & won't start")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "<script>" not in page
    assert "It stopped &amp; won&#x27;t start" in page
    assert "pywebview.api.restart()" in page and "pywebview.api.show_log()" in page


def test_the_window_size_is_remembered_within_the_minimum(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "WINDOW_STATE", tmp_path / "window.json")
    assert launcher.DesktopApp._saved_size() == launcher.DEFAULT_SIZE
    app = launcher.DesktopApp()
    app.resized(1500.0, 300.0)
    app._save_size()
    assert launcher.DesktopApp._saved_size() == (1500, launcher.MIN_SIZE[1])
    (tmp_path / "window.json").write_text("not json")
    assert launcher.DesktopApp._saved_size() == launcher.DEFAULT_SIZE


@can_build
def test_build_writes_a_signed_app_that_runs_the_launcher(tmp_path):
    app = build_app.build(tmp_path, register=False)
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == build_app.BUNDLE_ID
    assert info["CFBundleExecutable"] == build_app.APP_NAME
    assert info["NSAppTransportSecurity"] == {"NSAllowsLocalNetworking": True}
    executable = app / "Contents/MacOS" / build_app.APP_NAME
    assert os.access(executable, os.X_OK)
    assert str(build_app.ROOT / "deploy/macos/launcher.py").encode() in executable.read_bytes()
    assert (app / "Contents/Resources/AppIcon.icns").stat().st_size > 10_000
    subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
    assert build_app.build(tmp_path, icon=False, register=False) == app  # replaces its own earlier build


def test_build_leaves_an_app_it_did_not_make(tmp_path):
    contents = tmp_path / f"{build_app.APP_NAME}.app/Contents"
    contents.mkdir(parents=True)
    (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.example.other"}))
    with pytest.raises(SystemExit, match="wasn't built by this script"):
        build_app.build(tmp_path, icon=False, register=False)
    assert (contents / "Info.plist").exists()


def test_the_loading_page_follows_the_look_in_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "DATA_ROOT", tmp_path)
    assert launcher.look() == "glass"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/ui_prefs.json").write_text(json.dumps({"look": "instrument", "welcomed": True}))
    assert launcher.look() == "instrument"


def test_the_standalone_app_carries_the_code_it_runs_and_nothing_under_data(tmp_path):
    build_app.copy_app(build_app.ROOT, tmp_path)
    for name in (*build_app.APP_FILES, "opendetector/detector.py", "ui/guide.py", "ui/assets/mark.svg"):
        assert (tmp_path / name).is_file(), name
    assert not (tmp_path / "data").exists() and not (tmp_path / ".env").exists()
    assert not list(tmp_path.rglob("__pycache__"))


def test_the_standalone_app_carries_every_eval_module_the_app_imports():
    root = build_app.ROOT
    imported = set()
    shipped_eval = [root / name for name in build_app.APP_FILES if name.startswith("eval/")]
    for path in [root / "app.py", *(root / "ui").glob("*.py"), *(root / "opendetector").rglob("*.py"), *shipped_eval]:
        imported |= set(re.findall(r"^\s*(?:from|import) eval\.(\w+)", path.read_text(), re.MULTILINE))
    assert imported and {f"eval/{name}.py" for name in imported} <= set(build_app.APP_FILES)


@can_build
def test_the_standalone_executable_compiles(tmp_path):
    prefix, version = build_app.venv_python(build_app.ROOT)
    out = tmp_path / build_app.APP_NAME
    build_app._compile(build_app.STANDALONE_SOURCE.substitute(name=build_app.APP_NAME),
                       prefix / "lib" / f"libpython{version}.dylib", prefix / "include" / f"python{version}", out)
    assert os.access(out, os.X_OK)
    assert b"Library/Application Support/Open Detector" in out.read_bytes()


def test_the_standalone_app_leaves_out_headers_and_test_suites(tmp_path):
    for name in ("torch/include/ATen/core.h", "torch/share/cmake/Torch.cmake", "torch/fx/passes/tests/test_x.py",
                 "sympy/core/tests/test_basic.py", "sympy/core/tests/deeper/tests/x.py", "pandas/tests/test_x.py",
                 "torch/__init__.py", "torch/testing/__init__.py", "sympy/core/basic.py", "pkg/tests.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("")
    build_app.prune_packages(tmp_path)
    left = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert left == ["pkg/tests.py", "sympy/core/basic.py", "torch/__init__.py", "torch/testing/__init__.py"]
