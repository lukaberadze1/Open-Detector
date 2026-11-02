"""Build the Windows download, dist/Open-Detector-Setup.exe: the same app as the Mac's, in a window of its
own (deploy/macos/launcher.py), installed for one user with no administrator password.

    python deploy/windows/build_app.py             # on Windows, from a Visual Studio developer prompt
    python deploy/windows/build_app.py --pin       # anywhere: write the .venv's versions to constraints.txt

.github/workflows/windows-app.yml runs it on GitHub's Windows machines, which is how a release gets the
file: nothing here needs a Windows computer of your own. It needs uv, the MSVC tools (cl and rc) and Inno
Setup. The app carries uv's standalone CPython, the packages in deploy/macos/requirements.txt at the
versions in constraints.txt (the ones the tests ran against) and a copy of the code in build_app.py's
APP_FILES and APP_PACKAGES. It keeps what it writes in %LOCALAPPDATA%\\Open Detector, and the models
download on first use. PyPI's torch for Windows runs on the CPU, so the app does too.

The executable is a small C program that loads the app's Python and runs the launcher inside its own
process, so the window and the taskbar show the app's name and icon. Given arguments, it runs them as
python would: that is how the launcher starts the server.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("mac_build_app", HERE.parent / "macos/build_app.py")
mac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mac)

APP_NAME = mac.APP_NAME
VERSION = mac.VERSION
PYTHON = "3.12"  # as the .venv
# Fixed, so releases/latest/download/Open-Detector-Setup.exe always works.
INSTALLER_NAME = "Open-Detector-Setup"
CONSTRAINTS = HERE / "constraints.txt"
# Microsoft's small online installer.
WEBVIEW2_BOOTSTRAPPER = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"
INNO_SETUP = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")

EXECUTABLE_SOURCE = r"""/* Open Detector.exe, written by deploy/windows/build_app.py. The app's Python is in
   python\ and the code it runs in app\, next to this program, wherever it was installed. The C library is linked in,
   so nothing else has to be found before Python: Python's own reads the variables set here. */
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <wchar.h>

#define SIZE 4096
typedef int (*PyMain)(int, wchar_t **);

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR line, int show) {
    static wchar_t folder[SIZE], path[SIZE], data[SIZE];
    int argc;
    if (!GetModuleFileNameW(NULL, folder, SIZE)) return 1;
    *wcsrchr(folder, L'\\') = L'\0';
    swprintf(path, SIZE, L"%ls\\python", folder);
    SetEnvironmentVariableW(L"PYTHONHOME", path);
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONDONTWRITEBYTECODE", L"1");  /* compiled at build time */
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");  /* the code reads and writes UTF-8, as on a Mac */
    SetEnvironmentVariableW(L"OPENDETECTOR_PACKAGED", L"on");
    if (!GetEnvironmentVariableW(L"OPENDETECTOR_HOME", NULL, 0)
        && GetEnvironmentVariableW(L"LOCALAPPDATA", data, SIZE)) {
        swprintf(path, SIZE, L"%ls\\Open Detector", data);
        SetEnvironmentVariableW(L"OPENDETECTOR_HOME", path);
    }
    swprintf(path, SIZE, L"%ls\\python\\python$short.dll", folder);
    HMODULE python = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    PyMain py_main = python ? (PyMain)GetProcAddress(python, "Py_Main") : NULL;
    if (!py_main) {
        MessageBoxW(NULL, L"Part of the app is missing. Please install it again.", L"Open Detector can't start",
                    MB_OK | MB_ICONERROR);
        return 1;
    }
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argc > 1) return py_main(argc, argv);
    swprintf(path, SIZE, L"%ls\\app\\deploy\\macos\\launcher.py", folder);
    wchar_t *args[] = {argv[0], path, NULL};
    return py_main(2, args);
}
"""


def _uv() -> str:
    found = shutil.which("uv")
    if not found:
        raise SystemExit("Building needs uv to install the app's packages: https://docs.astral.sh/uv/")
    return found


def pin() -> None:
    """Write the .venv's package versions to constraints.txt, for the machine that can't see the .venv."""
    frozen = subprocess.run([_uv(), "pip", "freeze", "--python", str(ROOT / ".venv/bin/python")], capture_output=True,
                            text=True, check=True).stdout
    CONSTRAINTS.write_text("# The .venv's versions, written by deploy/windows/build_app.py --pin.\n" + frozen)


def bundle_python(home: Path) -> None:
    """Put a Python of the app's own in `home`, with the app's packages, and only what they use."""
    uv = _uv()
    subprocess.run([uv, "python", "install", PYTHON], check=True)
    # --system: never a .venv, which holds only a launcher that looks for its pyvenv.cfg.
    found = subprocess.run([uv, "python", "find", "--managed-python", "--system", PYTHON], capture_output=True,
                           text=True, check=True).stdout.strip()
    prefix = subprocess.run([found, "-c", "import sys; print(sys.base_prefix)"], capture_output=True, text=True,
                            check=True).stdout.strip()
    shutil.copytree(prefix, home, ignore=shutil.ignore_patterns("__pycache__"))
    python = home / "python.exe"
    subprocess.run([uv, "pip", "install", "--python", str(python), "--link-mode", "copy", "--no-progress",
                    "-r", str(ROOT / "deploy/macos/requirements.txt"), "-c", str(CONSTRAINTS)], check=True)
    prune_python(home)


def prune_python(home: Path) -> None:
    """What the app never loads: Tk and IDLE, Python's own tests, and torch's files for building against it."""
    lib = home / "Lib"
    packages = lib / "site-packages"
    mac.prune_packages(packages)
    unused = [home / "tcl", home / "Scripts", home / "Doc", lib / "EXTERNALLY-MANAGED",
              *(lib / name for name in ("test", "tkinter", "idlelib", "turtledemo", "ensurepip")),
              *(home / "DLLs").glob("_tkinter*"), *(home / "DLLs").glob("tcl*"), *(home / "DLLs").glob("tk*"),
              *(packages / "torch/lib").glob("*.lib")]
    for path in unused:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def compile_executable(out: Path, short_version: str) -> None:
    """Compile the app's executable, with the app's icon, using the MSVC tools."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "main.c").write_text(EXECUTABLE_SOURCE.replace("$short", short_version))
        shutil.copy2(HERE / "app.ico", folder)
        (folder / "app.rc").write_text('1 ICON "app.ico"\n')
        subprocess.run(["rc", "/nologo", "app.rc"], cwd=folder, check=True)
        subprocess.run(["cl", "/nologo", "/O2", "/MT", "main.c", "app.res", "user32.lib", "shell32.lib", f"/Fe:{out}",
                        "/link", "/SUBSYSTEM:WINDOWS"], cwd=folder, check=True)


def build(stage: Path) -> None:
    """Write everything the installer installs to `stage`, replacing an earlier build."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    bundle_python(stage / "python")
    mac.copy_app(ROOT, stage / "app")
    # Byte-compile now: the app doesn't write to its own folder. Files that don't compile (test data
    # shipped in some packages) are never imported.
    python = stage / "python/python.exe"
    subprocess.run([str(python), "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash",
                    str(stage / "python/Lib"), str(stage / "app")], check=False, capture_output=True)
    # The executable is the app's Python from here on; the copy's own programs and headers can go.
    for name in ("python.exe", "pythonw.exe"):
        (stage / "python" / name).unlink()
    for name in ("include", "libs"):
        shutil.rmtree(stage / "python" / name, ignore_errors=True)
    compile_executable(stage / f"{APP_NAME}.exe", PYTHON.replace(".", ""))


def make_installer(stage: Path, out_dir: Path) -> Path:
    """Pack `stage` into the file people download, with Microsoft's installer for WebView2, the window's
    browser engine, which the setup runs only on the few computers that don't have it yet."""
    compiler = shutil.which("ISCC") or (str(INNO_SETUP) if INNO_SETUP.exists() else None)
    if not compiler:
        raise SystemExit("Packing the installer needs Inno Setup: https://jrsoftware.org/isinfo.php")
    bootstrapper = stage.parent / "MicrosoftEdgeWebview2Setup.exe"
    urllib.request.urlretrieve(WEBVIEW2_BOOTSTRAPPER, bootstrapper)
    subprocess.run([compiler, f"/DVersion={VERSION}", f"/DStage={stage}", f"/DBootstrapper={bootstrapper}",
                    f"/DOutputName={INSTALLER_NAME}", f"/O{out_dir}", str(HERE / "installer.iss")], check=True)
    return out_dir / f"{INSTALLER_NAME}.exe"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pin", action="store_true", help="write the .venv's versions to constraints.txt and stop")
    args = parser.parse_args()
    if args.pin:
        pin()
        print(f"Wrote {CONSTRAINTS}")
        return
    if sys.platform != "win32":
        raise SystemExit("This builds the Windows app, so it runs on Windows: see .github/workflows/windows-app.yml.")
    stage = ROOT / "dist/windows/stage"
    build(stage)
    print(f"Packed {make_installer(stage, ROOT / 'dist')}")


if __name__ == "__main__":
    main()
