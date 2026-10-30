"""Build Open Detector.app, so the detector opens like any other Mac app: from the Dock, Launchpad,
Spotlight or the Finder, in a window of its own (deploy/macos/launcher.py).

    .venv/bin/python deploy/macos/build_app.py                     # into ~/Applications
    .venv/bin/python deploy/macos/build_app.py --dest /Applications
    .venv/bin/python deploy/macos/build_app.py --standalone --dmg  # the download, into dist/

By default the app runs this checkout's code with its .venv and holds no copy of the project, so a
git pull, a refit or a new style profile shows up the next time it opens. Build it again after moving
the project folder. The icon is drawn from ui/assets/mark.svg.

--standalone builds the app other people download: it carries its own Python (the standalone CPython
uv installed for the .venv), the packages in deploy/macos/requirements.txt at the .venv's versions, and
a copy of the code the app runs, so it works on any Apple silicon Mac with macOS 14 or later. It keeps
what it writes in ~/Library/Application Support/Open Detector, and the models download on first use.
--dmg also packs it into dist/Open-Detector.dmg, the file a GitHub release offers. It needs uv.

The app's executable is a small C program, compiled here with the Xcode command line tools
(xcode-select --install), that runs the launcher on Python inside the app's own process. A script
that hands over to python with exec would lose the process: macOS then reports no process for the
app, which breaks accessibility and switching to the open window.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Open Detector"
BUNDLE_ID = "io.github.lukaberadze1.opendetector"
VERSION = "1.0.0"
DMG_NAME = "Open-Detector.dmg"  # fixed, so releases/latest/download/Open-Detector.dmg always works
LSREGISTER = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework"
              "/Support/lsregister")
# What the standalone app runs: the Streamlit app, the modules it imports, the fitted combiners and the
# theme. Development tools, evaluation data and everything under data/ stay out.
APP_FILES = ("app.py", "LICENSE", ".streamlit/config.toml", "models/combiner.json", "models/combiner_slim.json",
             "models/generation_tuning.json",
             "deploy/macos/launcher.py", "eval/__init__.py", "eval/generate_essays.py", "eval/redteam_batch.py",
             "eval/evasion_mode.py", "eval/generation_settings.py", "eval/tune_generation.py",
             "update_style_profile.py", "styles/human_default_profile.json", "styles/human_default/shapes.json",
             "styles/human_default/yardstick.json")
APP_PACKAGES = ("opendetector", "ui")
MACHO_MAGICS = {b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}

EXECUTABLE_SOURCE = Template("""\
/* Open Detector.app's executable, written by deploy/macos/build_app.py. */
#include <Python.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    const char *python = $python, *launcher = $launcher;
    if (access(python, X_OK) != 0 || access(launcher, R_OK) != 0) {
        char *alert[] = {"osascript", "-e", "on run argv", "-e",
                         "display alert (item 1 of argv) message (item 2 of argv) as critical", "-e", "end run",
                         $title, $missing, NULL};
        execv("/usr/bin/osascript", alert);
        return 1;
    }
    /* Python takes its packages and sys.executable from the .venv, as if started as its python. */
    setenv("PYTHONEXECUTABLE", python, 1);
    char *args[] = {argv[0], (char *)launcher, NULL};
    return Py_BytesMain(2, args);
}
""")

STANDALONE_SOURCE = Template("""\
/* Open Detector.app's executable, written by deploy/macos/build_app.py --standalone. The app carries
   its own Python in Contents/Resources/python and the code it runs in Contents/Resources/app, and
   finds both next to itself, wherever the app was put. Given arguments, it runs them as python
   would: that is how the launcher starts the app's server, inside the same signed app. */
#include <Python.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    char found[PATH_MAX], executable[PATH_MAX], contents[PATH_MAX], path[PATH_MAX];
    uint32_t size = sizeof found;
    if (_NSGetExecutablePath(found, &size) != 0 || realpath(found, executable) == NULL) return 1;
    strcpy(contents, executable);
    for (int i = 0; i < 2; i++) *strrchr(contents, '/') = '\\0';  /* .../Contents/MacOS/<name> */
    snprintf(path, sizeof path, "%s/Resources/python", contents);
    setenv("PYTHONHOME", path, 1);
    setenv("PYTHONEXECUTABLE", executable, 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);  /* the bundle must stay as it was signed */
    setenv("OPENDETECTOR_PACKAGED", "on", 1);
    const char *home = getenv("HOME");
    struct passwd *user = home ? NULL : getpwuid(getuid());
    if (user) home = user->pw_dir;
    if (home) {
        snprintf(path, sizeof path, "%s/Library/Application Support/$name", home);
        setenv("OPENDETECTOR_HOME", path, 0);
    }
    if (argc > 1 && strncmp(argv[1], "-psn_", 5) != 0) return Py_BytesMain(argc, argv);
    snprintf(path, sizeof path, "%s/Resources/app/deploy/macos/launcher.py", contents);
    char *args[] = {argv[0], path, NULL};
    return Py_BytesMain(2, args);
}
""")


def info_plist(version: str, short_version: str = "1.0", minimum_macos: str = "11.0") -> dict:
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": short_version,
        "CFBundleVersion": version,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": minimum_macos,
        "LSArchitecturePriority": ["arm64"],
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},  # the window loads http://127.0.0.1
    }


def git_version(root: Path) -> str:
    found = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                           check=False)
    return found.stdout.strip() or "0"


def venv_python(root: Path) -> tuple[Path, str]:
    """The folder of the Python the project's .venv was made from, and its version ("3.12")."""
    python = root / ".venv/bin/python"
    config = dict(line.split(" = ", 1) for line in (root / ".venv/pyvenv.cfg").read_text().splitlines()
                  if " = " in line)
    # pyvenv.cfg's home is uv's folder for the minor version, which survives patch updates.
    prefix = Path(config["home"]).parent
    version = subprocess.run([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, check=True).stdout.strip()
    return prefix, version


def _compile(source: str, library: Path, include: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        main = Path(tmp) / "main.c"
        main.write_text(source)
        subprocess.run(["clang", "-O2", "-mmacosx-version-min=11.0", "-Wl,-headerpad_max_install_names",
                        "-I", str(include), str(main), str(library), "-o", str(out)], check=True)


def compile_executable(root: Path, out: Path) -> None:
    """Compile the app's executable against the Python the project's .venv was made from."""
    python = root / ".venv/bin/python"
    prefix, version = venv_python(root)
    library = prefix / "lib" / f"libpython{version}.dylib"
    if not library.exists():
        raise SystemExit(f"{library} is missing: the .venv's Python has no shared library to build the app on.")

    def c_string(text: str) -> str:
        return json.dumps(text, ensure_ascii=False)

    source = EXECUTABLE_SOURCE.substitute(
        python=c_string(str(python)), launcher=c_string(str(root / "deploy/macos/launcher.py")),
        title=c_string(f"{APP_NAME} can't start"),
        missing=c_string(f"It runs the project at {root} with its .venv, and that is missing. If the project "
                         "moved, build the app again from its new folder: .venv/bin/python deploy/macos/build_app.py"))
    _compile(source, library, prefix / "include" / f"python{version}", out)
    # The library names itself by its patch-version folder; load it through the minor-version one.
    recorded = subprocess.run(["otool", "-D", str(library)], capture_output=True, text=True,
                              check=True).stdout.splitlines()[-1].strip()
    if recorded != str(library):
        subprocess.run(["install_name_tool", "-change", recorded, str(library), str(out)], check=True)


def copy_app(root: Path, dest: Path) -> None:
    """Copy the code the standalone app runs (APP_FILES and APP_PACKAGES) from `root` to `dest`."""
    for name in APP_FILES:
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest / name)
    for name in APP_PACKAGES:
        shutil.copytree(root / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _uv() -> str:
    found = shutil.which("uv") or next((str(path) for path in (Path("/opt/homebrew/bin/uv"),
                                                                Path.home() / ".local/bin/uv") if path.exists()), None)
    if not found:
        raise SystemExit("--standalone needs uv to install the app's packages: https://docs.astral.sh/uv/")
    return found


def bundle_python(root: Path, resources: Path, out: Path) -> None:
    """Put a Python of the app's own in resources/python, with the app's packages at the .venv's versions,
    and compile the executable `out` to run it."""
    prefix, version = venv_python(root)
    home = resources / "python"
    shutil.copytree(prefix.resolve(), home, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    # What the app never uses: headers, docs, Tk and IDLE, and the marker that keeps pip off a system Python.
    lib = home / "lib"
    for leftover in [home / "share", *lib.glob("tcl*"), *lib.glob("tk*"), *lib.glob("itcl*"), *lib.glob("thread*"),
                     *lib.glob("libtcl*"), *lib.glob("libtk*"), lib / f"python{version}/EXTERNALLY-MANAGED",
                     *[lib / f"python{version}" / name for name in ("tkinter", "idlelib", "turtledemo", "ensurepip")],
                     *(lib / f"python{version}/lib-dynload").glob("_tkinter*")]:
        if leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover)
        elif leftover.exists() or leftover.is_symlink():
            leftover.unlink()

    python = home / "bin" / f"python{version}"
    uv = _uv()
    with tempfile.TemporaryDirectory() as tmp:
        pins = Path(tmp) / "pins.txt"
        pins.write_text(subprocess.run([uv, "pip", "freeze", "--python", str(root / ".venv/bin/python")],
                                       capture_output=True, text=True, check=True).stdout)
        subprocess.run([uv, "pip", "install", "--python", str(python), "--link-mode", "copy", "--no-progress",
                        "-r", str(root / "deploy/macos/requirements.txt"), "-c", str(pins)], check=True)
    prune_packages(lib / f"python{version}" / "site-packages")
    # Byte-compile everything now, checked by hash rather than file times, since the app can't write
    # its own bundle. Files that don't compile (test data shipped in some packages) are never imported.
    subprocess.run([str(python), "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "unchecked-hash",
                    str(lib), str(resources / "app")], check=False, capture_output=True)

    library = lib / f"libpython{version}.dylib"
    subprocess.run(["install_name_tool", "-id", f"@executable_path/../Resources/python/lib/{library.name}",
                    str(library)], check=True)
    _compile(STANDALONE_SOURCE.substitute(name=APP_NAME), library, home / "include" / f"python{version}", out)
    # The executable is the app's Python from here on; the copy's own programs and headers can go.
    shutil.rmtree(home / "bin")
    shutil.rmtree(home / "include")


def prune_packages(packages: Path) -> None:
    """Remove what the app never loads from the installed packages: torch's C++ headers and CMake files,
    which only build extensions, and every package's own test suite. That is about 15,000 files and
    160 MB, and macOS checks every file of a downloaded app the first time it opens."""
    unused = [packages / "torch/include", packages / "torch/share",
              *(path for path in packages.rglob("tests") if path.is_dir())]
    for path in sorted(unused, key=lambda path: len(path.parts)):
        if path.is_dir():  # not already gone with a folder around it
            shutil.rmtree(path)


def sign_code(folder: Path) -> None:
    """Ad-hoc sign every program and library under `folder`: Apple silicon runs only signed code, and
    install_name_tool has changed some of them."""
    code = [str(path) for path in folder.rglob("*") if path.is_file() and not path.is_symlink()
            and path.suffix not in (".py", ".pyc", ".txt", ".json") and _is_macho(path)]
    for start in range(0, len(code), 200):
        subprocess.run(["codesign", "--force", "--sign", "-", *code[start:start + 200]], check=True,
                       capture_output=True)


def _is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(4) in MACHO_MAGICS
    except OSError:
        return False


def render_icon(svg: Path, out: Path) -> None:
    """Draw `svg` on macOS's icon grid (824 px of artwork on a 1024 px canvas, rounded corners, a soft
    shadow) and write every size of an .icns to `out`."""
    import AppKit

    size, inset = 1024, 100
    side = size - 2 * inset
    rect = ((inset, inset), (side, side))
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(  # noqa: E501
        None, size, size, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0)
    AppKit.NSGraphicsContext.saveGraphicsState()
    AppKit.NSGraphicsContext.setCurrentContext_(AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    shape = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, side * 0.2237, side * 0.2237)
    shadow = AppKit.NSShadow.alloc().init()
    shadow.setShadowColor_(AppKit.NSColor.colorWithWhite_alpha_(0, 0.28))
    shadow.setShadowBlurRadius_(22)
    shadow.setShadowOffset_((0, -10))
    AppKit.NSGraphicsContext.saveGraphicsState()
    shadow.set()
    AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0.39, 0.40, 0.95, 1).setFill()
    shape.fill()
    AppKit.NSGraphicsContext.restoreGraphicsState()
    # The mark's own gradient, filling the corners its rounder square leaves inside the icon shape.
    shape.addClip()
    AppKit.NSGradient.alloc().initWithStartingColor_endingColor_(
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0x63 / 255, 0x66 / 255, 0xF1 / 255, 1),
        AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0x8B / 255, 0x5C / 255, 0xF6 / 255, 1),
    ).drawInBezierPath_angle_(shape, -45)
    AppKit.NSImage.alloc().initWithContentsOfFile_(str(svg)).drawInRect_(rect)
    AppKit.NSGraphicsContext.restoreGraphicsState()

    with tempfile.TemporaryDirectory() as tmp:
        master = Path(tmp) / "icon.png"
        rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(
            str(master), True)
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for points in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                name = iconset / f"icon_{points}x{points}{suffix}.png"
                subprocess.run(["sips", "-z", str(points * scale), str(points * scale), str(master), "--out",
                                str(name)], check=True, capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)


def build(dest: Path, root: Path = ROOT, icon: bool = True, register: bool = True,
          standalone: bool = False) -> Path:
    """Write `dest`/Open Detector.app for the project at `root`, replacing an earlier build. A
    standalone build carries its own Python and code; the default one runs the checkout."""
    app = dest / f"{APP_NAME}.app"
    if app.exists():
        try:
            built_here = plistlib.loads((app / "Contents/Info.plist").read_bytes()).get("CFBundleIdentifier")
        except (OSError, plistlib.InvalidFileException):
            built_here = None
        if built_here != BUNDLE_ID:
            raise SystemExit(f"{app} exists and wasn't built by this script; move it away first.")
        shutil.rmtree(app)
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()
    executable = contents / "MacOS" / APP_NAME
    if standalone:
        # torch's wheels for Apple silicon need macOS 14.
        plist = info_plist(git_version(root), VERSION, "14.0")
        copy_app(root, contents / "Resources/app")
        bundle_python(root, contents / "Resources", executable)
    else:
        plist = info_plist(git_version(root))
        compile_executable(root, executable)
    (contents / "Info.plist").write_bytes(plistlib.dumps(plist))
    if icon:
        render_icon(root / "ui/assets/mark.svg", contents / "Resources/AppIcon.icns")
    if standalone:
        sign_code(contents / "Resources")
    # Apple silicon runs only signed code; an ad-hoc signature is enough on the Mac that built the app.
    # Other Macs ask their user to allow it once (see README.md).
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True, capture_output=True)
    if register and os.path.exists(LSREGISTER):
        # So the Finder, Dock and Spotlight pick up the new icon and settings straight away.
        subprocess.run([LSREGISTER, "-f", str(app)], check=False)
    return app


def make_dmg(app: Path, out: Path) -> Path:
    """Pack `app` into a disk image that opens on the app beside a link to /Applications, the usual
    drag-to-install window."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / APP_NAME
        stage.mkdir()
        subprocess.run(["ditto", str(app), str(stage / app.name)], check=True)
        (stage / "Applications").symlink_to("/Applications")
        out.unlink(missing_ok=True)
        subprocess.run(["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", str(stage), "-fs", "HFS+",
                        "-format", "UDZO", "-imagekey", "zlib-level=9", "-quiet", str(out)], check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, help="folder to put the app in (default: ~/Applications, or dist/ "
                                                  "with --standalone)")
    parser.add_argument("--standalone", action="store_true",
                        help="carry its own Python and code, for other Macs (needs uv)")
    parser.add_argument("--dmg", action="store_true", help=f"also pack a standalone app into dist/{DMG_NAME}")
    args = parser.parse_args()
    if args.dmg and not args.standalone:
        parser.error("--dmg packs the app other Macs download: add --standalone")
    dest = (args.dest or (ROOT / "dist" if args.standalone else Path.home() / "Applications")).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    app = build(dest, standalone=args.standalone, register=not args.standalone)
    print(f"Built {app}")
    if args.dmg:
        print(f"Packed {make_dmg(app, ROOT / 'dist' / DMG_NAME)}")
    elif not args.standalone:
        print(f"Open it from Launchpad, Spotlight or the Finder, or run: open {shlex.quote(str(app))}")


if __name__ == "__main__":
    main()
