"""Build Open Detector.app, so the detector opens like any other Mac app: from the Dock, Launchpad,
Spotlight or the Finder, in a window of its own (deploy/macos/launcher.py).

    .venv/bin/python deploy/macos/build_app.py                     # into ~/Applications
    .venv/bin/python deploy/macos/build_app.py --dest /Applications

The app runs this checkout's code with its .venv and holds no copy of the project, so a git pull, a
refit or a new style profile shows up the next time it opens. Build it again after moving the
project folder. The icon is drawn from ui/assets/mark.svg.

The app's executable is a small C program, compiled here with the Xcode command line tools
(xcode-select --install), that runs the launcher on the .venv's Python inside the app's own
process. A script that hands over to python with exec would lose the process: macOS then reports
no process for the app, which breaks accessibility and switching to the open window.
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
LSREGISTER = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework"
              "/Support/lsregister")

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


def info_plist(version: str) -> dict:
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": version,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "11.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},  # the window loads http://127.0.0.1
    }


def git_version(root: Path) -> str:
    found = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                           check=False)
    return found.stdout.strip() or "0"


def compile_executable(root: Path, out: Path) -> None:
    """Compile the app's executable against the Python the project's .venv was made from."""
    python = root / ".venv/bin/python"
    config = dict(line.split(" = ", 1) for line in (root / ".venv/pyvenv.cfg").read_text().splitlines()
                  if " = " in line)
    # pyvenv.cfg's home is uv's folder for the minor version, which survives patch updates.
    prefix = Path(config["home"]).parent
    version = subprocess.run([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, check=True).stdout.strip()
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
    with tempfile.TemporaryDirectory() as tmp:
        main = Path(tmp) / "main.c"
        main.write_text(source)
        subprocess.run(["clang", "-O2", "-mmacosx-version-min=11.0", "-Wl,-headerpad_max_install_names",
                        "-I", str(prefix / "include" / f"python{version}"), str(main), str(library), "-o", str(out)],
                       check=True)
    # The library names itself by its patch-version folder; load it through the minor-version one.
    recorded = subprocess.run(["otool", "-D", str(library)], capture_output=True, text=True,
                              check=True).stdout.splitlines()[-1].strip()
    if recorded != str(library):
        subprocess.run(["install_name_tool", "-change", recorded, str(library), str(out)], check=True)


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


def build(dest: Path, root: Path = ROOT, icon: bool = True, register: bool = True) -> Path:
    """Write `dest`/Open Detector.app for the project at `root`, replacing an earlier build."""
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
    (contents / "Info.plist").write_bytes(plistlib.dumps(info_plist(git_version(root))))
    compile_executable(root, contents / "MacOS" / APP_NAME)
    if icon:
        render_icon(root / "ui/assets/mark.svg", contents / "Resources/AppIcon.icns")
    # Apple silicon runs only signed code; an ad-hoc signature is enough on the Mac that built the app.
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True, capture_output=True)
    if register and os.path.exists(LSREGISTER):
        # So the Finder, Dock and Spotlight pick up the new icon and settings straight away.
        subprocess.run([LSREGISTER, "-f", str(app)], check=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=Path.home() / "Applications",
                        help="folder to put the app in (default: ~/Applications)")
    dest = parser.parse_args().dest.expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    app = build(dest)
    print(f"Built {app}\nOpen it from Launchpad, Spotlight or the Finder, or run: open {shlex.quote(str(app))}")


if __name__ == "__main__":
    main()
