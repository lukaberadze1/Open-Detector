"""Open Detector as a Mac app: the Streamlit app in a native window of its own.

    .venv/bin/python deploy/macos/launcher.py      # what Open Detector.app runs; works from a terminal too

Starts the app's server on 127.0.0.1 only, shows a loading page until it answers, then shows the
app in a WebKit window. Closing the window or quitting stops the server, after asking first if a Red
team batch is still running. If the launcher dies some other way, the server notices and stops
itself. The server's output goes to ~/Library/Logs/Open Detector/server.log, and the page offers it
if the server fails. build_app.py wraps this script in Open Detector.app.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The app's data folder, as opendetector/config.py finds it (this script runs before any of the app loads).
DATA_ROOT = Path(os.environ["OPENDETECTOR_HOME"]).expanduser() if os.environ.get("OPENDETECTOR_HOME") else ROOT
APP_NAME = "Open Detector"
LOG_PATH = Path.home() / "Library/Logs" / APP_NAME / "server.log"
WINDOW_STATE = Path.home() / "Library/Application Support" / APP_NAME / "window.json"
# A fixed port keeps the page's address the same between launches, so what the page stores in the
# browser, such as the chosen theme, lasts. Any free port is used when something else holds it.
PREFERRED_PORT = 8517
START_TIMEOUT = 300  # seconds; the first start after an update reads torch from a cold disk
DEFAULT_SIZE = (1360, 900)
MIN_SIZE = (960, 640)


def streamlit_args(port: int) -> list[str]:
    """`streamlit run` for the desktop: headless, reachable from this computer only, no file watcher,
    and the viewer's menu (settings and theme) without the developer and deploy options."""
    return ["run", str(ROOT / "app.py"), "--server.headless=true", "--server.address=127.0.0.1",
            f"--server.port={port}", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false",
            "--client.toolbarMode=viewer"]


def pick_port(preferred: int = PREFERRED_PORT) -> int:
    for port in (preferred, 0):
        with socket.socket() as sock:
            # As the server binds: a port still in use is refused, closed connections of a last run aren't.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise OSError("no free port on 127.0.0.1")


def serve(port: int, parent: int) -> None:
    """The server process: the Streamlit app on `port`, stopping itself once `parent` has gone (a
    force quit or crash of the launcher leaves no server behind)."""

    def watch_parent() -> None:
        while os.getppid() == parent:
            time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)  # Streamlit's own handler shuts the server down
        time.sleep(15)
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    from streamlit.web import cli

    sys.argv = ["streamlit", *streamlit_args(port)]
    sys.exit(cli.main())


def start_server(port: int) -> subprocess.Popen:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.replace(LOG_PATH.with_name("server.previous.log"))
    # Apps opened from the Finder get a bare PATH; add Homebrew's for the tools the app calls.
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "PATH": os.pathsep.join([os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
                                    "/opt/homebrew/bin", "/usr/local/bin"])}
    env.pop("PYTHONEXECUTABLE", None)  # set by the app bundle for the launcher only
    with LOG_PATH.open("w") as log:
        return subprocess.Popen([sys.executable, __file__, "--serve", str(port), "--parent", str(os.getpid())],
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def wait_until_ready(url: str, server: subprocess.Popen, timeout: float = START_TIMEOUT) -> bool:
    """True once the server answers its health check; False if it exits or the time runs out."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a system proxy
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and server.poll() is None:
        try:
            with opener.open(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.3)
    return False


def batch_running(server: subprocess.Popen) -> bool:
    """Whether the Red team page is running a batch: it keeps the Mac awake with caffeinate, a child
    of the server, for exactly that long."""
    if server.poll() is not None:
        return False
    found = subprocess.run(["pgrep", "-P", str(server.pid), "-x", "caffeinate"], capture_output=True, check=False)
    return found.returncode == 0


def log_tail(lines: int = 30) -> str:
    try:
        return "\n".join(LOG_PATH.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def dark_mode() -> bool:
    found = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True,
                           check=False)
    return found.stdout.strip() == "Dark"


def look() -> str:
    """The look picked under Settings in the app (ui/prefs.py keeps it in data/ui_prefs.json), so the
    loading and error pages match the app they open into."""
    try:
        chosen = json.loads((DATA_ROOT / "data/ui_prefs.json").read_text()).get("look")
    except (OSError, ValueError, AttributeError):
        chosen = None
    return chosen if chosen in ("glass", "instrument") else "glass"


# Window background behind the first paint, per look and mode.
BACKGROUNDS = {("glass", False): "#EEF0F8", ("glass", True): "#06070D",
               ("instrument", False): "#EEF1F0", ("instrument", True): "#05070A"}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
:root {{ --bg: #EEF0F8; --text: #0E1222; --muted: #545A72; --panel: rgba(255,255,255,.6); --edge: rgba(255,255,255,.85);
  --hair: rgba(20,24,48,.1); --accent: #5B57F5; --accent2: #0E9FB8;
  --sky: radial-gradient(50rem 34rem at 10% -10%, rgba(140,132,255,.38), transparent 60%),
         radial-gradient(44rem 30rem at 100% 0%, rgba(80,210,235,.3), transparent 60%),
         radial-gradient(48rem 34rem at 70% 115%, rgba(255,160,200,.28), transparent 60%); }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #06070D; --text: #ECEEF7; --muted: #A0A6BE; --panel: rgba(24,27,44,.55); --edge: rgba(255,255,255,.1);
    --hair: rgba(255,255,255,.08); --accent: #8F8CFF; --accent2: #3FD3EA;
    --sky: radial-gradient(50rem 34rem at 5% -10%, rgba(92,72,255,.3), transparent 60%),
           radial-gradient(44rem 30rem at 100% 5%, rgba(20,170,200,.2), transparent 60%),
           radial-gradient(48rem 34rem at 65% 115%, rgba(170,60,220,.2), transparent 60%); }}
}}
body.instrument {{ --bg: #EEF1F0; --text: #0A1E25; --muted: #4A6470; --panel: #fff; --edge: rgba(0,70,90,.24);
  --hair: rgba(0,70,90,.12); --accent: #00788C; --accent2: #4C7A00;
  --sky: linear-gradient(rgba(0,90,110,.06) 1px, transparent 1px) 0 0 / 32px 32px,
         linear-gradient(90deg, rgba(0,90,110,.06) 1px, transparent 1px) 0 0 / 32px 32px; }}
@media (prefers-color-scheme: dark) {{
  body.instrument {{ --bg: #05070A; --text: #D9F4FA; --muted: #7896A1; --panel: rgba(9,15,20,.9);
    --edge: rgba(90,230,255,.2); --hair: rgba(90,230,255,.1); --accent: #45E0FF; --accent2: #B6FF5C;
    --sky: linear-gradient(rgba(90,230,255,.045) 1px, transparent 1px) 0 0 / 32px 32px,
           linear-gradient(90deg, rgba(90,230,255,.045) 1px, transparent 1px) 0 0 / 32px 32px,
           radial-gradient(60rem 40rem at 50% -20%, rgba(69,224,255,.1), transparent 70%); }}
}}
body {{ margin: 0; height: 100vh; display: flex; align-items: center; justify-content: center;
  background: var(--sky), var(--bg); color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif; -webkit-user-select: none; }}
main {{ position: relative; max-width: 620px; padding: 40px 44px 36px; text-align: center; border-radius: 28px;
  background: var(--panel); border: 1px solid var(--edge);
  box-shadow: 0 1px 0 rgba(255,255,255,.12) inset, 0 30px 80px -30px rgba(0,0,0,.45);
  -webkit-backdrop-filter: blur(30px) saturate(170%); backdrop-filter: blur(30px) saturate(170%); }}
.mark {{ position: relative; width: 84px; height: 84px; margin: 0 auto; }}
.mark svg {{ position: absolute; inset: 12px; width: 60px; height: 60px; }}
.halo {{ position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(from 0deg, transparent 0 55%, var(--accent2), var(--accent), transparent 95%);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 3px), #000 calc(100% - 2px));
  animation: spin 1.4s linear infinite; }}
.stopped .halo {{ animation: none; background: var(--hair); }}
h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -.02em; margin: 18px 0 4px; }}
p {{ color: var(--muted); margin: 6px 0; }}
.bar {{ position: relative; height: 3px; width: 220px; margin: 22px auto 0; border-radius: 3px; background: var(--hair);
  overflow: hidden; }}
.bar i {{ position: absolute; inset: 0 auto 0 0; width: 40%; border-radius: 3px;
  background: linear-gradient(90deg, transparent, var(--accent), var(--accent2));
  animation: sweep 1.6s ease-in-out infinite; }}
.later {{ opacity: 0; animation: show .4s 15s forwards; font-size: 13px; }}
pre {{ text-align: left; background: rgba(127,127,127,.08); border: 1px solid var(--hair); border-radius: 12px;
  padding: 12px 14px; font: 12px/1.45 ui-monospace, Menlo, monospace; max-height: 42vh; overflow: auto;
  white-space: pre-wrap; -webkit-user-select: text; color: var(--text); }}
button {{ font: inherit; font-weight: 600; padding: 8px 18px; margin: 10px 4px 0; border-radius: 999px; cursor: pointer;
  border: 1px solid var(--hair); background: transparent; color: var(--text); }}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
body.instrument main {{ border-radius: 4px; box-shadow: 0 24px 60px -30px rgba(0,0,0,.8); backdrop-filter: none;
  -webkit-backdrop-filter: none;
  background-image: linear-gradient(var(--accent), var(--accent)), linear-gradient(var(--accent), var(--accent)),
    linear-gradient(var(--accent), var(--accent)), linear-gradient(var(--accent), var(--accent));
  background-size: 16px 1.5px, 1.5px 16px, 16px 1.5px, 1.5px 16px;
  background-position: top left, top left, bottom right, bottom right; background-repeat: no-repeat; }}
body.instrument h1, body.instrument p, body.instrument button {{ font-family: "SF Mono", ui-monospace, Menlo, monospace;
  text-transform: uppercase; letter-spacing: .1em; }}
body.instrument h1 {{ font-size: 17px; letter-spacing: .2em; }}
body.instrument p {{ font-size: 12px; }}
body.instrument button {{ border-radius: 3px; font-size: 12px; }}
body.instrument button.primary {{ color: var(--bg); }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes sweep {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(260%); }} }}
@keyframes show {{ to {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ .halo, .bar i {{ animation: none; }} }}
</style></head><body class="{look}"><main class="{state}"><div class="mark"><div class="halo"></div>{mark}</div>
{body}</main></body></html>"""


def loading_page() -> str:
    starting = "Booting the detector" if look() == "instrument" else "Starting the detector…"
    return PAGE.format(look=look(), state="loading", mark=_mark(), body=(
        f"<h1>{APP_NAME}</h1><p>{starting}</p><div class='bar'><i></i></div>"
        "<p class='later'>The first start after an update or restart can take a minute.</p>"))


def error_page(message: str) -> str:
    tail = log_tail()
    return PAGE.format(look=look(), state="stopped", mark=_mark(), body=(
        f"<h1>{APP_NAME} stopped</h1><p>{html.escape(message)}</p>"
        + (f"<pre>{html.escape(tail)}</pre>" if tail else "")
        + "<button class='primary' onclick='pywebview.api.restart()'>Start again</button>"
          "<button onclick='pywebview.api.show_log()'>Show the log</button>"))


def _mark() -> str:
    name = "mark-instrument.svg" if look() == "instrument" else "mark.svg"
    try:
        return (ROOT / "ui/assets" / name).read_text()
    except OSError:
        return ""


def add_undo_to_edit_menu() -> None:
    """pywebview's Edit menu has no Undo or Redo, and without them ⌘Z does nothing in a text box.
    Adds them on the main thread once pywebview has installed its menus."""
    import AppKit
    from PyObjCTools import AppHelper

    def add() -> None:
        main = AppKit.NSApplication.sharedApplication().mainMenu()
        menus = [item.submenu() for item in main.itemArray()] if main else []
        edit = next((menu for menu in menus if menu is not None and menu.title() == "Edit"), None)
        if edit is None:
            AppHelper.callLater(0.5, add)
        elif edit.itemWithTitle_("Undo") is None:
            edit.insertItem_atIndex_(AppKit.NSMenuItem.separatorItem(), 0)
            edit.insertItemWithTitle_action_keyEquivalent_atIndex_("Redo", "redo:", "Z", 0)  # ⇧⌘Z
            edit.insertItemWithTitle_action_keyEquivalent_atIndex_("Undo", "undo:", "z", 0)

    AppHelper.callAfter(add)


class DesktopApp:
    """The window, the server behind it, and what happens when either stops."""

    def __init__(self):
        self.window = None
        self.server: subprocess.Popen | None = None
        self.quitting = False
        self.size = self._saved_size()

    def run(self) -> None:
        import webview

        webview.settings["ALLOW_DOWNLOADS"] = True  # download buttons open a save panel
        self.window = webview.create_window(
            APP_NAME, html=loading_page(), js_api=PageApi(self), width=self.size[0], height=self.size[1],
            min_size=MIN_SIZE, text_select=True, zoomable=True,
            background_color=BACKGROUNDS[(look(), dark_mode())])
        self.window.events.closing += self.closing
        self.window.events.resized += self.resized
        try:
            webview.start(self.start, private_mode=False)
        finally:
            self.quitting = True
            self.stop_server(wait=True)

    def start(self) -> None:
        """Start a server and show it once it answers; show what went wrong if it stops."""
        add_undo_to_edit_menu()
        port = pick_port()
        server = self.server = start_server(port)
        url = f"http://127.0.0.1:{port}"
        if not wait_until_ready(url, server):
            if not self.quitting:
                self.stop_server()
                self.window.load_html(error_page(
                    "The detector's server stopped while starting." if server.poll() is not None
                    else f"The detector's server didn't answer within {START_TIMEOUT // 60} minutes."))
            return
        self.window.load_url(url)
        code = server.wait()
        if not self.quitting and server is self.server:
            how = f"was ended by signal {-code}" if code < 0 else f"stopped with exit code {code}"
            self.window.load_html(error_page(f"The detector's server {how}. A computer short of memory can end it."))

    def restart(self) -> None:
        self.window.load_html(loading_page())
        threading.Thread(target=self.start, daemon=True).start()

    def stop_server(self, wait: bool = False) -> None:
        server = self.server
        if server is None or server.poll() is not None:
            return
        server.terminate()
        if wait:
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

    def closing(self):
        """Runs on the main thread before the window closes, from its close button or from Quit."""
        if self.server and batch_running(self.server) and not self._confirm_quit():
            return False
        self.quitting = True
        self._save_size()
        self.stop_server()
        return None

    def resized(self, width: float, height: float) -> None:
        self.size = (int(width), int(height))

    @staticmethod
    def _confirm_quit() -> bool:
        import AppKit

        alert = AppKit.NSAlert.alloc().init()
        alert.setMessageText_("A Red team batch is still running")
        alert.setInformativeText_("Quitting stops it. The blocks written so far are saved: start the same batch "
                                  "again and press Resume to continue from there.")
        alert.addButtonWithTitle_("Keep Running")
        alert.addButtonWithTitle_("Quit")
        return alert.runModal() == AppKit.NSAlertSecondButtonReturn

    @staticmethod
    def _saved_size() -> tuple[int, int]:
        try:
            width, height = json.loads(WINDOW_STATE.read_text())["size"]
            return max(int(width), MIN_SIZE[0]), max(int(height), MIN_SIZE[1])
        except (OSError, ValueError, KeyError, TypeError):
            return DEFAULT_SIZE

    def _save_size(self) -> None:
        try:
            WINDOW_STATE.parent.mkdir(parents=True, exist_ok=True)
            WINDOW_STATE.write_text(json.dumps({"size": list(self.size)}))
        except OSError:
            pass


class PageApi:
    """What the loading and error pages can call, as pywebview.api.<name>()."""

    def __init__(self, app: DesktopApp):
        self._app = app

    def restart(self) -> None:
        self._app.restart()

    def show_log(self) -> None:
        subprocess.run(["open", "-R", str(LOG_PATH)], check=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--serve", type=int, metavar="PORT", help=argparse.SUPPRESS)
    parser.add_argument("--parent", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.serve:
        serve(args.serve, args.parent or os.getppid())
    else:
        DesktopApp().run()


if __name__ == "__main__":
    main()
