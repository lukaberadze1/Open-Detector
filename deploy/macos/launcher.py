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


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
:root {{ --bg: #FFFFFF; --text: #171A26; --muted: #5E6475; --panel: #F4F5F9; --border: #E1E3EB; --accent: #4F46E5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0D0F16; --text: #E7E9F0; --muted: #9AA0B2; --panel: #181B25; --border: #272B38; --accent: #8B93FF; }}
}}
body {{ margin: 0; height: 100vh; display: flex; align-items: center; justify-content: center;
  background: var(--bg); color: var(--text); font: 15px/1.5 -apple-system, BlinkMacSystemFont, sans-serif;
  -webkit-user-select: none; }}
main {{ max-width: 680px; padding: 0 32px; text-align: center; }}
.mark svg {{ width: 72px; height: 72px; }}
h1 {{ font-size: 22px; font-weight: 650; margin: 14px 0 6px; }}
p {{ color: var(--muted); margin: 6px 0; }}
.spinner {{ width: 22px; height: 22px; margin: 22px auto 0; border-radius: 50%; border: 2.5px solid var(--border);
  border-top-color: var(--accent); animation: spin .9s linear infinite; }}
.later {{ opacity: 0; animation: show .4s 15s forwards; }}
pre {{ text-align: left; background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; font: 12px/1.45 ui-monospace, Menlo, monospace; max-height: 42vh; overflow: auto;
  white-space: pre-wrap; -webkit-user-select: text; }}
button {{ font: inherit; font-weight: 550; padding: 7px 16px; margin: 6px 4px 0; border-radius: 9px; cursor: pointer;
  border: 1px solid var(--border); background: var(--panel); color: var(--text); }}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes show {{ to {{ opacity: 1; }} }}
</style></head><body><main><div class="mark">{mark}</div>{body}</main></body></html>"""


def loading_page() -> str:
    return PAGE.format(mark=_mark(), body=(
        f"<h1>{APP_NAME}</h1><p>Starting the detector…</p><div class='spinner'></div>"
        "<p class='later'>The first start after an update or restart can take a minute.</p>"))


def error_page(message: str) -> str:
    tail = log_tail()
    return PAGE.format(mark=_mark(), body=(
        f"<h1>{APP_NAME} stopped</h1><p>{html.escape(message)}</p>"
        + (f"<pre>{html.escape(tail)}</pre>" if tail else "")
        + "<button class='primary' onclick='pywebview.api.restart()'>Start again</button>"
          "<button onclick='pywebview.api.show_log()'>Show the log</button>"))


def _mark() -> str:
    try:
        return (ROOT / "ui/assets/mark.svg").read_text()
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
            background_color="#0D0F16" if dark_mode() else "#FFFFFF")
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
