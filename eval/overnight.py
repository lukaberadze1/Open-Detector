"""The overnight tests: one model-heavy job at a time, most decision-worthy first, and no failure ends the night.

Each stage has a time limit and a log of its own (eval/cache/overnight/logs/), a crash is tried again, and a stage that
still fails is noted and passed by, along with anything that needs its result. Stages save as they go, so running this
again carries on: finished stages are skipped. Gemini's comparisons run in rounds, one repeat of every setup each, so a
quota that runs out leaves the setups equally measured. A spent model hands on to the next, which starts its comparison
afresh, since scores aren't compared across models. An overloaded model isn't swapped out, since that would split the
comparison: the round is put off and a later round writes the repeats it missed. eval/cache/overnight/BRIEF.md is
rewritten after every stage, so
there is something to read whenever you look. It keeps the Mac awake on mains power itself.

    python -m eval.overnight --until 10:00
    python -m eval.overnight --status
    python -m eval.overnight --smoke          # every stage, tiny, in its own folder: about 30 minutes
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
QUOTA_EXIT = 3  # as eval.tune_revise exits on a spent quota
DOWN_EXIT = 4  # and on a model that stays overloaded
MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
QUOTA_RESET = (9, 5)  # local hour and minute: Gemini's daily quota resets at midnight Pacific, 09:00 in Georgia's west
QUOTA_WAIT = 600  # seconds before believing a spent quota, since a per-minute limit can read the same
CRASH_WAIT = 120
ATTEMPTS = 2
RESERVE = 20  # minutes kept back at the end for the closing stages
MEMORY_GB = 1.5  # wanted free before a stage starts; it starts anyway after MEMORY_PATIENCE seconds
MEMORY_PATIENCE = 300
DOCUMENTS = ROOT / "data/essays/sources/revise"
DAIGT_CSV = ROOT / "data/pairs/train_v2_drcat_02.csv"
DEMOS = ROOT / "data/pairs/humanize_text_demo"


@dataclass
class Stage:
    name: str
    question: str  # what its result lets us decide
    minutes: int = 60
    command: list[str] | None = None
    call: Callable[[], str | None] | None = None
    needs: list[Path] = field(default_factory=list)
    needs_rows: bool = False  # the Gemini runs have to have produced something
    fresh: list[Path] = field(default_factory=list)  # deleted before each try: it appends and can't resume
    gemini: int = 0  # the round of the comparison this is, or 0
    closing: bool = False  # runs even past the deadline
    again: bool = False  # runs on every start, since what it reads grows


def py(*arguments: object) -> list[str]:
    return [PYTHON, "-m", *map(str, arguments)]


class Paths:
    def __init__(self, smoke: bool) -> None:
        self.out = ROOT / "eval/cache/overnight" / ("smoke" if smoke else "")
        self.logs = self.out / "logs"
        self.state = self.out / "state.json"
        self.brief = self.out / "BRIEF.md"
        self.revise = self.out / "revise.jsonl"
        self.summary = self.out / "summary.md"
        self.revised = self.out / "revised.jsonl"
        self.revised_math = self.out / "revised_math.jsonl"
        self.samples = self.out / "redteam_samples.jsonl"
        self.humanizers = self.out / "humanizers.jsonl" if smoke else ROOT / "data/essays/humanizer_redteam.jsonl"
        self.humanizer_report = self.out / "humanizers.md" if smoke else ROOT / "eval/reports/humanizers.md"
        self.features = self.out / "redteam_features.jsonl"
        self.reference_before = self.out / "reference_before.md"
        self.anatomy = self.out / "anatomy.md"
        self.humanize_demo = self.out / "humanize_demo.md" if smoke else ROOT / "eval/reports/humanize_demo.md"
        self.local_hop = self.out / "local_hop.md" if smoke else ROOT / "eval/reports/local_hop.md"
        self.daigt = self.out / "daigt.jsonl" if smoke else ROOT / "eval/cache/daigt.jsonl"
        self.trial = self.out / "trial_combiner_slim.json" if smoke else ROOT / "models/trial_combiner_slim.json"
        self.trial_fit = self.out / "trial_fit.md" if smoke else ROOT / "eval/reports/trial_slim_fit.md"
        self.trial_report = self.out / "trial_compare.md" if smoke else ROOT / "eval/reports/trial_slim.md"
        self.redteam = self.out / "redteam.md" if smoke else ROOT / "eval/reports/redteam.md"
        self.reference = self.out / "reference.md" if smoke else ROOT / "eval/reports/reference.md"


def gemini_setups(paths: Paths, smoke: bool) -> list[tuple[str, list[str]]]:
    return [("baseline", []), ("multistart", ["--drafts", 2 if smoke else 3]),
            ("roundtrip", ["--attack", "roundtrip-german"]), ("masked", ["--mask-facts"])]


def gemini_command(paths: Paths, smoke: bool, setup: str, extra: list, repeats: int, model: str) -> list[str]:
    documents = sorted(DOCUMENTS.glob("*.md"))[:1 if smoke else None]
    tiny = ["--rounds", 1, "--no-log"] if smoke else []
    return py("eval.tune_revise", *documents, "--as-page", "--repeats", repeats, "--model", model,
              "--out", paths.revise, "--report", paths.out / f"{setup}_{model}.md", *extra, *tiny)


def write_samples(paths: Paths, limit: int | None) -> str:
    from eval.trial_compare import collect_samples

    return f"{collect_samples(paths.samples, limit)} saved samples"


def export_revised(paths: Paths) -> str:
    return "exported" if subprocess.run(py("eval.overnight_summary", paths.revise, "--export", paths.revised),
                                        cwd=ROOT, capture_output=True).returncode == 0 else "nothing to export"


def plan(paths: Paths, smoke: bool) -> list[Stage]:
    """In order of what the night is for. First where we stand, then what decides the generator's next step, then the
    slow work on the detector, and last whatever the remaining hours can buy."""
    small = smoke
    reference_small = ["--mage", 8, "--no-saved", "--daigt", 0, "--hc3", 0] if small else []
    board = lambda number: Stage(  # noqa: E731
        f"scoreboard_{number}", "Which setup to build on; refreshed after each round.", 20,
        py("eval.overnight_summary", paths.revise, "--report", paths.summary, "--math", paths.revised_math),
        needs_rows=True)
    return [
        Stage("tests", "Is the code sound before trusting anything below?", 10,
              py("pytest", "-q", "--no-header", "-p", "no:cacheprovider")),
        Stage("reference_before", "Where do we stand, before tonight's samples? Do guided samples fool only Desklib "
              "(the transfer gap against RoBERTa)?", 90,
              py("eval.reference_benchmark", "--detectors", "slim,roberta", "--report", paths.reference_before,
                 *reference_small)),
        Stage("humanize_demo", "What does a trained humanizer's output do to the detector, and does it flag the human "
              "rewrites?", 90,
              py("eval.score_demo", "--report", paths.humanize_demo, *(["--limit", 2] if small else [])),
              needs=[DEMOS / "kto_seed_demo.jsonl"]),
        Stage("local_hop", "Does the local translation hop move AI text, and does it bring text into the math script's "
              "range?", 120,
              py("eval.bench_hop", "--texts", 1 if small else 20, "--pivots", *(["German"] if small else
                 ["German", "Finnish"]), "--math", "--budget", 10 if small else 150, "--independent",
                 "--out", paths.out / "hop.jsonl", "--report", paths.local_hop)),
        Stage("humanizer_rules", "How do the public rule-based humanizers fare? No model, so it is quick.", 30,
              py("eval.humanizer_attack", "--n", 2 if small else 40, "--no-report", "--out", paths.humanizers)),
        Stage("gemini_round_1", "Does multi-start, the German round trip or holding names and numbers beat the current "
              "recipe?", 360, gemini=1),
        board(1),
        Stage("anatomy", "What does Desklib read: words, order, punctuation? Which edits can move it?", 150,
              py("eval.detector_anatomy", "--texts", 2 if small else 20, "--out", paths.out / "anatomy.jsonl",
                 "--report", paths.anatomy)),
        Stage("borderline_math", "With the ranking fixed, how far does math alone move borderline text, and does the "
              "move survive a detector it never saw?", 150,
              py("eval.bench_math", "--band", 0.6, 0.95, "--texts", 1 if small else 17, "--independent",
                 "--out", paths.out / "borderline.jsonl", *(["--budget", 10] if small else []))),
        Stage("humanizer_bart", "Do the public BART humanizers beat the detector without using its scores?", 180,
              py("eval.humanizer_attack", "--n", 1 if small else 40, "--no-report", "--humanizers",
                 "stealth-bart,cive-bart", "--out", paths.humanizers)),
        Stage("humanizer_report", "The humanizers' scores, the same texts before and after.", 45,
              py("eval.humanizer_attack", "--report-only", "--out", paths.humanizers, "--report",
                 paths.humanizer_report), needs=[paths.humanizers]),
        Stage("gemini_round_2", "The same comparison with a second repeat: is the first round's answer stable?", 300,
              gemini=2),
        Stage("export_revised", "Hands the revised texts to the next stage.", 2, call=lambda: export_revised(paths),
              needs_rows=True, again=True),
        Stage("math_after_revision", "Does the math script finish off what revision leaves between 15% and 95%?", 120,
              py("eval.math_attack", paths.revised, "--score", "--band", 0.15, 0.95, "--out", paths.revised_math,
                 *(["--budget", 10] if small else [])), needs=[paths.revised], fresh=[paths.revised_math]),
        board(2),
        Stage("daigt_scoring", "Scores the new training essays for the trial fit.", 270,
              py("eval.featurize", "daigt", "--n", 4 if small else 3000, "--engines", "desklib", "--device", "cpu",
                 "--dtype", "float32", "--out", paths.daigt), needs=[DAIGT_CSV]),
        Stage("trial_fit", "Fits a trial slim combiner on the wider data; the shipped ones stay as they are.", 10,
              py("eval.fit_combiner", "eval/cache/raid.jsonl", "eval/cache/mage.jsonl", paths.daigt, "--features",
                 "desklib_mean,desklib_max", "--out", paths.trial, "--report", paths.trial_fit), needs=[paths.daigt]),
        Stage("collect_samples", "Gathers every saved sample, including tonight's.", 5,
              call=lambda: write_samples(paths, 3 if small else None), again=True),
        Stage("redteam_scoring", "Scores those samples for the comparison below.", 120,
              py("eval.featurize", "local", "--path", paths.samples, "--engines", "desklib", "--device", "cpu",
                 "--dtype", "float32", "--out", paths.features), needs=[paths.samples]),
        Stage("trial_compare", "Does refitting the combiner catch guided samples, or is retraining the model the only "
              "way?", 10,
              py("eval.trial_compare", paths.trial, paths.daigt, "eval/cache/mage.jsonl", "eval/cache/raid.jsonl",
                 "--redteam", paths.features, "--report", paths.trial_report), needs=[paths.trial, paths.features]),
        Stage("redteam_report", "How much of everything saved gets through, by attack?", 45,
              py("eval.redteam_report", *([paths.samples] if small else []), "--report", paths.redteam)),
        Stage("gemini_round_3", "A third repeat, if quota and time allow.", 240, gemini=3),
        board(3),
        Stage("reference_full", "The same grading on the full build; it picks up from its saved scores and takes the "
              "hours left over.", 600,
              py("eval.reference_benchmark", "--report", paths.reference, *reference_small)),
        Stage("reference_report", "Reports on whatever the full build got through.", 15,
              py("eval.reference_benchmark", "--cached-only", "--report", paths.reference, *reference_small),
              closing=True, again=True),
        Stage("scoreboard_final", "The final table.", 20,
              py("eval.overnight_summary", paths.revise, "--report", paths.summary, "--math", paths.revised_math),
              needs_rows=True, closing=True, again=True),
    ]


def available_gb() -> float | None:
    try:
        output = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pages = {line.split(":")[0]: int(line.split(":")[1].strip().rstrip(".")) for line in output.splitlines()[1:]
             if ":" in line and line.split(":")[1].strip().rstrip(".").isdigit()}
    free = sum(pages.get(name, 0) for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return free * os.sysconf("SC_PAGE_SIZE") / 2 ** 30


def checks() -> list[str]:
    """What could spoil the night and can be seen now. Warnings, never reasons to stop."""
    from opendetector.env import load_env

    notes = []
    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        notes.append("GEMINI_API_KEY isn't set: the Gemini rounds will fail and the rest will run.")
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    if "AC Power" not in power:
        notes.append("The Mac is on battery: it can sleep or run out, and keeping it awake only works on mains power.")
    if (free := shutil.disk_usage(ROOT).free / 2 ** 30) < 5:
        notes.append(f"Only {free:.0f} GB of disk left.")
    apps = subprocess.run(["pgrep", "-f", "streamlit run"], capture_output=True, text=True).stdout.split()
    if apps:
        notes.append("The Streamlit app is running: quit it, since it takes memory the tests need on 8 GB.")
    if not sorted(DOCUMENTS.glob("*.md")):
        notes.append(f"No documents in {DOCUMENTS}: the Gemini rounds will have nothing to do.")
    return notes


class Night:
    def __init__(self, paths: Paths, stages: list[Stage], until: datetime, smoke: bool,
                 run: Callable[..., int] | None = None, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], datetime] = datetime.now, models: tuple[str, ...] = MODELS) -> None:
        self.paths, self.stages, self.until, self.smoke, self.models = paths, stages, until, smoke, models
        self.sleep, self.now = sleep, now
        self.run_command = run or self._run_command
        self.child: subprocess.Popen | None = None
        paths.logs.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(paths.state.read_text()) if paths.state.exists() else {"stages": {}, "spent": {}}
        self.warnings: list[str] = []

    def save(self) -> None:
        temporary = self.paths.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=1))
        temporary.replace(self.paths.state)
        self.write_brief()

    def note(self, stage: Stage, status: str, note: str = "", minutes: float = 0.0) -> None:
        entry = self.state["stages"].setdefault(stage.name, {"minutes": 0.0})
        entry.update(status=status, note=note, minutes=round(entry["minutes"] + minutes, 1),
                     at=self.now().strftime("%H:%M"))
        print(f"{self.now():%H:%M} {stage.name}: {status} {note}", flush=True)
        self.save()

    def minutes_left(self, stage: Stage) -> float:
        left = (self.until - self.now()).total_seconds() / 60
        return stage.minutes if stage.closing else min(stage.minutes, left - RESERVE)

    def _run_command(self, command: list[str], log: Path, minutes: float) -> int:
        """Its exit code, or None past its time limit. In a process group of its own, so nothing is left running."""
        with log.open("a", encoding="utf-8") as sink:
            sink.write(f"\n=== {self.now():%H:%M} {' '.join(command)[:300]}\n")
            sink.flush()
            environment = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT, env=environment,
                                          start_new_session=True)
            try:
                return self.child.wait(timeout=minutes * 60)
            except subprocess.TimeoutExpired:
                self.stop_child()
                return None
            finally:
                self.child = None

    def stop_child(self) -> None:
        if self.child and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
                self.child.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                with contextlib.suppress(OSError):
                    os.killpg(self.child.pid, signal.SIGKILL)

    def wait_for_memory(self) -> None:
        waited = 0
        while waited < MEMORY_PATIENCE and (free := available_gb()) is not None and free < MEMORY_GB:
            self.sleep(30)
            waited += 30

    def missing(self, stage: Stage) -> str | None:
        for path in stage.needs:
            if not path.exists() or not path.stat().st_size:
                return f"needs {path.name}"
        if stage.needs_rows and not (self.paths.revise.exists() and self.paths.revise.stat().st_size):
            return "no Gemini runs yet"
        return None

    def run_stage(self, stage: Stage) -> None:
        if (why := self.missing(stage)):
            self.note(stage, "skipped", why)
            return
        limit = self.minutes_left(stage)
        if limit <= 0:
            self.note(stage, "out of time", "resumes on the next start")
            return
        began = time.monotonic()
        self.note(stage, "running")
        if stage.call:
            try:
                note = stage.call() or ""
                self.note(stage, "done", note, (time.monotonic() - began) / 60)
            except Exception as exc:  # a closing step that fails mustn't end the night
                self.note(stage, "failed", f"{type(exc).__name__}: {str(exc)[:150]}", (time.monotonic() - began) / 60)
            return
        for attempt in range(1, ATTEMPTS + 1):
            self.wait_for_memory()
            for path in stage.fresh:
                path.unlink(missing_ok=True)
            code = self.run_command(stage.command, self.paths.logs / f"{stage.name}.log", self.minutes_left(stage))
            spent = (time.monotonic() - began) / 60
            if code == 0:
                self.note(stage, "done", "", spent)
                return
            if code is None:
                self.note(stage, "timed out", "what it saved is kept; resumes on the next start", spent)
                return
            if attempt < ATTEMPTS:
                self.sleep(CRASH_WAIT)
        self.note(stage, "failed", f"exit {code}, see logs/{stage.name}.log", spent)

    def spent(self, model: str) -> bool:
        """Spent today, as far as we know: a model that ran out before this morning's reset is back."""
        when = self.state["spent"].get(model)
        if not when:
            return False
        ran_out, now = datetime.fromisoformat(when), self.now()
        reset = now.replace(hour=QUOTA_RESET[0], minute=QUOTA_RESET[1], second=0, microsecond=0)
        return not (ran_out < reset <= now)

    def finished_runs(self) -> int:
        revise = self.paths.revise
        return sum(1 for _ in revise.open(encoding="utf-8")) if revise.exists() else 0

    def run_gemini(self, stage: Stage) -> None:
        began = time.monotonic()
        runs_before = self.finished_runs()
        self.note(stage, "running")
        finished, failures = 0, []
        setups = gemini_setups(self.paths, self.smoke)
        names = [name for name, _ in setups]
        model = next((m for m in self.models if not self.spent(m)), None)
        index = 0
        believed: set[str] = set()
        while model and index < len(setups):
            limit = self.minutes_left(stage) - (time.monotonic() - began) / 60
            if limit <= 0:
                self.note(stage, "timed out", f"{model}: reached {names[:index]}", (time.monotonic() - began) / 60)
                return
            name, extra = setups[index]
            share = limit / (len(setups) - index)  # so one slow setup can't eat the others' time
            self.wait_for_memory()
            code = self.run_command(gemini_command(self.paths, self.smoke, name, extra, stage.gemini, model),
                                    self.paths.logs / f"{stage.name}.log", share)
            if code == QUOTA_EXIT:
                if model not in believed and not self.smoke:
                    believed.add(model)
                    self.sleep(QUOTA_WAIT)
                    continue
                self.state["spent"][model] = self.now().isoformat()
                model = next((m for m in self.models if not self.spent(m)), None)
                index = 0  # a new model's comparison needs all its setups
                continue
            if code == DOWN_EXIT:
                self.note(stage, "model down", f"{model} overloaded; a later round catches up. "
                          f"{self.finished_runs() - runs_before} new runs", (time.monotonic() - began) / 60)
                return
            if code == 0:
                finished += 1
            elif code is None:
                failures.append(f"{name} timed out")
            else:
                failures.append(f"{name} exit {code}")
            index += 1
        spent = (time.monotonic() - began) / 60
        new = f"{self.finished_runs() - runs_before} new runs"
        if model is None:
            self.note(stage, "quota spent", f"{new}; every model is out until the reset", spent)
        else:
            self.note(stage, "done" if not failures else "partly done", "; ".join([model, new, *failures]), spent)

    def run_all(self) -> None:
        self.warnings = checks() if not self.smoke else []
        self.save()
        for stage in self.stages:
            status = self.state["stages"].get(stage.name, {}).get("status")
            if status == "done" and not stage.again:
                continue
            if not stage.closing and self.minutes_left(stage) <= 0:
                self.note(stage, "out of time", "resumes on the next start")
                continue
            if stage.gemini:
                self.run_gemini(stage)
            else:
                self.run_stage(stage)
        print(f"Finished. Read {self.paths.brief}", flush=True)

    def write_brief(self) -> None:
        paths = self.paths
        lines = [f"# Overnight brief, {self.now():%d %b %H:%M}", ""]
        if self.warnings:
            lines += ["## Warnings at the start", "", *[f"- {warning}" for warning in self.warnings], ""]
        lines += ["## Stages", "", "| Stage | Status | Minutes | At | What it decides | Note |",
                  "|---|---|---:|---|---|---|"]
        for stage in self.stages:
            entry = self.state["stages"].get(stage.name, {})
            lines.append(f"| {stage.name} | {entry.get('status', 'waiting')} | {entry.get('minutes', '')} | "
                         f"{entry.get('at', '')} | {stage.question} | {entry.get('note', '')} |")
        for title, path in (("Generator", paths.summary), ("A trained humanizer's output", paths.humanize_demo),
                            ("Local translation hop", paths.local_hop), ("What the detector reads", paths.anatomy),
                            ("Public humanizers", paths.humanizer_report), ("Combiner trial", paths.trial_report),
                            ("Reference, before tonight", paths.reference_before)):
            if path.exists():
                text = path.read_text(encoding="utf-8")
                cut = text.find("\n## slim")  # the per-genre tables are long; the file has them
                lines += ["", f"# {title}", f"`{path.relative_to(ROOT)}`", "", text[:cut] if cut > 0 else text]
        paths.brief.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deadline(text: str, now: datetime) -> datetime:
    hour, minute = map(int, text.split(":"))
    moment = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return moment if moment > now else moment + timedelta(days=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--until", default="10:00", help="stop starting work this long before the closing stages")
    parser.add_argument("--smoke", action="store_true", help="every stage, tiny, in its own folder")
    parser.add_argument("--status", action="store_true", help="print the brief's stage table and stop")
    parser.add_argument("--again", action="append", default=[], metavar="STAGE", help="run a finished stage once more")
    parser.add_argument("--models", default=",".join(MODELS), help="Gemini models in the order to use them")
    args = parser.parse_args(argv)
    paths = Paths(args.smoke)
    if args.status:
        print(paths.brief.read_text(encoding="utf-8").split("\n# ")[0] if paths.brief.exists() else "Not run yet.")
        return 0
    paths.out.mkdir(parents=True, exist_ok=True)
    lock = paths.out / "running.pid"
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            print(f"Already running as process {lock.read_text()}.")
            return 1
        except (OSError, ValueError):
            pass
    lock.write_text(str(os.getpid()))
    now = datetime.now()
    night = Night(paths, plan(paths, args.smoke), deadline(args.until, now), args.smoke,
                  models=tuple(args.models.split(",")))
    for name in args.again:
        night.state["stages"].pop(name, None)

    def stop(signum, _frame) -> None:
        print(f"Stopped by signal {signum}; what's saved is kept and the next start resumes.", flush=True)
        night.stop_child()
        night.save()
        lock.unlink(missing_ok=True)
        os._exit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)  # a closed terminal would end the run, so start it with nohup or leave it open
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
    if not args.smoke:  # a DAIGT run paused by hand would write the same cache; its finished essays are kept
        subprocess.run(["pkill", "-9", "-f", "eval.featurize daigt"], capture_output=True)
    try:
        night.run_all()
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
