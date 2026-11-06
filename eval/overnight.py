"""The overnight tests: one model-heavy job at a time, most decision-worthy first, and no failure ends the night.

Each stage has a time limit and a log of its own (eval/cache/overnight/logs/), a crash is tried again, and a stage
that still fails is noted and passed by, along with anything that needs its result. Stages save as they go, so running
this again carries on: finished stages are skipped. Gemini's comparisons run in rounds, one repeat of every setup
each, so a quota that runs out leaves the setups equally measured. A spent or overloaded model hands on to the next
(Gemini, then OpenRouter), which starts its comparison afresh, since scores aren't compared across models.
An overloaded one is tried first again next round, which writes the repeats it missed. eval/cache/overnight/BRIEF.md
is rewritten after every stage, so there is something to read whenever you look. It keeps the Mac awake on mains power
itself.

    python -m eval.overnight --until 07:40
    python -m eval.overnight --status
    python -m eval.overnight --smoke          # every stage, tiny, in its own folder: about 30 minutes
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
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
UNUSABLE_EXIT = 5  # and on one that refuses the request itself, which no wait fixes
# In the order to use them. Scores are only compared within a model, so each measures every setup itself. Tried on a
# real document on 2026-11-05: Groq's free tier refuses requests this size (HTTP 413, tokens per minute), and the other
# OpenRouter models were rate-limited or took over a minute a reply. A "provider:" prefix picks the API.
MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "openrouter:cohere/north-mini-code:free")
QUOTA_RESET = (9, 5)  # local hour and minute: Gemini's daily quota resets at midnight Pacific, 09:00 CET
QUOTA_WAIT = 600  # seconds before believing a spent quota, since a per-minute limit can read the same
CATCH_UP_PASSES = 12  # at most, so no clock or state can keep the night going round forever
RECHECK = timedelta(minutes=60)  # a spent model is tried again after this: one request finds out whether it is back
CRASH_WAIT = 120
ATTEMPTS = 2
RESERVE = 20  # minutes kept back at the end for the closing stages
MEMORY_GB = 1.5  # wanted free before a stage starts; it starts anyway after MEMORY_PATIENCE seconds
MEMORY_PATIENCE = 300
PROBLEM = re.compile(r"^.*(?:\b\w+Error\b|Traceback|Giving up|can't start).*$", re.M)
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
    provider, name = model.split(":", 1) if model.split(":")[0] in ("groq", "openrouter") else ("gemini", model)
    label = re.sub(r"[^\w.-]", "-", name)
    return py("eval.tune_revise", *documents, "--as-page", "--repeats", repeats, "--provider", provider,
              "--model", name,
              "--out", paths.revise, "--trace", paths.out / "trace.jsonl",
              "--report", paths.out / f"{setup}_{label}.md", *extra, *tiny)


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
              "(the transfer gap against RoBERTa)?", 50,
              py("eval.reference_benchmark", "--detectors", "slim,roberta", "--report", paths.reference_before,
                 *reference_small)),
        Stage("reference_outside", "Does an outside detector (Tabularis) catch what slim misses? The transfer gap on "
              "guided samples, with a detector strong enough to mean something.", 40,
              py("eval.reference_benchmark", "--detectors", "slim,tabularis", "--report", paths.out /
                 "reference_outside.md", *reference_small)),
        Stage("humanize_demo", "What does a trained humanizer's output do to the detector, and does it flag the human "
              "rewrites?", 40,
              py("eval.score_demo", "--report", paths.humanize_demo, *(["--limit", 2] if small else [])),
              needs=[DEMOS / "kto_seed_demo.jsonl"]),
        Stage("local_hop", "Does the local translation hop move AI text, and does it bring text into the math script's "
              "range?", 25,
              py("eval.bench_hop", "--texts", 1 if small else 10, "--pivots", *(["German"] if small else
                 ["German", "Finnish"]), "--math", "--budget", 10 if small else 150, "--independent",
                 "--out", paths.out / "hop.jsonl", "--report", paths.local_hop)),
        Stage("humanizer_rules", "How do the public rule-based humanizers fare? No model, so it is quick.", 10,
              py("eval.humanizer_attack", "--n", 2 if small else 40, "--no-report", "--out", paths.humanizers)),
        Stage("gemini_round_1", "Does multi-start, the German round trip or holding names and numbers beat the current "
              "recipe?", 130, gemini=1),
        board(1),
        Stage("anatomy", "What does Desklib read: words, order, punctuation? Which edits can move it?", 40,
              py("eval.detector_anatomy", "--texts", 2 if small else 16, "--out", paths.out / "anatomy.jsonl",
                 "--report", paths.anatomy)),
        Stage("daigt_scoring", "Scores the new training essays for the trial fit.", 150,
              py("eval.featurize", "daigt", "--n", 4 if small else 2000, "--engines", "desklib", "--device", "cpu",
                 "--dtype", "float32", "--out", paths.daigt), needs=[DAIGT_CSV]),
        Stage("trial_fit", "Fits a trial slim combiner on the wider data; the shipped ones stay as they are.", 10,
              py("eval.fit_combiner", "eval/cache/raid.jsonl", "eval/cache/mage.jsonl", paths.daigt, "--features",
                 "desklib_mean,desklib_max", "--out", paths.trial, "--report", paths.trial_fit), needs=[paths.daigt]),
        Stage("borderline_math", "With the ranking fixed, how far does math alone move borderline text, and does the "
              "move survive a detector it never saw?", 50,
              py("eval.bench_math", "--band", 0.6, 0.95, "--texts", 1 if small else 8, "--independent",
                 "--out", paths.out / "borderline.jsonl", *(["--budget", 10] if small else []))),
        Stage("humanizer_bart", "Do the public BART humanizers beat the detector without using its scores?", 45,
              py("eval.humanizer_attack", "--n", 1 if small else 15, "--no-report", "--humanizers",
                 "stealth-bart,cive-bart", "--out", paths.humanizers)),
        Stage("humanizer_report", "The humanizers' scores, the same texts before and after.", 15,
              py("eval.humanizer_attack", "--report-only", "--out", paths.humanizers, "--report",
                 paths.humanizer_report), needs=[paths.humanizers]),
        Stage("gemini_round_2", "The same comparison with a second repeat: is the first round's answer stable?", 50,
              gemini=2),
        Stage("export_revised", "Hands the revised texts to the next stage.", 2, call=lambda: export_revised(paths),
              needs_rows=True, again=True),
        Stage("math_after_revision", "Does the math script finish off what revision leaves between 15% and 95%?", 25,
              py("eval.math_attack", paths.revised, "--score", "--band", 0.15, 0.95, "--out", paths.revised_math,
                 *(["--budget", 10] if small else [])), needs=[paths.revised], fresh=[paths.revised_math]),
        board(2),
        Stage("collect_samples", "Gathers every saved sample, including tonight's.", 5,
              call=lambda: write_samples(paths, 3 if small else None), again=True),
        Stage("redteam_scoring", "Scores those samples for the comparison below.", 40,
              py("eval.featurize", "local", "--path", paths.samples, "--engines", "desklib", "--device", "cpu",
                 "--dtype", "float32", "--out", paths.features), needs=[paths.samples]),
        Stage("trial_compare", "Does refitting the combiner catch guided samples, or is retraining the model the only "
              "way?", 10,
              py("eval.trial_compare", paths.trial, paths.daigt, "eval/cache/mage.jsonl", "eval/cache/raid.jsonl",
                 "--redteam", paths.features, "--report", paths.trial_report), needs=[paths.trial, paths.features]),
        Stage("redteam_report", "How much of everything saved gets through, by attack?", 20,
              py("eval.redteam_report", *([paths.samples] if small else []), "--report", paths.redteam)),
        Stage("gemini_round_3", "A third repeat, if quota and time allow.", 30, gemini=3),
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


def problems_in(log: Path, start: int = 0) -> str:
    """A note if a stage that exited cleanly logged errors it caught: "done" shouldn't hide that."""
    if not log.exists():
        return ""
    with log.open("rb") as file:
        file.seek(start)
        text = file.read().decode("utf-8", "replace")
    count = len(PROBLEM.findall(text))
    return f"{count} errors logged, check logs/{log.name}" if count else ""


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
    commit = subprocess.run(["git", "log", "-1", "--format=%h %s"], cwd=ROOT, capture_output=True, text=True).stdout
    changed = subprocess.run(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                             text=True).stdout.split("\n")
    changed = [line[3:] for line in changed if line.strip()]
    notes.append(f"Code: commit {commit.strip()}" + (f", with uncommitted changes in {', '.join(changed)}: "
                                                      "they weren't part of what was tested" if changed else ""))
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
            log = self.paths.logs / f"{stage.name}.log"
            start = log.stat().st_size if log.exists() else 0
            code = self.run_command(stage.command, log, self.minutes_left(stage))
            spent = (time.monotonic() - began) / 60
            if code == 0:
                self.note(stage, "done", problems_in(log, start), spent)
                return
            if code is None:
                self.note(stage, "timed out", "what it saved is kept; resumes on the next start", spent)
                return
            if attempt < ATTEMPTS:
                self.sleep(CRASH_WAIT)
        self.note(stage, "failed", f"exit {code}, see logs/{stage.name}.log", spent)

    def spent(self, model: str) -> bool:
        """Out, as far as we know: refused for the night, or out of quota less than RECHECK ago and not since reset.
        Quota can come back before the daily reset (a per-minute limit read as daily, another key, a quota raised),
        and one request finds out."""
        if model in self.state.setdefault("refused", {}):
            return True
        when = self.state["spent"].get(model)
        if not when:
            return False
        return self.now() < self.back_at(model)

    def back_at(self, model: str) -> datetime:
        """When a spent model is worth trying again: RECHECK after it ran out, or the daily reset if that is sooner."""
        ran_out = datetime.fromisoformat(self.state["spent"][model])
        reset = ran_out.replace(hour=QUOTA_RESET[0], minute=QUOTA_RESET[1], second=0, microsecond=0)
        reset += timedelta(days=1) if reset <= ran_out else timedelta()
        return min(ran_out + RECHECK, reset)

    def finished_runs(self) -> int:
        revise = self.paths.revise
        return sum(1 for _ in revise.open(encoding="utf-8")) if revise.exists() else 0

    def run_gemini(self, stage: Stage) -> None:
        began = time.monotonic()
        runs_before = self.finished_runs()
        self.note(stage, "running")
        finished, failures, down = 0, [], False
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
                model = next((m for m in self.models[self.models.index(model) + 1:] if not self.spent(m)), None)
                index = 0  # a new model's comparison needs all its setups
                continue
            if code == UNUSABLE_EXIT:  # out for the night: no wait fixes it
                failures.append(f"{model} refused the request")
                self.state.setdefault("refused", {})[model] = self.now().isoformat()
                model = next((m for m in self.models[self.models.index(model) + 1:] if not self.spent(m)), None)
                index = 0
                continue
            if code == DOWN_EXIT:
                failures.append(f"{model} overloaded")
                untried = self.models[self.models.index(model) + 1:]
                model = next((m for m in untried if not self.spent(m)), None)
                index = 0
                down = True
                continue
            if code == 0:
                finished += 1
            elif code is None:
                failures.append(f"{name} timed out")
            else:
                failures.append(f"{name} exit {code}")
            index += 1
        spent = (time.monotonic() - began) / 60
        new = f"{self.finished_runs() - runs_before} new runs"
        if model is None and down:
            self.note(stage, "model down", f"{new}; {'; '.join(failures)}", spent)
        elif model is None:
            self.note(stage, "quota spent", f"{new}; every model is out for now; the night comes back to it", spent)
        else:
            self.note(stage, "done" if not failures else "partly done", "; ".join([model, new, *failures]), spent)

    def run_all(self) -> None:
        self.warnings = checks() if not self.smoke else []
        self.save()
        for stage in [stage for stage in self.stages if not stage.closing]:
            self.run_one(stage)
        self.catch_up()
        for stage in [stage for stage in self.stages if stage.closing]:
            self.run_one(stage)
        print(f"Finished. Read {self.paths.brief}", flush=True)

    def run_one(self, stage: Stage) -> None:
        status = self.state["stages"].get(stage.name, {}).get("status")
        if status == "done" and not stage.again:
            return
        if not stage.closing and self.minutes_left(stage) <= 0:
            self.note(stage, "out of time", "resumes on the next start")
            return
        if stage.gemini:
            self.run_gemini(stage)
        else:
            self.run_stage(stage)

    def unfinished_rounds(self) -> list[Stage]:
        return [stage for stage in self.stages if stage.gemini
                and self.state["stages"].get(stage.name, {}).get("status") != "done"]

    def catch_up(self) -> None:
        """Back to the Gemini rounds that quota, overload or time cut short, once everything else is done: waiting for
        a spent model to be worth trying again while there is time, and going round while a pass still adds runs."""
        for _ in range(CATCH_UP_PASSES):
            if not (rounds := self.unfinished_rounds()):
                return
            usable = [model for model in self.models if model not in self.state.get("refused", {})]
            if not usable:
                return
            if all(self.spent(model) for model in usable):
                back = min(self.back_at(model) for model in usable if self.state["spent"].get(model))
                if back >= self.until - timedelta(minutes=RESERVE + 30):
                    return  # it wouldn't be back in time to do anything
                print(f"{self.now():%H:%M} every model is out; trying again at {back:%H:%M}", flush=True)
                self.sleep(max(0.0, (back - self.now()).total_seconds()))
            before = self.finished_runs()
            for stage in rounds:
                if self.minutes_left(stage) <= 0:
                    return
                self.run_gemini(stage)
            if self.finished_runs() == before and not all(self.spent(model) for model in usable):
                return  # a pass that added nothing with a model available: going round again wouldn't either

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
                            ("Reference, before tonight", paths.reference_before),
                            ("Reference with an outside detector", paths.out / "reference_outside.md")):
            if path.exists():
                text = path.read_text(encoding="utf-8")
                cut = text.find("\n## slim")  # the per-genre tables are long; the file has them
                lines += ["", f"# {title}", f"`{path.relative_to(ROOT)}`", "", text[:cut] if cut > 0 else text]
        paths.brief.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deadline(text: str, now: datetime) -> datetime:
    hour, minute = map(int, text.split(":"))
    moment = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return moment if moment > now else moment + timedelta(days=1)


# Short, telling and mostly free of quota, most useful first: `--day` runs these, and the night skips what they did.
DAY = ("tests", "reference_before", "humanizer_rules", "anatomy", "gemini_round_1", "scoreboard_1", "humanize_demo",
       "local_hop", "humanizer_bart", "humanizer_report", "borderline_math")


def day_plan(stages: list[Stage]) -> list[Stage]:
    by_name = {stage.name: stage for stage in stages}
    return [by_name[name] for name in DAY]


def live_lock(lock: Path) -> int | None:
    """The process holding the lock, if it is still running."""
    try:
        pid = int(lock.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def wait_for_lock(lock: Path, sleep: Callable[[float], None] = time.sleep) -> None:
    """Wait while another run, a day run say, still holds the lock."""
    told = False
    while (pid := live_lock(lock)) is not None:
        if not told:
            print(f"{datetime.now():%H:%M} waiting for run {pid} to finish", flush=True)
            told = True
        sleep(30)


def wait_until(start: datetime, lock: Path) -> None:
    """Sleep until `start`, looking at the clock at least once a minute: a Mac that sleeps meanwhile stops the clock
    time.sleep counts, and the run should begin as soon as it wakes past `start`. Holds the lock, so no second run
    starts meanwhile, and a stop signal ends the wait like a run."""
    print(f"{datetime.now():%H:%M} waiting to start at {start:%H:%M}", flush=True)
    while (left := (start - datetime.now()).total_seconds()) > 0:
        time.sleep(min(60.0, left))
    print(f"{datetime.now():%H:%M} starting", flush=True)


def detach(argv: list[str], paths: Paths) -> int:
    """Start the run in a session of its own and return. Started with `&` from a shell, it died with that shell's
    process group on 2026-11-05 at 22:08, 18 minutes in; in its own session nothing the shell does reaches it."""
    log = paths.out / "night.log"
    with log.open("a", encoding="utf-8") as sink:
        child = subprocess.Popen([PYTHON, "-m", "eval.overnight", *argv, "--foreground"], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    time.sleep(5)
    if child.poll() is not None:
        print(f"It stopped at once (exit {child.returncode}). The last lines of {log}:")
        print("\n".join(log.read_text(encoding="utf-8").splitlines()[-10:]))
        return 1
    waits = f", waiting to start at {argv[argv.index('--at') + 1]}" if "--at" in argv else ""
    print(f"Running as process {child.pid}{waits}, apart from this shell: closing this window won't stop it.\n"
          f"Progress: {paths.brief}\nCheck with: .venv/bin/python -m eval.overnight --status")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--until", help="stop starting work by then (default 07:40, or 21:30 with --day)")
    parser.add_argument("--day", action="store_true",
                        help=f"only the short stages, most useful first: {', '.join(DAY)}")
    parser.add_argument("--smoke", action="store_true", help="every stage, tiny, in its own folder")
    parser.add_argument("--status", action="store_true", help="print the brief's stage table and stop")
    parser.add_argument("--again", action="append", default=[], metavar="STAGE", help="run a finished stage once more")
    parser.add_argument("--models", default=",".join(MODELS), help="Gemini models in the order to use them")
    parser.add_argument("--at", metavar="HH:MM", help="wait until then to start, e.g. 22:00")
    parser.add_argument("--foreground", action="store_true",
                        help="run in this shell; otherwise it starts itself apart from the shell and returns")
    args = parser.parse_args(argv)
    paths = Paths(args.smoke)
    if args.status:
        print(paths.brief.read_text(encoding="utf-8").split("\n# ")[0] if paths.brief.exists() else "Not run yet.")
        return 0
    paths.out.mkdir(parents=True, exist_ok=True)
    lock = paths.out / "running.pid"
    args.until = args.until or ("21:30" if args.day else "07:40")
    if (pid := live_lock(lock)) and not args.at:
        print(f"Already running as process {pid}.")
        return 1
    if not args.foreground:
        return detach(argv if argv is not None else sys.argv[1:], paths)
    if args.at:  # holds nothing while it waits, so a day run can go meanwhile; then waits for that to finish
        wait_until(deadline(args.at, datetime.now()), lock)
        wait_for_lock(lock)
    elif live_lock(lock):
        print("Already running.")
        return 1
    lock.write_text(str(os.getpid()))
    now = datetime.now()
    stages = plan(paths, args.smoke)
    night = Night(paths, day_plan(stages) if args.day else stages, deadline(args.until, now), args.smoke,
                  models=tuple(args.models.split(",")))
    for name in args.again:
        night.state["stages"].pop(name, None)

    def stop(signum, _frame) -> None:
        print(f"Stopped by signal {signum}; what's saved is kept and the next start resumes.", flush=True)
        night.stop_child()
        night.save()
        if live_lock(lock) == os.getpid():
            lock.unlink(missing_ok=True)
        os._exit(1)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a closed terminal mustn't end the night
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
