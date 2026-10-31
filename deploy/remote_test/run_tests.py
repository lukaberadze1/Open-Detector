"""Run the detector-against-generator tests on another computer and pack the results to send back.

    python deploy/remote_test/run_tests.py                # everything
    python deploy/remote_test/run_tests.py --no-generate  # no Gemini key: steps 1-3 only

1. The test suite (pytest, stand-in engines, no downloads).
2. The revised samples in data/essays/revise_samples/ scored by the slim build (Desklib alone, as they
   were scored on the 8 GB Mac where they were written), to check this machine reads them the same.
3. The same samples scored by the full build (GLYPH, Desklib and Binoculars), which the revision loop
   never saw: do they still pass?
4. With a Gemini key in .env, new samples written with the revision loop run against the full build
   (eval.tune_prompts --profile full --revise 3), for finished-work and quick-notes prompts: can the
   generator beat the full detector when it can see it? About 120 Gemini requests.

Everything goes to remote_test_results/ and is zipped as remote_test_results.zip, the one file to send
back. Steps already done are skipped on a rerun, so a run that stops can be started again.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "data/essays/revise_samples"
OUT = ROOT / "remote_test_results"
PROFILE = ROOT / "data/styles/alexandria_profile.json"
SETS = (("finished", "finalized"), ("informal", "notes"))  # prompt set, profile register


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    with (OUT / "run_log.txt").open("a", encoding="utf-8") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def machine() -> dict:
    """What the tests ran on."""
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "processor": platform.processor()}
    try:
        import torch

        info |= {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    except ImportError as exc:
        info["torch"] = f"missing: {exc}"
    return info


def run_pytest() -> str:
    log("Step 1/4: test suite")
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    (OUT / "pytest.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
    last = (done.stdout.strip().splitlines() or ["no output"])[-1]
    log(f"  {last}")
    return last


def score(profile: str) -> dict:
    """Every saved sample scored by one detector build, by file and sample key."""
    from opendetector import Detector, DetectorConfig

    log(f"  loading the {profile} build (the first time downloads its models)")
    detector = Detector.from_config(DetectorConfig.for_profile(profile))
    scores = {}
    for path in sorted(SAMPLES.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            started = time.monotonic()
            report = detector.analyze(sample["text"], sentences=False)
            scores[f"{path.stem}|{sample['key']}"] = {
                "ai": report.ai_probability, "verdict": report.verdict, "engines": report.engines,
                "seconds": round(time.monotonic() - started, 1)}
            log(f"  {path.stem} {sample['key'].split('|')[-2]}#{sample['key'].split('|')[-1]}: "
                f"{100 * (report.ai_probability or 0):.0f}% {report.verdict}")
    del detector
    gc.collect()
    return scores


def run_scoring(profile: str, step: str) -> dict:
    target = OUT / f"scores_{profile}.json"
    if target.exists():
        log(f"{step}: {profile} scores already done")
        return json.loads(target.read_text(encoding="utf-8"))
    log(f"{step}: score the revised samples with the {profile} build")
    scores = score(profile)
    target.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    return scores


def written(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0


def run_generation(models: list[str], repeats: int, tries: int, wait: int) -> list[str]:
    """Each prompt set written by the first model, asked again after a wait while Gemini is busy (HTTP 503,
    "high demand"), then by the next model. A run resumes, so only missing samples are asked for."""
    from eval.tune_prompts import PROMPTS

    log("Step 4/4: new samples, revised against the full build")
    outputs = []
    for prompts, register in SETS:
        out = OUT / f"generate_full_{prompts}.jsonl"
        wanted = len(PROMPTS[prompts]) * repeats
        for model in models:
            for attempt in range(1, tries + 1):
                if written(out) >= wanted:
                    break
                command = [sys.executable, "-m", "eval.tune_prompts", "--style", str(PROFILE), "--register",
                           register, "--set", prompts, "--restyle", "--revise", "3", "--profile", "full",
                           "--models", model, "--repeats", str(repeats), "--out", str(out)]
                log(f"  {prompts} with {model}, try {attempt} of {tries}: {written(out)} of {wanted} samples so far")
                with (OUT / f"generate_full_{prompts}.log").open("a", encoding="utf-8") as file:
                    subprocess.run(command, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT, text=True)
                if written(out) < wanted and attempt < tries:
                    log(f"  {written(out)} of {wanted} written (Gemini busy or out of quota, see "
                        f"generate_full_{prompts}.log); waiting {wait // 60} minutes")
                    time.sleep(wait)
            if written(out) >= wanted:
                break
        log(f"  {prompts}: {written(out)} of {wanted} samples")
        outputs.append(out.name)
    return outputs


def mean(values: list[float]) -> str:
    return f"{statistics.fmean(values):.0f} ± {statistics.pstdev(values):.0f}" if values else "n/a"


def table(rows: list[dict]) -> list[str]:
    cell = mean
    lines = ["| Samples | Mac slim % | Slim here % | Full here % | Full: Likely human | Full: Needs review "
             "| Full: Likely AI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted({row["file"] for row in rows}):
        mine = [row for row in rows if row["file"] == name]
        full = [row["full_verdict"] for row in mine]
        lines.append(
            f"| {name} ({len(mine)}) | {cell([r['mac_slim'] for r in mine if r['mac_slim'] is not None])} "
            f"| {cell([r['slim'] for r in mine if r['slim'] is not None])} "
            f"| {cell([r['full'] for r in mine if r['full'] is not None])} "
            f"| {full.count('likely_human')} | {full.count('needs_review')} | {full.count('likely_ai')} |")
    return lines


def generation_table(names: list[str]) -> list[str]:
    lines = ["| Prompts | Samples | Before revising % | After % | Likely human | Words |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        path = OUT / name
        samples = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                   if path.exists() else [])
        scored = [sample for sample in samples if sample.get("ai") is not None]
        if not scored:
            lines.append(f"| {name} | 0 | | | | |")
            continue
        before = [100 * s["rounds"][0] for s in scored if s.get("rounds") and s["rounds"][0] is not None]
        human = sum(sample.get("verdict") == "likely_human" for sample in scored)
        lines.append(f"| {name} | {len(scored)} | {mean(before)} | {mean([100 * s['ai'] for s in scored])} "
                     f"| {human} of {len(scored)} | {statistics.fmean(s['words'] for s in scored):.0f} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-generate", action="store_true", help="skip step 4, which needs a Gemini key")
    parser.add_argument("--models", default="gemini-3.1-flash-lite,gemini-3.5-flash-lite",
                        help="the models step 4 writes with, in order: the next is used only while one stays busy")
    parser.add_argument("--repeats", type=int, default=3, help="samples per prompt in step 4")
    parser.add_argument("--tries", type=int, default=4, help="tries per model in step 4 while Gemini is busy")
    parser.add_argument("--wait", type=int, default=600, help="seconds between those tries")
    args = parser.parse_args(argv)

    OUT.mkdir(exist_ok=True)
    os.chdir(ROOT)
    from opendetector.env import load_env

    load_env()
    info = machine()
    (OUT / "machine.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"Machine: {info}")
    if not any(SAMPLES.glob("*.jsonl")):
        log(f"No samples in {SAMPLES}: the package is incomplete.")
        return 1

    tests = run_pytest()
    slim = run_scoring("slim", "Step 2/4")
    full = run_scoring("full", "Step 3/4")

    generated: list[str] = []
    if args.no_generate:
        log("Step 4/4: skipped (--no-generate)")
    elif not os.environ.get("GEMINI_API_KEY"):
        log("Step 4/4: skipped, no GEMINI_API_KEY in .env")
    else:
        generated = run_generation([name.strip() for name in args.models.split(",") if name.strip()],
                                   args.repeats, args.tries, args.wait)

    rows = []
    for path in sorted(SAMPLES.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = json.loads(line)
                key = f"{path.stem}|{sample['key']}"
                rows.append({"file": path.stem,
                             "mac_slim": 100 * sample["ai"] if sample.get("ai") is not None else None,
                             "slim": 100 * slim[key]["ai"] if slim[key]["ai"] is not None else None,
                             "full": 100 * full[key]["ai"] if full[key]["ai"] is not None else None,
                             "full_verdict": full[key]["verdict"]})
    summary = ["# Remote test results", "", f"Machine: {json.dumps(info)}", "", f"Test suite: {tests}", "",
               "## The revised samples, slim and full builds", "",
               "Mac slim is the score saved when the samples were written; slim here should match it closely.", "",
               *table(rows), ""]
    if generated:
        summary += ["## New samples revised against the full build", "", *generation_table(generated), ""]
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    archive = shutil.make_archive(str(ROOT / "remote_test_results"), "zip", OUT)
    log(f"Done. Send back this one file: {archive}")
    print("\n" + "\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
