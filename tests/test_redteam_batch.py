"""Block-by-block checkpoints for the red-team generator app's batches."""

import threading

import pytest

from eval.redteam_batch import BatchWorker, Checkpoint, batch_key, generate_missing, score_missing

KEYS = ["001 answer_box Table 1, row 1", "002 answer_box Table 1, row 2", "003 answer_box Table 1, row 3"]


class Stopped(BaseException):
    """What Streamlit raises inside a Streamlit call when the user presses Stop."""


def test_checkpoint_survives_a_reload_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "batches" / "abc.json"
    checkpoint = Checkpoint.start(path, {"source_file": "form.docx"})
    checkpoint.update(KEYS[0], text="First answer.", model="m")
    checkpoint.update(KEYS[0], score={"verdict": "likely_ai"})
    loaded = Checkpoint.load(path)
    assert loaded.meta["source_file"] == "form.docx" and loaded.meta["created"]
    assert loaded.blocks[KEYS[0]] == {"text": "First answer.", "model": "m", "score": {"verdict": "likely_ai"}}
    assert loaded.generated(KEYS) == loaded.scored(KEYS) == [KEYS[0]]
    assert [p.name for p in path.parent.iterdir()] == ["abc.json"]
    loaded.discard()
    assert Checkpoint.load(path) is None


def test_unreadable_checkpoint_is_treated_as_none(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"meta": {}, "blocks": ')
    assert Checkpoint.load(path) is None and Checkpoint.load(tmp_path / "missing.json") is None


def test_a_failure_keeps_every_finished_block_and_a_rerun_resumes_after_it(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    calls = []

    def flaky(key):
        calls.append(key)
        if key == KEYS[1]:
            raise RuntimeError("every model failed")
        return f"text for {key[:3]}", "gemini"

    with pytest.raises(RuntimeError):
        generate_missing(checkpoint, KEYS, flaky)
    assert Checkpoint.load(checkpoint.path).generated(KEYS) == [KEYS[0]]

    resumed = Checkpoint.load(checkpoint.path)
    calls.clear()
    generate_missing(resumed, KEYS, lambda key: (calls.append(key) or f"text for {key[:3]}", "gemma3:4b"))
    assert calls == KEYS[1:], "the finished block is not generated again"
    assert [resumed.blocks[k]["model"] for k in KEYS] == ["gemini", "gemma3:4b", "gemma3:4b"]


def test_stop_inside_the_progress_report_comes_after_the_block_is_saved(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})

    def stop_on_first_report(key):
        raise Stopped

    with pytest.raises(Stopped):
        generate_missing(checkpoint, KEYS, lambda key: ("Saved before the stop.", "m"), stop_on_first_report)
    assert Checkpoint.load(checkpoint.path).blocks[KEYS[0]]["text"] == "Saved before the stop."


def test_scoring_skips_scored_and_ungenerated_blocks(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    checkpoint.update(KEYS[0], text="one", model="m", score={"verdict": "likely_human"})
    checkpoint.update(KEYS[1], text="two", model="m")
    scored, reported = [], []
    score_missing(checkpoint, KEYS, lambda text: scored.append(text) or {"verdict": "likely_ai"}, reported.append)
    assert scored == ["two"] and reported == [KEYS[1]]
    assert Checkpoint.load(checkpoint.path).scored(KEYS) == KEYS[:2]


def test_worker_generates_then_scores_on_its_own_thread(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    order = []
    worker = BatchWorker(checkpoint, KEYS, lambda key: (order.append("generate") or key[:3], "m"),
                         score=lambda text: order.append("score") or {"verdict": "likely_ai"},
                         before_scoring=lambda: order.append("unload"), finish=lambda: order.append("finish"))
    worker.start()
    worker.join(5)
    assert worker.phase == "done" and worker.error is None
    assert order == ["generate"] * 3 + ["unload"] + ["score"] * 3 + ["finish"], (
        "every block is generated before any is scored, and the whole is scored last")
    assert Checkpoint.load(checkpoint.path).scored(KEYS) == KEYS
    assert len(worker.seconds["generated"]) == len(worker.seconds["scored"]) == 3


def test_worker_stops_after_the_block_in_progress(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    in_progress, release = threading.Event(), threading.Event()

    def slow(key):
        in_progress.set()
        release.wait(5)
        return "written", "m"

    finished = []
    worker = BatchWorker(checkpoint, KEYS, slow, score=lambda text: {"verdict": "likely_ai"},
                         finish=lambda: finished.append(True))
    worker.start()
    assert in_progress.wait(5)
    worker.stop()  # pressed while the first block is still being written
    release.set()
    worker.join(5)
    assert worker.phase == "stopped" and worker.stopping and not finished
    saved = Checkpoint.load(checkpoint.path)
    assert saved.generated(KEYS) == [KEYS[0]] and saved.scored(KEYS) == [], "the block in progress is kept"


def test_worker_failure_is_reported_with_earlier_blocks_saved(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})

    def quota_runs_out(key):
        if key == KEYS[2]:
            raise RuntimeError("Every configured text model failed")
        return "written", "m"

    worker = BatchWorker(checkpoint, KEYS, quota_runs_out)
    worker.start()
    worker.join(5)
    assert worker.phase == "failed" and "Every configured" in str(worker.error)
    assert Checkpoint.load(checkpoint.path).generated(KEYS) == KEYS[:2]


def test_batch_key_is_stable_and_changes_with_the_settings():
    key = batch_key("hash", True, False, "Write formally.", 100)
    assert key == batch_key("hash", True, False, "Write formally.", 100) and len(key) == 16
    assert key != batch_key("hash", True, False, "Write casually.", 100)
