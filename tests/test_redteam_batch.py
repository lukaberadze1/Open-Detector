"""Block-by-block checkpoints for the red-team generator app's batches."""

import threading

import pytest

from eval.redteam_batch import (
    BatchWorker,
    Checkpoint,
    batch_key,
    generate_missing,
    parse_group_reply,
    score_missing,
)

KEYS = ["001 answer_box Table 1, row 1", "002 answer_box Table 1, row 2", "003 answer_box Table 1, row 3"]


class Stopped(BaseException):
    """What Streamlit raises inside a Streamlit call when the user presses Stop."""


def each(write):
    """A batch's `generate` that writes every block of a group with `write(key) -> (text, model)`."""
    def generate(group):
        for key in group:
            yield key, *write(key)
    return generate


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
        generate_missing(checkpoint, KEYS, each(flaky))
    assert Checkpoint.load(checkpoint.path).generated(KEYS) == [KEYS[0]]

    resumed = Checkpoint.load(checkpoint.path)
    calls.clear()
    generate_missing(resumed, KEYS, each(lambda key: (calls.append(key) or f"text for {key[:3]}", "gemma3:4b")))
    assert calls == KEYS[1:], "the finished block is not generated again"
    assert [resumed.blocks[k]["model"] for k in KEYS] == ["gemini", "gemma3:4b", "gemma3:4b"]


def test_stop_inside_the_progress_report_comes_after_the_block_is_saved(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})

    def stop_on_first_report(key):
        raise Stopped

    with pytest.raises(Stopped):
        generate_missing(checkpoint, KEYS, each(lambda key: ("Saved before the stop.", "m")), stop_on_first_report)
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
    worker = BatchWorker(checkpoint, KEYS, each(lambda key: (order.append("generate") or key[:3], "m")),
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
    worker = BatchWorker(checkpoint, KEYS, each(slow), score=lambda text: {"verdict": "likely_ai"},
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

    worker = BatchWorker(checkpoint, KEYS, each(quota_runs_out))
    worker.start()
    worker.join(5)
    assert worker.phase == "failed" and "Every configured" in str(worker.error)
    assert Checkpoint.load(checkpoint.path).generated(KEYS) == KEYS[:2]


def test_batch_key_is_stable_and_changes_with_the_settings():
    key = batch_key("hash", True, False, "Write formally.", 100)
    assert key == batch_key("hash", True, False, "Write formally.", 100) and len(key) == 16
    assert key != batch_key("hash", True, False, "Write casually.", 100)


def test_worker_runs_its_end_hook_however_the_batch_ends(tmp_path):
    ended, workers = [], []
    for name, generate in (("done", each(lambda key: ("text", "m"))), ("failed", each(lambda key: 1 / 0))):
        checkpoint = Checkpoint.start(tmp_path / f"{name}.json", {})
        workers.append(BatchWorker(checkpoint, ["a"], generate, on_end=lambda: ended.append(workers[-1].phase)))
        workers[-1].start()
        workers[-1].join(5)
    assert ended == ["done", "failed"]


def test_blocks_go_out_in_groups_and_a_resume_groups_only_the_missing_ones(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    keys = [f"{index:03d} text_block Paragraph {index}" for index in range(1, 6)]
    groups = []

    def generate(group):
        groups.append(group)
        for key in group:
            yield key, f"new {key[:3]}", "gemini"

    generate_missing(checkpoint, keys, generate, group_size=2)
    assert groups == [keys[0:2], keys[2:4], keys[4:]]
    assert checkpoint.generated(keys) == keys

    resumed = Checkpoint.start(tmp_path / "r.json", {})
    for key in (keys[0], keys[2]):
        resumed.update(key, text="kept", model="m")
    groups.clear()
    generate_missing(resumed, keys, generate, group_size=2)
    assert groups == [[keys[1], keys[3]], [keys[4]]]


def test_groups_are_written_at_the_same_time_and_a_failure_waits_for_the_others(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    keys = [f"{index:03d} block" for index in range(1, 7)]
    both_running, failure_seen, started = threading.Barrier(2, timeout=5), threading.Event(), []

    def generate(group):
        started.append(group[0])
        if len(started) <= 2:
            both_running.wait()  # the first two groups are in flight together
        if group[0] == keys[0]:
            yield group[0], "first", "m"
            raise RuntimeError("every model failed")
        assert failure_seen.wait(5)  # the second group finishes after the first has failed
        for key in group:
            yield key, "written", "m"

    with pytest.raises(RuntimeError, match="every model failed"):
        generate_missing(checkpoint, keys, generate, lambda key: failure_seen.set(), group_size=2, workers=2)
    saved = Checkpoint.load(checkpoint.path)
    assert saved.generated(keys) == [keys[0], keys[2], keys[3]], (
        "the failed group's finished block and the whole group beside it are saved; no new group starts")
    assert saved.first_missing(keys) == 2


def test_a_stop_lets_the_groups_in_flight_finish_and_starts_no_more(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    keys = [f"{index:03d} block" for index in range(1, 9)]
    release, reports = threading.Event(), []

    def generate(group):
        if group[0] != keys[0]:
            assert release.wait(5)
        for key in group:
            yield key, "written", "m"

    def report(key):
        reports.append(key)
        release.set()
        raise Stopped

    with pytest.raises(Stopped):
        generate_missing(checkpoint, keys, generate, report, group_size=2, workers=2)
    assert Checkpoint.load(checkpoint.path).generated(keys) == keys[:4]
    assert reports == [keys[0]], "nothing is reported once the stop is raised"


def test_blocks_saved_from_many_threads_at_once_all_reach_the_file(tmp_path):
    checkpoint = Checkpoint.start(tmp_path / "c.json", {})
    keys = [f"{index:03d} block" for index in range(200)]
    generate_missing(checkpoint, keys, each(lambda key: ("text " * 20, "m")), workers=8)
    assert Checkpoint.load(checkpoint.path).generated(keys) == keys


def test_the_time_left_averages_over_a_few_rounds_of_groups(tmp_path):
    worker = BatchWorker(Checkpoint.start(tmp_path / "c.json", {}), KEYS, each(lambda key: ("t", "m")),
                         group_size=6, workers=2)
    assert worker.window == 24
    assert BatchWorker(worker.checkpoint, KEYS, worker._generate).window == 8


def test_a_group_reply_gives_each_known_block_its_text():
    ids = ["B1", "B2", "B3"]
    fenced = '```json\n{"blocks": [{"id": "B1", "text": " One. "}, {"id": "B3", "text": "Three."}]}\n```'
    assert parse_group_reply(fenced, ids) == {"B1": "One.", "B3": "Three."}
    twice = '{"blocks": [{"id": "B1", "text": "a"}, {"id": "B1", "text": "b"}, {"id": "B2", "text": "c"}]}'
    assert parse_group_reply(twice, ids) == {"B2": "c"}, "a block given twice is written again on its own"
    odd = '{"blocks": [{"id": "B9", "text": "x"}, {"id": "B2", "text": "  "}, {"id": "B3"}, "B1"]}'
    assert parse_group_reply(odd, ids) == {}
    assert parse_group_reply("Sorry, I can't help with that.", ids) == {}
    assert parse_group_reply('{"blocks": [', ids) == {}


def test_only_prose_blocks_are_worth_rewriting():
    from eval.redteam_batch import worth_rewriting

    assert [worth_rewriting(text) for text in ("14 August", "Done", "1", "Osuke, Dima", "I met criteria 1, 2, 3",
                                               "Landed a Ollie .")] == [False] * 6
    assert worth_rewriting("My first proper skating session ever.")
    assert worth_rewriting("Notes from the hive on a warm and windy afternoon in late June")


def test_short_blocks_share_a_request_and_a_long_one_takes_the_room_of_several():
    from eval.redteam_batch import group_by_words

    words = {"a": 6, "b": 6, "c": 6, "d": 150, "e": 30, "f": 30}
    assert group_by_words(list(words), 2, words.get) == [["a", "b", "c", "d"][:3], ["d"], ["e", "f"]], (
        "6-word blocks take a quarter place, a 150-word one five places on its own, 30-word ones a place each")
    assert group_by_words(list(words), 1, words.get) == [[key] for key in words], "one block per request"
    assert group_by_words(list(words), 4) == [["a", "b", "c", "d"], ["e", "f"]], "without lengths, by count"
