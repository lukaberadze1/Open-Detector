import json
from datetime import datetime, timedelta

from eval import overnight
from eval.overnight import QUOTA_EXIT, Night, Paths, Stage


class Clock:
    def __init__(self):
        self.time = datetime(2026, 11, 5, 20, 0)

    def __call__(self):
        return self.time


def night(tmp_path, stages, run, monkeypatch, until_hours=10, clock=None):
    paths = Paths(True)
    paths.out = tmp_path
    paths.logs, paths.state, paths.brief = tmp_path / "logs", tmp_path / "state.json", tmp_path / "BRIEF.md"
    paths.revise, paths.summary = tmp_path / "revise.jsonl", tmp_path / "summary.md"
    paths.anatomy = paths.trial_report = paths.reference_before = tmp_path / "none.md"
    monkeypatch.setattr(overnight, "available_gb", lambda: 8.0)
    monkeypatch.setattr(overnight, "gemini_command", lambda p, s, name, extra, repeats, model: [name, model])
    clock = clock or Clock()
    return Night(paths, stages, clock() + timedelta(hours=until_hours), True, run=run, sleep=lambda _: None, now=clock)


def status(item, name):
    return item.state["stages"][name]["status"]


def test_a_failed_stage_is_retried_then_passed_by_and_the_rest_still_run(tmp_path, monkeypatch):
    calls = []

    def run(command, log, minutes):
        calls.append(command[0])
        return 1 if command[0] == "bad" else 0

    item = night(tmp_path, [Stage("bad", "?", command=["bad"]), Stage("good", "?", command=["good"])], run, monkeypatch)
    item.run_all()
    assert calls == ["bad", "bad", "good"]
    assert status(item, "bad") == "failed" and status(item, "good") == "done"
    assert "| bad | failed" in item.paths.brief.read_text()


def test_a_stage_past_its_limit_is_noted_and_the_next_runs(tmp_path, monkeypatch):
    item = night(tmp_path, [Stage("slow", "?", command=["slow"]), Stage("next", "?", command=["next"])],
                 lambda command, log, minutes: None if command[0] == "slow" else 0, monkeypatch)
    item.run_all()
    assert status(item, "slow") == "timed out" and status(item, "next") == "done"


def test_a_stage_needing_a_missing_file_is_skipped(tmp_path, monkeypatch):
    calls = []
    stage = Stage("late", "?", command=["late"], needs=[tmp_path / "absent"])
    item = night(tmp_path, [stage], lambda *a: calls.append(a) or 0, monkeypatch)
    item.run_all()
    assert not calls and status(item, "late") == "skipped"


def test_finished_stages_are_skipped_on_a_second_start_unless_they_read_what_grows(tmp_path, monkeypatch):
    calls = []
    stages = [Stage("once", "?", command=["once"]), Stage("each", "?", command=["each"], again=True)]

    def run(command, log, minutes):
        calls.append(command[0])
        return 0

    night(tmp_path, stages, run, monkeypatch).run_all()
    night(tmp_path, stages, run, monkeypatch).run_all()
    assert calls == ["once", "each", "each"]


def test_only_closing_stages_run_past_the_deadline(tmp_path, monkeypatch):
    calls = []
    stages = [Stage("work", "?", command=["work"]), Stage("close", "?", command=["close"], closing=True)]
    item = night(tmp_path, stages, lambda c, _log, _minutes: calls.append(c[0]) or 0, monkeypatch, until_hours=0)
    item.run_all()
    assert calls == ["close"] and status(item, "work") == "out of time"


def test_a_crash_in_an_in_process_step_is_only_a_note(tmp_path, monkeypatch):
    def boom():
        raise ValueError("broken")

    item = night(tmp_path, [Stage("step", "?", call=boom), Stage("next", "?", command=["next"])],
                 lambda *a: 0, monkeypatch)
    item.run_all()
    assert status(item, "step") == "failed" and "broken" in item.state["stages"]["step"]["note"]
    assert status(item, "next") == "done"


def test_a_spent_model_hands_on_and_the_next_measures_every_setup(tmp_path, monkeypatch):
    calls = []

    def run(command, log, minutes):
        calls.append(tuple(command))
        return QUOTA_EXIT if command == ["anchored_masked", overnight.MODELS[0]] else 0

    item = night(tmp_path, [Stage("round", "?", gemini=1)], run, monkeypatch)
    item.run_all()
    first, second = overnight.MODELS[:2]
    assert calls == [("anchored", first), ("anchored_masked", first), ("anchored", second),
                     ("anchored_masked", second), ("masked", second), ("baseline", second)]
    assert status(item, "round") == "done" and item.spent(first) and not item.spent(second)


def test_an_overloaded_model_hands_on_for_this_round_and_is_tried_first_again(tmp_path, monkeypatch):
    calls = []
    first, second = overnight.MODELS[:2]

    def run(command, log, minutes):
        calls.append(tuple(command))
        return overnight.DOWN_EXIT if command[1] == first else 0

    stages = [Stage("round", "?", gemini=1)]
    item = night(tmp_path, stages, run, monkeypatch)
    item.run_all()
    setups = [name for name, _ in overnight.gemini_setups(item.paths, True)]
    first_pass = [(setups[0], first), *[(name, second) for name in setups]]
    assert calls[:len(first_pass)] == first_pass
    assert calls[len(first_pass)] == (setups[0], first), "the night comes back to the round, first model first"
    assert status(item, "round") == "partly done" and not item.spent(first)
    calls.clear()
    item.state["stages"].pop("round")
    item.run_all()
    assert calls[0] == (setups[0], first), "next round, the first model gets another chance"


def test_a_model_that_refuses_the_request_is_out_for_the_night(tmp_path, monkeypatch):
    calls = []
    first, second = overnight.MODELS[:2]

    def run(command, log, minutes):
        calls.append(command[1])
        return overnight.UNUSABLE_EXIT if command[1] == first else 0

    item = night(tmp_path, [Stage("round", "?", gemini=1)], run, monkeypatch)
    item.run_all()
    assert calls[0] == first and set(calls[1:]) == {second} and item.spent(first)


def test_every_model_overloaded_only_notes_it(tmp_path, monkeypatch):
    stages = [Stage("round", "?", gemini=1), Stage("next", "?", command=["next"])]
    item = night(tmp_path, stages, lambda c, _l, _m: 0 if c[0] == "next" else overnight.DOWN_EXIT, monkeypatch)
    item.run_all()
    assert status(item, "round") == "model down" and status(item, "next") == "done"


def test_a_provider_model_goes_out_on_its_own_api(tmp_path):
    paths = Paths(True)
    command = [str(part) for part in overnight.gemini_command(paths, True, "baseline", [], 1,
                                                              "openrouter:cohere/north-mini-code:free")]
    assert command[command.index("--provider") + 1] == "openrouter"
    assert command[command.index("--model") + 1] == "cohere/north-mini-code:free"
    assert "/" not in command[command.index("--report") + 1].rsplit("/", 1)[-1]
    gemini = [str(part) for part in overnight.gemini_command(paths, True, "baseline", [], 1, overnight.MODELS[0])]
    assert gemini[gemini.index("--provider") + 1] == "gemini"


def test_when_every_model_is_spent_the_night_goes_on_and_the_reset_brings_them_back(tmp_path, monkeypatch):
    clock = Clock()
    stages = [Stage("round", "?", gemini=1), Stage("next", "?", command=["next"])]
    run = lambda c, _log, _minutes: 0 if c[0] == "next" else QUOTA_EXIT  # noqa: E731
    item = night(tmp_path, stages, run, monkeypatch, clock=clock)
    item.run_all()
    assert status(item, "round") == "quota spent" and status(item, "next") == "done"
    assert sum(item.spent(model) for model in overnight.MODELS) == len(overnight.MODELS)
    clock.time = datetime(2026, 11, 6, 9, 30)
    assert not item.spent(overnight.MODELS[0])


def test_the_plan_has_a_question_for_each_stage_and_keeps_gemini_rounds_in_order(tmp_path):
    stages = overnight.plan(Paths(True), True)
    assert all(stage.question for stage in stages)
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names))
    assert names.index("gemini_round_1") < names.index("gemini_round_2") < names.index("gemini_round_3")
    assert names.index("trial_fit") < names.index("trial_compare") and names[-2:] == ["scoreboard_final", "recipe"]
    assert all(stage.command or stage.call or stage.gemini for stage in stages)


def test_the_no_api_stages_run_before_the_gemini_rounds_and_the_demo_can_be_absent():
    stages = overnight.plan(Paths(True), True)
    names = [stage.name for stage in stages]
    assert names.index("humanize_demo") < names.index("local_hop") < names.index("gemini_round_1")
    assert next(stage for stage in stages if stage.name == "humanize_demo").needs


def test_other_work_is_in_the_plan(tmp_path):
    paths = Paths(True)
    names = [stage.name for stage in overnight.plan(paths, True)]
    wanted = ("humanize_demo", "local_hop", "humanizer_rules", "humanizer_bart", "humanizer_report", "daigt_scoring")
    for name in wanted:
        assert name in names
    assert names.index("humanizer_bart") < names.index("humanizer_report")
    assert "masked" in [setup for setup, _ in overnight.gemini_setups(paths, True)]


def test_errors_a_stage_caught_show_in_its_note(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("old ValueError: earlier\nfine\nstealth-bart, m: ValueError: no\nGiving up on a.md\n")
    assert overnight.problems_in(log, len("old ValueError: earlier\n")).startswith("2 errors logged")
    assert overnight.problems_in(tmp_path / "absent.log") == ""


def test_state_survives_as_json(tmp_path, monkeypatch):
    item = night(tmp_path, [Stage("s", "?", command=["s"])], lambda *a: 0, monkeypatch)
    item.run_all()
    assert json.loads(item.paths.state.read_text())["stages"]["s"]["status"] == "done"


def test_the_day_run_takes_short_stages_most_useful_first_and_the_night_skips_them(tmp_path, monkeypatch):
    stages = overnight.plan(Paths(True), True)
    day = overnight.day_plan(stages)
    assert [stage.name for stage in day] == list(overnight.DAY)
    assert "daigt_scoring" not in overnight.DAY and "reference_full" not in overnight.DAY
    calls = []
    small = [Stage("short", "?", command=["short"]), Stage("long", "?", command=["long"])]
    run = lambda c, _l, _m: calls.append(c[0]) or 0  # noqa: E731
    night(tmp_path, small[:1], run, monkeypatch).run_all()
    night(tmp_path, small, run, monkeypatch).run_all()
    assert calls == ["short", "long"], "the night doesn't redo what the day finished"


def test_a_waiting_run_waits_for_a_live_lock_and_ignores_a_dead_one(tmp_path):
    import os

    lock = tmp_path / "running.pid"
    lock.write_text("999999999")
    assert overnight.live_lock(lock) is None
    overnight.wait_for_lock(lock, sleep=lambda _: None)
    lock.write_text(str(os.getpid()))
    assert overnight.live_lock(lock) == os.getpid()
    naps = []

    def nap(_):
        naps.append(1)
        lock.write_text("999999999")  # the day run ends

    overnight.wait_for_lock(lock, sleep=nap)
    assert naps == [1]


def test_a_spent_model_is_tried_again_an_hour_later_and_the_night_comes_back_to_the_round(tmp_path, monkeypatch):
    clock = Clock()
    calls = []
    quota_left = {"value": False}

    def run(command, log, minutes):
        calls.append(command[0])
        if command[0] in ("other", "close"):
            return 0
        if not quota_left["value"]:
            return QUOTA_EXIT
        item.paths.revise.open("a").write("{}\n")
        return 0

    def sleep(seconds):
        clock.time += timedelta(seconds=seconds)
        quota_left["value"] = True  # an hour on, the quota is back

    stages = [Stage("round", "?", gemini=1), Stage("other", "?", command=["other"]),
              Stage("close", "?", command=["close"], closing=True)]
    item = night(tmp_path, stages, run, monkeypatch, clock=clock)
    item.sleep = sleep
    item.models = overnight.MODELS[:1]
    item.run_all()
    assert calls[0] == "anchored" and calls[1] == "other", "a spent quota moves the night on to other work"
    assert calls[-1] == "close" and calls.count("anchored") >= 2, "then it comes back to the round before closing"
    assert status(item, "round") == "done"


def test_the_night_doesnt_wait_for_quota_that_would_come_back_too_late(tmp_path, monkeypatch):
    clock = Clock()
    naps = []
    stages = [Stage("round", "?", gemini=1)]
    item = night(tmp_path, stages, lambda c, _l, _m: QUOTA_EXIT, monkeypatch, until_hours=1, clock=clock)
    item.sleep = naps.append
    item.run_all()
    assert not [nap for nap in naps if nap > overnight.QUOTA_WAIT], "an hour's wait wouldn't leave time to use it"


def test_a_refused_model_stays_out_while_a_spent_one_comes_back(tmp_path, monkeypatch):
    clock = Clock()
    item = night(tmp_path, [], lambda *a: 0, monkeypatch, clock=clock)
    item.state["refused"] = {"x": clock().isoformat()}
    item.state["spent"]["y"] = clock().isoformat()
    assert item.spent("x") and item.spent("y")
    clock.time += overnight.RECHECK
    assert item.spent("x") and not item.spent("y")


def test_no_api_holds_the_rounds_back_for_a_later_start_and_runs_the_rest(tmp_path, monkeypatch):
    calls = []
    stages = [Stage("round", "?", gemini=1), Stage("other", "?", command=["other"])]
    item = night(tmp_path, stages, lambda command, *a: calls.append(command[0]) or 0, monkeypatch)
    item.models = ()
    item.run_all()
    assert calls == ["other"] and status(item, "round") == "held back" and status(item, "other") == "done"
    later = night(tmp_path, stages, lambda command, *a: calls.append(command[0]) or 0, monkeypatch)
    later.run_all()
    assert status(later, "round") == "done" and "baseline" in calls


def test_no_api_is_on_the_command_line():
    import inspect
    assert "--no-api" in inspect.getsource(overnight.main)


def test_tonight_measures_the_pages_loop_first_and_the_recipe_is_rebuilt_last(tmp_path):
    paths = Paths(True)
    setups = overnight.gemini_setups(paths, True)
    assert [name for name, _ in setups][:2] == ["anchored", "anchored_masked"]
    assert all("anchored" in extra for name, extra in setups if name.startswith("anchored"))
    stages = overnight.plan(paths, True)
    assert stages[-1].name == "recipe" and stages[-1].closing
    assert str(paths.recipe).startswith(str(paths.out)), "a smoke run never writes the page's models/recipe.json"
