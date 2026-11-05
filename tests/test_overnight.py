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
        return QUOTA_EXIT if command == ["multistart", overnight.MODELS[0]] else 0

    item = night(tmp_path, [Stage("round", "?", gemini=1)], run, monkeypatch)
    item.run_all()
    first, second = overnight.MODELS
    assert calls == [("baseline", first), ("multistart", first), ("baseline", second), ("multistart", second),
                     ("roundtrip", second), ("masked", second)]
    assert status(item, "round") == "done" and item.spent(first) and not item.spent(second)


def test_an_overloaded_model_puts_the_round_off_and_a_later_start_tries_it_again(tmp_path, monkeypatch):
    calls = []

    def run(command, log, minutes):
        calls.append(command[0])
        return overnight.DOWN_EXIT if command[0] == "baseline" else 0

    stages = [Stage("round", "?", gemini=1), Stage("next", "?", command=["next"])]
    item = night(tmp_path, stages, run, monkeypatch)
    item.run_all()
    assert calls == ["baseline", "next"]
    assert status(item, "round") == "model down" and not item.spent(overnight.MODELS[0])
    night(tmp_path, stages, run, monkeypatch).run_all()
    assert calls == ["baseline", "next", "baseline"]


def test_when_every_model_is_spent_the_night_goes_on_and_the_reset_brings_them_back(tmp_path, monkeypatch):
    clock = Clock()
    stages = [Stage("round", "?", gemini=1), Stage("next", "?", command=["next"])]
    run = lambda c, _log, _minutes: 0 if c[0] == "next" else QUOTA_EXIT  # noqa: E731
    item = night(tmp_path, stages, run, monkeypatch, clock=clock)
    item.run_all()
    assert status(item, "round") == "quota spent" and status(item, "next") == "done"
    clock.time = datetime(2026, 11, 6, 9, 30)
    assert not item.spent(overnight.MODELS[0])


def test_the_plan_has_a_question_for_each_stage_and_keeps_gemini_rounds_in_order(tmp_path):
    stages = overnight.plan(Paths(True), True)
    assert all(stage.question for stage in stages)
    names = [stage.name for stage in stages]
    assert len(names) == len(set(names))
    assert names.index("gemini_round_1") < names.index("gemini_round_2") < names.index("gemini_round_3")
    assert names.index("trial_fit") < names.index("trial_compare") and names[-1] == "scoreboard_final"
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


def test_state_survives_as_json(tmp_path, monkeypatch):
    item = night(tmp_path, [Stage("s", "?", command=["s"])], lambda *a: 0, monkeypatch)
    item.run_all()
    assert json.loads(item.paths.state.read_text())["stages"]["s"]["status"] == "done"
