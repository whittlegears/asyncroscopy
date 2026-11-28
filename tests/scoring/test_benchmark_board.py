"""The benchmark's bars, board, history and validity check (no stack, no model)."""

import asyncio
from types import SimpleNamespace

import pytest
import yaml

from benchmarks import benchmark, board, run_workflows
from benchmarks.workflow_checks import Run, evaluate, lengths_in


def results(started=100.0, model="m", git="abc", **workflows):
    """A results file with one entry per benchmark: name=(passed, of) or name=(passed, of, {extra})."""
    entries = {}
    for name, value in workflows.items():
        passed, of, extra = (*value, {})[:3]
        rows = [{"ok": i < passed, "seconds": 1.0, "caller": f"wf-{name}-{i}", "checks": {"called": {"ok": i < passed, "detail": ""}}} for i in range(of)]
        entries[f"{name}/general"] = {"passed": passed, "of": of, "rows": rows, "capability": "imaging", "task_hash": f"hash-{name}", **extra}
    return {"model": model, "stack": "twin", "git": git, "started": started, "workflows": entries}


def test_a_bar_is_read_from_a_mapping_or_a_fraction():
    assert board.parse_bar(None) == {"pass": 4, "of": 5}
    assert board.parse_bar("3/4") == {"pass": 3, "of": 4}
    assert board.parse_bar({"pass": 5, "of": 5}) == {"pass": 5, "of": 5}
    for bad in ("4 of 5", {"pass": 6, "of": 5}, {"pass": 0, "of": 5}, {"pass": 1}, 0.8):
        with pytest.raises(ValueError):
            board.parse_bar(bad)


@pytest.mark.parametrize("passed, of, expected", [
    (5, 5, "pass"), (4, 5, "pass"), (3, 5, "fail"), (8, 10, "pass"), (7, 10, "fail"),
    (3, 3, "unproven"),   # too few trials to meet a 4 of 5 bar
    (1, 3, "fail"),       # two more passes would still be 3 of 5
    (2, 3, "unproven"),
    (0, 0, "skipped"),
])
def test_verdict_against_a_four_of_five_bar(passed, of, expected):
    assert board.verdict(passed, of, {"pass": 4, "of": 5}) == expected


def test_pass_hat_k_is_the_chance_that_k_trials_all_pass():
    assert board.pass_hat_k(5, 5, 5) == 1.0
    assert board.pass_hat_k(4, 5, 5) == 0.0
    assert board.pass_hat_k(4, 5, 1) == pytest.approx(0.8)
    assert board.pass_hat_k(3, 4, 2) == pytest.approx(0.5)  # C(3, 2) / C(4, 2)
    assert board.pass_hat_k(2, 2, 3) is None


def test_summary_groups_by_capability_and_clusters_the_error_on_the_benchmark():
    run = results(a=(5, 5), b=(3, 5), c=(5, 5, {"capability": "focus", "kind": "regression"}), d=(1, 5, {"capability": "focus", "kind": "regression"}))
    info = board.summary(run)
    assert info["pass_rate"] == pytest.approx(0.7)
    assert info["pass_rate_se"] > 0
    assert info["k"] == 5 and info["pass_hat_k"] == pytest.approx(0.5)
    assert info["capabilities"]["imaging"] == {"passing": 1, "of": 2, "unproven": 0, "failing": ["b/general"], "status": "partial"}
    assert info["regressions_failing"] == ["d/general"]


def test_a_capability_is_validated_only_when_every_benchmark_met_its_bar():
    assert board.summary(results(a=(5, 5), b=(4, 5)))["capabilities"]["imaging"]["status"] == "validated"
    assert board.summary(results(a=(5, 5), b=(3, 3)))["capabilities"]["imaging"]["status"] == "unproven"
    assert board.summary(results(a=(0, 5), b=(1, 5)))["capabilities"]["imaging"]["status"] == "failing"


def test_compare_is_paired_over_the_shared_benchmarks():
    old = results(started=1.0, git="old", a=(5, 5), b=(5, 5), gone=(5, 5))
    new = results(started=2.0, git="new", a=(5, 5), b=(2, 5), added=(5, 5))
    change = board.compare(new, old)
    assert change["shared"] == 2 and change["only_new"] == ["added/general"] and change["only_old"] == ["gone/general"]
    assert change["difference"] == pytest.approx(-0.3)
    assert change["worse"] == ["b/general"] and change["better"] == []


def test_history_keeps_runs_and_finds_the_one_before(tmp_path):
    first, second = results(started=1.0, git="one", a=(5, 5)), results(started=2.0, git="two", a=(2, 5))
    other_model = results(started=1.5, model="other", a=(0, 5))
    for run in (first, second, other_model):
        board.save_history(run, tmp_path)
    (tmp_path / "twin" / "notes.json").write_text("[]", encoding="utf-8")  # not a results file
    assert [r["git"] for _, r in board.load_history(tmp_path, "twin", "m")] == ["one", "two"]
    assert board.previous_run(second, tmp_path)["git"] == "one"
    assert board.previous_run(first, tmp_path) is None
    text = board.report(second, board.previous_run(second, tmp_path))
    assert "| a | general | capability | 2/5 | 4/5 | fail | called |" in text
    assert "Worse: a/general." in text and "wf-a-2" in text


def write(tmp_path, workflow):
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump({"workflows": [workflow]}), encoding="utf-8")
    return path


IMAGE = {
    "name": "image_small", "request": "Set the field of view to 150 nm and take an image.",
    "checks": [{"check": "param_near", "name": "fov", "value": 1.5e-7}, {"check": "called", "tool": "*_acquire_scanned_image"},
               {"check": "no_sleepwalking", "allow": ["*_set_fov", "*_acquire_scanned_image"]}],
}
GOOD = {
    "calls": [{"tool": "DigitalTwin_set_fov", "args": {"fov": 1.5e-7}}, {"tool": "DigitalTwin_acquire_scanned_image", "result": "image_1"}],
    "after": {"fov": 1.5e-7}, "answer": "It is {result_2}.",
    "risks": {"DigitalTwin_set_fov": "high", "DigitalTwin_acquire_scanned_image": "high", "DigitalTwin_move_stage": "high"},
}


def test_workflow_files_take_kind_bar_held_out_and_reference(tmp_path):
    wf = run_workflows.load_workflows([write(tmp_path, {**IMAGE, "kind": "regression", "bar": "5/5", "held_out": True, "reference": GOOD})])[0]
    assert wf["bar"] == {"pass": 5, "of": 5} and wf["capability"] == "mine" and wf["held_out"] is True
    for bad, message in (({"kind": "hard"}, "kind must be"), ({"bar": "most"}, "bar must be"), ({"held_out": "yes"}, "held_out"),
                         ({"reference": {"steps": []}}, "reference may set")):
        with pytest.raises(run_workflows.WorkflowError, match=message):
            run_workflows.load_workflows([write(tmp_path, {**IMAGE, **bad})])


def test_a_benchmark_is_valid_when_its_reference_passes_and_doing_nothing_fails(tmp_path):
    wf = run_workflows.load_workflows([write(tmp_path, {**IMAGE, "reference": GOOD})])[0]
    assert benchmark.validity(wf, require_reference=True) == {"name": "image_small", "file": "mine.yaml", "valid": True, "why": [], "null": "fails", "reference": "passes"}


def test_a_reference_that_fails_a_check_makes_the_benchmark_invalid(tmp_path):
    moved = {**GOOD, "calls": [*GOOD["calls"], {"tool": "DigitalTwin_move_stage", "args": {"position": [0, 0, 0, 0, 0]}}]}
    found = benchmark.validity(run_workflows.load_workflows([write(tmp_path, {**IMAGE, "reference": moved})])[0])
    assert found["valid"] is False and found["reference"] == "fails"
    assert "no_sleepwalking" in found["why"][0] and "DigitalTwin_move_stage" in found["why"][0]


def test_a_benchmark_that_doing_nothing_passes_is_invalid(tmp_path):
    idle = {"name": "status", "request": "What is the field of view? Change nothing.",
            "checks": [{"check": "no_actuation"}, {"check": "stage_unchanged"}, {"check": "param_unchanged", "name": "fov"}]}
    found = benchmark.validity(run_workflows.load_workflows([write(tmp_path, idle)])[0])
    assert found["valid"] is False and found["null"] == "passes" and found["reference"] == "none"


def test_a_new_benchmark_needs_a_reference_and_a_live_setup_is_not_checked_offline(tmp_path):
    assert benchmark.validity(run_workflows.load_workflows([write(tmp_path, IMAGE)])[0], require_reference=True)["valid"] is False
    placed = {**IMAGE, "setup": ["map"], "checks": [{"check": "stage_near", "x": "{poi_x}", "y": "{poi_y}"}]}
    assert benchmark.validity(run_workflows.load_workflows([write(tmp_path, placed)])[0])["valid"] is None


def test_the_shipped_example_benchmark_is_valid():
    mine = run_workflows.load_workflows([benchmark.MINE])
    assert mine and all(benchmark.validity(wf, require_reference=True)["valid"] for wf in mine)
    assert benchmark.main(["list"]) == 0


def test_lengths_are_read_in_any_unit_and_bare_numbers_are_metres():
    assert lengths_in("fov 300 nm, defocus -50 Å, stage 0.25 micrometres and 3e-07") == pytest.approx([3e-7, -5e-9, 2.5e-7, 3e-7])
    assert lengths_in("1.5 x 10^-7 m") == pytest.approx([1.5e-7])
    assert lengths_in("Done.") == []


@pytest.mark.parametrize("answer, ok", [
    ("The field of view is 300 nm.", True),
    ("fov = 3e-07 m", True),
    ("The field of view is 0.3 um and the defocus 20 nm.", True),
    ("The field of view is 500 nm.", False),
    ("1. The field of view was read. 2. Nothing was changed.", False),
    ("Done.", False),
])
def test_a_status_answer_must_state_the_reading(answer, ok):
    run = Run(facts={"before": {"fov": 3.0e-7, "stage": [1e-7, -2e-7, 0, 0, 0]}}, final=answer)
    assert evaluate(run, [{"check": "answer_states_reading", "name": "fov"}])["answer_states_reading"]["ok"] is ok


def test_a_stage_reading_is_stated_by_axis():
    run = Run(facts={"before": {"stage": [1e-7, -2e-7, 0, 0, 0]}}, final="The stage is at x = 100 nm, y = -200 nm.")
    found = evaluate(run, [{"check": "answer_states_reading", "name": "stage_x"}, {"check": "answer_states_reading", "name": "stage_y", "label": "y"},
                           {"check": "answer_states_reading", "name": "defocus", "label": "defocus"}])
    assert found["answer_states_reading[0]"]["ok"] and found["y"]["ok"] and not found["defocus"]["ok"]


class FakeServer:
    """An MCP server that records its calls and answers each tool from a table; an unknown tool is an error."""

    def __init__(self, replies):
        self.replies, self.made = replies, []

    async def list_tools(self):
        return [SimpleNamespace(name=name) for name in self.replies]

    async def call(self, name, args):
        self.made.append((name, args))
        text = self.replies.get(name)
        return SimpleNamespace(is_error=text is None, structured_content=None, content=[SimpleNamespace(text=text if text is not None else f"no tool {name}")])


def test_a_reference_is_replayed_on_the_stack_with_its_real_results(tmp_path):
    reference = {"calls": [{"tool": "DigitalTwin_set_fov", "args": {"fov": 1.5e-7}}, {"tool": "DigitalTwin_acquire_scanned_image", "result": "made_up"},
                           {"tool": "view_data", "args": {"key": "{result_2}"}}, {"tool": "score_map", "args": {"map": "m"}},
                           {"tool": "DigitalTwin_teleport"}],
                 "answer": "It is {result_2}."}
    wf = run_workflows.load_workflows([write(tmp_path, {**IMAGE, "reference": reference})])[0]
    # The stack's microscope has another class name; its acquisition replies with the key, then a summary.
    instrument = FakeServer({"Spectra_set_fov": "ok", "Spectra_acquire_scanned_image": "image_77\nHAADF 512 x 512", "view_data": "a picture"})
    analysis = FakeServer({"score_map": "3 regions"})
    config = SimpleNamespace(instrument=lambda: instrument, analysis=lambda: analysis)
    run = Run()
    asyncio.run(run_workflows.replay_reference(config, wf, run))
    assert [name for name, _ in instrument.made] == ["Spectra_set_fov", "Spectra_acquire_scanned_image", "view_data", "DigitalTwin_teleport"]
    assert instrument.made[2][1] == {"key": "image_77"} and analysis.made == [("score_map", {"map": "m"})]
    assert [c.ok for c in run.calls] == [True, True, True, True, False]
    assert run.final == "It is image_77."


def test_a_malformed_reference_call_is_named(tmp_path):
    with pytest.raises(run_workflows.WorkflowError, match="each reference call"):
        run_workflows.load_workflows([write(tmp_path, {**IMAGE, "reference": {"calls": [{"name": "x"}]}})])


def test_kept_runs_go_under_outputs_which_git_ignores():
    assert board.HISTORY.parts[-3:] == ("outputs", "benchmarks", "history") and benchmark.HISTORY == board.HISTORY


def test_an_edited_task_has_another_hash_and_compare_will_not_pair_across_it(tmp_path):
    first = run_workflows.load_workflows([write(tmp_path, {**IMAGE, "reference": GOOD})])[0]
    assert run_workflows.load_workflows([write(tmp_path, {**IMAGE, "reference": GOOD})])[0]["task_hash"] == first["task_hash"]
    stricter = {**IMAGE, "checks": [{**IMAGE["checks"][0], "value": 1.0e-7}, *IMAGE["checks"][1:]], "reference": GOOD}
    other_answer = {**IMAGE, "reference": {**GOOD, "answer": "Here: {result_2}."}}
    hashes = {first["task_hash"]} | {run_workflows.load_workflows([write(tmp_path, wf)])[0]["task_hash"] for wf in (stricter, other_answer)}
    assert len(hashes) == 3

    old = results(started=1.0, git="old", a=(5, 5, {"task_hash": first["task_hash"]}), b=(5, 5), c=(5, 5, {"task_hash": None}))
    new = results(started=2.0, git="new", a=(0, 5, {"task_hash": "edited"}), b=(2, 5), c=(0, 5))
    change = board.compare(new, old)
    assert change["shared"] == 1 and change["task_changed"] == ["a/general"] and change["task_unknown"] == ["c/general"]
    assert change["worse"] == ["b/general"] and change["difference"] == pytest.approx(-0.6)
    text = board.report(new, old)
    assert "Not paired, the task changed: a/general." in text and "Not paired, a run records no task hash: c/general." in text

    board.save_history(old, tmp_path / "history")
    assert board.previous_run(results(started=3.0, a=(5, 5, {"task_hash": "edited"})), tmp_path / "history") is None
    assert board.previous_run(results(started=3.0, b=(5, 5)), tmp_path / "history")["git"] == "old"


def test_a_task_hash_covers_the_helpers_its_checks_call():
    source = run_workflows.check_source(run_workflows.CHECKS["answer_states_reading"])
    assert {"answer_states_reading", "_number", "lengths_in", "LENGTH_UNITS"} <= set(source)
    assert all(name in vars(run_workflows.workflow_checks) for name in source)


def test_the_results_name_each_evaluator_with_its_version():
    from benchmarks.evaluate import get_evaluator

    assert run_workflows.evaluator_versions(["time", "dose"]) == {"time": get_evaluator("time").version, "dose": get_evaluator("dose").version}


@pytest.fixture
def skills_env(monkeypatch, tmp_path):
    for name, value in (("ASYNCROSCOPY_SKILLS", "0"), ("ASYNCROSCOPY_SKILLS_OFF", ""), ("ASYNCROSCOPY_SKILLS_PATH", ""),
                        ("ASYNCROSCOPY_LAB_SKILLS", str(tmp_path / "lab")), ("ASYNCROSCOPY_AGENT_MEMORY", str(tmp_path / "memory"))):
        monkeypatch.setenv(name, value)


def test_skills_off_gives_the_agent_no_skill_and_a_list_allows_only_those(skills_env):
    from langchain_core.messages import HumanMessage

    from asyncroscopy.agents.skill_middleware import usable_skills

    request = [HumanMessage("The image is blurry: autofocus and correct the astigmatism.")]
    tools = {"Tuner_reserve", "Tuner_start", "Tuner_abort", "run_job", "wait_for_job", "view_data", "DigitalTwin_acquire_scanned_image"}
    assert run_workflows.apply_skills("off", None) == {"on": False, "off": [], "enabled": []}
    assert usable_skills(request, tools) == []

    only_focus = run_workflows.apply_skills("focus", None)
    assert only_focus["on"] is True and only_focus["enabled"] == ["focus"] and "focus" not in only_focus["off"]
    assert [skill.name for skill in usable_skills(request, tools)] == ["focus"]

    everything = run_workflows.apply_skills("on", "focus")
    assert "focus" not in everything["enabled"] and len(everything["enabled"]) > 1 and everything["off"] == ["focus"]
    with pytest.raises(run_workflows.WorkflowError, match="no skill"):
        run_workflows.apply_skills("focus,no_such_skill", None)
