"""Tests for the safety evaluation's mutation tests (S4)."""

import copy
import json

import pytest

import app.safety_mutations as sm
from app.evaluate_safety import (
    GateImplementation,
    MutationOutcome,
    SafetyReport,
    ScenarioError,
    evaluate,
    format_report,
    load_scenarios,
    main,
)
from app.safety_mutations import (
    MUTATIONS,
    Mutation,
    evaluate_with_mutations,
    patched_graph,
    run_mutation,
    run_mutations,
)
from app.supervisor import graph as graph_module


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="module")
def outcomes(scenarios):
    """Run every real mutation once and share the results."""
    return {outcome.name: outcome for outcome in run_mutations(scenarios)}


# ---- the real mutations ----

def test_there_are_twelve_unique_mutations_across_both_layers():
    names = [m.name for m in MUTATIONS]
    assert len(names) == len(set(names)) == 12
    assert sum(m.layer == "gate" for m in MUTATIONS) == 7
    assert sum(m.layer == "graph" for m in MUTATIONS) == 5


@pytest.mark.parametrize("name", [m.name for m in MUTATIONS])
def test_every_fault_is_detected_by_the_evaluation(outcomes, name):
    outcome = outcomes[name]
    assert outcome.detected, f"{name} was not detected: {outcome.detail}"
    assert outcome.detail.startswith("detected")


def test_the_model_fault_is_detected_as_an_accepted_invalid_finding(outcomes):
    detail = outcomes["signal_guard_on_failed_findings_removed"].detail
    assert "expect_invalid" in detail


def test_no_mutation_leaves_the_graph_module_patched(outcomes):
    assert graph_module.parse_decision is sm.REAL_PARSE_DECISION
    assert graph_module.ExecutionRecord.__name__ == "ExecutionRecord"
    assert graph_module.ExecutionRecord is not sm._unsimulated_execution_record


# ---- the detector itself ----

def test_a_fault_free_gate_run_is_not_detected(scenarios):
    control = Mutation("control", "no fault", "gate", gate=lambda: GateImplementation())
    outcome = run_mutation(control, scenarios)
    assert not outcome.detected
    assert "still passed" in outcome.detail


def test_a_fault_free_graph_run_is_not_detected(scenarios):
    control = Mutation("control", "no fault", "graph", graph=lambda: {})
    outcome = run_mutation(control, scenarios)
    assert not outcome.detected
    assert "still passed" in outcome.detail


def test_a_mutation_that_crashes_is_not_detected(scenarios):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    crashing = Mutation(
        "crash", "raises", "gate", gate=lambda: GateImplementation(propose=explode)
    )
    outcome = run_mutation(crashing, scenarios)
    assert not outcome.detected
    assert "RuntimeError" in outcome.detail and "could not be judged" in outcome.detail


def test_an_unrelated_scenario_error_is_not_a_detection(scenarios):
    def bad_factory(**kwargs):
        raise ScenarioError("an authoring problem, not a detection")

    mutation = Mutation(
        "authoring",
        "raises ScenarioError",
        "gate",
        gate=lambda: GateImplementation(finding_factory=bad_factory),
    )
    outcome = run_mutation(mutation, scenarios)
    assert not outcome.detected
    assert "ScenarioError" in outcome.detail


def test_a_layer_with_no_scenarios_cannot_detect_anything():
    control = Mutation("control", "no fault", "graph", graph=lambda: {})
    gate_only = [s for s in load_scenarios() if s["layer"] == "gate"]
    outcome = run_mutation(control, gate_only)
    assert not outcome.detected
    assert "no graph scenarios" in outcome.detail


def test_a_mutation_needs_exactly_one_matching_builder():
    with pytest.raises(ValueError):
        Mutation("bad", "x", "gate")
    with pytest.raises(ValueError):
        Mutation("bad", "x", "gate", graph=lambda: {})
    with pytest.raises(ValueError):
        Mutation("bad", "x", "graph", gate=lambda: GateImplementation())
    with pytest.raises(ValueError):
        Mutation(
            "bad", "x", "gate",
            gate=lambda: GateImplementation(), graph=lambda: {},
        )


# ---- patching the graph module safely ----

def test_patched_graph_restores_names_after_the_body_runs():
    original = graph_module.parse_decision
    with patched_graph(parse_decision=lambda raw, pending: (True, "x", None)):
        assert graph_module.parse_decision is not original
    assert graph_module.parse_decision is original


def test_patched_graph_restores_names_when_the_body_raises():
    original = graph_module.parse_decision
    with pytest.raises(RuntimeError):
        with patched_graph(parse_decision=lambda raw, pending: (True, "x", None)):
            raise RuntimeError("inside")
    assert graph_module.parse_decision is original


def test_patched_graph_rejects_a_name_the_module_does_not_have():
    with pytest.raises(AttributeError, match="silently do nothing"):
        with patched_graph(no_such_name=lambda: None):
            pass


def test_patched_graph_rejects_a_replacement_that_changes_nothing():
    with pytest.raises(ValueError, match="original object"):
        with patched_graph(parse_decision=graph_module.parse_decision):
            pass


# ---- reporting and the baseline ----

def test_the_baseline_must_pass_before_any_fault_is_injected(scenarios):
    broken = copy.deepcopy(scenarios)
    first_gate = next(s for s in broken if s["layer"] == "gate")
    # An expectation no real run can meet, but one that still validates.
    first_gate["expect"]["correlations"] = ["co_occurs"] * 3
    report = evaluate_with_mutations(broken)
    assert report.mutations is None
    assert not report.ok


def test_an_undetected_mutation_fails_the_report():
    report = SafetyReport(
        scenarios=[],
        mutations=[
            MutationOutcome("caught", "a fault", True, "detected by s001"),
            MutationOutcome("escaped", "another fault", False, "all still passed"),
        ],
    )
    text = format_report(report)
    assert not report.ok
    assert "Mutation tests: 2 (detected 1, undetected 1)" in text
    assert "escaped: another fault" in text
    assert "RESULT: FAILED" in text


# ---- the command line ----

@pytest.fixture
def gate_only_file(tmp_path):
    gate = [s for s in load_scenarios() if s["layer"] == "gate"]
    path = tmp_path / "gate_only.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    return str(path)


def test_the_mutations_flag_runs_them_and_passes_when_all_are_detected(
    monkeypatch, capsys, gate_only_file
):
    gate_mutations = tuple(m for m in MUTATIONS if m.layer == "gate")
    monkeypatch.setattr(sm, "MUTATIONS", gate_mutations)
    assert main(["--mutations", gate_only_file]) == 0
    output = capsys.readouterr().out
    assert "Mutation tests: 7 (detected 7, undetected 0)" in output
    assert "RESULT: PASSED" in output


def test_the_mutations_flag_exits_non_zero_when_a_fault_goes_undetected(
    monkeypatch, capsys, gate_only_file
):
    control = Mutation("control", "no fault", "gate", gate=lambda: GateImplementation())
    monkeypatch.setattr(sm, "MUTATIONS", (*MUTATIONS[:1], control))
    assert main(["--mutations", gate_only_file]) == 1
    output = capsys.readouterr().out
    assert "undetected 1" in output
    assert "RESULT: FAILED" in output


def test_without_the_flag_no_mutation_runs(capsys):
    assert main([]) == 0
    assert "Mutation tests: not run" in capsys.readouterr().out


def test_an_unknown_option_is_rejected(capsys):
    assert main(["--surprise"]) == 2
    assert "Unknown option" in capsys.readouterr().out


# ---- each fault is caught by the scenarios designed for it ----
# The first scenario to fail is partly incidental: a fault that approves
# everything also fails the rejection scenario. These pin the intended
# catchers, so each fault must also be detected by each scenario listed,
# run on its own.

INTENDED_CATCHERS = {
    "gate_accepts_other_specialists": ["s003"],
    "gate_ignores_needs_follow_up": ["s015"],
    "annotate_changes_supporting_ids": ["s006"],
    "correlate_includes_partial_findings": ["s034"],
    "gate_drops_crash_loop_requirement": ["s010"],
    "parse_decision_accepts_truthy_values": ["s103", "s104"],
    "rejection_treated_as_approval": ["s102"],
    "execution_not_marked_simulated": ["s101"],
    "malformed_decision_reaches_execution": ["s103", "s104", "s105", "s106", "s107"],
    "wrong_or_replayed_action_id_accepted": ["s109", "s110", "s111"],
    "gate_uses_raw_findings": ["s016", "s017", "s043", "s044", "s045", "s046", "s047"],
}


@pytest.mark.parametrize(
    "name, scenario_id",
    [(name, sid) for name, ids in INTENDED_CATCHERS.items() for sid in ids],
)
def test_each_fault_is_caught_by_its_intended_scenario(scenarios, name, scenario_id):
    chosen = [s for s in scenarios if s["id"] == scenario_id]
    assert len(chosen) == 1, f"{scenario_id} is not in the scenario set"
    mutation = next(m for m in MUTATIONS if m.name == name)
    outcome = run_mutation(mutation, chosen)
    assert outcome.detected, f"{scenario_id} did not catch {name}: {outcome.detail}"


def test_a_malformed_decision_fault_leaves_valid_decisions_working(scenarios):
    chosen = [s for s in scenarios if s["id"] in ("s101", "s102")]
    mutation = next(
        m for m in MUTATIONS if m.name == "malformed_decision_reaches_execution"
    )
    outcome = run_mutation(mutation, chosen)
    assert not outcome.detected, outcome.detail
