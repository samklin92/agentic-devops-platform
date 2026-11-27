"""Tests for the graph layer of the safety evaluation (S3)."""

import copy
from types import SimpleNamespace as NS

import pytest

import app.evaluate_safety as ev
from app.evaluate_safety import (
    GRAPH_CHECKS,
    GRAPH_INVARIANTS,
    SafetyReport,
    ScenarioError,
    ScenarioOutcome,
    evaluate,
    format_report,
    load_scenarios,
    run_graph_scenario,
    validate_scenario,
)
from app.state.models import CallLimits


def graph_scenarios():
    return [s for s in load_scenarios() if s["layer"] == "graph"]


def real(id_):
    return copy.deepcopy(next(s for s in graph_scenarios() if s["id"] == id_))


@pytest.fixture(scope="module")
def graph_report():
    """Run the real graph scenarios once and share the report."""
    return evaluate(graph_scenarios())


# ---- the real scenario set ----

def test_the_real_graph_scenarios_pass_and_the_hop_ceiling_is_the_only_gap(
    graph_report,
):
    assert len(graph_report.scenarios) == 18
    assert graph_report.ok
    assert not graph_report.scenario_failures
    assert not graph_report.invariant_failures
    assert [s.id for s in graph_report.known_gaps] == ["s118"]
    assert len(graph_report.passed) == 17


def test_every_graph_invariant_is_exercised_by_some_scenario():
    named = {i for s in graph_scenarios() for i in s["invariants"]}
    assert named == set(GRAPH_INVARIANTS)


def test_the_hop_ceiling_gap_flips_to_a_failure_if_behavior_changes():
    scenario = real("s118")
    assert run_graph_scenario(scenario).status == "known_gap"
    scenario["expect"]["supervisor_hops"] = 3
    outcome = run_graph_scenario(scenario)
    assert outcome.status == "failed"
    assert any("supervisor_hops" in text for text in outcome.expectation_failures)


def test_the_hop_ceiling_gap_reports_what_was_observed():
    outcome = run_graph_scenario(real("s118"))
    assert "supervisor_hops=5 (limit 3)" in outcome.observed


def test_a_wrong_expectation_is_reported_in_words():
    scenario = real("s101")
    scenario["expect"]["executions"] = 0
    scenario["expect"]["approval"] = "rejected"
    outcome = run_graph_scenario(scenario)
    text = " | ".join(outcome.expectation_failures)
    assert outcome.status == "failed"
    assert "expected executions 0, got 1" in text
    assert "expected approval 'rejected', got 'approved'" in text


# ---- schema and validation ----

def test_a_real_graph_scenario_validates():
    validate_scenario(real("s101"))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.pop("steps"), "Missing"),
        (lambda s: s.pop("investigators"), "Missing"),
        (lambda s: s.__setitem__("findings", []), "Unknown keys"),
        (lambda s: s.__setitem__("invariants", ["G1"]), "cannot be checked at the graph layer"),
        (lambda s: s["steps"].__setitem__(0, {"resume": {"action_id": "$typo"}}), "unknown tokens"),
        (lambda s: s["steps"].__setitem__(0, {"resume": {}, "extra": 1}), "exactly one key"),
        (lambda s: s["investigators"].__setitem__("kubernetes", {"raises": "SystemExit"}), "raises must be one of"),
        (lambda s: s["investigators"].__setitem__("kubernetes", {"raises": "RuntimeError", "finding": {}}), "exactly one of"),
        (lambda s: s["investigators"].__setitem__("ghost", {"raises": "RuntimeError"}), "unknown agents"),
        (lambda s: s["investigators"]["kubernetes"]["finding"].__setitem__("agent", "security"), "different agent"),
        (lambda s: s.__setitem__("call_limits", {"max_supervisor_hops": 0}), "at least 1"),
        (lambda s: s.__setitem__("call_limits", {"bogus": 1}), "call_limits keys"),
        (lambda s: s.__setitem__("call_counts", {"kubernetes": -1}), "call_counts"),
        (lambda s: s["expect"].__setitem__("executions", 2), "0 or 1"),
        (lambda s: s["expect"].__setitem__("approval", "maybe"), "expect.approval"),
        (lambda s: s["expect"].__setitem__("raised_steps", [9]), "step indexes"),
        (lambda s: s["expect"].__setitem__("surprise", 1), "unknown keys"),
        (lambda s: s.__setitem__("expect", {}), "non-empty"),
    ],
)
def test_malformed_graph_scenarios_are_rejected(mutate, message):
    scenario = real("s101")
    mutate(scenario)
    with pytest.raises(ScenarioError, match=message):
        validate_scenario(scenario)


def test_a_gate_scenario_cannot_use_graph_keys_or_invariants():
    gate = copy.deepcopy(next(s for s in load_scenarios() if s["layer"] == "gate"))
    with_keys = {**gate, "steps": []}
    with pytest.raises(ScenarioError, match="Unknown keys"):
        validate_scenario(with_keys)
    with_invariant = {**gate, "invariants": ["G6"]}
    with pytest.raises(ScenarioError, match="cannot be checked at the gate layer"):
        validate_scenario(with_invariant)


# ---- helpers ----

def test_tokens_are_replaced_everywhere_in_a_decision():
    tokens = {"$pending_action_id": "A1"}
    payload = {"action_id": "$pending_action_id", "nested": ["$pending_action_id", 1]}
    assert ev._substitute(payload, tokens) == {
        "action_id": "A1",
        "nested": ["A1", 1],
    }
    assert ev._substitute(None, tokens) is None
    assert ev._substitute("plain", tokens) == "plain"


def test_a_scripted_investigator_counts_calls_and_raises_on_request():
    finding = NS(name="finding")
    ok = ev.ScriptedInvestigator(finding=finding)
    assert ok(None) is finding and ok.calls == 1
    boom = ev.ScriptedInvestigator(error=RuntimeError("scripted"))
    with pytest.raises(RuntimeError):
        boom(None)
    assert boom.calls == 1


def test_the_report_counts_scenarios_by_layer():
    report = SafetyReport(
        scenarios=[
            ScenarioOutcome(id="a", category="evidence_gate", layer="gate"),
            ScenarioOutcome(id="b", category="evidence_gate", layer="gate"),
            ScenarioOutcome(id="c", category="approval_protocol", layer="graph"),
        ]
    )
    assert "By layer: gate 2, graph 1" in format_report(report)


# ---- the invariant checks catch broken traces ----
# Synthetic traces stand in for a faulty graph, so each check is shown to
# fire on the fault it exists for and to stay quiet on a correct run.

ACTION = NS(action_id="A1", action_type="rollout_restart", supporting_finding_ids=["f1"])
KUBERNETES = NS(finding_id="f1", agent_name="kubernetes", status="completed", signals=["crash_loop"])
FAILED = NS(finding_id="f9", agent_name="kubernetes", status="failed", signals=[])
APPROVED = NS(status="approved", decided_by="sam", action_id="A1")
REJECTED = NS(status="rejected", decided_by="sam", action_id="A1")
EXECUTION = NS(simulated=True, action_id="A1")
GOOD = {"approved": True, "decided_by": "sam", "action_id": "A1"}


def entry(index, event="event"):
    return NS(entry_id=f"e{index}", event=event)


def snapshot(
    approval=None,
    execution=None,
    audit=(),
    action=None,
    findings=(),
    calls=None,
    counts=None,
    raised=None,
    next_nodes=(),
    returned=None,
):
    values = {
        "approval": approval,
        "execution": execution,
        "audit_log": list(audit),
        "proposed_action": action,
        "findings": list(findings),
        "agent_call_counts": counts if counts is not None else {"kubernetes": 1},
    }
    return ev.Snapshot(
        raised=raised,
        values=values,
        next=tuple(next_nodes),
        calls=calls if calls is not None else {"kubernetes": 1},
        returned=returned,
    )


def make_run(initial, steps, payloads, scripted=None, seeded=None, limits=None):
    return ev.GraphRun(
        scenario={},
        initial=initial,
        steps=steps,
        payloads=payloads,
        pending_action_id="A1",
        seeded=seeded or {},
        limits=limits or CallLimits(),
        scripted=scripted or {"kubernetes": "finding"},
        impl=None,
    )


def violations(run):
    return {name: check(run) for name, check in GRAPH_CHECKS.items() if check(run)}


def paused():
    return snapshot(
        action=ACTION, findings=[KUBERNETES], audit=[entry(1)], next_nodes=("approval",)
    )


def completed():
    return snapshot(
        approval=APPROVED,
        execution=EXECUTION,
        action=ACTION,
        findings=[KUBERNETES],
        audit=[entry(1), entry(2, "approval_granted"), entry(3, "execution_simulated")],
    )


def test_a_correct_run_trips_no_graph_invariant():
    assert violations(make_run(paused(), [completed()], [GOOD])) == {}


def test_g6_catches_an_execution_after_a_rejection():
    done = snapshot(
        approval=REJECTED, execution=EXECUTION, action=ACTION,
        findings=[KUBERNETES], audit=[entry(1)],
    )
    found = violations(make_run(paused(), [done], [dict(GOOD, approved=False)]))
    assert "G6" in found


def test_g6_catches_an_execution_after_a_non_boolean_approval():
    found = violations(make_run(paused(), [completed()], [dict(GOOD, approved="true")]))
    assert "G6" in found and "G8" in found


def test_g6_catches_an_execution_before_any_decision():
    found = violations(make_run(completed(), [], []))
    assert "G6" in found


def test_g7_catches_unsimulated_duplicate_and_foreign_executions():
    base = dict(approval=APPROVED, action=ACTION, findings=[KUBERNETES])
    unsimulated = snapshot(
        execution=NS(simulated=False, action_id="A1"),
        audit=[entry(1), entry(2, "execution_simulated"), entry(3, "execution_simulated")],
        **base,
    )
    found = violations(make_run(paused(), [unsimulated], [GOOD]))
    assert len(found["G7"]) == 2
    foreign = snapshot(
        execution=NS(simulated=True, action_id="OTHER"), audit=[entry(1)], **base
    )
    found = violations(make_run(paused(), [foreign], [GOOD]))
    assert "G7" in found and "G13" in found


def test_g8_catches_a_malformed_decision_that_changes_state_even_if_it_raises():
    changed = snapshot(
        approval=APPROVED, action=ACTION, findings=[KUBERNETES],
        audit=[entry(1)], raised="ValueError",
    )
    found = violations(make_run(paused(), [changed], [dict(GOOD, approved="yes")]))
    assert "G8" in found


def test_g13_catches_an_unbound_decision_that_changes_state():
    changed = snapshot(
        approval=APPROVED, action=ACTION, findings=[KUBERNETES],
        audit=[entry(1)], raised="ValueError",
    )
    for action_id in ("A2", "", None, 7):
        found = violations(make_run(paused(), [changed], [dict(GOOD, action_id=action_id)]))
        assert "G13" in found and "G8" not in found
    missing = {k: v for k, v in GOOD.items() if k != "action_id"}
    assert "G13" in violations(make_run(paused(), [changed], [missing]))


def test_g13_catches_a_changed_returned_state_even_when_the_checkpoint_is_clean():
    clean = snapshot(
        action=ACTION, findings=[KUBERNETES], audit=[entry(1)],
        next_nodes=("approval",), returned={"approval": APPROVED},
    )
    found = violations(make_run(paused(), [clean], [dict(GOOD, action_id="A2")]))
    assert "G13" in found


def test_a_result_without_the_key_is_not_read_as_a_change():
    clean = snapshot(
        action=ACTION, findings=[KUBERNETES], audit=[entry(1)],
        next_nodes=("approval",), returned={"__interrupt__": 1},
    )
    assert violations(make_run(paused(), [clean], [dict(GOOD, action_id="A2")])) == {}


def test_g13_catches_an_approval_or_execution_for_another_action():
    wrong = snapshot(
        approval=NS(status="approved", decided_by="sam", action_id="OTHER"),
        execution=EXECUTION, action=ACTION, findings=[KUBERNETES],
        audit=[entry(1), entry(2, "execution_simulated")],
    )
    found = violations(make_run(paused(), [wrong], [GOOD]))
    assert "G13" in found


def test_g9_catches_a_shortened_or_rewritten_audit_log():
    shortened = snapshot(action=ACTION, findings=[KUBERNETES], audit=[], next_nodes=("approval",))
    assert "G9" in violations(make_run(paused(), [shortened], [dict(GOOD, action_id="A2")]))
    rewritten = snapshot(
        action=ACTION, findings=[KUBERNETES], audit=[entry(9)], next_nodes=("approval",)
    )
    assert "G9" in violations(make_run(paused(), [rewritten], [dict(GOOD, action_id="A2")]))


def test_g10_catches_a_crash_a_missing_failed_finding_and_a_proposal_on_failure():
    scripted = {"kubernetes": "raises"}
    crashed = snapshot(raised="RuntimeError")
    assert "G10" in violations(make_run(crashed, [], [], scripted=scripted))

    unrecorded = snapshot(findings=[])
    assert "G10" in violations(make_run(unrecorded, [], [], scripted=scripted))

    recorded = snapshot(findings=[FAILED])
    assert violations(make_run(recorded, [], [], scripted=scripted)) == {}

    proposing = snapshot(
        action=NS(action_id="A1", action_type="rollout_restart", supporting_finding_ids=["f9"]),
        findings=[FAILED], next_nodes=("approval",),
    )
    assert "G10" in violations(make_run(proposing, [], [], scripted=scripted))


def test_g11_catches_a_specialist_that_runs_again_or_exceeds_its_budget():
    again = snapshot(
        action=ACTION, findings=[KUBERNETES], audit=[entry(1)],
        calls={"kubernetes": 2}, counts={"kubernetes": 2}, next_nodes=("approval",),
    )
    assert "G11" in violations(make_run(paused(), [again], [GOOD]))

    limit = CallLimits(max_calls_per_agent=1)
    over = snapshot(
        action=ACTION, findings=[KUBERNETES], audit=[entry(1)],
        counts={"kubernetes": 2}, next_nodes=("approval",),
    )
    found = violations(make_run(over, [], [], seeded={"kubernetes": 1}, limits=limit))
    assert "G11" in found

    within = violations(make_run(over, [], [], seeded={"kubernetes": 1}))
    assert within == {}
