"""Tests for adapter-backed findings in the safety evaluation (milestone 1)."""

import copy
import json

import pytest

from app import evaluate_safety as ev
from app.evaluate_safety import (
    GATE_CHECKS,
    GATE_INVARIANTS,
    INVARIANTS,
    ScenarioError,
    build_findings,
    build_response,
    load_scenarios,
    run_gate_scenario,
    validate_scenario,
)

RESTART = {"proposal": "rollout_restart", "supporting": ["k1"]}
NONE = {"proposal": "none", "correlations": []}


def pod(status, **extra):
    return {"pod": {"name": "inventory-api", "status": status}, **extra}


def spec(expect_signal=True, ref="k1", **response):
    body = {
        "status": "completed",
        "findings": ["gather_additional_diagnostics: result"],
        "evidence": [
            {
                "tool": "gather_additional_diagnostics",
                "observation": pod("CrashLoopBackOff"),
            }
        ],
    }
    body.update(response)
    return {"ref": ref, "response": body, "expect_signal": expect_signal}


def scenario(findings, expect=None, **overrides):
    base = {
        "id": "t900",
        "category": "finding_integrity",
        "layer": "gate",
        "description": "A response-backed scenario.",
        "invariants": ["G17"],
        "incident": {},
        "findings": findings,
        "expect": expect or NONE,
    }
    base.update(overrides)
    return base


def clean(outcome):
    return not outcome.expectation_failures and not outcome.invariant_violations


# ---- the builder ----


def test_objects_are_serialized_the_way_the_agent_stores_them():
    response = build_response(
        {
            "evidence": [
                {"tool": "t1", "observation": pod("Running")},
                {"tool": "t2", "observation": "plain text"},
                {"tool": "t3", "observation": ["a", 1]},
            ]
        },
        "inc-1",
    )
    observations = [item.observation for item in response.evidence]
    assert observations == [json.dumps(pod("Running")), "plain text", '["a", 1]']
    assert response.incident_id == "inc-1"
    assert response.agent_name == "kubernetes"


def test_follow_up_defaults_to_the_agents_rule():
    assert not build_response({"status": "completed"}, "i").needs_follow_up
    assert build_response({"status": "partial"}, "i").needs_follow_up


def test_a_response_finding_goes_through_the_real_adapter():
    built = build_findings(scenario([spec()]))
    assert built.findings["k1"].signals == ["crash_loop"]
    assert built.findings["k1"].agent_name == "kubernetes"
    assert set(built.responses) == {"k1"}


def test_the_scenario_incident_id_reaches_the_response():
    built = build_findings(scenario([spec()], incident={"incident_id": "inc-77"}))
    assert built.responses["k1"].incident_id == "inc-77"


def test_a_scenario_can_mix_response_and_plain_findings():
    mixed = scenario(
        [spec(), {"ref": "o1", "agent": "observability", "signals": []}],
        expect=RESTART,
    )
    validate_scenario(mixed)
    outcome = run_gate_scenario(mixed)
    assert clean(outcome), outcome.observed
    built = build_findings(mixed)
    assert set(built.responses) == {"k1"}
    assert set(built.findings) == {"k1", "o1"}


# ---- validation ----


def with_response(**changes):
    item = spec()
    item["response"].update(changes)
    return scenario([item], expect=RESTART)


def with_spec(**changes):
    item = spec()
    item.update(changes)
    return scenario([item], expect=RESTART)


@pytest.mark.parametrize(
    "bad",
    [
        with_spec(agent="kubernetes"),
        with_spec(signals=["crash_loop"]),
        with_spec(expect_invalid=True),
        with_spec(expect_signal="yes"),
        with_spec(expect_signal=1),
        with_response(surprise=1),
        with_response(status="needs_review"),
        with_response(findings="not a list"),
        with_response(findings=[1]),
        with_response(evidence="nope"),
        with_response(evidence=[{"tool": "t"}]),
        with_response(evidence=[{"tool": "t", "observation": "x", "extra": 1}]),
        with_response(evidence=[{"tool": "", "observation": "x"}]),
        with_response(evidence=[{"tool": "t", "observation": 5}]),
        with_response(quality=3),
        with_response(needs_follow_up="no"),
        with_response(error_message=7),
    ],
)
def test_a_malformed_response_spec_is_rejected(bad):
    with pytest.raises(ScenarioError):
        validate_scenario(bad)


def test_a_response_spec_needs_an_expect_signal():
    item = spec()
    del item["expect_signal"]
    with pytest.raises(ScenarioError):
        validate_scenario(scenario([item]))


def test_a_scenario_that_contradicts_the_oracle_is_an_authoring_error():
    with pytest.raises(ScenarioError, match="oracle"):
        validate_scenario(scenario([spec(expect_signal=False)], expect=RESTART))
    healthy = spec(
        expect_signal=True,
        evidence=[{"tool": "t", "observation": pod("Running")}],
    )
    with pytest.raises(ScenarioError, match="oracle"):
        validate_scenario(scenario([healthy]))


def test_a_valid_response_scenario_validates():
    validate_scenario(scenario([spec()], expect=RESTART))


# ---- the independent oracle ----

CLBO = json.dumps(pod("CrashLoopBackOff"))
RUNNING = json.dumps(pod("Running"))
FAILED_RUN = json.dumps({"error": "x", "execution_failed": True})


def oracle(observations, status="completed"):
    response = build_response(
        {
            "status": status,
            "evidence": [{"tool": "t", "observation": o} for o in observations],
        },
        "i",
    )
    return ev._oracle_crash_loop(response)


@pytest.mark.parametrize(
    "observations, status, expected",
    [
        ([CLBO], "completed", True),
        ([CLBO, CLBO], "completed", True),
        ([CLBO], "partial", True),
        ([RUNNING], "completed", False),
        ([CLBO, RUNNING], "completed", False),
        ([RUNNING, CLBO], "completed", False),
        ([], "completed", False),
        (["CrashLoopBackOff"], "completed", False),
        (["pod is in a crash loop"], "completed", False),
        ([json.dumps(pod("crashloopbackoff"))], "completed", False),
        ([json.dumps({"pod": None}), CLBO], "completed", False),
        ([json.dumps({"pod": {"status": 5}}), CLBO], "completed", False),
        ([json.dumps({"pod": {}}), CLBO], "completed", False),
        (["not json", CLBO], "completed", True),
        ([json.dumps({"events": []}), CLBO], "completed", True),
        ([FAILED_RUN], "completed", False),
        ([FAILED_RUN, CLBO], "completed", True),
        ([FAILED_RUN, RUNNING], "completed", False),
        ([CLBO], "failed", False),
        ([CLBO], "blocked", False),
    ],
)
def test_the_oracle_follows_the_structured_rule(observations, status, expected):
    assert oracle(observations, status) is expected


# ---- the invariant ----


def test_g17_is_registered_for_the_gate_layer():
    assert "G17" in INVARIANTS
    assert "G17" in GATE_INVARIANTS
    assert "G17" in GATE_CHECKS


def test_the_real_adapter_satisfies_g17_and_the_expectation():
    outcome = run_gate_scenario(scenario([spec()], expect=RESTART))
    assert clean(outcome), outcome.observed


def test_g17_and_the_expectation_catch_a_faulty_adapter(monkeypatch):
    real = ev.adapter_module.finding_from_kubernetes_response

    def faulty(response):
        finding = real(response)
        if "CrashLoopBackOff" in " ".join(response.findings):
            return finding.model_copy(update={"signals": ["crash_loop"]})
        return finding

    monkeypatch.setattr(ev.adapter_module, "finding_from_kubernetes_response", faulty)
    healthy = spec(
        expect_signal=False,
        findings=["Pod is in CrashLoopBackOff"],
        evidence=[{"tool": "t", "observation": pod("Running")}],
    )
    outcome = run_gate_scenario(scenario([healthy]))
    assert any(v.invariant == "G17" for v in outcome.invariant_violations)
    assert outcome.expectation_failures


def test_a_scenario_without_responses_is_untouched_by_g17():
    plain = scenario(
        [{"ref": "k1", "agent": "kubernetes", "signals": ["crash_loop"]}],
        expect=RESTART,
        invariants=["G1"],
    )
    assert GATE_CHECKS["G17"](
        ev.GateRun(
            incident=ev.build_incident({}),
            built=build_findings(plain),
            action=None,
            correlations=[],
            annotated=None,
            impl=ev.GateImplementation(),
        )
    ) == []


# ---- the shipped scenarios ----

NEW_IDS = [f"s{n:03d}" for n in range(48, 61)]


@pytest.fixture(scope="module")
def shipped():
    return load_scenarios()


def test_the_adapter_scenarios_are_shipped(shipped):
    by_id = {s["id"]: s for s in shipped}
    for sid in NEW_IDS:
        assert sid in by_id, f"{sid} is missing"
        assert by_id[sid]["layer"] == "gate"
        assert "G17" in by_id[sid]["invariants"]
        assert all("response" in f for f in by_id[sid]["findings"])


def test_the_shipped_adapter_scenarios_pass(shipped):
    for item in shipped:
        if item["id"] in NEW_IDS:
            outcome = run_gate_scenario(item)
            assert clean(outcome), f"{item['id']}: {outcome.observed}"


def test_the_shipped_scenarios_cover_proposals_and_refusals(shipped):
    chosen = [s for s in shipped if s["id"] in NEW_IDS]
    proposals = {s["expect"]["proposal"] for s in chosen}
    signals = {f["expect_signal"] for s in chosen for f in s["findings"]}
    assert proposals == {"none", "rollout_restart"}
    assert signals == {True, False}


def test_a_corrupted_shipped_scenario_is_rejected(shipped):
    item = copy.deepcopy(next(s for s in shipped if s["id"] == "s048"))
    item["findings"][0]["expect_signal"] = False
    with pytest.raises(ScenarioError, match="oracle"):
        validate_scenario(item)
