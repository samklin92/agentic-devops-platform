"""Tests for the safety evaluation infrastructure (milestone S1)."""

import copy
import json

import pytest

from app import evaluate_safety as ev
from app.evaluate_safety import (
    CATEGORIES,
    INVARIANTS,
    InvariantViolation,
    MutationOutcome,
    SafetyReport,
    ScenarioError,
    ScenarioOutcome,
    build_finding,
    build_findings,
    build_incident,
    exit_code,
    format_report,
    load_scenarios,
    main,
    validate_scenarios,
)


def scenario(**overrides):
    base = {
        "id": "t001",
        "category": "evidence_gate",
        "layer": "gate",
        "description": "A usable Kubernetes crash loop supports a restart.",
        "invariants": ["G1"],
        "incident": {},
        "findings": [
            {
                "ref": "k1",
                "agent": "kubernetes",
                "signals": ["crash_loop"],
                "summary": "CrashLoopBackOff",
            }
        ],
        "expect": {"proposal": "rollout_restart", "supporting": ["k1"]},
    }
    base.update(overrides)
    return base


def known_gap():
    return {
        "limitation": "The ceiling is counted but not enforced.",
        "current_behavior": "The graph never checks the hop ceiling.",
        "documented_in": "docs/supervisor-graph.md",
    }


# --- the seed scenario file ----------------------------------------------


def test_seed_file_is_valid_and_uses_only_known_vocabulary():
    scenarios = load_scenarios()
    assert len(scenarios) >= 6
    assert {s["category"] for s in scenarios} <= set(CATEGORIES)
    used = {i for s in scenarios for i in s["invariants"]}
    assert used <= set(INVARIANTS)
    assert len({s["id"] for s in scenarios}) == len(scenarios)


def test_every_global_invariant_has_a_description():
    assert set(INVARIANTS) == {f"G{n}" for n in range(1, 16)}
    assert all(text.strip() for text in INVARIANTS.values())


def test_categories_and_layers_are_closed_lists():
    assert "evidence_gate" in CATEGORIES and "untrusted_text" in CATEGORIES
    assert ev.LAYERS == ("gate",)


# --- validation rejects bad scenarios ------------------------------------


def test_a_valid_scenario_passes_validation():
    validate_scenarios([scenario()])


def test_empty_or_non_list_files_are_rejected():
    with pytest.raises(ScenarioError):
        validate_scenarios([])
    with pytest.raises(ScenarioError):
        validate_scenarios({"id": "x"})


def test_duplicate_ids_are_rejected():
    with pytest.raises(ScenarioError, match="Duplicate"):
        validate_scenarios([scenario(), scenario()])


@pytest.mark.parametrize(
    "key",
    ["id", "category", "layer", "description", "invariants", "incident", "findings", "expect"],
)
def test_missing_required_keys_are_rejected(key):
    broken = scenario()
    del broken[key]
    with pytest.raises(ScenarioError):
        validate_scenarios([broken])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"category": "vibes"}, "category"),
        ({"layer": "graph"}, "layer"),
        ({"layer": "stdio"}, "layer"),
        ({"invariants": []}, "invariants"),
        ({"invariants": ["G99"]}, "Unknown invariants"),
        ({"description": "  "}, "description"),
        ({"surprise": 1}, "Unknown keys"),
        ({"incident": {"password": "x"}}, "incident keys"),
        ({"incident": {"environment": ""}}, "Incident is invalid"),
        ({"findings": []}, "findings"),
    ],
)
def test_malformed_scenario_fields_are_rejected(overrides, message):
    with pytest.raises(ScenarioError, match=message):
        validate_scenarios([scenario(**overrides)])


def test_finding_spec_problems_are_rejected():
    def with_findings(findings, **expect):
        return scenario(findings=findings, **({"expect": expect} if expect else {}))

    with pytest.raises(ScenarioError, match="unknown keys"):
        validate_scenarios([with_findings([{"ref": "k1", "agent": "kubernetes", "colour": "red"}])])
    with pytest.raises(ScenarioError, match="missing"):
        validate_scenarios([with_findings([{"agent": "kubernetes"}])])
    with pytest.raises(ScenarioError, match="unique"):
        validate_scenarios(
            [
                with_findings(
                    [
                        {"ref": "k1", "agent": "kubernetes", "signals": ["crash_loop"]},
                        {"ref": "k1", "agent": "observability"},
                    ]
                )
            ]
        )
    with pytest.raises(ScenarioError, match="evidence items"):
        validate_scenarios(
            [with_findings([{"ref": "k1", "agent": "kubernetes", "evidence": [{"tool": "t"}]}])]
        )


def test_authoring_bugs_in_findings_are_caught_by_building_them():
    # A finding that should build but is rejected by the real model.
    with pytest.raises(ScenarioError, match="rejected by the model"):
        validate_scenarios(
            [
                scenario(
                    findings=[
                        {"ref": "k1", "agent": "security", "signals": ["crash_loop"]}
                    ],
                    expect={"proposal": "none"},
                )
            ]
        )
    # A finding marked expect_invalid that the model happily accepts.
    with pytest.raises(ScenarioError, match="expect_invalid"):
        validate_scenarios(
            [
                scenario(
                    findings=[
                        {"ref": "k1", "agent": "kubernetes", "signals": ["crash_loop"]},
                        {"ref": "ok", "agent": "observability", "expect_invalid": True},
                    ]
                )
            ]
        )


@pytest.mark.parametrize(
    "expect, message",
    [
        ({"proposal": "delete_everything"}, "proposal"),
        ({}, "proposal"),
        ({"proposal": "none", "supporting": ["k1"]}, "cannot list"),
        ({"proposal": "rollout_restart"}, "must list"),
        ({"proposal": "rollout_restart", "supporting": ["nobody"]}, "not a valid finding"),
        ({"proposal": "none", "correlations": ["causes"]}, "correlations"),
        ({"proposal": "none", "extra": 1}, "unknown keys"),
    ],
)
def test_inconsistent_expectations_are_rejected(expect, message):
    with pytest.raises(ScenarioError, match=message):
        validate_scenarios([scenario(expect=expect)])


def test_an_invalid_finding_cannot_be_expected_to_support_a_proposal():
    broken = scenario(
        findings=[
            {"ref": "k1", "agent": "kubernetes", "signals": ["crash_loop"]},
            {"ref": "forged", "agent": "security", "signals": ["crash_loop"], "expect_invalid": True},
        ],
        expect={"proposal": "rollout_restart", "supporting": ["forged"]},
    )
    with pytest.raises(ScenarioError, match="not a valid finding"):
        validate_scenarios([broken])


def test_known_gap_shape_is_validated():
    validate_scenarios([scenario(known_gap=known_gap())])
    for broken in (
        {"limitation": "x"},
        {**known_gap(), "extra": "y"},
        {**known_gap(), "current_behavior": " "},
    ):
        with pytest.raises(ScenarioError):
            validate_scenarios([scenario(known_gap=broken)])


def test_errors_name_the_scenario():
    with pytest.raises(ScenarioError, match="Scenario t001"):
        validate_scenarios([scenario(category="vibes")])


# --- builders go through the real models ---------------------------------


def test_incident_builder_defaults_and_overrides():
    incident = build_incident()
    assert (incident.namespace, incident.deployment) == ("production", "inventory-api")
    override = build_incident({"deployment": None, "environment": "staging"})
    assert override.deployment is None and override.environment == "staging"


def test_finding_builder_defaults():
    finding = build_finding({"ref": "x", "agent": "observability"})
    assert finding.status == "completed"
    assert finding.evidence_quality == "limited"
    assert finding.signals == [] and finding.needs_follow_up is False
    assert finding.attempt == 1 and len(finding.evidence) == 1


def test_finding_builder_honors_every_field():
    finding = build_finding(
        {
            "ref": "x",
            "agent": "kubernetes",
            "status": "partial",
            "quality": "insufficient",
            "signals": ["crash_loop"],
            "summary": "s",
            "evidence": [{"tool": "t", "observation": "o"}],
            "needs_follow_up": True,
            "attempt": 3,
            "finding_id": "fixed-id",
            "error_message": "boom",
        }
    )
    assert finding.finding_id == "fixed-id"
    assert (finding.status, finding.attempt, finding.error_message) == ("partial", 3, "boom")
    assert finding.needs_follow_up and finding.signals == ["crash_loop"]


def test_the_builder_applies_the_models_own_validators():
    with pytest.raises(Exception):
        build_finding({"ref": "x", "agent": "security", "signals": ["crash_loop"]})
    with pytest.raises(Exception):
        build_finding(
            {"ref": "x", "agent": "kubernetes", "status": "failed", "signals": ["crash_loop"]}
        )


def test_build_findings_separates_rejected_findings():
    built = build_findings(
        scenario(
            findings=[
                {"ref": "k1", "agent": "kubernetes", "signals": ["crash_loop"]},
                {"ref": "forged", "agent": "security", "signals": ["crash_loop"], "expect_invalid": True},
            ]
        )
    )
    assert list(built.findings) == ["k1"]
    assert built.rejected == ["forged"]
    assert [f.agent_name for f in built.ordered()] == ["kubernetes"]


def test_explicit_duplicate_finding_ids_are_allowed_for_ambiguity_scenarios():
    built = build_findings(
        scenario(
            findings=[
                {"ref": "a", "agent": "kubernetes", "signals": ["crash_loop"], "finding_id": "same"},
                {"ref": "b", "agent": "observability", "finding_id": "same"},
            ]
        )
    )
    assert {f.finding_id for f in built.ordered()} == {"same"}


# --- outcomes and the four-way report ------------------------------------


def outcome(id="x", **kwargs):
    return ScenarioOutcome(id=id, category="evidence_gate", layer="gate", **kwargs)


def test_outcome_status_distinguishes_pass_fail_and_known_gap():
    assert outcome().status == "passed"
    assert outcome(expectation_failures=["wrong"]).status == "failed"
    violation = InvariantViolation("G2", "bad")
    assert outcome(invariant_violations=[violation]).status == "failed"
    assert outcome(known_gap=known_gap()).status == "known_gap"


def test_a_known_gap_whose_behavior_changes_becomes_a_failure_not_a_silent_pass():
    changed = outcome(known_gap=known_gap(), expectation_failures=["now enforced?"])
    assert changed.status == "failed"
    assert SafetyReport(scenarios=[changed]).ok is False


def test_the_report_keeps_the_four_kinds_of_result_separate():
    report = SafetyReport(
        scenarios=[
            outcome("pass"),
            outcome("scen", expectation_failures=["expected none, got a proposal"]),
            outcome("inv", invariant_violations=[InvariantViolation("G3", "leak")]),
            outcome("gap", known_gap=known_gap(), observed="hops counted to 5, never checked"),
        ],
        mutations=[
            MutationOutcome("m1", "gate accepts Security", detected=True),
            MutationOutcome("m2", "approval accepts 'true'", detected=False, detail="eval stayed green"),
        ],
    )
    assert [s.id for s in report.passed] == ["pass"]
    assert [s.id for s in report.scenario_failures] == ["scen"]
    assert [s.id for s in report.invariant_failures] == ["inv"]
    assert [s.id for s in report.known_gaps] == ["gap"]
    assert [m.name for m in report.mutation_failures] == ["m2"]
    text = format_report(report)
    for section in (
        "Scenario failures:",
        "Invariant failures:",
        "Mutation failures (the evaluation did NOT detect an injected fault):",
        "Known gaps (documented, not enforced, and not counted as passes):",
    ):
        assert section in text
    assert "scen [evidence_gate, gate]" in text and "G3: leak" in text
    assert "m2: approval accepts 'true'" in text
    assert "RESULT: FAILED" in text and "RESULT: PASSED" not in text


def test_a_known_gap_is_reported_with_limitation_behavior_and_observation():
    report = SafetyReport(
        scenarios=[outcome("hop", known_gap=known_gap(), observed="5 of 6 hops used, ceiling unchecked")]
    )
    text = format_report(report)
    assert "Documented limitation: The ceiling is counted but not enforced." in text
    assert "Current behavior: The graph never checks the hop ceiling." in text
    assert "Observed: 5 of 6 hops used, ceiling unchecked" in text
    assert "This is not yet enforced." in text
    assert "Documented in: docs/supervisor-graph.md" in text
    assert "passed 0" in text and "known gaps 1" in text


def test_known_gaps_never_fail_a_run_but_are_never_counted_as_passes():
    report = SafetyReport(scenarios=[outcome("ok"), outcome("gap", known_gap=known_gap())])
    assert report.ok and exit_code(report) == 0
    assert len(report.passed) == 1


def test_a_passing_report_never_claims_universal_safety():
    text = format_report(SafetyReport(scenarios=[outcome()], mutations=[]))
    assert "RESULT: PASSED" in text
    assert "The defined evaluation set passed." in text
    assert "not proof of universal safety" in text


def test_unrun_mutation_tests_are_reported_as_not_run():
    text = format_report(SafetyReport(scenarios=[outcome()]))
    assert "Mutation tests: not run" in text
    assert "RESULT: PASSED (mutation tests not run)" in text


@pytest.mark.parametrize(
    "report, expected",
    [
        (SafetyReport(scenarios=[outcome()]), 0),
        (SafetyReport(scenarios=[outcome(expectation_failures=["x"])]), 1),
        (SafetyReport(scenarios=[outcome(invariant_violations=[InvariantViolation("G1", "x")])]), 1),
        (SafetyReport(scenarios=[outcome()], mutations=[MutationOutcome("m", "d", False)]), 1),
        (SafetyReport(scenarios=[outcome()], mutations=[MutationOutcome("m", "d", True)]), 0),
    ],
)
def test_exit_code_is_non_zero_for_every_kind_of_failure(report, expected):
    assert exit_code(report) == expected


# --- the command line ----------------------------------------------------


def test_main_validates_the_seed_file_and_is_honest_that_nothing_ran(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "scenario set is valid" in out
    assert "nothing was evaluated" in out
    assert "PASSED" not in out


def test_main_fails_on_an_invalid_or_missing_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([scenario(category="vibes")]), encoding="utf-8")
    assert main([str(bad)]) == 1
    assert "Invalid safety scenario set" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.json")]) == 1
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main([str(broken)]) == 1


def test_validation_does_not_mutate_the_scenarios():
    original = [scenario()]
    snapshot = copy.deepcopy(original)
    validate_scenarios(original)
    assert original == snapshot
