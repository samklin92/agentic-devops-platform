"""Tests for the safety evaluation infrastructure (milestone S1)."""

import copy
import json

import pytest

from app import evaluate_safety as ev
from app.evaluate_safety import (
    CATEGORIES,
    GATE_CHECKS,
    GATE_INVARIANTS,
    INVARIANTS,
    BuiltFindings,
    GateImplementation,
    GateRun,
    InvariantViolation,
    MutationOutcome,
    SafetyReport,
    ScenarioError,
    ScenarioOutcome,
    build_finding,
    build_findings,
    build_incident,
    evaluate,
    exit_code,
    format_report,
    load_scenarios,
    main,
    run_gate_scenario,
    validate_scenarios,
)
from app.supervisor.proposals import is_usable, propose_action


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


def test_main_runs_the_evaluation_and_prints_the_four_way_report(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Safety evaluation" in out
    assert "Scenario failures:" in out and "Known gaps" in out
    assert "RESULT: PASSED (mutation tests not run)" in out
    assert "not proof of universal safety" in out


def test_main_exits_non_zero_when_a_scenario_fails(tmp_path, capsys):
    wrong = scenario(expect={"proposal": "none"})
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps([wrong]), encoding="utf-8")
    assert main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "RESULT: FAILED" in out
    assert "expected proposal 'none'" in out


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


# --- S2: the gate layer ---------------------------------------------------

REAL = GateImplementation()


def seed(id_):
    return next(s for s in load_scenarios() if s["id"] == id_)


def run(id_, impl=None):
    return run_gate_scenario(seed(id_), impl)


def violated(outcome):
    return {v.invariant for v in outcome.invariant_violations}


def test_the_real_gate_set_passes_every_scenario_and_every_invariant():
    report = evaluate()
    assert len(report.scenarios) >= 40
    assert report.ok and exit_code(report) == 0
    assert not report.scenario_failures and not report.invariant_failures
    assert len(report.passed) == len(report.scenarios)


def test_the_set_exercises_every_gate_invariant_and_category():
    scenarios = load_scenarios()
    for invariant in GATE_INVARIANTS:
        count = sum(invariant in s["invariants"] for s in scenarios)
        assert count >= 3, f"{invariant} is declared by only {count} scenarios"
    gate_categories = {
        "evidence_gate",
        "authority_separation",
        "untrusted_text",
        "correlation",
        "finding_integrity",
    }
    for category in gate_categories:
        count = sum(s["category"] == category for s in scenarios)
        assert count >= 5, f"{category} has only {count} scenarios"


def test_every_gate_check_runs_on_every_scenario_not_just_declared_ones():
    assert set(GATE_CHECKS) == set(GATE_INVARIANTS)
    seen = []
    original = dict(GATE_CHECKS)
    try:
        for name, check in original.items():
            GATE_CHECKS[name] = lambda run, n=name, c=check: (seen.append(n), c(run))[1]
        run_gate_scenario(seed("s018"))
    finally:
        GATE_CHECKS.update(original)
    assert sorted(seen) == sorted(GATE_INVARIANTS)


def test_gate_scenarios_are_order_independent():
    for original in load_scenarios():
        reversed_scenario = copy.deepcopy(original)
        reversed_scenario["findings"].reverse()
        assert run_gate_scenario(reversed_scenario).status == "passed", original["id"]


def test_running_a_scenario_twice_gives_the_same_outcome():
    for original in load_scenarios()[:10]:
        first, second = run_gate_scenario(original), run_gate_scenario(original)
        assert first == second


def test_validation_rejects_graph_only_invariants_and_bad_risk_levels():
    with pytest.raises(ScenarioError, match="cannot be checked at the gate layer"):
        validate_scenarios([scenario(invariants=["G6"])])
    with pytest.raises(ScenarioError, match="risk_level"):
        validate_scenarios(
            [scenario(expect={"proposal": "rollout_restart", "supporting": ["k1"], "risk_level": "spicy"})]
        )
    with pytest.raises(ScenarioError, match="needs a proposal"):
        validate_scenarios([scenario(expect={"proposal": "none", "risk_level": "low"})])


def test_expectation_mismatches_are_reported_in_words():
    bad = scenario(
        findings=[
            {"ref": "k1", "agent": "kubernetes", "signals": ["crash_loop"]},
            {"ref": "k2", "agent": "kubernetes", "signals": ["crash_loop"]},
        ],
        expect={
            "proposal": "rollout_restart",
            "supporting": ["k1"],
            "risk_level": "low",
            "correlations": ["corroborates"],
        },
    )
    outcome = run_gate_scenario(bad)
    text = " | ".join(outcome.expectation_failures)
    assert outcome.status == "failed"
    assert "expected supporting ['k1'], got ['k1', 'k2']" in text
    assert "expected risk 'low', got 'medium'" in text
    assert "expected correlations ['corroborates'], got []" in text
    assert "expected proposal 'none', got 'rollout_restart'" in " | ".join(
        run_gate_scenario(scenario(expect={"proposal": "none"})).expectation_failures
    )


def test_a_gate_known_gap_reports_what_was_observed_and_flips_if_behavior_changes():
    gap = scenario(known_gap=known_gap())
    outcome = run_gate_scenario(gap)
    assert outcome.status == "known_gap"
    assert "proposal=rollout_restart, supporting=['k1']" in outcome.observed
    changed = scenario(known_gap=known_gap(), expect={"proposal": "none"})
    assert run_gate_scenario(changed).status == "failed"


def test_evaluate_rejects_a_malformed_scenario_set():
    with pytest.raises(ScenarioError):
        evaluate([scenario(category="vibes")])


# --- the evaluation has teeth: injected faults are noticed ----------------


def _real_action(id_="s001"):
    spec = seed(id_)
    return propose_action(build_incident(spec["incident"]), build_findings(spec).ordered())


def _text_based_gate(incident, findings):
    """The old bug: a gate that reads crash-loop wording instead of the signal."""
    import re

    for finding in findings:
        text = " ".join([finding.summary] + [e.observation for e in finding.evidence])
        if (
            finding.agent_name == "kubernetes"
            and is_usable(finding)
            and re.search(r"crash[\s-]?loop", text, re.IGNORECASE)
        ):
            return propose_action(
                incident, [finding.model_copy(update={"signals": ["crash_loop"]})]
            )
    return None


def _any_agent_gate(incident, findings):
    """A gate that lets any specialist's usable finding authorize."""
    disguised = [
        f.model_copy(update={"agent_name": "kubernetes", "signals": ["crash_loop"]})
        for f in findings
        if is_usable(f)
    ]
    return propose_action(incident, disguised)


def _ignores_usability_gate(incident, findings):
    healed = [
        f.model_copy(
            update={"status": "completed", "evidence_quality": "limited", "needs_follow_up": False}
        )
        for f in findings
    ]
    return propose_action(incident, healed)


def _with_support(transform):
    def gate(incident, findings):
        action = propose_action(incident, findings)
        return None if action is None else transform(action, findings)

    return gate


def test_a_text_reading_gate_fails_scenarios_and_g14():
    impl = GateImplementation(propose=_text_based_gate)
    # G14 catches the fault on every untrusted-text scenario, even those
    # whose own wording does not happen to trigger it, because its poisoned
    # variant does.
    for id_ in ("s023", "s024", "s025", "s026", "s027", "s028"):
        outcome = run(id_, impl)
        assert outcome.status == "failed", id_
        assert "G14" in violated(outcome), id_
    # Where the scenario's own text contains crash-loop wording, the
    # scenario's expectation fails as well.
    for id_ in ("s023", "s024", "s025"):
        outcome = run(id_, impl)
        assert any("expected proposal 'none'" in f for f in outcome.expectation_failures), id_


def test_a_gate_that_lets_any_specialist_authorize_fails_g1_g2_and_g3():
    impl = GateImplementation(propose=_any_agent_gate)
    outcome = run("s018", impl)
    assert outcome.status == "failed"
    assert {"G1", "G2", "G3"} <= violated(outcome)


def test_a_gate_that_ignores_usability_fails_g15_and_g1():
    impl = GateImplementation(propose=_ignores_usability_gate)
    for id_ in ("s004", "s013", "s014", "s015"):
        outcome = run(id_, impl)
        assert outcome.status == "failed", id_
        assert {"G1", "G15"} <= violated(outcome), id_


def test_repeated_or_foreign_supporting_ids_fail_g2():
    repeats = GateImplementation(
        propose=_with_support(
            lambda a, f: a.model_copy(
                update={"supporting_finding_ids": a.supporting_finding_ids * 2}
            )
        )
    )
    assert "G2" in violated(run("s001", repeats))
    foreign = GateImplementation(
        propose=_with_support(
            lambda a, f: a.model_copy(
                update={
                    "supporting_finding_ids": a.supporting_finding_ids
                    + [x.finding_id for x in f if x.agent_name != "kubernetes"]
                }
            )
        )
    )
    assert "G2" in violated(run("s006", foreign))
    empty = GateImplementation(
        propose=_with_support(lambda a, f: a.model_copy(update={"supporting_finding_ids": []}))
    )
    assert "G2" in violated(run("s001", empty))


def test_a_non_restart_action_fails_g1():
    impl = GateImplementation(
        propose=_with_support(lambda a, f: a.model_copy(update={"action_type": "delete_pod"}))
    )
    assert "G1" in violated(run("s001", impl))


@pytest.mark.parametrize(
    "name, annotate",
    [
        (
            "changes the supporting findings",
            lambda action, correlations, findings: action
            and action.model_copy(update={"supporting_finding_ids": []}),
        ),
        (
            "changes the risk level",
            lambda action, correlations, findings: action
            and action.model_copy(update={"risk_level": "critical"}),
        ),
        (
            "removes the proposal",
            lambda action, correlations, findings: None,
        ),
        (
            "rewrites the rationale",
            lambda action, correlations, findings: action
            and action.model_copy(update={"rationale": "something else entirely"}),
        ),
    ],
)
def test_annotation_that_changes_more_than_the_text_fails_g4(name, annotate):
    outcome = run("s006", GateImplementation(annotate=annotate))
    assert "G4" in violated(outcome), name


def test_annotation_that_creates_a_proposal_fails_g4():
    fabricated = _real_action()
    impl = GateImplementation(
        annotate=lambda action, correlations, findings: action or fabricated
    )
    outcome = run("s018", impl)
    assert "G4" in violated(outcome)


def test_correlation_over_unusable_or_single_specialist_findings_fails_g5():
    from app.state.models import Correlation

    def bad(rule_ids):
        def correlator(findings, now=None):
            return [
                Correlation(
                    kind="corroborates",
                    rule="faulty",
                    finding_ids=rule_ids(findings),
                    explanation="Faulty correlation for the test.",
                    correlated_at=now,
                )
            ]

        return correlator

    partial = GateImplementation(correlate=bad(lambda f: [x.finding_id for x in f][:2]))
    assert "G5" in violated(run("s034", partial))  # includes the partial finding
    same_specialist = GateImplementation(
        correlate=bad(lambda f: [x.finding_id for x in f][:2])
    )
    assert "G5" in violated(run("s016", same_specialist))  # two Kubernetes findings
    unknown = GateImplementation(correlate=bad(lambda f: ["ghost-1", "ghost-2"]))
    assert "G5" in violated(run("s001", unknown))


def test_the_g12_check_notices_foreign_signals_and_signals_on_failed_findings():
    built = build_findings(seed("s001"))
    good = built.findings["k1"]
    foreign = good.model_copy(update={"agent_name": "observability"})  # bypasses validation
    run_ = GateRun(
        incident=build_incident({}),
        built=BuiltFindings({"x": foreign}, []),
        action=None,
        correlations=[],
        annotated=None,
        impl=REAL,
    )
    assert any("may not emit" in m for m in GATE_CHECKS["G12"](run_))
    failed = good.model_copy(update={"status": "failed"})
    run_.built = BuiltFindings({"x": failed}, [])
    assert any("failed but carries" in m for m in GATE_CHECKS["G12"](run_))


def test_the_g14_variants_keep_structure_and_change_only_text():
    from app.evaluate_safety import _POISON, _with_text

    original = build_findings(seed("s029")).findings["k1"]
    for text in (None, _POISON):
        variant = _with_text(original, text)
        assert variant.finding_id == original.finding_id
        assert (variant.agent_name, variant.status, variant.signals, variant.evidence_quality) == (
            original.agent_name,
            original.status,
            original.signals,
            original.evidence_quality,
        )
        assert [e.source_tool for e in variant.evidence] == [
            e.source_tool for e in original.evidence
        ]
        assert variant.summary != original.summary
    assert _with_text(original, _POISON).error_message == _POISON
    assert _with_text(original, None).error_message is None


def test_the_real_functions_pass_the_same_scenarios_the_faults_fail():
    for id_ in ("s018", "s023", "s004", "s006", "s034", "s016", "s001"):
        assert run(id_).status == "passed", id_
