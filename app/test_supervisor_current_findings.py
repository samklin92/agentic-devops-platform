"""Tests for the shared current-findings selection and the gate built on it."""

import itertools

import pytest

from app.contracts.agent_messages import AgentEvidence
from app.state.models import Finding, IncidentContext
from app.supervisor.current_findings import current_findings
from app.supervisor.proposals import propose_action


def make_incident() -> IncidentContext:
    return IncidentContext(
        incident_id="inc-001",
        description="Pod inventory-api-7d9f is in CrashLoopBackOff",
        severity="high",
        environment="production",
        namespace="production",
        pod_name="inventory-api-7d9f",
        deployment="inventory-api",
    )


def evidence(observation="CrashLoopBackOff"):
    return AgentEvidence(source_tool="get_pod_status", observation=observation)


def crash_loop_finding(**overrides) -> Finding:
    data = {
        "agent_name": "kubernetes",
        "summary": "Pod is in CrashLoopBackOff",
        "evidence": [evidence()],
        "evidence_quality": "limited",
        "signals": ["crash_loop"],
    }
    data.update(overrides)
    return Finding(**data)


def healthy_finding(**overrides) -> Finding:
    return crash_loop_finding(
        summary="Pod is Running",
        evidence=[evidence("Running")],
        signals=[],
        **overrides,
    )


def other_agent_finding(agent: str, **overrides) -> Finding:
    data = {
        "agent_name": agent,
        "summary": f"{agent} finding",
        "evidence": [evidence("ok")],
        "evidence_quality": "limited",
    }
    data.update(overrides)
    return Finding(**data)


# ---- selection ----

def test_the_latest_attempt_is_the_current_finding():
    first = crash_loop_finding(attempt=1)
    second = crash_loop_finding(attempt=2)
    for ordering in itertools.permutations([first, second]):
        assert current_findings(ordering)["kubernetes"] is second


def test_a_single_finding_is_current():
    finding = crash_loop_finding()
    assert current_findings([finding]) == {"kubernetes": finding}


def test_exact_duplicates_collapse_to_one_finding():
    finding = crash_loop_finding()
    current = current_findings([finding, finding.model_copy()])
    assert current == {"kubernetes": finding}


def test_one_id_on_two_different_records_excludes_the_specialist():
    a = crash_loop_finding(finding_id="dup")
    b = healthy_finding(finding_id="dup")
    for ordering in itertools.permutations([a, b]):
        assert "kubernetes" not in current_findings(ordering)


def test_an_ambiguous_id_does_not_salvage_a_later_clean_attempt():
    a = crash_loop_finding(finding_id="dup", attempt=1)
    b = healthy_finding(finding_id="dup", attempt=1)
    later = crash_loop_finding(attempt=2)
    assert "kubernetes" not in current_findings([a, b, later])


def test_different_ids_at_the_same_attempt_exclude_the_specialist():
    a = crash_loop_finding(attempt=1)
    b = healthy_finding(attempt=1)
    for ordering in itertools.permutations([a, b]):
        assert "kubernetes" not in current_findings(ordering)


def test_a_same_attempt_conflict_at_an_older_attempt_still_excludes():
    old_a = crash_loop_finding(attempt=1)
    old_b = healthy_finding(attempt=1)
    later = crash_loop_finding(attempt=2)
    assert "kubernetes" not in current_findings([old_a, old_b, later])


def test_an_id_shared_across_specialists_excludes_both():
    k8s = crash_loop_finding(finding_id="shared")
    obs = other_agent_finding("observability", finding_id="shared")
    assert current_findings([k8s, obs]) == {}


def test_ambiguity_in_one_specialist_leaves_the_others_alone():
    a = crash_loop_finding(attempt=1)
    b = healthy_finding(attempt=1)
    security = other_agent_finding("security")
    assert current_findings([a, b, security]) == {"security": security}


def test_the_result_does_not_depend_on_input_order():
    findings = [
        crash_loop_finding(attempt=1),
        healthy_finding(attempt=2),
        other_agent_finding("security"),
        other_agent_finding("observability"),
    ]
    expected = current_findings(findings)
    for ordering in itertools.permutations(findings):
        assert current_findings(ordering) == expected


def test_no_findings_means_no_current_findings():
    assert current_findings([]) == {}


# ---- the gate uses only current findings ----

def test_a_superseded_crash_loop_does_not_support_a_proposal():
    stale = crash_loop_finding(attempt=1)
    healthy = healthy_finding(attempt=2)
    for ordering in itertools.permutations([stale, healthy]):
        assert propose_action(make_incident(), list(ordering)) is None


def test_the_latest_attempt_supports_the_proposal_alone():
    stale = crash_loop_finding(attempt=1)
    latest = crash_loop_finding(attempt=2)
    action = propose_action(make_incident(), [stale, latest])
    assert action.supporting_finding_ids == [latest.finding_id]


def test_a_later_partial_attempt_removes_the_proposal():
    usable = crash_loop_finding(attempt=1)
    partial = crash_loop_finding(attempt=2, status="partial")
    assert propose_action(make_incident(), [usable, partial]) is None


def test_duplicate_records_give_one_unique_supporting_id():
    finding = crash_loop_finding()
    action = propose_action(
        make_incident(), [finding, finding.model_copy(), finding.model_copy()]
    )
    assert action.supporting_finding_ids == [finding.finding_id]


@pytest.mark.parametrize("attempt", [1, 2])
def test_an_ambiguous_id_never_proposes_even_if_one_record_has_the_signal(attempt):
    signalled = crash_loop_finding(finding_id="dup", attempt=attempt)
    conflicting = healthy_finding(finding_id="dup", attempt=attempt)
    for ordering in itertools.permutations([signalled, conflicting]):
        assert propose_action(make_incident(), list(ordering)) is None


def test_same_attempt_conflicting_records_never_propose():
    signalled = crash_loop_finding(attempt=1)
    conflicting = healthy_finding(attempt=1)
    for ordering in itertools.permutations([signalled, conflicting]):
        assert propose_action(make_incident(), list(ordering)) is None


def test_every_proposal_has_exactly_one_supporting_id():
    findings = [
        crash_loop_finding(attempt=1),
        crash_loop_finding(attempt=2),
        other_agent_finding("security"),
    ]
    action = propose_action(make_incident(), findings)
    assert len(action.supporting_finding_ids) == 1
