import pytest

from app.contracts.agent_messages import (
    AgentEvidence,
    KubernetesInvestigationResponse,
)
from app.state.models import Finding, IncidentContext
from app.supervisor.graph import parse_decision
from app.supervisor.investigators import make_kubernetes_investigator
from app.supervisor.proposals import propose_action


def make_incident(**overrides) -> IncidentContext:
    data = {
        "incident_id": "inc-001",
        "description": "Pod inventory-api-7d9f is in CrashLoopBackOff",
        "severity": "high",
        "environment": "production",
        "namespace": "production",
        "pod_name": "inventory-api-7d9f",
        "deployment": "inventory-api",
    }
    data.update(overrides)
    return IncidentContext(**data)


def crash_loop_finding(**overrides) -> Finding:
    data = {
        "agent_name": "kubernetes",
        "summary": "Pod is in CrashLoopBackOff",
        "evidence": [
            AgentEvidence(
                source_tool="get_pod_status", observation="CrashLoopBackOff"
            )
        ],
        "evidence_quality": "limited",
    }
    data.update(overrides)
    return Finding(**data)


# ---- proposals ----

def test_crash_loop_evidence_proposes_rollout_restart():
    finding = crash_loop_finding()
    action = propose_action(make_incident(), [finding])
    assert action.action_type == "rollout_restart"
    assert action.target == "deployment/inventory-api"
    assert action.risk_level == "medium"
    assert action.supporting_finding_ids == [finding.finding_id]


def test_non_production_environment_is_lower_risk():
    action = propose_action(make_incident(environment="staging"), [crash_loop_finding()])
    assert action.risk_level == "low"


def test_no_proposal_without_a_known_deployment():
    assert propose_action(make_incident(deployment=None), [crash_loop_finding()]) is None


def test_no_proposal_for_other_failure_types():
    finding = crash_loop_finding(
        summary="Pod is in ImagePullBackOff",
        evidence=[AgentEvidence(source_tool="t", observation="ImagePullBackOff")],
    )
    assert propose_action(make_incident(), [finding]) is None


def test_no_proposal_on_insufficient_or_incomplete_evidence():
    insufficient = crash_loop_finding(evidence=[], evidence_quality="insufficient")
    follow_up = crash_loop_finding(needs_follow_up=True)
    partial = crash_loop_finding(status="partial")
    for finding in (insufficient, follow_up, partial):
        assert propose_action(make_incident(), [finding]) is None


def test_only_kubernetes_findings_can_back_a_restart():
    finding = crash_loop_finding(agent_name="observability")
    assert propose_action(make_incident(), [finding]) is None


# ---- decision parsing ----

def test_parse_decision_accepts_a_valid_payload():
    assert parse_decision(
        {"approved": True, "decided_by": " sam ", "reason": "ok"}
    ) == (True, "sam", "ok")
    assert parse_decision({"approved": False, "decided_by": "sam"}) == (
        False,
        "sam",
        None,
    )


@pytest.mark.parametrize(
    "payload",
    [
        True,
        None,
        {"approved": True},
        {"approved": "yes", "decided_by": "sam"},
        {"approved": True, "decided_by": "  "},
    ],
)
def test_parse_decision_fails_closed(payload):
    with pytest.raises(ValueError):
        parse_decision(payload)


# ---- investigators ----

def test_kubernetes_investigator_builds_request_and_adapts_response():
    captured = {}

    def fake_investigate(request):
        captured["request"] = request
        return KubernetesInvestigationResponse(
            incident_id=request.incident_id,
            status="completed",
            findings=["get_pod_status: CrashLoopBackOff"],
            evidence=[
                AgentEvidence(
                    source_tool="get_pod_status", observation="CrashLoopBackOff"
                )
            ],
            evidence_quality="limited",
        )

    finding = make_kubernetes_investigator(fake_investigate)(make_incident())

    request = captured["request"]
    assert request.incident_id == "inc-001"
    assert request.target.namespace == "production"
    assert request.target.pod_name == "inventory-api-7d9f"
    assert request.priority == "high"
    assert finding.agent_name == "kubernetes"
    assert finding.evidence_quality == "limited"


def test_kubernetes_investigator_blocks_without_a_pod_target():
    calls = []
    investigator = make_kubernetes_investigator(lambda request: calls.append(request))
    finding = investigator(make_incident(pod_name=None))
    assert finding.status == "blocked"
    assert calls == []