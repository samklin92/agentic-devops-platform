import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.contracts.agent_messages import AgentEvidence
from app.state.graph_state import new_platform_state
from app.state.models import CallLimits, Finding, IncidentContext
from app.state.serde import platform_serializer
from app.supervisor.graph import build_supervisor_graph

CONFIG = {"configurable": {"thread_id": "inc-001"}}


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


def crash_loop_finding() -> Finding:
    return Finding(
        agent_name="kubernetes",
        summary="Pod is in CrashLoopBackOff with 7 restarts",
        evidence=[
            AgentEvidence(
                source_tool="get_pod_status",
                observation="CrashLoopBackOff, 7 restarts",
            ),
            AgentEvidence(
                source_tool="get_pod_events",
                observation="Back-off restarting failed container",
            ),
        ],
        evidence_quality="corroborated",
    )


class FakeInvestigator:
    def __init__(self, finding: Finding | None = None, error: Exception | None = None):
        self.finding = finding
        self.error = error
        self.calls = 0

    def __call__(self, incident: IncidentContext) -> Finding:
        self.calls += 1
        if self.error:
            raise self.error
        return self.finding


def open_saver(path: str):
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection, serde=platform_serializer()), connection


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "checkpoints.sqlite")


def run(db_path, investigators, state):
    """Invoke on a fresh connection; return (result, snapshot)."""
    saver, connection = open_saver(db_path)
    try:
        graph = build_supervisor_graph(investigators, saver)
        result = graph.invoke(state, CONFIG)
        return result, graph.get_state(CONFIG)
    finally:
        connection.close()


def resume(db_path, investigators, decision):
    """Resume on a new connection and graph object (simulated restart)."""
    saver, connection = open_saver(db_path)
    try:
        graph = build_supervisor_graph(investigators, saver)
        return graph.invoke(Command(resume=decision), CONFIG)
    finally:
        connection.close()


def events(result) -> list[str]:
    return [entry.event for entry in result["audit_log"]]


def test_approved_incident_runs_end_to_end(db_path):
    investigator = FakeInvestigator(crash_loop_finding())
    investigators = {"kubernetes": investigator}

    _, snapshot = run(db_path, investigators, new_platform_state(make_incident()))
    assert snapshot.next == ("approval",)
    action = snapshot.values["proposed_action"]
    assert action.action_type == "rollout_restart"
    assert action.target == "deployment/inventory-api"
    assert action.risk_level == "medium"
    assert action.supporting_finding_ids == [
        snapshot.values["findings"][0].finding_id
    ]

    final = resume(db_path, investigators, {"approved": True, "decided_by": "samklin"})

    assert investigator.calls == 1  # the specialist must not re-run on resume
    assert final["approval"].status == "approved"
    assert final["execution"].simulated is True
    assert final["report"].outcome == "action_executed_simulated"
    assert final["status"] == "completed"
    assert final["agent_call_counts"] == {"kubernetes": 1}
    assert final["supervisor_hops"] == 4
    assert events(final) == [
        "alert_received",
        "routed",
        "kubernetes_investigated",
        "action_proposed",
        "approval_granted",
        "execution_simulated",
        "report_generated",
    ]


def test_rejected_incident_is_not_executed(db_path):
    investigators = {"kubernetes": FakeInvestigator(crash_loop_finding())}
    run(db_path, investigators, new_platform_state(make_incident()))

    final = resume(
        db_path,
        investigators,
        {"approved": False, "decided_by": "samklin", "reason": "Wait for deploy"},
    )

    assert final["approval"].status == "rejected"
    assert final["approval"].reason == "Wait for deploy"
    assert final["execution"] is None
    assert final["report"].outcome == "action_rejected"
    assert final["status"] == "completed"
    assert "execution_simulated" not in events(final)


def test_malformed_approval_payload_does_not_approve(db_path):
    investigators = {"kubernetes": FakeInvestigator(crash_loop_finding())}
    run(db_path, investigators, new_platform_state(make_incident()))

    with pytest.raises(ValueError):
        resume(db_path, investigators, {"approved": "yes", "decided_by": "sam"})


def test_out_of_scope_incident_is_escalated_without_a_pause(db_path):
    investigator = FakeInvestigator(crash_loop_finding())
    incident = make_incident(
        description="Failed deployment pipeline on the main branch",
        namespace=None,
        pod_name=None,
        deployment=None,
    )

    result, snapshot = run(
        db_path, {"kubernetes": investigator}, new_platform_state(incident)
    )

    assert snapshot.next == ()
    assert result["routing"] is None
    assert investigator.calls == 0
    assert result["report"].outcome == "no_action_proposed"
    assert "escalate" in result["report"].summary
    assert events(result) == [
        "alert_received",
        "no_specialist_matched",
        "report_generated",
    ]


def test_insufficient_evidence_never_reaches_approval(db_path):
    weak = Finding(agent_name="kubernetes", summary="Pod status unavailable")
    investigator = FakeInvestigator(weak)

    result, snapshot = run(
        db_path, {"kubernetes": investigator}, new_platform_state(make_incident())
    )

    assert snapshot.next == ()
    assert result["proposed_action"] is None
    assert result["approval"] is None
    assert result["report"].outcome == "investigation_failed"
    assert result["status"] == "failed"


def test_specialist_exception_is_contained_and_reported(db_path):
    investigator = FakeInvestigator(error=RuntimeError("API down"))

    result, snapshot = run(
        db_path, {"kubernetes": investigator}, new_platform_state(make_incident())
    )

    assert snapshot.next == ()
    assert result["findings"][0].status == "failed"
    assert result["findings"][0].error_message == "API down"
    assert "kubernetes_failed" in events(result)
    assert result["agent_call_counts"] == {"kubernetes": 1}
    assert result["report"].outcome == "investigation_failed"


def test_unregistered_specialist_produces_a_blocked_finding(db_path):
    incident = make_incident(
        description="Suspicious unauthorized access to the admin console",
        namespace=None,
        pod_name=None,
        deployment=None,
    )

    result, _ = run(db_path, {}, new_platform_state(incident))

    assert result["routing"].selected_agents == ["security"]
    assert result["findings"][0].status == "blocked"
    assert "specialist_unavailable" in events(result)
    assert result["report"].outcome == "investigation_failed"


def test_exhausted_call_budget_blocks_the_specialist(db_path):
    investigator = FakeInvestigator(crash_loop_finding())
    state = new_platform_state(make_incident(), CallLimits(max_calls_per_agent=1))
    state["agent_call_counts"] = {"kubernetes": 1}

    result, _ = run(db_path, {"kubernetes": investigator}, state)

    assert investigator.calls == 0
    assert result["findings"][0].status == "blocked"
    assert "call_budget_exhausted" in events(result)
    assert result["report"].outcome == "investigation_failed"


def test_multi_specialist_incident_continues_when_one_is_unavailable(db_path):
    investigator = FakeInvestigator(crash_loop_finding())
    incident = make_incident(
        description="Unauthorized kubectl exec into a CrashLoopBackOff pod"
    )

    _, snapshot = run(
        db_path, {"kubernetes": investigator}, new_platform_state(incident)
    )

    findings = snapshot.values["findings"]
    assert {f.agent_name for f in findings} == {"kubernetes", "security"}
    assert {f.agent_name: f.status for f in findings}["security"] == "blocked"
    assert snapshot.next == ("approval",)
    assert snapshot.values["proposed_action"] is not None