import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.state.graph_state import new_platform_state
from app.state.models import IncidentContext
from app.state.serde import platform_serializer
from app.supervisor.graph import build_supervisor_graph
from app.supervisor.kubernetes_bridge import (
    KUBERNETES_TOOL_ALLOWLIST,
    build_kubernetes_investigator,
    static_tools_provider,
    validate_tool_allowlist,
)

CONFIG = {"configurable": {"thread_id": "inc-001"}}


def make_incident(**overrides) -> IncidentContext:
    data = {
        "incident_id": "inc-001",
        "description": (
            "Investigate why inventory-api is crashing: "
            "pod inventory-api-7d9f is in CrashLoopBackOff"
        ),
        "severity": "high",
        "environment": "production",
        "namespace": "production",
        "pod_name": "inventory-api-7d9f",
        "deployment": "inventory-api",
    }
    data.update(overrides)
    return IncidentContext(**data)


def make_crash_loop_tools(calls: list[str]) -> dict:
    async def inspect_pod_events():
        calls.append("inspect_pod_events")
        return {
            "reason": "CrashLoopBackOff",
            "message": "Container repeatedly restarting",
            "terminal": False,
        }

    async def gather_additional_diagnostics():
        calls.append("gather_additional_diagnostics")
        return {
            "pod": {"status": "CrashLoopBackOff"},
            "logs": {"logs": ["ERROR: Database connection refused"]},
            "deployment": {
                "desired_replicas": 3,
                "ready_replicas": 1,
                "status": "Degraded",
            },
            "terminal": True,
        }

    return {
        "inspect_pod_events": inspect_pod_events,
        "gather_additional_diagnostics": gather_additional_diagnostics,
    }


def make_failing_tools() -> dict:
    async def inspect_pod_events():
        raise RuntimeError("API unreachable")

    async def gather_additional_diagnostics():
        raise RuntimeError("API unreachable")

    return {
        "inspect_pod_events": inspect_pod_events,
        "gather_additional_diagnostics": gather_additional_diagnostics,
    }


def open_saver(path: str):
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection, serde=platform_serializer()), connection


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "checkpoints.sqlite")


def run(db_path, investigators, state):
    saver, connection = open_saver(db_path)
    try:
        graph = build_supervisor_graph(investigators, saver)
        result = graph.invoke(state, CONFIG)
        return result, graph.get_state(CONFIG)
    finally:
        connection.close()


def resume(db_path, investigators, decision):
    saver, connection = open_saver(db_path)
    try:
        graph = build_supervisor_graph(investigators, saver)
        return graph.invoke(Command(resume=decision), CONFIG)
    finally:
        connection.close()


def kubernetes_investigators(tools: dict) -> dict:
    return {
        "kubernetes": build_kubernetes_investigator(static_tools_provider(tools))
    }

def describe(snapshot, calls: list[str]) -> str:
    """Summarise what the real agent produced, for assertion messages."""
    values = snapshot.values
    findings = [
        (f.status, f.evidence_quality, f.needs_follow_up, f.error_message)
        for f in values["findings"]
    ]
    report = values["report"].outcome if values["report"] else None
    return (
        f"calls={calls} findings={findings} report={report} "
        f"audit={[e.event for e in values['audit_log']]}"
    )

def test_real_kubernetes_agent_drives_an_approved_incident(db_path):
    calls: list[str] = []
    investigators = kubernetes_investigators(make_crash_loop_tools(calls))

    _, snapshot = run(db_path, investigators, new_platform_state(make_incident()))

    assert snapshot.next == ("approval",), describe(snapshot, calls)
    finding = snapshot.values["findings"][0]
    assert finding.agent_name == "kubernetes"
    assert finding.status == "completed"
    assert finding.evidence_quality == "limited"
    assert len(finding.evidence) == 2
    assert calls == ["inspect_pod_events", "gather_additional_diagnostics"]

    action = snapshot.values["proposed_action"]
    assert action.action_type == "rollout_restart"
    assert action.target == "deployment/inventory-api"
    assert action.supporting_finding_ids == [finding.finding_id]

    final = resume(
        db_path, investigators, {"approved": True, "decided_by": "samklin"}
    )

    # The agent must not run again when the graph resumes.
    assert calls == ["inspect_pod_events", "gather_additional_diagnostics"]
    assert final["execution"].simulated is True
    assert final["report"].outcome == "action_executed_simulated"
    assert final["agent_call_counts"] == {"kubernetes": 1}


def test_failing_tools_never_reach_approval(db_path):
    investigators = kubernetes_investigators(make_failing_tools())

    result, snapshot = run(
        db_path, investigators, new_platform_state(make_incident())
    )

    assert snapshot.next == ()
    assert result["proposed_action"] is None
    assert result["approval"] is None
    assert result["report"].outcome == "investigation_failed"
    assert result["status"] == "failed"


def test_disallowed_tool_is_rejected_before_the_agent_runs(db_path):
    calls: list[str] = []
    tools = make_crash_loop_tools(calls)

    async def restart_deployment():
        calls.append("restart_deployment")
        return {}

    tools["restart_deployment"] = restart_deployment
    investigators = kubernetes_investigators(tools)

    result, snapshot = run(
        db_path, investigators, new_platform_state(make_incident())
    )

    assert calls == []
    assert snapshot.next == ()
    finding = result["findings"][0]
    assert finding.status == "failed"
    assert "restart_deployment" in finding.error_message
    assert result["proposed_action"] is None
    assert result["report"].outcome == "investigation_failed"
    assert "kubernetes_failed" in [e.event for e in result["audit_log"]]


def test_validate_tool_allowlist_accepts_known_and_rejects_unknown_tools():
    validate_tool_allowlist({"inspect_pod_events": None})
    with pytest.raises(ValueError, match="delete_pod"):
        validate_tool_allowlist({"inspect_pod_events": None, "delete_pod": None})


def test_allowlist_contains_only_the_known_read_only_tools():
    assert KUBERNETES_TOOL_ALLOWLIST == {
        "inspect_pod_events",
        "gather_additional_diagnostics",
    }