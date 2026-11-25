"""Supervisor graph against the real simulated MCP server over stdio.

These tests start the server as a subprocess, so run pytest from the
repository root.
"""

import sqlite3
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.state.graph_state import new_platform_state
from app.state.models import IncidentContext
from app.state.serde import platform_serializer
from app.supervisor.graph import build_supervisor_graph
from app.supervisor.kubernetes_bridge import build_kubernetes_investigator
from app.supervisor.mcp_tools import (
    mcp_session_tools_provider,
    stdio_session,
    stdio_tools_provider,
)
from app.supervisor.observability import make_observability_investigator

CONFIG = {"configurable": {"thread_id": "inc-001"}}


def make_incident(**overrides) -> IncidentContext:
    data = {
        "incident_id": "inc-001",
        "description": (
            "Investigate why inventory-api is crashing: "
            "pod inventory-api is in CrashLoopBackOff"
        ),
        "severity": "high",
        "environment": "production",
        "namespace": "production",
        "pod_name": "inventory-api",
        "deployment": "inventory-api",
    }
    data.update(overrides)
    return IncidentContext(**data)


def counting(provider, opened: list[str]):
    """Wrap a tools provider to record each investigation it serves."""

    @asynccontextmanager
    async def wrapped(request):
        opened.append(request.target.pod_name)
        async with provider(request) as tools:
            yield tools

    return wrapped


def describe(snapshot) -> str:
    values = snapshot.values
    findings = [
        (f.status, f.evidence_quality, f.needs_follow_up, f.error_message)
        for f in values["findings"]
    ]
    report = values["report"].outcome if values["report"] else None
    return (
        f"findings={findings} report={report} "
        f"audit={[e.event for e in values['audit_log']]}"
    )


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


def stdio_investigators(opened: list[str]) -> dict:
    provider = counting(stdio_tools_provider(), opened)
    return {"kubernetes": build_kubernetes_investigator(provider)}

def all_stdio_investigators(opened: list[str]) -> dict:
    investigators = stdio_investigators(opened)
    investigators["observability"] = make_observability_investigator(
        stdio_session
    )
    return investigators

def make_metrics_incident(**overrides) -> IncidentContext:
    """An incident with no pod, so only metrics specialists can apply."""
    data = {
        "description": "High p95 latency on payment-api",
        "pod_name": None,
        "deployment": None,
        "service": "payment-api",
    }
    data.update(overrides)
    return make_incident(**data)

def test_crash_loop_is_approved_end_to_end_over_stdio(db_path):
    opened: list[str] = []
    investigators = stdio_investigators(opened)

    _, snapshot = run(db_path, investigators, new_platform_state(make_incident()))

    assert snapshot.next == ("approval",), describe(snapshot)
    finding = snapshot.values["findings"][0]
    assert finding.status == "completed"
    assert {e.source_tool for e in finding.evidence} == {
        "inspect_pod_events",
        "gather_additional_diagnostics",
    }
    action = snapshot.values["proposed_action"]
    assert (action.action_type, action.target, action.risk_level) == (
        "rollout_restart",
        "deployment/inventory-api",
        "medium",
    )

    final = resume(
        db_path, investigators, {"approved": True, "decided_by": "samklin"}
    )

    assert opened == ["inventory-api"]  # resume must not start the server again
    assert final["execution"].simulated is True
    assert final["report"].outcome == "action_executed_simulated"


def test_healthy_staging_pod_produces_no_proposal(db_path):
    opened: list[str] = []
    incident = make_incident(
        description="Investigate pod inventory-api in staging",
        environment="staging",
        namespace="staging",
    )

    result, snapshot = run(
        db_path, stdio_investigators(opened), new_platform_state(incident)
    )

    assert snapshot.next == (), describe(snapshot)
    assert result["findings"][0].status == "completed"
    assert result["proposed_action"] is None
    assert result["approval"] is None
    assert result["report"].outcome == "no_action_proposed"


def test_unknown_pod_fails_without_a_proposal(db_path):
    opened: list[str] = []
    incident = make_incident(
        description="Investigate pod unknown-api in production",
        pod_name="unknown-api",
        deployment="unknown-api",
    )

    result, snapshot = run(
        db_path, stdio_investigators(opened), new_platform_state(incident)
    )

    assert snapshot.next == (), describe(snapshot)
    finding = result["findings"][0]
    assert finding.status == "failed"
    assert "not found" in finding.error_message
    assert result["proposed_action"] is None
    assert result["report"].outcome == "investigation_failed"
    assert result["status"] == "failed"


def test_server_that_cannot_start_is_contained(db_path):
    @asynccontextmanager
    async def broken_session():
        raise RuntimeError("cannot start MCP server")
        yield  # makes this an async generator

    investigators = {
        "kubernetes": build_kubernetes_investigator(
            mcp_session_tools_provider(broken_session)
        )
    }

    result, snapshot = run(
        db_path, investigators, new_platform_state(make_incident())
    )

    assert snapshot.next == (), describe(snapshot)
    finding = result["findings"][0]
    assert finding.status == "failed"
    assert "cannot start MCP server" in finding.error_message
    assert "kubernetes_failed" in [e.event for e in result["audit_log"]]
    assert result["proposed_action"] is None
    assert result["report"].outcome == "investigation_failed"

def test_metrics_only_incident_ends_without_a_proposal(db_path):
    opened: list[str] = []

    result, snapshot = run(
        db_path,
        all_stdio_investigators(opened),
        new_platform_state(make_metrics_incident()),
    )

    assert snapshot.next == (), describe(snapshot)
    assert opened == []  # Kubernetes was not routed
    assert [f.agent_name for f in result["findings"]] == ["observability"]
    finding = result["findings"][0]
    assert finding.status == "completed"
    assert finding.summary == "Metrics within normal thresholds."
    assert result["proposed_action"] is None
    assert result["report"].outcome == "no_action_proposed"
    assert result["status"] == "completed"


def test_unknown_service_ends_as_investigation_failed(db_path):
    incident = make_metrics_incident(
        description="Elevated error rate on ghost-api",
        service="ghost-api",
    )

    result, snapshot = run(
        db_path,
        all_stdio_investigators([]),
        new_platform_state(incident),
    )

    assert snapshot.next == (), describe(snapshot)
    finding = result["findings"][0]
    assert finding.agent_name == "observability"
    assert finding.status == "failed"
    assert "ghost-api" in finding.error_message
    assert result["proposed_action"] is None
    assert result["report"].outcome == "investigation_failed"
    assert result["status"] == "failed"


def test_mixed_incident_proposal_is_supported_only_by_kubernetes(db_path):
    opened: list[str] = []
    incident = make_incident(
        description=(
            "inventory-api is in CrashLoopBackOff with elevated error "
            "rate and high p95 latency"
        ),
        service="inventory-api",
    )

    _, snapshot = run(
        db_path,
        all_stdio_investigators(opened),
        new_platform_state(incident),
    )

    assert snapshot.next == ("approval",), describe(snapshot)
    by_agent = {f.agent_name: f for f in snapshot.values["findings"]}
    assert set(by_agent) == {"kubernetes", "observability"}
    assert by_agent["observability"].summary == (
        "Errors elevated, latency elevated, resources normal."
    )
    action = snapshot.values["proposed_action"]
    assert action.action_type == "rollout_restart"
    assert action.supporting_finding_ids == [
        by_agent["kubernetes"].finding_id
    ]
    assert opened == ["inventory-api"]

def test_observability_session_failure_is_contained(db_path):
    @asynccontextmanager
    async def broken_session():
        raise RuntimeError("cannot start MCP server")
        yield  # makes this an async generator

    investigators = {
        "observability": make_observability_investigator(broken_session)
    }

    result, snapshot = run(
        db_path, investigators, new_platform_state(make_metrics_incident())
    )

    assert snapshot.next == (), describe(snapshot)
    finding = result["findings"][0]
    assert finding.status == "failed"
    assert "cannot start MCP server" in finding.error_message
    assert "observability_failed" in [e.event for e in result["audit_log"]]
    assert result["proposed_action"] is None
    assert result["report"].outcome == "investigation_failed"