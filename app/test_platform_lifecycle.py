import logging
import sqlite3
import warnings

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.contracts.agent_messages import (
    AgentEvidence,
    KubernetesInvestigationResponse,
)
from app.state.adapters import finding_from_kubernetes_response
from app.state.graph_state import (
    PlatformState,
    check_call_budget,
    new_platform_state,
    summarize_usage,
)
from app.state.serde import platform_serializer
from app.state.models import (
    ApprovalRecord,
    AuditEntry,
    ExecutionRecord,
    IncidentContext,
    IncidentReport,
    ProposedAction,
    RoutingDecision,
    UsageRecord,
)

CONFIG = {"configurable": {"thread_id": "inc-001"}}


def make_incident() -> IncidentContext:
    return IncidentContext(
        incident_id="inc-001",
        description="inventory-api pod in CrashLoopBackOff",
        severity="high",
        environment="production",
        namespace="production",
        pod_name="inventory-api-7d9f",
        source="cloudwatch-alarm",
    )


def audit(state: PlatformState, actor: str, event: str, detail: str = "") -> AuditEntry:
    return AuditEntry(
        incident_id=state["incident"].incident_id,
        actor=actor,
        event=event,
        detail=detail,
    )


def build_lifecycle_graph(checkpointer):
    """Stub graph exercising every state field. Real agents come later."""

    def receive_alert(state: PlatformState) -> dict:
        return {
            "status": "received",
            "supervisor_hops": 1,
            "audit_log": [
                audit(state, "supervisor", "alert_received",
                      state["incident"].description)
            ],
        }

    def route(state: PlatformState) -> dict:
        routing = RoutingDecision(
            selected_agents=["kubernetes"],
            reason="CrashLoopBackOff is a Kubernetes workload failure.",
        )
        return {
            "routing": routing,
            "status": "investigating",
            "supervisor_hops": 1,
            "audit_log": [audit(state, "supervisor", "routed", routing.reason)],
        }

    def kubernetes_agent(state: PlatformState) -> dict:
        assert check_call_budget(state, "kubernetes") is None
        incident = state["incident"]
        response = KubernetesInvestigationResponse(
            incident_id=incident.incident_id,
            status="completed",
            findings=[
                "get_pod_status: CrashLoopBackOff, 7 restarts",
                "get_pod_events: Back-off restarting failed container",
            ],
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
        return {
            "findings": [finding_from_kubernetes_response(response)],
            "usage": [
                UsageRecord(
                    incident_id=incident.incident_id,
                    agent_name="kubernetes",
                    model="test-model",
                    input_tokens=900,
                    output_tokens=250,
                )
            ],
            "agent_call_counts": {"kubernetes": 1},
            "audit_log": [audit(state, "kubernetes", "kubernetes_investigated")],
        }

    def evaluate_and_propose(state: PlatformState) -> dict:
        usable = [
            f for f in state["findings"]
            if f.evidence_quality == "corroborated" and not f.needs_follow_up
        ]
        assert usable, "stub scenario expects corroborated evidence"
        incident = state["incident"]
        action = ProposedAction(
            action_type="rollout_restart",
            namespace=incident.namespace,
            target="deployment/inventory-api",
            rationale="Container is crash-looping with corroborated evidence.",
            risk_level="medium",
            supporting_finding_ids=[f.finding_id for f in usable],
        )
        return {
            "proposed_action": action,
            "status": "awaiting_approval",
            "supervisor_hops": 1,
            "audit_log": [audit(state, "supervisor", "action_proposed",
                                action.rationale)],
        }

    def approval(state: PlatformState) -> dict:
        action = state["proposed_action"]
        # On resume this node re-runs from the top, so everything before
        # interrupt() must be free of side effects.
        decision = interrupt({
            "action_id": action.action_id,
            "action_type": action.action_type,
            "target": action.target,
            "risk_level": action.risk_level,
        })
        approved = bool(decision["approved"])
        record = ApprovalRecord(
            action_id=action.action_id,
            status="approved" if approved else "rejected",
            decided_by=decision["decided_by"],
            reason=decision.get("reason"),
        )
        event = "approval_granted" if approved else "approval_rejected"
        return {
            "approval": record,
            "audit_log": [audit(state, decision["decided_by"], event)],
        }

    def after_approval(state: PlatformState) -> str:
        return "execute" if state["approval"].status == "approved" else "report"

    def execute(state: PlatformState) -> dict:
        action = state["proposed_action"]
        record = ExecutionRecord(
            action_id=action.action_id,
            status="simulated",
            message=f"Simulated {action.action_type} on {action.target}.",
        )
        return {
            "execution": record,
            "status": "executing",
            "audit_log": [audit(state, "supervisor", "execution_simulated",
                                record.message)],
        }

    def report(state: PlatformState) -> dict:
        action = state["proposed_action"]
        approval_record = state["approval"]
        if state["execution"] is not None:
            outcome = "action_executed_simulated"
        elif approval_record and approval_record.status == "rejected":
            outcome = "action_rejected"
        else:
            outcome = "no_action_proposed"
        final = IncidentReport(
            incident_id=state["incident"].incident_id,
            outcome=outcome,
            summary=f"Outcome: {outcome}.",
            root_cause="Container crash loop on inventory-api.",
            finding_ids=[f.finding_id for f in state["findings"]],
            action_id=action.action_id if action else None,
            approval_status=approval_record.status if approval_record else None,
        )
        return {
            "report": final,
            "status": "completed",
            "supervisor_hops": 1,
            "audit_log": [audit(state, "supervisor", "report_generated")],
        }

    builder = StateGraph(PlatformState)
    for name, node in [
        ("receive_alert", receive_alert),
        ("route", route),
        ("kubernetes", kubernetes_agent),
        ("evaluate", evaluate_and_propose),
        ("approval", approval),
        ("execute", execute),
        ("report", report),
    ]:
        builder.add_node(name, node)

    builder.add_edge(START, "receive_alert")
    builder.add_edge("receive_alert", "route")
    builder.add_edge("route", "kubernetes")
    builder.add_edge("kubernetes", "evaluate")
    builder.add_edge("evaluate", "approval")
    builder.add_conditional_edges(
        "approval", after_approval, {"execute": "execute", "report": "report"}
    )
    builder.add_edge("execute", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)


def open_saver(path: str):
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection, serde=platform_serializer()), connection

@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "checkpoints.sqlite")


def pause_at_approval(db_path: str):
    """Run until the approval interrupt, then close the connection."""
    saver, connection = open_saver(db_path)
    graph = build_lifecycle_graph(saver)
    graph.invoke(new_platform_state(make_incident()), CONFIG)
    snapshot = graph.get_state(CONFIG)
    connection.close()
    return snapshot


def resume_in_fresh_process(db_path: str, decision: dict):
    """New connection, new graph object: simulates a process restart."""
    saver, connection = open_saver(db_path)
    graph = build_lifecycle_graph(saver)
    restored = graph.get_state(CONFIG)
    final = graph.invoke(Command(resume=decision), CONFIG)
    connection.close()
    return restored, final


def test_incident_lifecycle_approved(db_path, caplog):
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        paused = pause_at_approval(db_path)
        assert paused.next == ("approval",)
        assert paused.values["status"] == "awaiting_approval"
        assert paused.values["proposed_action"].action_type == "rollout_restart"
        assert paused.values["approval"] is None
        assert paused.values["execution"] is None

        restored, final = resume_in_fresh_process(
            db_path, {"approved": True, "decided_by": "samklin"}
        )

    # Checkpoint round-trip: state reloaded from SQLite equals state at pause.
    assert restored.values == paused.values

    messages = [str(w.message) for w in caught] + [
        r.getMessage() for r in caplog.records
    ]
    bad = [m for m in messages
           if "msgpack" in m.lower() or "deserializ" in m.lower()]
    assert not bad, bad

    assert final["approval"].status == "approved"
    assert final["approval"].decided_by == "samklin"
    assert final["execution"].simulated is True
    assert final["report"].outcome == "action_executed_simulated"
    assert final["report"].action_id == final["proposed_action"].action_id
    assert final["status"] == "completed"

    assert [e.event for e in final["audit_log"]] == [
        "alert_received",
        "routed",
        "kubernetes_investigated",
        "action_proposed",
        "approval_granted",
        "execution_simulated",
        "report_generated",
    ]
    assert final["agent_call_counts"] == {"kubernetes": 1}
    assert final["supervisor_hops"] == 4
    assert summarize_usage(final["usage"])["kubernetes"]["total_tokens"] == 1150


def test_incident_lifecycle_rejected(db_path):
    paused = pause_at_approval(db_path)
    assert paused.next == ("approval",)

    _, final = resume_in_fresh_process(
        db_path,
        {"approved": False, "decided_by": "samklin", "reason": "Wait for deploy"},
    )

    assert final["approval"].status == "rejected"
    assert final["approval"].reason == "Wait for deploy"
    assert final["execution"] is None
    assert final["report"].outcome == "action_rejected"
    events = [e.event for e in final["audit_log"]]
    assert "approval_rejected" in events
    assert "execution_simulated" not in events