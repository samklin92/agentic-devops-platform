import sqlite3
from typing import TypedDict, get_type_hints

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt

from app.contracts.agent_messages import AgentEvidence
from app.state.graph_state import PlatformState, new_platform_state
from app.state.models import CallLimits, Correlation, Finding, IncidentContext
from app.state.serde import platform_serializer
from app.supervisor.graph import build_supervisor_graph
from app.supervisor.routing import route_incident

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
    # receive_alert, route, correlate, evaluate and report each count once.
    assert final["supervisor_hops"] == 5
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


# --- correlation ---------------------------------------------------------

CORROBORATED_INCIDENT = (
    "Pod inventory-api-7d9f is in CrashLoopBackOff with elevated error rate"
)


def signalled(agent, *signals, summary=None, status="completed") -> Finding:
    return Finding(
        agent_name=agent,
        summary=summary or f"{agent} finding",
        status=status,
        evidence=[AgentEvidence(source_tool="some_tool", observation="ok")],
        evidence_quality="limited",
        signals=list(signals),
    )


def kubernetes_crash() -> Finding:
    return Finding(
        agent_name="kubernetes",
        summary="Pod is in CrashLoopBackOff",
        evidence=[
            AgentEvidence(
                source_tool="get_pod_status", observation="CrashLoopBackOff"
            )
        ],
        evidence_quality="limited",
        signals=["crash_loop"],
    )


def crash_loop_proposal(finding: Finding):
    from app.supervisor.proposals import propose_action

    return propose_action(make_incident(), [finding])


def pause_payload(snapshot) -> dict:
    return snapshot.tasks[0].interrupts[0].value


def test_corroboration_is_recorded_without_changing_the_proposal(db_path, tmp_path):
    kubernetes = kubernetes_crash()
    observability = signalled("observability", "errors_elevated")
    incident = make_incident(description=CORROBORATED_INCIDENT)

    _, snapshot = run(
        db_path,
        {
            "kubernetes": FakeInvestigator(kubernetes),
            "observability": FakeInvestigator(observability),
        },
        new_platform_state(incident),
    )

    (correlation,) = snapshot.values["correlations"]
    assert correlation.kind == "corroborates"
    assert correlation.finding_ids == [
        kubernetes.finding_id,
        observability.finding_id,
    ]
    action = snapshot.values["proposed_action"]
    assert action.supporting_finding_ids == [kubernetes.finding_id]
    assert "Evidence relationships" in action.rationale
    payload = pause_payload(snapshot)
    assert payload["correlations"] == [
        {
            "kind": "corroborates",
            "rule": "shared_incident_impact",
            "finding_ids": correlation.finding_ids,
            "explanation": correlation.explanation,
        }
    ]
    assert payload["rationale"] == action.rationale

    # Control: the same Kubernetes finding with no Observability finding.
    control_db = str(tmp_path / "control.sqlite")
    _, control = run(
        control_db,
        {"kubernetes": FakeInvestigator(kubernetes)},
        new_platform_state(incident),
    )
    assert control.values["correlations"] == []
    baseline = control.values["proposed_action"]
    for field in (
        "action_type",
        "namespace",
        "target",
        "risk_level",
        "supporting_finding_ids",
    ):
        assert getattr(action, field) == getattr(baseline, field)
    assert "Evidence relationships" not in baseline.rationale

    final = resume(
        db_path,
        {
            "kubernetes": FakeInvestigator(kubernetes),
            "observability": FakeInvestigator(observability),
        },
        {"approved": True, "decided_by": "samklin"},
    )
    assert final["report"].outcome == "action_executed_simulated"
    assert (
        "Evidence relationships: corroborates "
        "(kubernetes + observability, shared_incident_impact)."
    ) in final["report"].summary
    assert final["supervisor_hops"] == 5
    assert final["agent_call_counts"] == {"kubernetes": 1, "observability": 1}
    names = events(final)
    assert names.index("observability_investigated") < names.index(
        "correlation_recorded"
    ) < names.index("action_proposed")


def test_a_contradiction_is_shown_but_does_not_block_the_proposal(db_path):
    kubernetes = kubernetes_crash()
    clean = signalled("observability", summary="Metrics within normal thresholds.")
    incident = make_incident(description=CORROBORATED_INCIDENT)

    _, snapshot = run(
        db_path,
        {
            "kubernetes": FakeInvestigator(kubernetes),
            "observability": FakeInvestigator(clean),
        },
        new_platform_state(incident),
    )

    (correlation,) = snapshot.values["correlations"]
    assert correlation.kind == "contradicts"
    assert snapshot.next == ("approval",)
    action = snapshot.values["proposed_action"]
    assert action.supporting_finding_ids == [kubernetes.finding_id]
    assert "contradicts (kubernetes + observability" in action.rationale
    assert pause_payload(snapshot)["correlations"][0]["kind"] == "contradicts"


def test_co_occurrence_alone_never_proposes_or_pauses(db_path):
    incident = make_incident(
        description="Failed logins and CPU saturation on checkout-api",
        namespace="production",
        pod_name=None,
        deployment=None,
        service="checkout-api",
    )
    result, snapshot = run(
        db_path,
        {
            "observability": FakeInvestigator(
                signalled("observability", "cpu_saturated")
            ),
            "security": FakeInvestigator(
                signalled("security", "failed_auth_elevated")
            ),
        },
        new_platform_state(incident),
    )

    assert snapshot.next == ()
    (correlation,) = result["correlations"]
    assert correlation.kind == "co_occurs"
    assert result["proposed_action"] is None
    assert result["approval"] is None
    assert result["execution"] is None
    assert result["report"].outcome == "no_action_proposed"
    assert (
        "co occurs (observability + security, load_with_failed_auth)"
        in result["report"].summary
    )
    assert "correlation_recorded" in events(result)


def test_stale_correlations_are_replaced_not_accumulated(db_path):
    stale = Correlation(
        kind="corroborates",
        rule="shared_incident_impact",
        finding_ids=["stale-1", "stale-2"],
        explanation="Left over from an earlier investigation.",
    )
    state = new_platform_state(make_incident())
    state["correlations"] = [stale]
    healthy = Finding(
        agent_name="kubernetes",
        summary="Pod is Running",
        evidence=[
            AgentEvidence(source_tool="get_pod_status", observation="Running")
        ],
        evidence_quality="limited",
    )

    result, _ = run(
        db_path, {"kubernetes": FakeInvestigator(healthy)}, state
    )

    assert result["correlations"] == []
    assert "correlation_recorded" not in events(result)


def test_no_correlation_step_when_no_specialist_applies(db_path):
    incident = make_incident(
        description="Failed deployment pipeline on the main branch",
        namespace=None,
        pod_name=None,
        deployment=None,
    )
    result, _ = run(db_path, {}, new_platform_state(incident))
    assert result["correlations"] == []
    assert result["supervisor_hops"] == 3  # receive_alert, route, report


def test_state_without_the_correlations_key_still_runs(db_path):
    state = new_platform_state(make_incident())
    del state["correlations"]
    investigator = FakeInvestigator(crash_loop_finding())

    _, snapshot = run(db_path, {"kubernetes": investigator}, state)

    assert snapshot.next == ("approval",)
    assert snapshot.values["correlations"] == []


def test_resume_from_a_checkpoint_written_before_correlations_existed(db_path):
    # A checkpoint paused at approval, written by a version of the state
    # that had no `correlations` field. The current graph must resume it.
    hints = get_type_hints(PlatformState, include_extras=True)
    legacy_hints = {k: v for k, v in hints.items() if k != "correlations"}
    LegacyState = TypedDict("LegacyState", legacy_hints)

    finding = crash_loop_finding()
    action = crash_loop_proposal(finding)

    def legacy_approval(state):
        interrupt({"action_id": state["proposed_action"].action_id})
        return {}

    legacy = StateGraph(LegacyState)
    legacy.add_node("approval", legacy_approval)
    legacy.add_edge(START, "approval")

    state = new_platform_state(make_incident())
    del state["correlations"]
    state["routing"] = route_incident(make_incident())
    state["findings"] = [finding]
    state["proposed_action"] = action
    state["status"] = "awaiting_approval"

    saver, connection = open_saver(db_path)
    try:
        legacy.compile(checkpointer=saver).invoke(state, CONFIG)
    finally:
        connection.close()

    final = resume(
        db_path,
        {"kubernetes": FakeInvestigator(finding)},
        {"approved": True, "decided_by": "samklin"},
    )

    assert final["approval"].status == "approved"
    assert final["report"].outcome == "action_executed_simulated"
    assert "Evidence relationships" not in final["report"].summary
