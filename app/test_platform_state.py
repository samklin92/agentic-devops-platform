from langgraph.graph import END, START, StateGraph

from app.contracts.agent_messages import (
    AgentEvidence,
    KubernetesInvestigationResponse,
)
from app.state.adapters import (
    MAX_SUMMARY_CHARS,
    finding_from_kubernetes_response,
)
from app.state.graph_state import (
    PlatformState,
    check_call_budget,
    merge_counts,
    new_platform_state,
    summarize_usage,
    supervisor_hops_exhausted,
)
from app.state.models import CallLimits, Finding, IncidentContext, UsageRecord

from app.state.adapters import (
    MAX_SUMMARY_CHARS,
    finding_from_kubernetes_response,
    is_execution_failure,
)

def make_incident() -> IncidentContext:
    return IncidentContext(
        incident_id="inc-001",
        description="inventory-api pod in CrashLoopBackOff",
        environment="production",
    )


def make_finding(agent_name: str) -> Finding:
    return Finding(agent_name=agent_name, summary=f"{agent_name} finding")


def test_merge_counts_sums_per_key_and_handles_none():
    assert merge_counts({"a": 1}, {"a": 2, "b": 1}) == {"a": 3, "b": 1}
    assert merge_counts(None, {"a": 1}) == {"a": 1}
    assert merge_counts({"a": 1}, None) == {"a": 1}


def test_new_state_starts_empty_with_default_limits():
    state = new_platform_state(make_incident())
    assert state["status"] == "received"
    assert state["findings"] == []
    assert state["agent_call_counts"] == {}
    assert state["limits"] == CallLimits()


def test_call_budget_enforces_agent_and_incident_limits():
    state = new_platform_state(
        make_incident(),
        CallLimits(max_calls_per_agent=2, max_calls_per_incident=3),
    )
    assert check_call_budget(state, "kubernetes") is None

    state["agent_call_counts"] = {"kubernetes": 2}
    assert "kubernetes" in check_call_budget(state, "kubernetes")

    state["agent_call_counts"] = {
        "kubernetes": 1,
        "observability": 1,
        "security": 1,
    }
    assert "Incident call limit" in check_call_budget(state, "kubernetes")


def test_supervisor_hop_limit():
    state = new_platform_state(make_incident(), CallLimits(max_supervisor_hops=2))
    assert not supervisor_hops_exhausted(state)
    state["supervisor_hops"] = 2
    assert supervisor_hops_exhausted(state)


def test_summarize_usage_aggregates_per_agent():
    records = [
        UsageRecord(
            incident_id="i",
            agent_name="kubernetes",
            model="test-model",
            input_tokens=900,
            output_tokens=250,
            estimated_cost_usd=0.01,
        ),
        UsageRecord(
            incident_id="i",
            agent_name="kubernetes",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
        ),
    ]
    summary = summarize_usage(records)
    assert summary["kubernetes"]["calls"] == 2
    assert summary["kubernetes"]["total_tokens"] == 1300
    assert summary["kubernetes"]["estimated_cost_usd"] == 0.01


def test_parallel_specialists_do_not_overwrite_each_other():
    def make_node(name: str):
        def node(state: PlatformState) -> dict:
            return {
                "findings": [make_finding(name)],
                "agent_call_counts": {name: 1},
            }

        return node

    builder = StateGraph(PlatformState)
    builder.add_node("kubernetes", make_node("kubernetes"))
    builder.add_node("observability", make_node("observability"))
    builder.add_edge(START, "kubernetes")
    builder.add_edge(START, "observability")
    builder.add_edge("kubernetes", END)
    builder.add_edge("observability", END)
    graph = builder.compile()

    result = graph.invoke(new_platform_state(make_incident()))

    assert {f.agent_name for f in result["findings"]} == {
        "kubernetes",
        "observability",
    }
    assert result["agent_call_counts"] == {"kubernetes": 1, "observability": 1}


def test_adapter_maps_kubernetes_response_to_finding():
    response = KubernetesInvestigationResponse(
        incident_id="inc-001",
        status="completed",
        findings=["get_pod_status: CrashLoopBackOff"],
        evidence=[
            AgentEvidence(
                source_tool="get_pod_status",
                observation="CrashLoopBackOff",
            )
        ],
        evidence_quality="limited",
    )
    finding = finding_from_kubernetes_response(response)
    assert finding.agent_name == "kubernetes"
    assert finding.status == "completed"
    assert finding.evidence_quality == "limited"
    assert len(finding.evidence) == 1


def test_adapter_downgrades_quality_when_there_is_no_evidence():
    response = KubernetesInvestigationResponse(
        incident_id="inc-001",
        status="failed",
        evidence_quality="limited",
        error_message="Tool call limit reached.",
    )
    finding = finding_from_kubernetes_response(response)
    assert finding.evidence_quality == "insufficient"
    assert finding.summary == "Tool call limit reached."
    assert finding.error_message == "Tool call limit reached."


def test_adapter_truncates_long_summaries():
    response = KubernetesInvestigationResponse(
        incident_id="inc-001",
        status="partial",
        findings=["x" * 1000],
    )
    finding = finding_from_kubernetes_response(response)
    assert len(finding.summary) == MAX_SUMMARY_CHARS
    assert finding.summary.endswith("...")

def failure_evidence(tool: str = "inspect_pod_events") -> AgentEvidence:
    return AgentEvidence(
        source_tool=tool,
        observation='{"error": "MCP server unreachable", "execution_failed": true}',
    )


def test_adapter_does_not_count_failed_tool_runs_as_evidence():
    response = KubernetesInvestigationResponse(
        incident_id="inc-001",
        status="completed",
        findings=["inspect_pod_events: failed"],
        evidence=[failure_evidence()],
        evidence_quality="limited",
    )
    finding = finding_from_kubernetes_response(response)
    assert finding.evidence_quality == "insufficient"
    assert len(finding.evidence) == 1  # kept for the audit trail


def test_adapter_keeps_quality_when_real_evidence_sits_beside_a_failure():
    response = KubernetesInvestigationResponse(
        incident_id="inc-001",
        status="partial",
        findings=["mixed results"],
        evidence=[
            AgentEvidence(
                source_tool="get_pod_status", observation="CrashLoopBackOff"
            ),
            failure_evidence("gather_additional_diagnostics"),
        ],
        evidence_quality="limited",
    )
    assert finding_from_kubernetes_response(response).evidence_quality == "limited"


def test_is_execution_failure_only_matches_failure_records():
    assert is_execution_failure('{"execution_failed": true, "error": "x"}')
    assert not is_execution_failure('{"execution_failed": false}')
    assert not is_execution_failure("CrashLoopBackOff")
    assert not is_execution_failure("[1, 2]")