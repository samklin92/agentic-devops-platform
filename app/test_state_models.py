import pytest
from pydantic import ValidationError

from app.contracts.agent_messages import AgentEvidence
from app.state.models import (
    ApprovalRecord,
    AuditEntry,
    CallLimits,
    ExecutionRecord,
    Finding,
    IncidentContext,
    IncidentReport,
    ProposedAction,
    RoutingDecision,
    UsageRecord,
)
from app.tool_executor import MAX_TOOL_CALLS


def make_evidence() -> AgentEvidence:
    return AgentEvidence(
        source_tool="get_pod_events",
        observation="Back-off restarting failed container",
    )


def make_finding(**overrides) -> Finding:
    data = {
        "agent_name": "kubernetes",
        "summary": "Pod is in CrashLoopBackOff",
        "evidence": [make_evidence()],
        "evidence_quality": "limited",
    }
    data.update(overrides)
    return Finding(**data)


def make_action(**overrides) -> ProposedAction:
    data = {
        "action_type": "rollout_restart",
        "namespace": "production",
        "target": "deployment/inventory-api",
        "rationale": "Container repeatedly crashing after config change",
        "risk_level": "medium",
        "supporting_finding_ids": ["finding-1"],
    }
    data.update(overrides)
    return ProposedAction(**data)


def make_incident() -> IncidentContext:
    return IncidentContext(
        incident_id="inc-001",
        description="inventory-api pod in CrashLoopBackOff",
        severity="high",
        environment="production",
        namespace="production",
        pod_name="inventory-api-7d9f",
    )


def test_incident_context_valid_and_defaults():
    incident = make_incident()
    assert incident.source == "manual"
    assert incident.received_at.tzinfo is not None


def test_incident_context_rejects_empty_description_and_extra_fields():
    with pytest.raises(ValidationError):
        IncidentContext(incident_id="i", description="  ", environment="prod")
    with pytest.raises(ValidationError):
        IncidentContext(
            incident_id="i",
            description="x",
            environment="prod",
            surprise="field",
        )

def test_incident_context_service_defaults_to_none():
    assert make_incident().service is None


def test_incident_context_accepts_service():
    incident = IncidentContext(
        incident_id="i",
        description="p95 latency is elevated",
        environment="production",
        namespace="production",
        service="checkout-api",
    )
    assert incident.service == "checkout-api"


def test_incident_context_loads_payload_without_service():
    # Models serialized before the field existed must still load.
    payload = make_incident().model_dump(mode="json")
    payload.pop("service")
    restored = IncidentContext.model_validate(payload)
    assert restored.service is None


def test_incident_context_service_round_trips():
    incident = IncidentContext(
        incident_id="i",
        description="p95 latency is elevated",
        environment="production",
        service="checkout-api",
    )
    restored = IncidentContext.model_validate(
        incident.model_dump(mode="json")
    )
    assert restored.service == "checkout-api"


def test_incident_context_whitespace_service_is_stripped_to_empty():
    incident = IncidentContext(
        incident_id="i",
        description="p95 latency is elevated",
        environment="production",
        service="   ",
    )
    # Empty is falsy, so `service or deployment` falls back to deployment.
    assert incident.service == ""
    assert not incident.service

def test_routing_requires_at_least_one_specialist():
    with pytest.raises(ValidationError):
        RoutingDecision(selected_agents=[], reason="nothing")


def test_routing_cannot_target_supervisor_or_unknown_agents():
    with pytest.raises(ValidationError):
        RoutingDecision(selected_agents=["supervisor"], reason="loop")
    with pytest.raises(ValidationError):
        RoutingDecision(selected_agents=["terraform"], reason="not yet")


def test_finding_quality_claim_requires_evidence():
    with pytest.raises(ValidationError):
        make_finding(evidence=[], evidence_quality="limited")
    assert make_finding(evidence=[], evidence_quality="insufficient")


def test_finding_supports_retry_and_conflict_tracking():
    finding = make_finding(attempt=2, contradicts=["finding-0"])
    assert finding.attempt == 2
    assert finding.contradicts == ["finding-0"]
    with pytest.raises(ValidationError):
        make_finding(attempt=0)


def test_proposed_action_requires_supporting_findings():
    with pytest.raises(ValidationError):
        make_action(supporting_finding_ids=[])


def test_proposed_action_rejects_unknown_type():
    with pytest.raises(ValidationError):
        make_action(action_type="drop_database")


def test_approval_pending_cannot_carry_a_decision():
    with pytest.raises(ValidationError):
        ApprovalRecord(action_id="a", status="pending", decided_by="sam")


def test_approval_decision_requires_approver():
    with pytest.raises(ValidationError):
        ApprovalRecord(action_id="a", status="approved")


def test_approval_decision_sets_decided_at():
    record = ApprovalRecord(
        action_id="a", status="rejected", decided_by="sam", reason="too risky"
    )
    assert record.decided_at is not None


def test_execution_can_only_be_simulated_in_phase_1():
    assert ExecutionRecord(
        action_id="a", status="simulated", message="Restart simulated."
    ).simulated is True
    with pytest.raises(ValidationError):
        ExecutionRecord(
            action_id="a",
            status="simulated",
            simulated=False,
            message="real run",
        )


def test_usage_record_totals_and_rejects_negative_tokens():
    usage = UsageRecord(
        incident_id="inc-001",
        agent_name="kubernetes",
        model="claude-sonnet-4-6",
        input_tokens=1200,
        output_tokens=300,
    )
    assert usage.total_tokens == 1500
    with pytest.raises(ValidationError):
        UsageRecord(
            incident_id="inc-001",
            agent_name="kubernetes",
            model="m",
            input_tokens=-1,
            output_tokens=0,
        )


def test_call_limits_default_matches_existing_tool_executor_limit():
    assert CallLimits().max_calls_per_agent == MAX_TOOL_CALLS


def test_incident_report_rejects_unknown_outcome():
    with pytest.raises(ValidationError):
        IncidentReport(incident_id="i", outcome="fixed", summary="done")


@pytest.mark.parametrize(
    "model",
    [
        make_incident(),
        RoutingDecision(selected_agents=["kubernetes"], reason="CrashLoop"),
        make_finding(),
        make_action(),
        ApprovalRecord(action_id="a"),
        ApprovalRecord(action_id="a", status="approved", decided_by="sam"),
        ExecutionRecord(action_id="a", status="simulated", message="ok"),
        UsageRecord(
            incident_id="i",
            agent_name="supervisor",
            model="m",
            input_tokens=1,
            output_tokens=1,
        ),
        CallLimits(),
        AuditEntry(incident_id="i", actor="supervisor", event="routed"),
        IncidentReport(
            incident_id="i", outcome="no_action_proposed", summary="done"
        ),
    ],
    ids=lambda m: type(m).__name__,
)
def test_json_round_trip_preserves_every_record(model):
    """Checkpointing serializes state, so every record must survive JSON."""
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model