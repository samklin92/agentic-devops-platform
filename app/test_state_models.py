from typing import get_args

import pytest
from pydantic import ValidationError

from app.contracts.agent_messages import AgentEvidence
from app.state.models import (
    ApprovalRecord,
    AuditEntry,
    CallLimits,
    Correlation,
    CorrelationKind,
    ExecutionRecord,
    Finding,
    IncidentContext,
    IncidentReport,
    ProposedAction,
    RoutingDecision,
    SIGNALS_BY_AGENT,
    SignalName,
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


def make_correlation(**overrides) -> Correlation:
    data = {
        "kind": "corroborates",
        "rule": "unhealthy_workload",
        "finding_ids": ["finding-1", "finding-2"],
        "explanation": "Two specialists report signals about the same workload.",
    }
    data.update(overrides)
    return Correlation(**data)


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
        make_finding(
            agent_name="observability",
            signals=["errors_elevated", "latency_elevated"],
        ),
        make_correlation(),
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


# ---- structured signals ----


def test_finding_signals_default_to_empty():
    assert make_finding().signals == []


def test_finding_loads_payload_without_signals():
    # Findings serialized before the field existed must still load.
    payload = make_finding().model_dump(mode="json")
    payload.pop("signals")
    restored = Finding.model_validate(payload)
    assert restored.signals == []


@pytest.mark.parametrize(
    "agent, signals",
    [
        ("kubernetes", ["crash_loop"]),
        ("observability", ["errors_elevated", "cpu_saturated"]),
        ("security", ["failed_auth_elevated", "open_ingress"]),
    ],
)
def test_finding_accepts_signals_from_its_own_vocabulary(agent, signals):
    finding = make_finding(agent_name=agent, signals=signals)
    assert finding.signals == signals


def test_finding_rejects_unknown_signals():
    with pytest.raises(ValidationError):
        make_finding(signals=["on_fire"])


@pytest.mark.parametrize(
    "agent, signal",
    [
        ("kubernetes", "cpu_saturated"),
        ("observability", "crash_loop"),
        ("observability", "publicly_exposed"),
        ("security", "errors_elevated"),
    ],
)
def test_finding_rejects_another_agents_signal(agent, signal):
    with pytest.raises(ValidationError, match="cannot emit"):
        make_finding(agent_name=agent, signals=[signal])


def test_finding_rejects_repeated_signals():
    with pytest.raises(ValidationError, match="must not repeat"):
        make_finding(
            agent_name="observability",
            signals=["errors_elevated", "errors_elevated"],
        )


def test_finding_signals_require_evidence():
    with pytest.raises(ValidationError, match="require evidence"):
        make_finding(
            evidence=[], evidence_quality="insufficient", signals=["crash_loop"]
        )


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_failed_or_blocked_finding_cannot_carry_signals(status):
    with pytest.raises(ValidationError, match="missing data"):
        make_finding(status=status, signals=["crash_loop"])


@pytest.mark.parametrize("status", ["completed", "partial", "needs_review"])
def test_other_statuses_may_carry_signals(status):
    assert make_finding(status=status, signals=["crash_loop"]).signals


def test_signal_vocabulary_is_closed_and_owned_by_exactly_one_agent():
    assert set(SIGNALS_BY_AGENT) == {"kubernetes", "observability", "security"}
    owned = [signal for group in SIGNALS_BY_AGENT.values() for signal in group]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(get_args(SignalName))


# ---- correlations ----


def test_correlation_valid_and_defaults():
    correlation = make_correlation()
    assert correlation.correlation_id
    assert correlation.kind == "corroborates"
    assert correlation.correlated_at is not None


def test_correlation_ids_are_unique():
    assert make_correlation().correlation_id != make_correlation().correlation_id


def test_correlation_kinds_are_a_closed_list():
    assert set(get_args(CorrelationKind)) == {
        "corroborates",
        "contradicts",
        "co_occurs",
    }
    with pytest.raises(ValidationError):
        make_correlation(kind="causes")


@pytest.mark.parametrize("field", ["rule", "explanation"])
@pytest.mark.parametrize("value", ["", "   "])
def test_correlation_requires_rule_and_explanation(field, value):
    with pytest.raises(ValidationError):
        make_correlation(**{field: value})


@pytest.mark.parametrize(
    "ids",
    [[], ["finding-1"], ["finding-1", "finding-1"], ["a", "b", "a"]],
)
def test_correlation_needs_two_distinct_finding_ids(ids):
    with pytest.raises(ValidationError):
        make_correlation(finding_ids=ids)


@pytest.mark.parametrize("ids", [["a", ""], ["a", "   "]])
def test_correlation_rejects_blank_finding_ids(ids):
    with pytest.raises(ValidationError):
        make_correlation(finding_ids=ids)


def test_correlation_rejects_extra_fields():
    with pytest.raises(ValidationError):
        make_correlation(authorizes_action=True)


def test_correlation_accepts_more_than_two_distinct_findings():
    correlation = make_correlation(finding_ids=["a", "b", "c"])
    assert len(correlation.finding_ids) == 3
