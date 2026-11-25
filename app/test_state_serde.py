from app.state.serde import ALLOWED_MSGPACK_MODULES, platform_serializer


def test_allowlist_covers_state_records_and_nested_evidence():
    allowed = set(ALLOWED_MSGPACK_MODULES)
    for name in [
        "IncidentContext",
        "Finding",
        "ProposedAction",
        "ApprovalRecord",
        "ExecutionRecord",
        "UsageRecord",
        "AuditEntry",
        "IncidentReport",
        "Correlation",
    ]:
        assert ("app.state.models", name) in allowed
    assert ("app.contracts.agent_messages", "AgentEvidence") in allowed


def test_allowlist_contains_only_our_own_modules():
    for module, _ in ALLOWED_MSGPACK_MODULES:
        assert module.startswith("app.")


def test_serializer_round_trips_a_state_record():
    from app.state.models import AuditEntry

    serde = platform_serializer()
    entry = AuditEntry(incident_id="i", actor="supervisor", event="routed")
    restored = serde.loads_typed(serde.dumps_typed(entry))
    assert restored == entry


def test_serializer_round_trips_a_finding_with_signals_and_a_correlation():
    from app.contracts.agent_messages import AgentEvidence
    from app.state.models import Correlation, Finding

    serde = platform_serializer()
    finding = Finding(
        agent_name="observability",
        summary="Errors elevated.",
        evidence=[AgentEvidence(source_tool="get_error_rate", observation="x")],
        evidence_quality="limited",
        signals=["errors_elevated"],
    )
    correlation = Correlation(
        kind="corroborates",
        rule="unhealthy_workload",
        finding_ids=[finding.finding_id, "other"],
        explanation="Two specialists report signals about the same workload.",
    )
    for record in (finding, correlation):
        assert serde.loads_typed(serde.dumps_typed(record)) == record
