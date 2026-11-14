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