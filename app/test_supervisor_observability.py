"""Tests for the Observability investigator."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from app.contracts.agent_messages import AgentEvidence
from app.mcp_server import server
from app.state.adapters import is_execution_failure
from app.state.models import Finding, IncidentContext
from app.supervisor import observability as obs
from app.supervisor.proposals import is_usable, propose_action


def make_incident(**overrides) -> IncidentContext:
    data = {
        "incident_id": "inc-obs",
        "description": "p95 latency is elevated",
        "severity": "high",
        "environment": "production",
        "namespace": "production",
        "service": "payment-api",
    }
    data.update(overrides)
    return IncidentContext(**data)


def server_backend(calls=None):
    """Call the simulated server's metric functions directly."""

    async def call_tool(name, arguments):
        if calls is not None:
            calls.append((name, dict(arguments)))
        return getattr(server, name)(**arguments)

    return call_tool


def fixed_backend(
    error_rate_pct=0.1, p95_ms=100, cpu=10, memory=10, raises=(), calls=None
):
    payloads = {
        obs.TOOL_ERROR_RATE: {"error_rate_pct": error_rate_pct},
        obs.TOOL_LATENCY: {"p95_ms": p95_ms, "p99_ms": p95_ms * 2},
        obs.TOOL_RESOURCES: {
            "cpu_pct_of_limit": cpu,
            "memory_pct_of_limit": memory,
        },
    }

    async def call_tool(name, arguments):
        if calls is not None:
            calls.append(name)
        if name in raises:
            raise RuntimeError("boom")
        return payloads[name]

    return call_tool


def collect(call_tool, namespace="production", service="payment-api"):
    return obs.finding_from_metrics(
        asyncio.run(obs.collect_metrics(call_tool, namespace, service))
    )


class FakeResponse:
    def __init__(self, content, is_error=False):
        self.structured_content = content
        self.is_error = is_error


class FakeSession:
    def __init__(self, error_tools=()):
        self.calls = []
        self.error_tools = set(error_tools)

    async def call_tool(self, name, arguments=None):
        self.calls.append(name)
        if name in self.error_tools:
            return FakeResponse({}, is_error=True)
        return FakeResponse(getattr(server, name)(**arguments))


def factory_for(session, opened=None):
    @asynccontextmanager
    async def factory():
        if opened is not None:
            opened.append(True)
        yield session

    return factory


# --- target resolution ---------------------------------------------------


def test_target_prefers_service_over_deployment():
    incident = make_incident(service="checkout-api", deployment="other")
    assert obs.resolve_target(incident) == ("production", "checkout-api")


def test_target_falls_back_to_deployment():
    incident = make_incident(service=None, deployment="inventory-api")
    assert obs.resolve_target(incident) == ("production", "inventory-api")


def test_blank_service_counts_as_missing():
    incident = make_incident(service="   ", deployment=None)
    assert obs.resolve_target(incident) is None


def test_blank_service_falls_back_to_deployment():
    incident = make_incident(service="   ", deployment="inventory-api")
    assert obs.resolve_target(incident) == ("production", "inventory-api")


def test_missing_namespace_counts_as_missing():
    assert obs.resolve_target(make_incident(namespace=None)) is None


# --- scenarios from the simulated server ---------------------------------


def test_healthy_service_is_completed_and_usable():
    finding = collect(server_backend())
    assert finding.status == "completed"
    assert finding.evidence_quality == "limited"
    assert finding.summary == "Metrics within normal thresholds."
    assert len(finding.evidence) == 3
    assert is_usable(finding)
    assert not any(is_execution_failure(e.observation) for e in finding.evidence)


def test_inventory_api_reports_symptoms_not_causes():
    finding = collect(server_backend(), service="inventory-api")
    assert finding.status == "completed"
    assert finding.summary == (
        "Errors elevated, latency elevated, resources normal."
    )
    lowered = finding.summary.lower()
    assert "cause" not in lowered and "because" not in lowered
    assert "crash" not in lowered


def test_checkout_api_reports_cpu_saturation():
    finding = collect(server_backend(), service="checkout-api")
    assert finding.summary == "Latency elevated, CPU saturated."


# --- thresholds ----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"error_rate_pct": 5.0}, "Errors elevated, resources normal."),
        ({"error_rate_pct": 4.9}, "Metrics within normal thresholds."),
        ({"p95_ms": 1000}, "Latency elevated, resources normal."),
        ({"p95_ms": 999}, "Metrics within normal thresholds."),
        ({"cpu": 90}, "CPU saturated."),
        ({"cpu": 89}, "Metrics within normal thresholds."),
        ({"memory": 90}, "Memory near limit."),
        ({"cpu": 90, "memory": 95}, "CPU saturated, memory near limit."),
    ],
)
def test_thresholds_are_inclusive(kwargs, expected):
    assert collect(fixed_backend(**kwargs)).summary == expected


# --- bounded, ordered calls ----------------------------------------------


def test_exactly_three_calls_in_fixed_order_with_window():
    calls = []
    collect(server_backend(calls))
    assert [name for name, _ in calls] == [
        "get_error_rate",
        "get_latency_percentiles",
        "get_resource_usage",
    ]
    for name, arguments in calls:
        assert arguments["service"] == "payment-api"
        assert arguments["namespace"] == "production"
        if name == "get_resource_usage":
            assert "window_minutes" not in arguments
        else:
            assert arguments["window_minutes"] == obs.WINDOW_MINUTES


# --- failures ------------------------------------------------------------


def test_unknown_service_fails_with_only_failure_evidence():
    finding = collect(server_backend(), service="ghost")
    assert finding.status == "failed"
    assert finding.evidence_quality == "insufficient"
    assert len(finding.evidence) == 3
    assert all(is_execution_failure(e.observation) for e in finding.evidence)
    assert "ghost" in finding.error_message
    assert not is_usable(finding)


def test_one_failing_tool_gives_partial_and_the_others_still_run():
    calls = []
    finding = collect(
        fixed_backend(raises={obs.TOOL_LATENCY}, calls=calls)
    )
    assert calls == list(obs._SEQUENCE)
    assert finding.status == "partial"
    assert finding.evidence_quality == "limited"
    assert "Unavailable: get_latency_percentiles" in finding.summary
    failures = [
        e for e in finding.evidence if is_execution_failure(e.observation)
    ]
    assert len(failures) == 1
    payload = json.loads(failures[0].observation)
    assert payload["tool"] == "get_latency_percentiles"
    assert payload["execution_failed"] is True
    assert not is_usable(finding)


def test_malformed_results_are_recorded_as_failures():
    async def call_tool(name, arguments):
        if name == obs.TOOL_ERROR_RATE:
            return []
        if name == obs.TOOL_LATENCY:
            return {"p95_ms": True, "p99_ms": 1}
        return {"cpu_pct_of_limit": 10, "memory_pct_of_limit": 10}

    finding = collect(call_tool)
    assert finding.status == "partial"
    failures = [
        e for e in finding.evidence if is_execution_failure(e.observation)
    ]
    assert len(failures) == 2


# --- allowlist -----------------------------------------------------------


def test_allowlist_matches_the_sequence_and_exists_on_the_server():
    registered = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    assert obs.OBSERVABILITY_TOOL_ALLOWLIST == set(obs._SEQUENCE)
    assert obs.OBSERVABILITY_TOOL_ALLOWLIST <= registered


@pytest.mark.parametrize(
    "tool",
    [
        "inspect_pod",
        "get_pod_events",
        "get_container_logs",
        "check_deployment_status",
        "delete_pod",
    ],
)
def test_guard_rejects_kubernetes_and_unknown_tools(tool):
    guarded = obs.guard_call_tool(server_backend())
    with pytest.raises(obs.ToolNotAllowedError):
        asyncio.run(guarded(tool, {}))


def test_allowlist_violation_is_not_swallowed_as_a_tool_failure(monkeypatch):
    monkeypatch.setattr(obs, "_SEQUENCE", ("inspect_pod",))
    with pytest.raises(obs.ToolNotAllowedError):
        asyncio.run(obs.collect_metrics(server_backend(), "production", "x"))


# --- the investigator ----------------------------------------------------


def test_investigator_runs_through_a_session():
    session = FakeSession()
    investigator = obs.make_observability_investigator(factory_for(session))
    finding = investigator(make_incident(service="inventory-api"))
    assert finding.agent_name == "observability"
    assert finding.status == "completed"
    assert finding.summary == (
        "Errors elevated, latency elevated, resources normal."
    )
    assert session.calls == list(obs._SEQUENCE)


def test_investigator_falls_back_to_deployment():
    session = FakeSession()
    investigator = obs.make_observability_investigator(factory_for(session))
    finding = investigator(
        make_incident(service=None, deployment="inventory-api")
    )
    assert finding.status == "completed"
    assert "Errors elevated" in finding.summary


def test_investigator_records_mcp_error_responses_as_failures():
    session = FakeSession(error_tools={obs.TOOL_ERROR_RATE})
    investigator = obs.make_observability_investigator(factory_for(session))
    finding = investigator(make_incident())
    assert finding.status == "partial"
    assert is_execution_failure(finding.evidence[0].observation)
    assert session.calls == list(obs._SEQUENCE)


@pytest.mark.parametrize(
    "overrides",
    [
        {"service": None, "deployment": None},
        {"service": "   ", "deployment": None},
        {"namespace": None},
    ],
)
def test_investigator_blocks_without_a_target_and_opens_no_session(overrides):
    opened = []
    investigator = obs.make_observability_investigator(
        factory_for(FakeSession(), opened)
    )
    finding = investigator(make_incident(**overrides))
    assert finding.status == "blocked"
    assert finding.evidence == []
    assert finding.error_message
    assert opened == []


# --- observability can never authorize remediation -----------------------


def crash_loop_incident():
    return make_incident(
        description="inventory-api crash loop with HTTP 500 errors",
        service="inventory-api",
        deployment="inventory-api",
    )


def finding_mentioning_crash_loop(agent_name):
    return Finding(
        agent_name=agent_name,
        summary="crash loop symptoms",
        evidence=[
            AgentEvidence(source_tool="some_tool", observation="crash loop")
        ],
        evidence_quality="limited",
    )


def test_usable_observability_finding_never_produces_a_proposal():
    finding = finding_mentioning_crash_loop("observability")
    assert is_usable(finding)
    assert propose_action(crash_loop_incident(), [finding]) is None


def test_mixed_findings_are_supported_only_by_the_kubernetes_finding():
    kubernetes = finding_mentioning_crash_loop("kubernetes")
    observability = finding_mentioning_crash_loop("observability")
    action = propose_action(
        crash_loop_incident(), [observability, kubernetes]
    )
    assert action is not None
    assert action.action_type == "rollout_restart"
    assert action.supporting_finding_ids == [kubernetes.finding_id]


# --- structured signals --------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, []),
        ({"error_rate_pct": 5.0}, ["errors_elevated"]),
        ({"error_rate_pct": 4.9}, []),
        ({"p95_ms": 1000}, ["latency_elevated"]),
        ({"p95_ms": 999}, []),
        ({"cpu": 90}, ["cpu_saturated"]),
        ({"cpu": 89}, []),
        ({"memory": 90}, ["memory_near_limit"]),
        ({"memory": 89}, []),
        (
            {"error_rate_pct": 9, "p95_ms": 2000, "cpu": 95, "memory": 95},
            [
                "errors_elevated",
                "latency_elevated",
                "cpu_saturated",
                "memory_near_limit",
            ],
        ),
    ],
)
def test_signals_follow_the_inclusive_thresholds(kwargs, expected):
    assert collect(fixed_backend(**kwargs)).signals == expected


@pytest.mark.parametrize(
    "service, expected",
    [
        ("payment-api", []),
        ("inventory-api", ["errors_elevated", "latency_elevated"]),
        ("checkout-api", ["latency_elevated", "cpu_saturated"]),
    ],
)
def test_simulated_services_carry_the_expected_signals(service, expected):
    assert collect(server_backend(), service=service).signals == expected


_OWNED_SIGNALS = {
    obs.TOOL_ERROR_RATE: {"errors_elevated"},
    obs.TOOL_LATENCY: {"latency_elevated"},
    obs.TOOL_RESOURCES: {"cpu_saturated", "memory_near_limit"},
}


@pytest.mark.parametrize("tool", sorted(_OWNED_SIGNALS))
def test_a_failed_metric_call_never_produces_its_signals(tool):
    finding = collect(
        fixed_backend(
            error_rate_pct=50, p95_ms=5000, cpu=99, memory=99, raises={tool}
        )
    )
    assert finding.status == "partial"
    assert not _OWNED_SIGNALS[tool] & set(finding.signals)
    # The metrics that did succeed still report their own signals.
    others = set().union(
        *(v for k, v in _OWNED_SIGNALS.items() if k != tool)
    )
    assert others <= set(finding.signals)


def test_all_calls_failing_carries_no_signals():
    finding = collect(
        fixed_backend(error_rate_pct=50, raises=set(obs._SEQUENCE))
    )
    assert finding.status == "failed"
    assert finding.signals == []


def test_malformed_metric_results_carry_no_signals():
    async def call_tool(name, arguments):
        return {"error_rate_pct": "50", "p95_ms": True, "cpu_pct_of_limit": None}

    finding = collect(call_tool)
    assert finding.status == "failed"
    assert finding.signals == []


def test_unknown_service_and_blocked_findings_carry_no_signals():
    assert collect(server_backend(), service="ghost").signals == []
    blocked = obs.make_observability_investigator(factory_for(FakeSession()))(
        make_incident(namespace=None)
    )
    assert blocked.status == "blocked"
    assert blocked.signals == []


def test_signals_and_summary_always_agree():
    import itertools

    tools = sorted(obs._SEQUENCE)
    failure_sets = [
        set(subset)
        for size in range(len(tools) + 1)
        for subset in itertools.combinations(tools, size)
    ]
    for error, p95, cpu, memory in itertools.product(
        (0.1, 9), (100, 2000), (10, 95), (10, 95)
    ):
        for raises in failure_sets:
            finding = collect(
                fixed_backend(
                    error_rate_pct=error,
                    p95_ms=p95,
                    cpu=cpu,
                    memory=memory,
                    raises=raises,
                )
            )
            summary = finding.summary.lower()
            named = {
                signal
                for signal, text in obs._SIGNAL_TEXT.items()
                if text.lower() in summary
            }
            assert set(finding.signals) == named
