"""Tests for the Kubernetes MCP server."""

import asyncio

import pytest

from app.mcp_server.server import mcp

from types import SimpleNamespace

from app.mcp_server.adapter import build_mcp_tools

import app.langgraph_agent as langgraph_agent

from app.langgraph_agent import build_devops_graph_async

from app.mcp_server import server as kubernetes_server 

KUBERNETES_TOOLS = {
    "inspect_pod",
    "get_pod_events",
    "get_container_logs",
    "check_deployment_status",
}

METRICS_TOOLS = {
    "get_error_rate",
    "get_latency_percentiles",
    "get_resource_usage",
}

SECURITY_TOOLS = {
    "get_failed_auth_attempts",
    "get_iam_activity",
    "check_network_exposure",
}


def test_mcp_server_registers_all_diagnostic_tools():

    result = asyncio.run(mcp.list_tools())

    registered_tool_names = {
        tool.name for tool in result
    }

    assert registered_tool_names == (
        KUBERNETES_TOOLS | METRICS_TOOLS | SECURITY_TOOLS
    )
    assert KUBERNETES_TOOLS.isdisjoint(METRICS_TOOLS)
    assert KUBERNETES_TOOLS.isdisjoint(SECURITY_TOOLS)
    assert METRICS_TOOLS.isdisjoint(SECURITY_TOOLS)

def test_adapter_inspects_pod_events():

    class FakeSession:

        async def call_tool(self, name, arguments):

            assert name == "get_pod_events"
            assert arguments == {
    "pod_name": "inventory-api",
    "namespace": "production",
}
            return SimpleNamespace(
                is_error=False,
                structured_content={
                    "events": [
                        {
                            "type": "Warning",
                            "reason": "BackOff",
                            "message": "Back-off restarting failed container",
                        }
                    ]
                },
            )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
        namespace="production",
    )

    result = asyncio.run(
        tools["inspect_pod_events"]()
    )

    assert result["events"][0]["reason"] == "BackOff"

def test_adapter_gathers_additional_diagnostics():

    class FakeSession:

        async def call_tool(self, name, arguments):

            responses = {
                "inspect_pod": {
                    "pod_name": "inventory-api",
                    "status": "CrashLoopBackOff",
                },
                "get_container_logs": {
                    "logs": ["ERROR: Database connection refused."]
                },
                "check_deployment_status": {
                    "desired_replicas": 3,
                    "ready_replicas": 1,
                    "status": "Degraded",
                },
            }

            assert name in responses

            return SimpleNamespace(
                is_error=False,
                structured_content=responses[name],
            )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
        namespace="production",
    )

    result = asyncio.run(
        tools["gather_additional_diagnostics"]()
    )

    assert result["pod"]["status"] == "CrashLoopBackOff"
    assert result["logs"]["logs"][0] == (
        "ERROR: Database connection refused."
    )
    assert result["deployment"]["status"] == "Degraded"
    assert result["terminal"] is True

def test_adapter_raises_error_when_mcp_tool_fails():

    class FakeSession:

        async def call_tool(self, name, arguments):

            return SimpleNamespace(
                is_error=True,
                structured_content=None,
            )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
        namespace="production",
    )

    with pytest.raises(
        RuntimeError,
        match="MCP tool 'get_pod_events' returned an error.",
    ):
        asyncio.run(
            tools["inspect_pod_events"]()
        )
def test_mcp_tools_integrate_with_async_langgraph(monkeypatch):

    calls = []

    class FakeSession:

        async def call_tool(self, name, arguments):

            calls.append(name)

            responses = {
                "get_pod_events": {
                    "events": [
                        {
                            "type": "Warning",
                            "reason": "BackOff",
                            "message": "Back-off restarting failed container",
                        }
                    ]
                },
                "inspect_pod": {
                    "pod_name": "inventory-api",
                    "status": "CrashLoopBackOff",
                },
                "get_container_logs": {
                    "logs": [
                        "ERROR: Database connection refused."
                    ]
                },
                "check_deployment_status": {
                    "desired_replicas": 3,
                    "ready_replicas": 1,
                    "status": "Degraded",
                },
            }

            return SimpleNamespace(
                is_error=False,
                structured_content=responses[name],
            )

    # Mock RAG retrieval to avoid loading the embedding model.
    monkeypatch.setattr(
        langgraph_agent,
        "retrieve_agent_knowledge",
        lambda **kwargs: [
            {
                "source": "crashloopbackoff.md",
                "section": "Possible Causes",
                "content": "Database connectivity problems can cause crashes.",
            }
        ],
    )

    # Mock response generation to avoid calling Claude's API.
    monkeypatch.setattr(
        langgraph_agent,
        "generate_agent_response",
        lambda state, client: {
            "answer": "The pod is experiencing database connectivity problems.",
            "sources": state.knowledge,
            "grounded_in_retrieved_knowledge": True,
        },
    )

    tools = build_mcp_tools(
        session=FakeSession(),
        pod_name="inventory-api",
        namespace="production",
    )

    graph = build_devops_graph_async(
        embedding_model=object(),
        tools=tools,
        client=object(),
    )

    initial_state = {
        "task": "Investigate why inventory-api is in CrashLoopBackOff",
        "task_id": "mcp-integration-test",
        "status": "pending",
        "findings": [],
        "actions_taken": [],
        "knowledge": [],
        "tool_calls": 0,
        "next_action": None,
        "response": "",
        "sources": [],
        "grounded_in_retrieved_knowledge": False,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert result["status"] == "completed"
    assert result["tool_calls"] == 2

    assert calls == [
        "get_pod_events",
        "inspect_pod",
        "get_container_logs",
        "check_deployment_status",
    ]

    assert result["response"] == (
        "The pod is experiencing database connectivity problems."
    )

    assert result["grounded_in_retrieved_knowledge"] is True

def test_same_pod_name_returns_different_resources_by_namespace():
    production_pod = kubernetes_server.inspect_pod(
        pod_name="inventory-api",
        namespace="production",
    )

    staging_pod = kubernetes_server.inspect_pod(
        pod_name="inventory-api",
        namespace="staging",
    )

    assert production_pod["status"] == "CrashLoopBackOff"
    assert staging_pod["status"] == "Running"

    assert production_pod["namespace"] == "production"
    assert staging_pod["namespace"] == "staging"

def test_pod_lookup_rejects_unknown_namespace():
    result = kubernetes_server.inspect_pod(
        pod_name="inventory-api",
        namespace="development",
    )

    assert "error" in result

def test_container_logs_are_namespace_specific():
    production_logs = kubernetes_server.get_container_logs(
        pod_name="inventory-api",
        namespace="production",
    )

    staging_logs = kubernetes_server.get_container_logs(
        pod_name="inventory-api",
        namespace="staging",
    )

    assert any(
        "Database connection refused" in log
        for log in production_logs
    )

    assert any(
        "Connected to staging database" in log
        for log in staging_logs
    )


def test_deployment_lookup_is_namespace_specific():
    production_deployment = kubernetes_server.check_deployment_status(
        deployment_name="inventory-api",
        namespace="production",
    )

    staging_deployment = kubernetes_server.check_deployment_status(
        deployment_name="inventory-api",
        namespace="staging",
    )

    assert production_deployment["status"] == "Degraded"
    assert staging_deployment["status"] == "Healthy"

def test_metric_tools_return_distinct_scenarios():
    healthy = kubernetes_server.get_error_rate(
        service="payment-api", namespace="production"
    )
    failing = kubernetes_server.get_error_rate(
        service="inventory-api", namespace="production"
    )
    assert healthy["error_rate_pct"] < 1
    assert failing["error_rate_pct"] > 90


def test_metric_tools_are_namespace_specific():
    production = kubernetes_server.get_latency_percentiles(
        service="inventory-api", namespace="production"
    )
    staging = kubernetes_server.get_latency_percentiles(
        service="inventory-api", namespace="staging"
    )
    assert production["p95_ms"] == 5000
    assert staging["p95_ms"] == 90
    assert production["namespace"] == "production"
    assert staging["namespace"] == "staging"


def test_resource_usage_shows_cpu_saturation_scenario():
    usage = kubernetes_server.get_resource_usage(
        service="checkout-api", namespace="production"
    )
    assert usage["cpu_pct_of_limit"] >= 90
    assert usage["memory_pct_of_limit"] < 90


def test_metric_tools_return_error_for_unknown_service():
    for result in (
        kubernetes_server.get_error_rate("ghost", "production", 15),
        kubernetes_server.get_latency_percentiles("ghost", "production", 15),
        kubernetes_server.get_resource_usage("ghost", "production"),
    ):
        assert "error" in result


def test_metric_tools_echo_the_window():
    result = kubernetes_server.get_error_rate(
        service="payment-api", namespace="production", window_minutes=30
    )
    assert result["window_minutes"] == 30


def test_metric_window_accepts_the_bounds():
    for window in (
        kubernetes_server.MIN_WINDOW_MINUTES,
        kubernetes_server.MAX_WINDOW_MINUTES,
    ):
        result = kubernetes_server.get_error_rate(
            "payment-api", "production", window
        )
        assert "error" not in result


@pytest.mark.parametrize("window", [0, -5, 1441, True, 1.5, "15"])
def test_metric_window_rejects_invalid_values(window):
    for result in (
        kubernetes_server.get_error_rate("payment-api", "production", window),
        kubernetes_server.get_latency_percentiles(
            "payment-api", "production", window
        ),
    ):
        assert "error" in result


def test_security_tools_return_distinct_scenarios():
    quiet = kubernetes_server.get_failed_auth_attempts(
        service="payment-api", namespace="production"
    )
    noisy = kubernetes_server.get_failed_auth_attempts(
        service="checkout-api", namespace="production"
    )
    assert quiet["failed_auth_count"] < 10
    assert noisy["failed_auth_count"] >= 1000
    assert noisy["distinct_source_ips"] > quiet["distinct_source_ips"]


def test_security_tools_are_namespace_specific():
    production = kubernetes_server.get_iam_activity(
        service="inventory-api", namespace="production"
    )
    staging = kubernetes_server.get_iam_activity(
        service="inventory-api", namespace="staging"
    )
    assert production["credential_age_days"] == 20
    assert staging["credential_age_days"] == 12
    assert production["namespace"] == "production"
    assert staging["namespace"] == "staging"


def test_iam_activity_shows_stale_credential_scenario():
    stale = kubernetes_server.get_iam_activity(
        service="payment-api", namespace="production"
    )
    denied = kubernetes_server.get_iam_activity(
        service="checkout-api", namespace="production"
    )
    assert stale["credential_age_days"] >= 90
    assert stale["access_denied_calls"] == 0
    assert denied["access_denied_calls"] >= 10


def test_network_exposure_returns_booleans_and_one_exposed_scenario():
    exposed = kubernetes_server.check_network_exposure(
        service="checkout-api", namespace="production"
    )
    private = kubernetes_server.check_network_exposure(
        service="payment-api", namespace="production"
    )
    for result in (exposed, private):
        assert isinstance(result["publicly_exposed"], bool)
        assert isinstance(result["open_ingress_cidr"], bool)
    assert exposed["publicly_exposed"] and exposed["open_ingress_cidr"]
    assert not private["publicly_exposed"]
    assert not private["open_ingress_cidr"]


def test_every_metrics_service_has_security_data():
    metrics_keys = set(kubernetes_server.SIMULATED_ERROR_RATES)
    assert metrics_keys == set(kubernetes_server.SIMULATED_FAILED_AUTH)
    assert metrics_keys == set(kubernetes_server.SIMULATED_IAM_ACTIVITY)
    assert metrics_keys == set(kubernetes_server.SIMULATED_NETWORK_EXPOSURE)


def test_security_tools_return_error_not_zeros_for_unknown_service():
    for result in (
        kubernetes_server.get_failed_auth_attempts("ghost", "production", 15),
        kubernetes_server.get_iam_activity("ghost", "production", 15),
        kubernetes_server.check_network_exposure("ghost", "production"),
    ):
        assert "error" in result
        assert not set(result) - {"error"}


def test_security_tools_echo_the_window():
    for tool in (
        kubernetes_server.get_failed_auth_attempts,
        kubernetes_server.get_iam_activity,
    ):
        result = tool(
            service="payment-api", namespace="production", window_minutes=30
        )
        assert result["window_minutes"] == 30


def test_security_window_accepts_the_bounds():
    for tool in (
        kubernetes_server.get_failed_auth_attempts,
        kubernetes_server.get_iam_activity,
    ):
        for window in (
            kubernetes_server.MIN_WINDOW_MINUTES,
            kubernetes_server.MAX_WINDOW_MINUTES,
        ):
            assert "error" not in tool("payment-api", "production", window)


@pytest.mark.parametrize("window", [0, -5, 1441, True, 1.5, "15"])
def test_security_window_rejects_invalid_values(window):
    for tool in (
        kubernetes_server.get_failed_auth_attempts,
        kubernetes_server.get_iam_activity,
    ):
        assert "error" in tool("payment-api", "production", window)
