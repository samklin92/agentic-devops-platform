import asyncio

from app.agents.kubernetes_agent import investigate_kubernetes
from app.contracts.agent_messages import (
    KubernetesInvestigationRequest,
)


def test_kubernetes_agent_completes_investigation():

    request = KubernetesInvestigationRequest(
        incident_id="INC-001",
        task="Investigate why inventory-api is crashing",
        target={
            "namespace": "production",
            "pod_name": "inventory-api",
        },
        priority="high",
    )

    calls = []

    async def inspect_pod_events():
        calls.append("inspect_pod_events")

        return {
            "reason": "CrashLoopBackOff",
            "message": "Container repeatedly restarting",
            "terminal": False,
        }

    async def gather_additional_diagnostics():
        calls.append("gather_additional_diagnostics")

        return {
            "pod": {"status": "CrashLoopBackOff"},
            "logs": {
                "logs": ["ERROR: Database connection refused"]
            },
            "deployment": {
                "desired_replicas": 3,
                "ready_replicas": 1,
                "status": "Degraded",
            },
            "terminal": True,
        }

    tools = {
        "inspect_pod_events": inspect_pod_events,
        "gather_additional_diagnostics": (
            gather_additional_diagnostics
        ),
    }

    result = asyncio.run(
        investigate_kubernetes(
            request=request,
            tools=tools,
        )
    )

    assert result.incident_id == "INC-001"
    assert result.agent_name == "kubernetes"
    assert result.status == "completed"
    assert result.needs_follow_up is False

    assert len(result.findings) == 2

    assert calls == [
        "inspect_pod_events",
        "gather_additional_diagnostics",
    ]


def test_kubernetes_agent_preserves_incident_id():

    request = KubernetesInvestigationRequest(
        incident_id="INC-002",
        task="Investigate payment-api",
        target={
            "namespace": "production",
            "pod_name": "payment-api",
        },
    )

    async def inspect_pod_events():
        return {
            "reason": "Healthy",
            "terminal": True,
        }

    result = asyncio.run(
        investigate_kubernetes(
            request=request,
            tools={
                "inspect_pod_events": inspect_pod_events,
            },
        )
    )

    assert result.incident_id == "INC-002"