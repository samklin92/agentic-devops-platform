import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

from app.contracts.agent_messages import (
    KubernetesInvestigationRequest,
    KubernetesTarget,
)
from app.state.models import IncidentContext
from app.supervisor.kubernetes_bridge import (
    KUBERNETES_TOOL_ALLOWLIST,
    build_kubernetes_investigator,
)
from app.supervisor.mcp_tools import (
    mcp_session_tools_provider,
    server_parameters,
)
from app.supervisor.proposals import is_usable

class FakeSession:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise RuntimeError("MCP server unreachable")
        payloads = {
            "get_pod_events": {
                "result": [
                    {
                        "type": "Warning",
                        "reason": "BackOff",
                        "message": "Back-off restarting failed container",
                    }
                ]
            },
            "inspect_pod": {"name": "inventory-api", "status": "CrashLoopBackOff"},
            "get_container_logs": {"result": ["ERROR: Database connection refused."]},
            "check_deployment_status": {
                "name": "inventory-api",
                "status": "Degraded",
            },
        }
        return SimpleNamespace(is_error=False, structured_content=payloads[name])


def make_session_factory(session, log: list[str]):
    @asynccontextmanager
    async def factory():
        log.append("opened")
        try:
            yield session
        finally:
            log.append("closed")

    return factory


def make_incident() -> IncidentContext:
    return IncidentContext(
        incident_id="inc-001",
        description=(
            "Investigate why inventory-api is crashing: "
            "pod inventory-api is in CrashLoopBackOff"
        ),
        severity="high",
        environment="production",
        namespace="production",
        pod_name="inventory-api",
        deployment="inventory-api",
    )


def test_provider_builds_allowlisted_tools_and_closes_the_session():
    session = FakeSession()
    log: list[str] = []
    provider = mcp_session_tools_provider(make_session_factory(session, log))
    request = KubernetesInvestigationRequest(
        incident_id="inc-001",
        task="Investigate inventory-api",
        target=KubernetesTarget(namespace="production", pod_name="inventory-api"),
    )

    async def use():
        async with provider(request) as tools:
            assert set(tools) == KUBERNETES_TOOL_ALLOWLIST
            await tools["inspect_pod_events"]()
            return list(log)

    assert asyncio.run(use()) == ["opened"]
    assert log == ["opened", "closed"]
    assert session.calls == [
        (
            "get_pod_events",
            {"pod_name": "inventory-api", "namespace": "production"},
        )
    ]


def test_session_wraps_a_real_agent_investigation():
    session = FakeSession()
    log: list[str] = []
    investigator = build_kubernetes_investigator(
        mcp_session_tools_provider(make_session_factory(session, log))
    )

    finding = investigator(make_incident())

    assert log == ["opened", "closed"]
    assert finding.agent_name == "kubernetes"
    assert finding.evidence
    assert "get_pod_events" in [name for name, _ in session.calls]
    for _, arguments in session.calls:
        assert arguments["namespace"] == "production"


def test_failing_session_is_closed_and_no_evidence_is_claimed():
    session = FakeSession(fail=True)
    log: list[str] = []
    investigator = build_kubernetes_investigator(
        mcp_session_tools_provider(make_session_factory(session, log))
    )

    finding = investigator(make_incident())

    assert log == ["opened", "closed"]
    assert finding.evidence_quality == "insufficient"
    assert not is_usable(finding), (
        finding.status,
        finding.evidence_quality,
        finding.needs_follow_up,
    )


def test_stdio_server_starts_with_the_current_interpreter():
    params = server_parameters()
    assert params.command == sys.executable
    assert params.args == ["-m", "app.mcp_server.server"]