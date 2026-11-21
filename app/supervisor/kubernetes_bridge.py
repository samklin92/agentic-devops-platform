"""Synchronous bridge from the Supervisor graph to the async Kubernetes agent.

The Supervisor graph is synchronous; the Kubernetes agent is async. The
bridge runs one agent investigation per call with asyncio.run, inside an
async context where the tools provider can open and close resources such
as an MCP session on the same event loop.
"""

import asyncio
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from app.agents.kubernetes_agent import investigate_kubernetes
from app.contracts.agent_messages import (
    KubernetesInvestigationRequest,
    KubernetesInvestigationResponse,
)
from app.supervisor.investigators import (
    Investigator,
    make_kubernetes_investigator,
)

# The only tools the Kubernetes agent may ever receive. Both are read-only
# diagnostics. Anything else is rejected before the agent runs.
KUBERNETES_TOOL_ALLOWLIST: frozenset[str] = frozenset(
    {"inspect_pod_events", "gather_additional_diagnostics"}
)

ToolsProvider = Callable[
    [KubernetesInvestigationRequest],
    AbstractAsyncContextManager[Mapping[str, Any]],
]


def validate_tool_allowlist(tools: Mapping[str, Any]) -> None:
    """Reject any tool the Kubernetes agent is not allowed to use."""
    unexpected = sorted(set(tools) - KUBERNETES_TOOL_ALLOWLIST)
    if unexpected:
        raise ValueError(
            f"Tools not allowed for the Kubernetes agent: {unexpected}"
        )


def static_tools_provider(tools: Mapping[str, Any]) -> ToolsProvider:
    """Provider that always yields the same tools. Useful for tests."""

    @asynccontextmanager
    async def provider(request: KubernetesInvestigationRequest):
        yield tools

    return provider


def build_kubernetes_investigator(
    tools_provider: ToolsProvider,
    embedding_model: Any = None,
) -> Investigator:
    """Create an Investigator backed by the real Kubernetes agent."""

    async def run(
        request: KubernetesInvestigationRequest,
    ) -> KubernetesInvestigationResponse:
        async with tools_provider(request) as tools:
            validate_tool_allowlist(tools)
            return await investigate_kubernetes(
                request=request,
                tools=dict(tools),
                embedding_model=embedding_model,
                approved_actions=set(),
            )

    def investigate(
        request: KubernetesInvestigationRequest,
    ) -> KubernetesInvestigationResponse:
        # asyncio.run raises if this thread already has a running event
        # loop. The Supervisor's investigate node contains that as a
        # failed finding instead of crashing the incident.
        return asyncio.run(run(request))

    return make_kubernetes_investigator(investigate)