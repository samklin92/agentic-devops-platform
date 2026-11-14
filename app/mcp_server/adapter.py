"""Adapter connecting the Kubernetes agent to MCP tools."""

from typing import Any


def _validate_response(response: Any, tool_name: str) -> dict:
    """Validate MCP transport and application-level responses."""

    if response.is_error:
        raise RuntimeError(
            f"MCP tool '{tool_name}' returned an error."
        )

    content = response.structured_content

    if not isinstance(content, dict):
        raise RuntimeError(
            f"MCP tool '{tool_name}' returned an invalid response."
        )

    if content.get("error"):
        raise RuntimeError(
            f"MCP tool '{tool_name}': {content['error']}"
        )

    return content


def build_mcp_tools(
    session: Any,
    pod_name: str,
    namespace: str,
) -> dict:
    """Build namespace-aware diagnostic tools for the Kubernetes agent."""

    async def inspect_pod_events() -> dict:
        response = await session.call_tool(
            "get_pod_events",
            arguments={
                "pod_name": pod_name,
                "namespace": namespace,
            },
        )

        return _validate_response(
            response,
            "get_pod_events",
        )

    async def gather_additional_diagnostics() -> dict:
        pod_response = await session.call_tool(
            "inspect_pod",
            arguments={
                "pod_name": pod_name,
                "namespace": namespace,
            },
        )

        logs_response = await session.call_tool(
            "get_container_logs",
            arguments={
                "pod_name": pod_name,
                "namespace": namespace,
            },
        )

        deployment_response = await session.call_tool(
            "check_deployment_status",
            arguments={
                "deployment_name": pod_name,
                "namespace": namespace,
            },
        )

        pod = _validate_response(
            pod_response,
            "inspect_pod",
        )

        logs = _validate_response(
            logs_response,
            "get_container_logs",
        )

        deployment = _validate_response(
            deployment_response,
            "check_deployment_status",
        )

        return {
            "pod": pod,
            "logs": logs,
            "deployment": deployment,
            "terminal": True,
        }

    return {
        "inspect_pod_events": inspect_pod_events,
        "gather_additional_diagnostics": gather_additional_diagnostics,
    }