"""Helpers shared by the MCP-backed specialist investigators.

The Observability and Security investigators follow the same pattern:
resolve a workload target, call a fixed sequence of allowlisted
read-only MCP tools, contain each failure separately and turn the
results into a Finding. This module holds the parts that are identical,
so each specialist keeps only its own tools, thresholds and wording.

Nothing here knows which tools a specialist may call. Each specialist
passes its own allowlist, so the allowlists stay independent.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from app.state.models import IncidentContext

CallTool = Callable[[str, dict[str, Any]], Awaitable[Mapping[str, Any]]]
SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


class ToolNotAllowedError(ValueError):
    """A specialist tried to call a tool outside its allowlist."""


@dataclass(frozen=True)
class ToolResult:
    tool: str
    data: Mapping[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def resolve_workload_target(incident: IncidentContext) -> tuple[str, str] | None:
    """Return (namespace, service), or None if the incident lacks either.

    Target: `incident.service`, falling back to `incident.deployment`.
    The fallback is a Phase 1 convenience: in real infrastructure a
    Kubernetes Deployment name and a monitoring or security service
    identity are not necessarily the same. A namespace is also required.
    """
    namespace = (incident.namespace or "").strip()
    service = (incident.service or "").strip() or (
        incident.deployment or ""
    ).strip()
    if not namespace or not service:
        return None
    return namespace, service


def guard_call_tool(
    call_tool: CallTool, allowlist: frozenset[str], agent_label: str
) -> CallTool:
    """Wrap a tool caller so only allowlisted tools can be called."""

    async def guarded(name: str, arguments: dict[str, Any]):
        if name not in allowlist:
            raise ToolNotAllowedError(
                f"Tool not allowed for the {agent_label} agent: {name}"
            )
        return await call_tool(name, arguments)

    return guarded


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def validation_error(
    result: Any,
    required: Sequence[str],
    predicate: Callable[[Any], bool],
    kind: str,
) -> str | None:
    """Describe why a tool result is unusable, or None if it is valid.

    `kind` names the expected field type in the message ("numeric",
    "boolean"). `predicate` decides whether one field value is valid.
    """
    if not isinstance(result, Mapping):
        return "invalid response"
    if result.get("error"):
        return str(result["error"])
    missing = [field for field in required if not predicate(result.get(field))]
    if missing:
        return f"response missing {kind} fields: {missing}"
    return None


async def run_tool_sequence(
    call_tool: CallTool,
    calls: Sequence[tuple[str, dict[str, Any]]],
    validate: Callable[[str, Any], str | None],
) -> list[ToolResult]:
    """Run the calls in order. One failure never stops the others.

    `call_tool` must already be guarded. A ToolNotAllowedError is a
    programming error, not a tool failure, so it is never contained.
    """
    results: list[ToolResult] = []
    for tool, arguments in calls:
        try:
            data = await call_tool(tool, dict(arguments))
        except ToolNotAllowedError:
            raise
        except Exception as exc:  # contain per-tool failures
            results.append(ToolResult(tool=tool, error=str(exc)))
            continue
        error = validate(tool, data)
        if error is not None:
            results.append(ToolResult(tool=tool, error=error))
        else:
            results.append(ToolResult(tool=tool, data=data))
    return results


def session_call_tool(session: Any) -> CallTool:
    """Adapt an MCP client session into a CallTool returning structured data."""

    async def call(name: str, arguments: dict[str, Any]):
        response = await session.call_tool(name, arguments=arguments)
        if response.is_error:
            raise RuntimeError(f"MCP tool '{name}' returned an error.")
        content = response.structured_content
        if not isinstance(content, Mapping):
            raise RuntimeError(f"MCP tool '{name}' returned an invalid response.")
        return content

    return call