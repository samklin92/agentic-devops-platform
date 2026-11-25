"""Observability investigator: bounded, read-only metrics symptoms.

The investigator calls three allowlisted metrics tools in a fixed order,
records each result as evidence and reports which symptoms are present.
It never states a cause. Relating a symptom to a Kubernetes finding is
the Supervisor's job, not this specialist's.

Target: `incident.service`, falling back to `incident.deployment`. The
fallback is a Phase 1 convenience: in real infrastructure a Kubernetes
Deployment name and a monitoring service identity are not necessarily
the same. A namespace is also required.

Why a partial finding is not usable evidence: the evidence gate
(`is_usable`) requires status "completed". If one of the three metrics
could not be collected the picture is incomplete, so the finding is
"partial" and does not pass the gate, even though the metrics that were
collected are kept in the finding and the audit trail. This is
deliberately conservative for Phase 1.

Observability findings can support investigation and, later, evidence
correlation. They can never authorize a remediation: only an eligible
Kubernetes finding can support a rollout restart proposal.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from app.contracts.agent_messages import AgentEvidence
from app.state.adapters import MAX_SUMMARY_CHARS
from app.state.models import Finding, IncidentContext
from app.supervisor.investigators import Investigator

WINDOW_MINUTES = 15

# Thresholds are inclusive: a value at the threshold counts as a symptom.
ERROR_RATE_ELEVATED_PCT = 5.0
P95_ELEVATED_MS = 1000
CPU_SATURATED_PCT = 90
MEMORY_NEAR_LIMIT_PCT = 90

TOOL_ERROR_RATE = "get_error_rate"
TOOL_LATENCY = "get_latency_percentiles"
TOOL_RESOURCES = "get_resource_usage"

# The only MCP tools the Observability agent may ever call. All read-only.
OBSERVABILITY_TOOL_ALLOWLIST: frozenset[str] = frozenset(
    {TOOL_ERROR_RATE, TOOL_LATENCY, TOOL_RESOURCES}
)

# Fixed order. The investigation is exactly these three calls.
_SEQUENCE = (TOOL_ERROR_RATE, TOOL_LATENCY, TOOL_RESOURCES)

_REQUIRED_FIELDS = {
    TOOL_ERROR_RATE: ("error_rate_pct",),
    TOOL_LATENCY: ("p95_ms", "p99_ms"),
    TOOL_RESOURCES: ("cpu_pct_of_limit", "memory_pct_of_limit"),
}

CallTool = Callable[[str, dict[str, Any]], Awaitable[Mapping[str, Any]]]
SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]


class ToolNotAllowedError(ValueError):
    """The Observability agent tried to call a tool outside its allowlist."""


@dataclass(frozen=True)
class MetricResult:
    tool: str
    data: Mapping[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def resolve_target(incident: IncidentContext) -> tuple[str, str] | None:
    """Return (namespace, service), or None if the incident lacks either."""
    namespace = (incident.namespace or "").strip()
    service = (incident.service or "").strip() or (
        incident.deployment or ""
    ).strip()
    if not namespace or not service:
        return None
    return namespace, service


def guard_call_tool(call_tool: CallTool) -> CallTool:
    """Wrap a tool caller so only allowlisted tools can be called."""

    async def guarded(name: str, arguments: dict[str, Any]):
        if name not in OBSERVABILITY_TOOL_ALLOWLIST:
            raise ToolNotAllowedError(
                f"Tool not allowed for the Observability agent: {name}"
            )
        return await call_tool(name, arguments)

    return guarded


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validation_error(tool: str, result: Any) -> str | None:
    if not isinstance(result, Mapping):
        return "invalid response"
    if result.get("error"):
        return str(result["error"])
    missing = [
        field
        for field in _REQUIRED_FIELDS[tool]
        if not _is_number(result.get(field))
    ]
    if missing:
        return f"response missing numeric fields: {missing}"
    return None


async def collect_metrics(
    call_tool: CallTool, namespace: str, service: str
) -> list[MetricResult]:
    """Run the three metrics calls. One failure never stops the others."""
    call_tool = guard_call_tool(call_tool)
    results: list[MetricResult] = []
    for tool in _SEQUENCE:
        arguments: dict[str, Any] = {
            "service": service,
            "namespace": namespace,
        }
        if tool != TOOL_RESOURCES:
            arguments["window_minutes"] = WINDOW_MINUTES
        try:
            data = await call_tool(tool, arguments)
        except ToolNotAllowedError:
            raise
        except Exception as exc:  # contain per-tool failures
            results.append(MetricResult(tool=tool, error=str(exc)))
            continue
        error = _validation_error(tool, data)
        if error is not None:
            results.append(MetricResult(tool=tool, error=error))
        else:
            results.append(MetricResult(tool=tool, data=data))
    return results


def _observation(result: MetricResult) -> str:
    if not result.ok:
        return json.dumps(
            {
                "execution_failed": True,
                "tool": result.tool,
                "error": result.error,
            }
        )
    data = result.data
    if result.tool == TOOL_ERROR_RATE:
        return (
            f"error_rate={data['error_rate_pct']}% over {WINDOW_MINUTES}m"
        )
    if result.tool == TOOL_LATENCY:
        return (
            f"latency p95={data['p95_ms']}ms p99={data['p99_ms']}ms "
            f"over {WINDOW_MINUTES}m"
        )
    return (
        f"cpu={data['cpu_pct_of_limit']}% "
        f"memory={data['memory_pct_of_limit']}% of limits"
    )


def _describe(results: list[MetricResult]) -> str:
    """Name the symptoms present in the metrics that were collected."""
    data = {r.tool: r.data for r in results if r.ok}
    abnormal: list[str] = []

    errors = data.get(TOOL_ERROR_RATE)
    if errors is not None and errors["error_rate_pct"] >= ERROR_RATE_ELEVATED_PCT:
        abnormal.append("errors elevated")

    latency = data.get(TOOL_LATENCY)
    if latency is not None and latency["p95_ms"] >= P95_ELEVATED_MS:
        abnormal.append("latency elevated")

    resources = data.get(TOOL_RESOURCES)
    resource_flags: list[str] = []
    if resources is not None:
        if resources["cpu_pct_of_limit"] >= CPU_SATURATED_PCT:
            resource_flags.append("CPU saturated")
        if resources["memory_pct_of_limit"] >= MEMORY_NEAR_LIMIT_PCT:
            resource_flags.append("memory near limit")
    abnormal.extend(resource_flags)

    parts = list(abnormal)
    if resources is not None and not resource_flags and abnormal:
        parts.append("resources normal")

    if parts:
        text = ", ".join(parts)
    elif len(data) == len(_SEQUENCE):
        text = "metrics within normal thresholds"
    else:
        text = "no elevated symptoms in the metrics that were collected"
    return text[0].upper() + text[1:] + "."


def finding_from_metrics(results: list[MetricResult]) -> Finding:
    """Build the Observability finding from the collected metric results."""
    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    summary = _describe(results) if succeeded else "No metrics could be collected."
    if failed:
        summary += " Unavailable: " + ", ".join(r.tool for r in failed) + "."
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 3] + "..."

    if not succeeded:
        status, quality = "failed", "insufficient"
    elif failed:
        status, quality = "partial", "limited"
    else:
        status, quality = "completed", "limited"

    error_message = (
        "; ".join(f"{r.tool}: {r.error}" for r in failed) or None
    )
    return Finding(
        agent_name="observability",
        summary=summary,
        status=status,
        evidence=[
            AgentEvidence(source_tool=r.tool, observation=_observation(r))
            for r in results
        ],
        evidence_quality=quality,
        error_message=error_message,
    )


def _blocked_finding() -> Finding:
    return Finding(
        agent_name="observability",
        summary=(
            "Incident does not identify a namespace and a service; "
            "cannot investigate."
        ),
        status="blocked",
        error_message="Missing namespace or service/deployment.",
    )


def _session_call_tool(session: Any) -> CallTool:
    async def call(name: str, arguments: dict[str, Any]):
        response = await session.call_tool(name, arguments=arguments)
        if response.is_error:
            raise RuntimeError(f"MCP tool '{name}' returned an error.")
        content = response.structured_content
        if not isinstance(content, Mapping):
            raise RuntimeError(f"MCP tool '{name}' returned an invalid response.")
        return content

    return call


def make_observability_investigator(
    session_factory: SessionFactory,
) -> Investigator:
    """Create an Investigator that opens one MCP session per investigation."""

    async def run(namespace: str, service: str) -> list[MetricResult]:
        async with session_factory() as session:
            return await collect_metrics(
                _session_call_tool(session), namespace, service
            )

    def investigator(incident: IncidentContext) -> Finding:
        target = resolve_target(incident)
        if target is None:
            return _blocked_finding()
        namespace, service = target
        # asyncio.run raises if this thread already has a running event
        # loop. The Supervisor's investigate node contains that as a
        # failed finding instead of crashing the incident.
        return finding_from_metrics(asyncio.run(run(namespace, service)))

    return investigator