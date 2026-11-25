"""Security investigator: bounded, read-only security signals.

The investigator calls three allowlisted security tools in a fixed order,
records each result as evidence and reports which signals are present.
It states observed conditions only, never a cause, an intent or a
remedy. Deciding what a signal means, and relating it to a Kubernetes
finding, is the Supervisor's job, not this specialist's.

Target: `incident.service`, falling back to `incident.deployment`, with
a namespace required (shared with Observability). Incidents that name no
workload, such as a leaked key in a repository, produce a blocked
finding. That is a known Phase 1 limitation, not a routing failure.

Absence of a signal is not safety. The clean summary says only that no
elevated signals were found in the checks that ran, never that the
service is secure.

Why a partial finding is not usable evidence: the evidence gate
(`is_usable`) requires status "completed". If one of the three checks
could not run, the picture is incomplete (a missing exposure check could
hide a hole), so the finding is "partial" and does not pass the gate,
even though the results that were collected stay in the finding and the
audit trail.

Evidence text is built only from validated numeric and boolean fields,
never from free text returned by a tool, so tool output cannot inject
words into a finding.

Security findings can support investigation and, later, evidence
correlation. They can never authorize a remediation: only an eligible
Kubernetes finding can support a rollout restart proposal.
"""

import asyncio
import json
from typing import Any

from app.contracts.agent_messages import AgentEvidence
from app.state.adapters import MAX_SUMMARY_CHARS
from app.state.models import Finding, IncidentContext
from app.supervisor.investigators import Investigator
from app.supervisor.specialist_common import (
    CallTool,
    SessionFactory,
    ToolNotAllowedError,  # noqa: F401  (re-exported for callers and tests)
    ToolResult,
    guard_call_tool as _guard_call_tool,
    is_bool,
    is_number,
    resolve_workload_target,
    run_tool_sequence,
    session_call_tool,
    validation_error,
)

CheckResult = ToolResult
resolve_target = resolve_workload_target

WINDOW_MINUTES = 15

# Thresholds are inclusive: a value at the threshold counts as a signal.
FAILED_AUTH_ELEVATED = 100
ACCESS_DENIED_ELEVATED = 10
CREDENTIAL_AGE_STALE_DAYS = 90

TOOL_FAILED_AUTH = "get_failed_auth_attempts"
TOOL_IAM = "get_iam_activity"
TOOL_EXPOSURE = "check_network_exposure"

# The only MCP tools the Security agent may ever call. All read-only.
SECURITY_TOOL_ALLOWLIST: frozenset[str] = frozenset(
    {TOOL_FAILED_AUTH, TOOL_IAM, TOOL_EXPOSURE}
)

# Fixed order. The investigation is exactly these three calls.
_SEQUENCE = (TOOL_FAILED_AUTH, TOOL_IAM, TOOL_EXPOSURE)

_NUMERIC_FIELDS = {
    TOOL_FAILED_AUTH: ("failed_auth_count", "distinct_source_ips"),
    TOOL_IAM: ("access_denied_calls", "credential_age_days"),
}
_BOOLEAN_FIELDS = {
    TOOL_EXPOSURE: ("publicly_exposed", "open_ingress_cidr"),
}


def guard_call_tool(call_tool: CallTool) -> CallTool:
    """Wrap a tool caller so only Security tools can be called."""
    return _guard_call_tool(call_tool, SECURITY_TOOL_ALLOWLIST, "Security")


def _validation_error(tool: str, result: Any) -> str | None:
    if tool in _BOOLEAN_FIELDS:
        return validation_error(
            result, _BOOLEAN_FIELDS[tool], is_bool, "boolean"
        )
    return validation_error(result, _NUMERIC_FIELDS[tool], is_number, "numeric")


async def collect_checks(
    call_tool: CallTool, namespace: str, service: str
) -> list[CheckResult]:
    """Run the three security calls. One failure never stops the others."""
    calls: list[tuple[str, dict[str, Any]]] = []
    for tool in _SEQUENCE:
        arguments: dict[str, Any] = {
            "service": service,
            "namespace": namespace,
        }
        if tool != TOOL_EXPOSURE:
            arguments["window_minutes"] = WINDOW_MINUTES
        calls.append((tool, arguments))
    return await run_tool_sequence(
        guard_call_tool(call_tool), calls, _validation_error
    )


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _observation(result: CheckResult) -> str:
    if not result.ok:
        return json.dumps(
            {
                "execution_failed": True,
                "tool": result.tool,
                "error": result.error,
            }
        )
    data = result.data
    if result.tool == TOOL_FAILED_AUTH:
        return (
            f"failed_auth={data['failed_auth_count']} from "
            f"{data['distinct_source_ips']} source IPs over {WINDOW_MINUTES}m"
        )
    if result.tool == TOOL_IAM:
        return (
            f"iam access_denied={data['access_denied_calls']} over "
            f"{WINDOW_MINUTES}m credential_age={data['credential_age_days']}d"
        )
    return (
        f"publicly_exposed={_flag(data['publicly_exposed'])} "
        f"open_ingress_0.0.0.0/0={_flag(data['open_ingress_cidr'])}"
    )


def _describe(results: list[CheckResult]) -> str:
    """Name the signals present in the checks that were collected."""
    data = {r.tool: r.data for r in results if r.ok}
    signals: list[str] = []

    auth = data.get(TOOL_FAILED_AUTH)
    if auth is not None and auth["failed_auth_count"] >= FAILED_AUTH_ELEVATED:
        signals.append("failed authentication elevated")

    iam = data.get(TOOL_IAM)
    if iam is not None:
        if iam["access_denied_calls"] >= ACCESS_DENIED_ELEVATED:
            signals.append("IAM access-denied calls elevated")
        if iam["credential_age_days"] >= CREDENTIAL_AGE_STALE_DAYS:
            signals.append(
                f"credential age at or above {CREDENTIAL_AGE_STALE_DAYS} days"
            )

    exposure = data.get(TOOL_EXPOSURE)
    if exposure is not None:
        if exposure["publicly_exposed"]:
            signals.append("publicly exposed")
        if exposure["open_ingress_cidr"]:
            signals.append("ingress open to 0.0.0.0/0")

    if signals:
        text = ", ".join(signals)
    elif len(data) == len(_SEQUENCE):
        text = "no elevated security signals in the checks run"
    else:
        text = "no elevated security signals in the checks that were collected"
    return text[0].upper() + text[1:] + "."


def finding_from_checks(results: list[CheckResult]) -> Finding:
    """Build the Security finding from the collected check results."""
    succeeded = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    summary = (
        _describe(results) if succeeded else "No security checks could be collected."
    )
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
        agent_name="security",
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
        agent_name="security",
        summary=(
            "Incident does not identify a namespace and a service; "
            "cannot investigate."
        ),
        status="blocked",
        error_message="Missing namespace or service/deployment.",
    )


def make_security_investigator(
    session_factory: SessionFactory,
) -> Investigator:
    """Create an Investigator that opens one MCP session per investigation."""

    async def run(namespace: str, service: str) -> list[CheckResult]:
        async with session_factory() as session:
            return await collect_checks(
                session_call_tool(session), namespace, service
            )

    def investigator(incident: IncidentContext) -> Finding:
        target = resolve_target(incident)
        if target is None:
            return _blocked_finding()
        namespace, service = target
        # asyncio.run raises if this thread already has a running event
        # loop. The Supervisor's investigate node contains that as a
        # failed finding instead of crashing the incident.
        return finding_from_checks(asyncio.run(run(namespace, service)))

    return investigator
