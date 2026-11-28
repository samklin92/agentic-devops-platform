"""Top-level LangGraph state for the Agentic DevOps Platform."""

import operator
from typing import Annotated, Literal, TypedDict

from app.state.models import (
    ApprovalRecord,
    AuditEntry,
    CallLimits,
    Correlation,
    ExecutionRecord,
    Finding,
    IncidentContext,
    IncidentReport,
    ProposedAction,
    RoutingDecision,
    UsageRecord,
)

IncidentStatus = Literal[
    "received",
    "investigating",
    "awaiting_approval",
    "executing",
    "completed",
    "failed",
]


def merge_counts(
    left: dict[str, int] | None,
    right: dict[str, int] | None,
) -> dict[str, int]:
    """Reducer: sum counters key by key."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = merged.get(key, 0) + value
    return merged


class PlatformState(TypedDict):
    """Shared state. Only the Supervisor writes the last-write-wins fields."""

    incident: IncidentContext
    status: IncidentStatus
    limits: CallLimits
    routing: RoutingDecision | None
    proposed_action: ProposedAction | None
    approval: ApprovalRecord | None
    execution: ExecutionRecord | None
    report: IncidentReport | None
    # Written only by the Supervisor, with no reducer. Recomputed from the
    # current findings every time, never accumulated, so a repeated
    # investigation cannot leave stale relationships behind.
    correlations: list[Correlation]

    # Safe for specialists to write: reducers merge concurrent updates.
    findings: Annotated[list[Finding], operator.add]
    usage: Annotated[list[UsageRecord], operator.add]
    audit_log: Annotated[list[AuditEntry], operator.add]
    agent_call_counts: Annotated[dict[str, int], merge_counts]
    supervisor_hops: Annotated[int, operator.add]


def new_platform_state(
    incident: IncidentContext,
    limits: CallLimits | None = None,
) -> PlatformState:
    """Build the initial state for a new incident."""
    return {
        "incident": incident,
        "status": "received",
        "limits": limits or CallLimits(),
        "routing": None,
        "proposed_action": None,
        "approval": None,
        "execution": None,
        "report": None,
        "correlations": [],
        "findings": [],
        "usage": [],
        "audit_log": [],
        "agent_call_counts": {},
        "supervisor_hops": 0,
    }


def check_call_budget(state: PlatformState, agent_name: str) -> str | None:
    """Return a reason if the agent may not make another call, else None."""
    limits = state["limits"]
    counts = state["agent_call_counts"]

    agent_calls = counts.get(agent_name, 0)
    if agent_calls >= limits.max_calls_per_agent:
        return f"Agent '{agent_name}' reached its call limit ({agent_calls})."

    incident_calls = sum(counts.values())
    if incident_calls >= limits.max_calls_per_incident:
        return f"Incident call limit reached ({incident_calls})."

    return None


def supervisor_hops_exhausted(state: PlatformState) -> bool:
    return state["supervisor_hops"] >= state["limits"].max_supervisor_hops


def summarize_usage(
    records: list[UsageRecord],
) -> dict[str, dict[str, float | int]]:
    """Aggregate LLM usage and estimated cost per agent.

    `estimated_cost_usd` sums only the calls that have a price, so when
    `unpriced_calls` is above zero it is a lower bound and not a total.
    A call with no price is counted, never treated as free.
    """
    summary: dict[str, dict[str, float | int]] = {}
    for record in records:
        entry = summary.setdefault(
            record.agent_name,
            {
                "calls": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "unpriced_calls": 0,
            },
        )
        entry["calls"] += 1
        entry["total_tokens"] += record.total_tokens
        if record.estimated_cost_usd is None:
            entry["unpriced_calls"] += 1
        else:
            entry["estimated_cost_usd"] += record.estimated_cost_usd
    return summary