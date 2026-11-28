"""The Supervisor graph: routing, investigation, correlation, approval and
reporting.

Flow: receive_alert, route, investigate, correlate, evaluate, then
approval, execute and report. Correlation is a Supervisor step. It records
how findings relate and never decides, changes or authorizes an action:
the proposal gate (`propose_action`) alone decides eligibility, and a
human approves.
"""

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.state.graph_state import PlatformState, check_call_budget
from app.state.models import (
    ApprovalRecord,
    AuditEntry,
    Correlation,
    ExecutionRecord,
    Finding,
    IncidentReport,
    SpecialistName,
    UsageRecord,
)
from app.supervisor.correlation import annotate_action
from app.supervisor.correlation import correlate as compute_correlations
from app.supervisor.investigators import Investigator
from app.supervisor.proposals import is_usable, propose_action
from app.supervisor.routing import route_incident
from app.usage import track_usage

MAX_REPORT_CHARS = 800


def audit(
    state: PlatformState, actor: str, event: str, detail: str = ""
) -> AuditEntry:
    return AuditEntry(
        incident_id=state["incident"].incident_id,
        actor=actor,
        event=event,
        detail=detail,
    )


def parse_decision(
    raw: Any, pending_action_id: str
) -> tuple[bool, str, str | None]:
    """Validate a resume payload against the pending action.

    Fails closed on anything malformed. The decision must carry the exact
    action_id of the pending proposal: a missing, blank, non-string or
    different ID is rejected, so a decision can never be applied to an
    action it was not made for. Error messages never echo the supplied
    value.
    """
    if not isinstance(raw, dict):
        raise ValueError("Approval decision must be a dict.")
    approved = raw.get("approved")
    decided_by = raw.get("decided_by")
    if not isinstance(approved, bool):
        raise ValueError("Approval decision needs a boolean 'approved'.")
    if not isinstance(decided_by, str) or not decided_by.strip():
        raise ValueError("Approval decision needs 'decided_by'.")
    action_id = raw.get("action_id")
    if not isinstance(action_id, str) or not action_id.strip():
        raise ValueError("Approval decision needs 'action_id'.")
    if action_id != pending_action_id:
        raise ValueError(
            "Approval decision action_id does not match the pending action."
        )
    reason = raw.get("reason")
    return approved, decided_by.strip(), reason if isinstance(reason, str) else None


def _blocked_finding(agent: SpecialistName, reason: str) -> Finding:
    return Finding(
        agent_name=agent,
        summary=f"{agent} agent did not run: {reason}",
        status="blocked",
        error_message=reason,
    )


def current_correlations(state: PlatformState) -> list[Correlation]:
    """The correlations in state.

    Tolerates a checkpoint written before the field existed, which simply
    has none.
    """
    return state.get("correlations") or []


def _relation_label(
    correlation: Correlation, findings: list[Finding]
) -> str:
    agent_of = {f.finding_id: f.agent_name for f in findings}
    agents = " + ".join(
        agent_of.get(finding_id, "unknown")
        for finding_id in correlation.finding_ids
    )
    return f"{correlation.kind.replace('_', ' ')} ({agents}, {correlation.rule})"


def _correlation_clause(
    correlations: list[Correlation], findings: list[Finding]
) -> str:
    if not correlations:
        return ""
    labels = "; ".join(_relation_label(c, findings) for c in correlations)
    return f" Evidence relationships: {labels}."


def _digest(findings: list[Finding]) -> str:
    text = "; ".join(f"{f.agent_name}: {f.summary}" for f in findings)
    if len(text) > MAX_REPORT_CHARS:
        text = text[: MAX_REPORT_CHARS - 3] + "..."
    return text


def build_supervisor_graph(
    investigators: Mapping[SpecialistName, Investigator],
    checkpointer: Any = None,
):
    """Compile the Supervisor graph with the given specialist investigators."""

    def receive_alert(state: PlatformState) -> dict:
        return {
            "status": "received",
            "supervisor_hops": 1,
            "audit_log": [
                audit(state, "supervisor", "alert_received",
                      state["incident"].description)
            ],
        }

    def route(state: PlatformState) -> dict:
        decision = route_incident(state["incident"])
        if decision is None:
            return {
                "routing": None,
                "supervisor_hops": 1,
                "audit_log": [
                    audit(state, "supervisor", "no_specialist_matched")
                ],
            }
        return {
            "routing": decision,
            "status": "investigating",
            "supervisor_hops": 1,
            "audit_log": [audit(state, "supervisor", "routed", decision.reason)],
        }

    def after_route(state: PlatformState) -> str:
        return "investigate" if state["routing"] is not None else "report"

    def investigate(state: PlatformState) -> dict:
        incident = state["incident"]
        counts = dict(state["agent_call_counts"])
        new_counts: dict[str, int] = {}
        findings: list[Finding] = []
        entries: list[AuditEntry] = []
        usage: list[UsageRecord] = []

        for agent in state["routing"].selected_agents:
            investigator = investigators.get(agent)
            if investigator is None:
                findings.append(_blocked_finding(agent, "no investigator registered"))
                entries.append(audit(state, "supervisor", "specialist_unavailable", agent))
                continue

            reason = check_call_budget({**state, "agent_call_counts": counts}, agent)
            if reason:
                findings.append(_blocked_finding(agent, reason))
                entries.append(audit(state, "supervisor", "call_budget_exhausted", reason))
                continue

            counts[agent] = counts.get(agent, 0) + 1
            new_counts[agent] = 1
            # Usage is observation only: the collector records what the
            # specialist spends, including when it fails, and nothing here
            # reads it back to decide anything.
            with track_usage(incident.incident_id, agent) as collector:
                try:
                    finding = investigator(incident)
                    event = f"{agent}_investigated"
                except Exception as exc:  # contain specialist failures
                    finding = Finding(
                        agent_name=agent,
                        summary=f"{agent} investigation failed.",
                        status="failed",
                        error_message=str(exc),
                    )
                    event = f"{agent}_failed"
            usage.extend(collector.records)
            findings.append(finding)
            entries.append(audit(state, agent, event))
            if collector.dropped or collector.warnings:
                entries.append(
                    audit(
                        state,
                        "supervisor",
                        "usage_not_recorded",
                        f"{agent}: {collector.dropped} call(s) not recorded, "
                        f"{len(collector.warnings)} warning(s)",
                    )
                )

        return {
            "findings": findings,
            "agent_call_counts": new_counts,
            "usage": usage,
            "audit_log": entries,
        }

    def correlate(state: PlatformState) -> dict:
        # Recomputed from the current findings every time, never
        # accumulated, so a repeated investigation leaves nothing stale.
        findings = state["findings"]
        correlations = compute_correlations(findings)
        return {
            "correlations": correlations,
            "supervisor_hops": 1,
            "audit_log": [
                audit(
                    state,
                    "supervisor",
                    "correlation_recorded",
                    _relation_label(c, findings),
                )
                for c in correlations
            ],
        }

    def evaluate(state: PlatformState) -> dict:
        # The gate decides on the findings alone. Correlations may only
        # add explanatory text to a proposal that already exists.
        proposed = propose_action(state["incident"], state["findings"])
        action = annotate_action(
            proposed, current_correlations(state), state["findings"]
        )
        if action is None:
            usable = any(is_usable(f) for f in state["findings"])
            detail = (
                "Evidence does not support an automated remediation proposal."
                if usable
                else "No usable evidence from any specialist."
            )
            return {
                "supervisor_hops": 1,
                "audit_log": [audit(state, "supervisor", "no_action_proposed", detail)],
            }
        return {
            "proposed_action": action,
            "status": "awaiting_approval",
            "supervisor_hops": 1,
            "audit_log": [
                audit(state, "supervisor", "action_proposed", action.rationale)
            ],
        }

    def after_evaluate(state: PlatformState) -> str:
        return "approval" if state["proposed_action"] is not None else "report"

    def approval(state: PlatformState) -> dict:
        action = state["proposed_action"]
        # This node re-runs from the top on resume: no side effects before
        # interrupt().
        raw = interrupt({
            "action_id": action.action_id,
            "action_type": action.action_type,
            "target": action.target,
            "risk_level": action.risk_level,
            "rationale": action.rationale,
            "correlations": [
                {
                    "kind": c.kind,
                    "rule": c.rule,
                    "finding_ids": list(c.finding_ids),
                    "explanation": c.explanation,
                }
                for c in current_correlations(state)
            ],
        })
        # The decision must name this exact action; anything else raises
        # here, before any record or audit entry exists (fails closed).
        approved, decided_by, reason = parse_decision(raw, action.action_id)
        record = ApprovalRecord(
            action_id=action.action_id,
            status="approved" if approved else "rejected",
            decided_by=decided_by,
            reason=reason,
        )
        event = "approval_granted" if approved else "approval_rejected"
        return {
            "approval": record,
            "audit_log": [audit(state, decided_by, event, reason or "")],
        }

    def after_approval(state: PlatformState) -> str:
        return "execute" if state["approval"].status == "approved" else "report"

    def execute(state: PlatformState) -> dict:
        action = state["proposed_action"]
        record = ExecutionRecord(
            action_id=action.action_id,
            status="simulated",
            message=f"Simulated {action.action_type} on {action.target}.",
        )
        return {
            "execution": record,
            "status": "executing",
            "audit_log": [
                audit(state, "supervisor", "execution_simulated", record.message)
            ],
        }

    def report(state: PlatformState) -> dict:
        findings = state["findings"]
        action = state["proposed_action"]
        approval_record = state["approval"]
        clause = _correlation_clause(current_correlations(state), findings)

        if state["routing"] is None:
            outcome = "no_action_proposed"
            summary = (
                "No specialist applies to this incident; "
                "escalate to a human engineer."
            )
        elif state["execution"] is not None:
            outcome = "action_executed_simulated"
            summary = (
                f"{state['execution'].message} "
                f"Findings: {_digest(findings)}{clause}"
            )
        elif approval_record is not None and approval_record.status == "rejected":
            outcome = "action_rejected"
            summary = (
                "Proposed action was rejected. "
                f"Findings: {_digest(findings)}{clause}"
            )
        elif action is None and any(is_usable(f) for f in findings):
            outcome = "no_action_proposed"
            summary = (
                "Evidence gathered but no automated remediation applies. "
                f"Findings: {_digest(findings)}{clause}"
            )
        else:
            outcome = "investigation_failed"
            summary = (
                "Evidence was insufficient for a remediation proposal; "
                f"human review needed. Findings: {_digest(findings)}"
            )

        final = IncidentReport(
            incident_id=state["incident"].incident_id,
            outcome=outcome,
            summary=summary,
            finding_ids=[f.finding_id for f in findings],
            action_id=action.action_id if action else None,
            approval_status=approval_record.status if approval_record else None,
        )
        return {
            "report": final,
            "status": "failed" if outcome == "investigation_failed" else "completed",
            "supervisor_hops": 1,
            "audit_log": [audit(state, "supervisor", "report_generated", outcome)],
        }

    builder = StateGraph(PlatformState)
    for name, node in [
        ("receive_alert", receive_alert),
        ("route", route),
        ("investigate", investigate),
        ("correlate", correlate),
        ("evaluate", evaluate),
        ("approval", approval),
        ("execute", execute),
        ("report", report),
    ]:
        builder.add_node(name, node)

    builder.add_edge(START, "receive_alert")
    builder.add_edge("receive_alert", "route")
    builder.add_conditional_edges(
        "route", after_route, {"investigate": "investigate", "report": "report"}
    )
    builder.add_edge("investigate", "correlate")
    builder.add_edge("correlate", "evaluate")
    builder.add_conditional_edges(
        "evaluate", after_evaluate, {"approval": "approval", "report": "report"}
    )
    builder.add_conditional_edges(
        "approval", after_approval, {"execute": "execute", "report": "report"}
    )
    builder.add_edge("execute", "report")
    builder.add_edge("report", END)
    return builder.compile(checkpointer=checkpointer)