"""Deterministic remediation proposals. Proposal only, never execution."""

import re

from app.state.models import (
    Finding,
    IncidentContext,
    ProposedAction,
    RiskLevel,
)

_CRASH_LOOP = re.compile(r"\bcrash[\s-]?loop", re.IGNORECASE)
_PRODUCTION = {"production", "prod"}


def is_usable(finding: Finding) -> bool:
    """A finding may back a proposal only if it is complete and evidenced."""
    return (
        finding.status == "completed"
        and finding.evidence_quality != "insufficient"
        and not finding.needs_follow_up
    )


def _finding_text(finding: Finding) -> str:
    parts = [finding.summary] + [e.observation for e in finding.evidence]
    return " ".join(parts)


def propose_action(
    incident: IncidentContext,
    findings: list[Finding],
) -> ProposedAction | None:
    """Return a proposal if the evidence supports one, otherwise None."""
    if not incident.namespace or not incident.deployment:
        return None

    supporting = [
        finding
        for finding in findings
        if finding.agent_name == "kubernetes"
        and is_usable(finding)
        and _CRASH_LOOP.search(_finding_text(finding))
    ]
    if not supporting:
        return None

    risk: RiskLevel = (
        "medium" if incident.environment.lower() in _PRODUCTION else "low"
    )
    return ProposedAction(
        action_type="rollout_restart",
        namespace=incident.namespace,
        target=f"deployment/{incident.deployment}",
        rationale=(
            "Kubernetes evidence shows a crash loop; a rollout restart is "
            "proposed for human review."
        ),
        risk_level=risk,
        supporting_finding_ids=[f.finding_id for f in supporting],
    )