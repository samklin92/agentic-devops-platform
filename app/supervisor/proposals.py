"""Deterministic remediation proposals. Proposal only, never execution."""

from app.state.models import (
    Finding,
    IncidentContext,
    ProposedAction,
    RiskLevel,
)
from app.state.signals import CRASH_LOOP_SIGNAL
from app.supervisor.current_findings import current_findings

_PRODUCTION = {"production", "prod"}


def is_usable(finding: Finding) -> bool:
    """A finding may back a proposal only if it is complete and evidenced."""
    return (
        finding.status == "completed"
        and finding.evidence_quality != "insufficient"
        and not finding.needs_follow_up
    )


def propose_action(
    incident: IncidentContext,
    findings: list[Finding],
) -> ProposedAction | None:
    """Return a proposal if the evidence supports one, otherwise None.

    The gate decides on structured, validated evidence only, and only on
    the CURRENT Kubernetes finding (see `current_findings`): the latest
    attempt, never a superseded one, and never a specialist whose records
    are ambiguous. That finding must be usable and carry the `crash_loop`
    signal. Summaries, evidence text and error messages are never read, so
    untrusted text cannot create a proposal. Anything short of that fails
    closed.
    """
    if not incident.namespace or not incident.deployment:
        return None

    kubernetes = current_findings(findings).get("kubernetes")
    if (
        kubernetes is None
        or not is_usable(kubernetes)
        or CRASH_LOOP_SIGNAL not in kubernetes.signals
    ):
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
        supporting_finding_ids=[kubernetes.finding_id],
    )
