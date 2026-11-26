"""Adapters from existing agent contracts to platform state records."""

import json

from app.contracts.agent_messages import KubernetesInvestigationResponse
from app.state.models import Finding
from app.state.signals import CRASH_LOOP_SIGNAL, mentions_crash_loop

MAX_SUMMARY_CHARS = 500


def is_execution_failure(observation: str) -> bool:
    """True if an observation records a failed tool run, not a diagnostic."""
    try:
        payload = json.loads(observation)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("execution_failed") is True


def finding_from_kubernetes_response(
    response: KubernetesInvestigationResponse,
) -> Finding:
    """Convert the Kubernetes agent's response into a Finding record."""
    if response.findings:
        summary = "; ".join(response.findings)
    else:
        summary = (
            response.error_message or "Kubernetes agent returned no findings."
        )

    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 3] + "..."

    # The agent records failed tool runs as observations. They stay in the
    # finding for the audit trail but never count as diagnostic evidence.
    diagnostic = [
        item
        for item in response.evidence
        if not is_execution_failure(item.observation)
    ]
    quality = response.evidence_quality if diagnostic else "insufficient"

    # A signal needs a successful diagnostic behind it. Error messages and
    # failed tool runs never produce one, and neither does a failed or
    # blocked finding.
    signals = []
    if diagnostic and response.status not in ("failed", "blocked"):
        signal_text = " ".join(
            [summary if response.findings else ""]
            + [item.observation for item in diagnostic]
        )
        if mentions_crash_loop(signal_text):
            signals.append(CRASH_LOOP_SIGNAL)

    return Finding(
        agent_name="kubernetes",
        summary=summary,
        status=response.status,
        evidence=list(response.evidence),
        evidence_quality=quality,
        needs_follow_up=response.needs_follow_up,
        signals=signals,
        error_message=response.error_message,
    )