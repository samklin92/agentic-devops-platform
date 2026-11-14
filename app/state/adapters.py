"""Adapters from existing agent contracts to platform state records."""

from app.contracts.agent_messages import KubernetesInvestigationResponse
from app.state.models import Finding

MAX_SUMMARY_CHARS = 500


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

    # Never claim evidence quality without evidence.
    quality = response.evidence_quality if response.evidence else "insufficient"

    return Finding(
        agent_name="kubernetes",
        summary=summary,
        status=response.status,
        evidence=list(response.evidence),
        evidence_quality=quality,
        needs_follow_up=response.needs_follow_up,
        error_message=response.error_message,
    )