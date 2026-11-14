"""Wrap concrete specialist agents into the Supervisor's Investigator form."""

from collections.abc import Callable

from app.contracts.agent_messages import (
    KubernetesInvestigationRequest,
    KubernetesInvestigationResponse,
    KubernetesTarget,
)
from app.state.adapters import finding_from_kubernetes_response
from app.state.models import Finding, IncidentContext

Investigator = Callable[[IncidentContext], Finding]


def make_kubernetes_investigator(
    investigate: Callable[
        [KubernetesInvestigationRequest], KubernetesInvestigationResponse
    ],
) -> Investigator:
    """Adapt a Kubernetes investigation function into an Investigator."""

    def investigator(incident: IncidentContext) -> Finding:
        if not incident.namespace or not incident.pod_name:
            return Finding(
                agent_name="kubernetes",
                summary=(
                    "Incident does not identify a namespace and pod; "
                    "cannot investigate."
                ),
                status="blocked",
                error_message="Missing namespace or pod_name.",
            )

        request = KubernetesInvestigationRequest(
            incident_id=incident.incident_id,
            task=incident.description,
            target=KubernetesTarget(
                namespace=incident.namespace,
                pod_name=incident.pod_name,
            ),
            priority=incident.severity,
        )
        return finding_from_kubernetes_response(investigate(request))

    return investigator