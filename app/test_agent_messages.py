import pytest
from pydantic import ValidationError

from app.contracts.agent_messages import (
    AgentEvidence,
    KubernetesInvestigationRequest,
    KubernetesInvestigationResponse,
    KubernetesTarget,
)

def test_valid_kubernetes_request():
    request = KubernetesInvestigationRequest(
        incident_id="INC-001",
        task="Investigate pod failure",
        target={
            "namespace": "production",
            "pod_name": "inventory-api",
        },
        priority="high",
    )

    assert request.incident_id == "INC-001"
    assert request.target.pod_name == "inventory-api"


def test_valid_kubernetes_response():
    response = KubernetesInvestigationResponse(
        incident_id="INC-001",
        status="completed",
        findings=["Pod is in CrashLoopBackOff"],
        evidence=[
            AgentEvidence(
                source_tool="inspect_pod",
                observation="Pod status is CrashLoopBackOff",
            ),
            AgentEvidence(
                source_tool="get_container_logs",
                observation="Database connection refused",
            ),
        ],
        evidence_quality="corroborated",
        needs_follow_up=False,
    )

    assert response.incident_id == "INC-001"
    assert response.agent_name == "kubernetes"
    assert response.status == "completed"
    assert len(response.findings) == 1
    assert len(response.evidence) == 2
    assert response.evidence[0].source_tool == "inspect_pod"
    assert response.evidence_quality == "corroborated"
    assert response.needs_follow_up is False


def test_request_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        KubernetesInvestigationRequest(
            incident_id="INC-001",
            task="Investigate pod failure",
            target={
                "namespace": "production",
                "pod_name": "inventory-api",
            },
            priority="urgent",
        )


def test_response_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        KubernetesInvestigationResponse(
            incident_id="INC-001",
            status="completed",
            confidence=1.5,
        )


def test_request_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        KubernetesInvestigationRequest(
            incident_id="INC-001",
            task="Investigate pod failure",
            target={
                "namespace": "production",
                "pod_name": "inventory-api",
            },
            unexpected_field="not allowed",
        )


def test_response_rejects_invalid_status():
    with pytest.raises(ValidationError):
        KubernetesInvestigationResponse(
            incident_id="INC-001",
            status="unknown",
            confidence=0.8,
        )