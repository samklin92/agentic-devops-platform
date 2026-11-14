from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentContract(BaseModel):
    """Base configuration shared by agent communication models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class KubernetesTarget(AgentContract):
    """Identifies the Kubernetes resource to investigate."""

    namespace: str = Field(min_length=1)
    pod_name: str = Field(min_length=1)


class KubernetesInvestigationRequest(AgentContract):
    """Investigation request sent from the Supervisor."""

    incident_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    target: KubernetesTarget
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class AgentEvidence(AgentContract):
    """A diagnostic observation with its originating tool."""

    source_tool: str = Field(min_length=1)
    observation: str = Field(min_length=1)


class KubernetesInvestigationResponse(AgentContract):
    """Structured investigation findings returned to the Supervisor."""

    incident_id: str = Field(min_length=1)

    agent_name: Literal["kubernetes"] = "kubernetes"

    status: Literal[
        "completed",
        "partial",
        "failed",
        "blocked",
        "needs_review",
    ]

    findings: list[str] = Field(default_factory=list)

    evidence: list[AgentEvidence] = Field(default_factory=list)

    evidence_quality: Literal[
        "insufficient",
        "limited",
        "corroborated",
    ] = "insufficient"

    needs_follow_up: bool = False

    error_message: str | None = None