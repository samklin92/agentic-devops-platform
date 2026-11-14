"""Typed records for the shared state of the Agentic DevOps Platform.

These models are the platform's common language: the Supervisor,
specialist agents, approval flow and audit trail all exchange
information as these records.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from app.contracts.agent_messages import AgentContract, AgentEvidence

Severity = Literal["low", "medium", "high", "critical"]
RiskLevel = Literal["low", "medium", "high", "critical"]
SpecialistName = Literal["kubernetes", "observability", "security"]
AgentName = Literal["supervisor", "kubernetes", "observability", "security"]
FindingStatus = Literal[
    "completed", "partial", "failed", "blocked", "needs_review"
]
EvidenceQuality = Literal["insufficient", "limited", "corroborated"]
ActionType = Literal[
    "rollout_restart", "scale_deployment", "delete_pod", "other"
]
ApprovalStatus = Literal["pending", "approved", "rejected"]
ExecutionStatus = Literal["simulated", "skipped", "failed"]
IncidentOutcome = Literal[
    "action_executed_simulated",
    "action_rejected",
    "no_action_proposed",
    "investigation_failed",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class IncidentContext(AgentContract):
    """The alert or request that started the investigation."""

    incident_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Severity = "medium"
    environment: str = Field(min_length=1)
    namespace: str | None = None
    pod_name: str | None = None
    deployment: str | None = None
    source: str = "manual"
    received_at: datetime = Field(default_factory=utc_now)


class RoutingDecision(AgentContract):
    """The Supervisor's choice of specialists, with its reasoning."""

    selected_agents: list[SpecialistName] = Field(min_length=1)
    reason: str = Field(min_length=1)
    decided_at: datetime = Field(default_factory=utc_now)


class Finding(AgentContract):
    """One specialist's conclusion plus the evidence behind it."""

    finding_id: str = Field(default_factory=new_id)
    agent_name: SpecialistName
    summary: str = Field(min_length=1)
    status: FindingStatus = "completed"
    evidence: list[AgentEvidence] = Field(default_factory=list)
    evidence_quality: EvidenceQuality = "insufficient"
    needs_follow_up: bool = False
    attempt: int = Field(default=1, ge=1)
    contradicts: list[str] = Field(default_factory=list)
    error_message: str | None = None
    recorded_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def evidence_must_back_quality_claim(self) -> "Finding":
        if self.evidence_quality != "insufficient" and not self.evidence:
            raise ValueError(
                "Evidence quality above 'insufficient' requires evidence."
            )
        return self


class ProposedAction(AgentContract):
    """A remediation the Supervisor proposes. Never executed unapproved."""

    action_id: str = Field(default_factory=new_id)
    action_type: ActionType
    namespace: str = Field(min_length=1)
    target: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    risk_level: RiskLevel
    supporting_finding_ids: list[str] = Field(min_length=1)
    proposed_at: datetime = Field(default_factory=utc_now)


class ApprovalRecord(AgentContract):
    """The human decision on one proposed action."""

    action_id: str = Field(min_length=1)
    status: ApprovalStatus = "pending"
    decided_by: str | None = None
    reason: str | None = None
    requested_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def decision_fields_match_status(self) -> "ApprovalRecord":
        if self.status == "pending":
            if self.decided_by or self.decided_at:
                raise ValueError("A pending approval cannot have a decision.")
        else:
            if not self.decided_by:
                raise ValueError("A decided approval must record who decided.")
            if self.decided_at is None:
                self.decided_at = utc_now()
        return self


class ExecutionRecord(AgentContract):
    """Outcome of executing an approved action."""

    action_id: str = Field(min_length=1)
    status: ExecutionStatus
    # Phase 1 is propose-only. Relax this in the phase that adds real execution.
    simulated: Literal[True] = True
    message: str = Field(min_length=1)
    executed_at: datetime = Field(default_factory=utc_now)


class UsageRecord(AgentContract):
    """One LLM call, attributed to an incident and an agent."""

    incident_id: str = Field(min_length=1)
    agent_name: AgentName
    model: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    recorded_at: datetime = Field(default_factory=utc_now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CallLimits(AgentContract):
    """Safety ceilings for one incident."""

    max_calls_per_agent: int = Field(default=5, ge=1)
    max_calls_per_incident: int = Field(default=15, ge=1)
    max_supervisor_hops: int = Field(default=6, ge=1)


class AuditEntry(AgentContract):
    """One immutable line in the incident's audit trail."""

    entry_id: str = Field(default_factory=new_id)
    incident_id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    event: str = Field(min_length=1)
    detail: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class IncidentReport(AgentContract):
    """The Supervisor's final report for the incident."""

    incident_id: str = Field(min_length=1)
    outcome: IncidentOutcome
    summary: str = Field(min_length=1)
    root_cause: str | None = None
    finding_ids: list[str] = Field(default_factory=list)
    action_id: str | None = None
    approval_status: ApprovalStatus | None = None
    generated_at: datetime = Field(default_factory=utc_now)