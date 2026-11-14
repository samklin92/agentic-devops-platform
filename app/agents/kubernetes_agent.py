"""Kubernetes specialist agent for the Agentic DevOps Platform."""

import json
from typing import Any

from app.agent_state import AgentState
from app.contracts.agent_messages import (
    AgentEvidence,
    KubernetesInvestigationRequest,
    KubernetesInvestigationResponse,
)
from app.langgraph_agent import (
    build_devops_graph_async,
    to_graph_state,
)


def extract_evidence(
    findings: list[dict[str, Any]],
) -> list[AgentEvidence]:
    """Convert diagnostic findings into structured evidence."""

    evidence = []

    for finding in findings:
        tool = finding.get("tool", "unknown diagnostic tool")
        result = finding.get("result", {})

        observation = json.dumps(result, default=str)

        evidence.append(
            AgentEvidence(
                source_tool=tool,
                observation=observation,
            )
        )

    return evidence


def extract_findings(
    evidence: list[AgentEvidence],
) -> list[str]:
    """Generate readable findings from structured evidence."""

    return [
        f"{item.source_tool}: {item.observation}"
        for item in evidence
    ]


def determine_status(
    raw_status: str,
    findings: list[str],
) -> str:
    """Translate internal graph statuses into response statuses."""

    if raw_status == "completed":
        return "completed" if findings else "failed"

    if raw_status == "partial":
        return "partial"

    if raw_status == "needs_review":
        return "needs_review" if findings else "failed"

    if raw_status == "blocked":
        return "blocked"

    if raw_status == "failed":
        return "failed"

    return "failed"


def extract_error_message(result: dict[str, Any]) -> str | None:
    """Extract an execution failure or review explanation."""

    message = result.get("reason") or result.get("error_message")

    if message:
        return str(message)

    for action in reversed(result.get("actions_taken", [])):
        if action.get("status") in {"failed", "blocked"}:
            return action.get("error") or action.get("message")

    return None


async def investigate_kubernetes(
    request: KubernetesInvestigationRequest,
    tools: dict[str, Any],
    embedding_model: Any = None,
    approved_actions: set[str] | None = None,
) -> KubernetesInvestigationResponse:
    """Investigate a Kubernetes incident and return structured findings."""

    agent_state = AgentState(
        task=(
            f"{request.task}. "
            f"Target namespace: {request.target.namespace}. "
            f"Target pod: {request.target.pod_name}."
        ),
        task_id=request.incident_id,
        status="pending",
    )

    graph = build_devops_graph_async(
        embedding_model=embedding_model,
        tools=tools,
        client=None,
        approved_actions=approved_actions,
        include_response=False,
    )

    result = await graph.ainvoke(to_graph_state(agent_state))

    evidence = extract_evidence(result.get("findings", []))
    findings = extract_findings(evidence)

    status = determine_status(
        result.get("status", "failed"),
        findings,
    )

    return KubernetesInvestigationResponse(
        incident_id=request.incident_id,
        agent_name="kubernetes",
        status=status,
        findings=findings,
        evidence=evidence,
        evidence_quality="limited" if evidence else "insufficient",
        needs_follow_up=status != "completed",
        error_message=extract_error_message(result),
    )