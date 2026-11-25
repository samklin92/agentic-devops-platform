"""Supervisor correlation: relationships between specialist findings.

Specialists produce observations and findings. This module, run only by
the Supervisor, records which findings agree, disagree or co-occur. It
annotates evidence and nothing else:

- It never changes a Finding. `effective_quality` is a derived
  interpretation, never stored and never used by the proposal gate.
- It never creates, removes or reshapes a proposal. `propose_action`
  decides eligibility on its own. `annotate_action` may only extend the
  rationale text of a proposal that already exists.
- It never authorizes or recommends an action.

Only usable findings take part (completed, evidenced, no follow-up
needed), so partial, failed and blocked findings never correlate, and
missing data is never read as negative evidence. Every relation needs two
different specialists. A repeated investigation is handled by looking at
only the latest attempt per specialist, and the whole result is
recomputed each time from the current findings.

All findings in one incident share its target, which holds while an
incident is a single workload.
"""

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime

from app.state.adapters import is_execution_failure
from app.state.models import (
    Correlation,
    EvidenceQuality,
    Finding,
    ProposedAction,
    utc_now,
)
from app.supervisor.proposals import is_usable

RULE_SHARED_IMPACT = "shared_incident_impact"
RULE_IMPACT_NOT_OBSERVED = "impact_not_observed"
RULE_LOAD_WITH_FAILED_AUTH = "load_with_failed_auth"

# Sentences that negate a claim. They are part of the explanations on
# purpose, and tests treat them as the only allowed mention of "cause".
R1_CAVEAT = (
    "This does not establish the cause of the crash loop or confirm it "
    "from the metrics."
)
R2_CAVEAT = "Co-occurrence does not establish causation; no remediation is implied."
CAVEATS = (R1_CAVEAT, R2_CAVEAT)

_IMPACT_SIGNALS = ("errors_elevated", "latency_elevated")
_LOAD_SIGNALS = ("latency_elevated", "cpu_saturated")
_NAMESPACE = uuid.UUID("5d0a7f3e-6c1b-4f0e-9a52-0d6f1c2b9e11")


def _current_findings(findings: Iterable[Finding]) -> dict[str, Finding]:
    """The latest unambiguous finding per specialist.

    Exact duplicates collapse to one. A finding ID that maps to two
    different records is ambiguous, and that specialist is left out.
    """
    by_id: dict[str, list[Finding]] = {}
    for finding in findings:
        by_id.setdefault(finding.finding_id, []).append(finding)

    ambiguous_agents: set[str] = set()
    unique: list[Finding] = []
    for group in by_id.values():
        if all(item == group[0] for item in group):
            unique.append(group[0])
        else:
            ambiguous_agents.update(item.agent_name for item in group)

    latest: dict[str, Finding] = {}
    ordered = sorted(
        unique, key=lambda f: (f.attempt, f.recorded_at, f.finding_id)
    )
    for finding in ordered:
        if finding.agent_name not in ambiguous_agents:
            latest[finding.agent_name] = finding
    return latest


def _completed_without_failed_runs(finding: Finding) -> bool:
    """True only if every check of the finding ran successfully."""
    return finding.status == "completed" and not any(
        is_execution_failure(item.observation) for item in finding.evidence
    )


def _correlation(
    kind: str,
    rule: str,
    findings: Sequence[Finding],
    explanation: str,
    now: datetime,
) -> Correlation:
    finding_ids = [finding.finding_id for finding in findings]
    return Correlation(
        correlation_id=str(
            uuid.uuid5(_NAMESPACE, rule + ":" + "|".join(finding_ids))
        ),
        kind=kind,
        rule=rule,
        finding_ids=finding_ids,
        explanation=explanation,
        correlated_at=now,
    )


def correlate(
    findings: Iterable[Finding], now: datetime | None = None
) -> list[Correlation]:
    """Return the relationships between the current findings.

    Pure and deterministic: the same findings give the same correlations,
    with the same IDs, in any input order.
    """
    now = now or utc_now()
    current = _current_findings(findings)
    usable = {
        agent: finding
        for agent, finding in current.items()
        if is_usable(finding)
    }
    kubernetes = usable.get("kubernetes")
    observability = usable.get("observability")
    security = usable.get("security")
    correlations: list[Correlation] = []

    if (
        kubernetes is not None
        and observability is not None
        and "crash_loop" in kubernetes.signals
    ):
        impact = [s for s in _IMPACT_SIGNALS if s in observability.signals]
        if impact:
            correlations.append(
                _correlation(
                    "corroborates",
                    RULE_SHARED_IMPACT,
                    [kubernetes, observability],
                    "Kubernetes signals crash_loop and Observability "
                    f"signals {', '.join(impact)}. Independent specialists "
                    "support a shared incident-impact hypothesis. "
                    + R1_CAVEAT,
                    now,
                )
            )
        elif not observability.signals and _completed_without_failed_runs(
            observability
        ):
            correlations.append(
                _correlation(
                    "contradicts",
                    RULE_IMPACT_NOT_OBSERVED,
                    [kubernetes, observability],
                    "Kubernetes signals crash_loop. A completed "
                    "Observability check reports no elevated error, "
                    "latency or resource signals, so it does not confirm "
                    "service impact. Normal metrics can coexist with a "
                    "crash-looping pod.",
                    now,
                )
            )

    if observability is not None and security is not None:
        load = [s for s in _LOAD_SIGNALS if s in observability.signals]
        if load and "failed_auth_elevated" in security.signals:
            correlations.append(
                _correlation(
                    "co_occurs",
                    RULE_LOAD_WITH_FAILED_AUTH,
                    [observability, security],
                    f"Observability signals {', '.join(load)} and Security "
                    "signals failed_auth_elevated. " + R2_CAVEAT,
                    now,
                )
            )

    return correlations


def effective_quality(
    finding: Finding, correlations: Iterable[Correlation]
) -> EvidenceQuality:
    """Derived reading of a finding's quality in light of correlations.

    A usable finding that another specialist corroborates reads as
    'corroborated'. The finding itself is not changed, and the proposal
    gate never uses this value.
    """
    if is_usable(finding) and any(
        c.kind == "corroborates" and finding.finding_id in c.finding_ids
        for c in correlations
    ):
        return "corroborated"
    return finding.evidence_quality


def correlation_notes(
    correlations: Iterable[Correlation], findings: Iterable[Finding]
) -> list[str]:
    """One readable line per correlation, for rationale and reports."""
    agent_of = {f.finding_id: f.agent_name for f in findings}
    notes: list[str] = []
    for correlation in correlations:
        agents = " + ".join(
            agent_of.get(finding_id, "unknown")
            for finding_id in correlation.finding_ids
        )
        notes.append(
            f"{correlation.kind.replace('_', ' ')} ({agents}, "
            f"{correlation.rule}): {correlation.explanation}"
        )
    return notes


def annotate_action(
    action: ProposedAction | None,
    correlations: Sequence[Correlation],
    findings: Iterable[Finding],
) -> ProposedAction | None:
    """Extend an existing proposal's rationale with the correlations.

    Never creates a proposal, and changes nothing but the rationale text.
    """
    if action is None or not correlations:
        return action
    notes = correlation_notes(correlations, findings)
    rationale = (
        f"{action.rationale} Evidence relationships (informational, not "
        f"an authorization): " + " ".join(notes)
    )
    return action.model_copy(update={"rationale": rationale})
