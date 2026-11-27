"""Mutation tests for the safety evaluation.

A passing evaluation shows the real code satisfies the scenarios. It does
not show the evaluation can fail. A mutation test injects one known fault
and requires the evaluation itself to notice it. Every fault in
MUTATIONS must be detected, or the run fails and exits non-zero.

A fault counts as detected only when the evaluation reports a scenario
failure or an invariant failure, or when it shows that the faulty model
accepted a finding that a scenario expects to be rejected. A fault that
crashes for any other reason is NOT detected, so a bug in a mutation can
never pass as a detection.

Gate-layer faults are injected through GateImplementation. Graph-layer
faults temporarily replace names in app.supervisor.graph, which the graph
nodes look up when they run, and every replaced name is restored in a
finally block. The mutations cannot express structural faults in the graph
(for example an edge from evaluate to execute), because the conditional
edges are closures inside build_supervisor_graph.

Usage: python -m app.evaluate_safety --mutations
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from app.evaluate_safety import (
    ExpectedInvalidAccepted,
    GateImplementation,
    MutationOutcome,
    SafetyReport,
    ScenarioOutcome,
    evaluate,
    load_scenarios,
    run_gate_scenario,
    run_graph_scenario,
)
from app.state.models import ExecutionRecord, Finding, ProposedAction
from app.state.signals import CRASH_LOOP_SIGNAL
from app.supervisor import graph as graph_module
from app.supervisor.correlation import annotate_action, correlate
from app.supervisor.current_findings import current_findings
from app.supervisor.proposals import is_usable

# Captured before any mutation runs, so faulty wrappers never call a patched
# name and the tests can prove nothing leaked.
REAL_PARSE_DECISION = graph_module.parse_decision

_PRODUCTION = ("production", "prod")


# --- injecting graph faults ------------------------------------------------


@contextmanager
def patched_graph(**replacements: Any) -> Iterator[None]:
    """Replace names in app.supervisor.graph, always restoring them.

    A name the module does not have, or a replacement that is the original
    object, would make a mutation silently do nothing, so both are errors.
    """
    missing = [name for name in replacements if not hasattr(graph_module, name)]
    if missing:
        raise AttributeError(
            f"app.supervisor.graph has no attribute(s) {missing}; "
            "a mutation would silently do nothing."
        )
    originals = {name: getattr(graph_module, name) for name in replacements}
    unchanged = [n for n, value in replacements.items() if value is originals[n]]
    if unchanged:
        raise ValueError(f"The replacement for {unchanged} is the original object.")
    try:
        for name, value in replacements.items():
            setattr(graph_module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(graph_module, name, value)
        leaked = [
            name
            for name, value in originals.items()
            if getattr(graph_module, name) is not value
        ]
        if leaked:
            raise RuntimeError(f"Failed to restore app.supervisor.graph {leaked}.")


# --- the faults -------------------------------------------------------------


def _has_target(incident: Any) -> bool:
    return bool(incident.namespace and incident.deployment)


def _action(incident: Any, supporting: Sequence[Finding]) -> ProposedAction:
    risk = "medium" if incident.environment.lower() in _PRODUCTION else "low"
    return ProposedAction(
        action_type="rollout_restart",
        namespace=incident.namespace,
        target=f"deployment/{incident.deployment}",
        rationale="Faulty gate: a rollout restart is proposed.",
        risk_level=risk,
        supporting_finding_ids=[f.finding_id for f in supporting],
    )


def _propose_from_any_specialist(incident: Any, findings: list[Finding]):
    """Fault 1: any usable finding with a structured signal supports a proposal."""
    if not _has_target(incident):
        return None
    supporting = [f for f in findings if is_usable(f) and f.signals]
    return _action(incident, supporting) if supporting else None


def _propose_ignoring_follow_up(incident: Any, findings: list[Finding]):
    """Fault 2: needs_follow_up no longer makes a finding unusable."""
    if not _has_target(incident):
        return None
    kubernetes = current_findings(findings).get("kubernetes")
    if (
        kubernetes is None
        or kubernetes.status != "completed"
        or kubernetes.evidence_quality == "insufficient"
        or CRASH_LOOP_SIGNAL not in kubernetes.signals
    ):
        return None
    return _action(incident, [kubernetes])


def _annotate_changing_supporting_ids(action: Any, correlations: Any, findings: Any):
    """Fault 3: correlation adds the findings it relates to the supporting IDs."""
    annotated = annotate_action(action, correlations, findings)
    if annotated is None or not correlations:
        return annotated
    extra = [
        finding_id
        for correlation in correlations
        for finding_id in correlation.finding_ids
        if finding_id not in annotated.supporting_finding_ids
    ]
    return annotated.model_copy(
        update={"supporting_finding_ids": [*annotated.supporting_finding_ids, *extra]}
    )


def _correlate_including_partial(findings: Any, now: Any = None):
    """Fault 4: partial findings are relabelled as complete before correlation."""
    relabelled = [
        finding.model_copy(
            update={
                "status": "completed",
                "needs_follow_up": False,
                "evidence_quality": (
                    "limited"
                    if finding.evidence_quality == "insufficient"
                    else finding.evidence_quality
                ),
            }
        )
        if finding.status == "partial"
        else finding
        for finding in findings
    ]
    return correlate(relabelled, now=now)


def _propose_without_signal_requirement(incident: Any, findings: list[Finding]):
    """Fault 5: a usable Kubernetes finding is enough, no crash_loop signal needed."""
    if not _has_target(incident):
        return None
    kubernetes = current_findings(findings).get("kubernetes")
    if kubernetes is None or not is_usable(kubernetes):
        return None
    return _action(incident, [kubernetes])


def _finding_without_signal_guard(**kwargs: Any) -> Finding:
    """Fault 6: failed or blocked findings may carry signals.

    The finding is first built through the real model without its signals,
    so every other field is still validated. The signals are then added
    with model_copy, which skips validation, so only the rule against
    signals on failed or blocked findings is bypassed.
    """
    signals = kwargs.get("signals") or []
    if signals and kwargs.get("status") in ("failed", "blocked"):
        finding = Finding(**{**kwargs, "signals": []})
        return finding.model_copy(update={"signals": list(signals)})
    return Finding(**kwargs)


def _propose_from_raw_findings(incident: Any, findings: list[Finding]):
    """Fault 12: the gate reads every finding again, not the current ones."""
    if not _has_target(incident):
        return None
    supporting = [
        f
        for f in findings
        if f.agent_name == "kubernetes"
        and is_usable(f)
        and CRASH_LOOP_SIGNAL in f.signals
    ]
    return _action(incident, supporting) if supporting else None


def _parse_accepting_truthy(raw: Any, pending_action_id: str):
    """Fault 7: any truthy value approves, so 'true', 'yes' and 1 all count."""
    if not isinstance(raw, dict):
        raise ValueError("Approval decision must be a dict.")
    decided_by = raw.get("decided_by")
    if not isinstance(decided_by, str) or not decided_by.strip():
        raise ValueError("Approval decision needs 'decided_by'.")
    if raw.get("action_id") != pending_action_id:
        raise ValueError("Approval decision action_id does not match the pending action.")
    reason = raw.get("reason")
    return (
        bool(raw.get("approved")),
        decided_by.strip(),
        reason if isinstance(reason, str) else None,
    )


def _parse_ignoring_rejection(raw: Any, pending_action_id: str):
    """Fault 8: a valid rejection is turned into an approval."""
    _, decided_by, reason = REAL_PARSE_DECISION(raw, pending_action_id)
    return True, decided_by, reason


def _unsimulated_execution_record(**kwargs: Any) -> ExecutionRecord:
    """Fault 9: the execution record is not marked simulated."""
    return ExecutionRecord.model_construct(**{**kwargs, "simulated": False})


def _is_malformed(raw: Any) -> bool:
    return not (
        isinstance(raw, dict)
        and isinstance(raw.get("approved"), bool)
        and isinstance(raw.get("decided_by"), str)
        and raw["decided_by"].strip()
    )


def _parse_failing_open_on_malformed(raw: Any, pending_action_id: str):
    """Fault 10: a malformed decision approves instead of being rejected.

    Well-formed decisions, including rejections and wrong action IDs, are
    still handled by the real parser, so only malformed-decision handling
    is broken.
    """
    try:
        return REAL_PARSE_DECISION(raw, pending_action_id)
    except ValueError:
        if _is_malformed(raw):
            return True, "unknown", None
        raise


def _parse_without_binding(raw: Any, pending_action_id: str):
    """Fault 11: the action_id in the decision is ignored."""
    if isinstance(raw, dict):
        raw = {**raw, "action_id": pending_action_id}
    return REAL_PARSE_DECISION(raw, pending_action_id)


@dataclass(frozen=True)
class Mutation:
    """One injected fault, for the gate layer or the graph layer."""

    name: str
    description: str
    layer: str
    gate: Callable[[], GateImplementation] | None = None
    graph: Callable[[], dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        gate_ok = self.layer == "gate" and self.gate and not self.graph
        graph_ok = self.layer == "graph" and self.graph and not self.gate
        if not (gate_ok or graph_ok):
            raise ValueError(
                f"Mutation '{self.name}' needs a layer of 'gate' with a gate "
                "builder, or 'graph' with a graph builder, and no other."
            )


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "gate_accepts_other_specialists",
        "Any usable finding with a structured signal supports a proposal, "
        "whichever specialist produced it.",
        "gate",
        gate=lambda: GateImplementation(propose=_propose_from_any_specialist),
    ),
    Mutation(
        "gate_ignores_needs_follow_up",
        "A finding that still needs follow-up is treated as usable.",
        "gate",
        gate=lambda: GateImplementation(propose=_propose_ignoring_follow_up),
    ),
    Mutation(
        "annotate_changes_supporting_ids",
        "Correlation adds the findings it relates to a proposal's supporting IDs.",
        "gate",
        gate=lambda: GateImplementation(annotate=_annotate_changing_supporting_ids),
    ),
    Mutation(
        "correlate_includes_partial_findings",
        "Partial findings are relabelled as complete before correlation.",
        "gate",
        gate=lambda: GateImplementation(correlate=_correlate_including_partial),
    ),
    Mutation(
        "gate_drops_crash_loop_requirement",
        "A usable Kubernetes finding proposes a restart without the "
        "structured crash_loop signal.",
        "gate",
        gate=lambda: GateImplementation(propose=_propose_without_signal_requirement),
    ),
    Mutation(
        "signal_guard_on_failed_findings_removed",
        "The finding model accepts signals on failed or blocked findings.",
        "gate",
        gate=lambda: GateImplementation(finding_factory=_finding_without_signal_guard),
    ),
    Mutation(
        "parse_decision_accepts_truthy_values",
        "The strings 'true' and 'yes' and the integer 1 count as an approval.",
        "graph",
        graph=lambda: {"parse_decision": _parse_accepting_truthy},
    ),
    Mutation(
        "rejection_treated_as_approval",
        "A valid rejection is recorded as an approval and executed.",
        "graph",
        graph=lambda: {"parse_decision": _parse_ignoring_rejection},
    ),
    Mutation(
        "execution_not_marked_simulated",
        "The execution record is not marked simulated.",
        "graph",
        graph=lambda: {"ExecutionRecord": _unsimulated_execution_record},
    ),
    Mutation(
        "malformed_decision_reaches_execution",
        "A malformed decision approves and executes, while well-formed decisions still behave normally.",
        "graph",
        graph=lambda: {"parse_decision": _parse_failing_open_on_malformed},
    ),
    Mutation(
        "wrong_or_replayed_action_id_accepted",
        "The action_id in a decision is ignored, so another proposal's "
        "approval is accepted.",
        "graph",
        graph=lambda: {"parse_decision": _parse_without_binding},
    ),
    Mutation(
        "gate_uses_raw_findings",
        "The gate reads every finding, so a superseded, duplicate or "
        "ambiguous finding can support a proposal.",
        "gate",
        gate=lambda: GateImplementation(propose=_propose_from_raw_findings),
    ),
)


# --- running them -----------------------------------------------------------


def _first_failure(outcome: ScenarioOutcome) -> str:
    if outcome.expectation_failures:
        return outcome.expectation_failures[0]
    violation = outcome.invariant_violations[0]
    return f"{violation.invariant}: {violation.detail}"


def run_mutation(mutation: Mutation, scenarios: list[dict[str, Any]]) -> MutationOutcome:
    """Run the scenarios of the mutation's layer with its fault injected."""
    layer_scenarios = [s for s in scenarios if s["layer"] == mutation.layer]
    if not layer_scenarios:
        return MutationOutcome(
            mutation.name,
            mutation.description,
            False,
            f"there are no {mutation.layer} scenarios to detect it",
        )
    try:
        if mutation.layer == "gate":
            impl = mutation.gate()
            for scenario in layer_scenarios:
                outcome = run_gate_scenario(scenario, impl)
                if outcome.status == "failed":
                    return MutationOutcome(
                        mutation.name,
                        mutation.description,
                        True,
                        f"detected by {outcome.id}: {_first_failure(outcome)}",
                    )
        else:
            with patched_graph(**mutation.graph()):
                for scenario in layer_scenarios:
                    outcome = run_graph_scenario(scenario)
                    if outcome.status == "failed":
                        return MutationOutcome(
                            mutation.name,
                            mutation.description,
                            True,
                            f"detected by {outcome.id}: {_first_failure(outcome)}",
                        )
    except ExpectedInvalidAccepted as exc:
        return MutationOutcome(
            mutation.name,
            mutation.description,
            True,
            f"detected: the faulty model accepted a finding that a scenario "
            f"expects it to reject ({exc})",
        )
    except Exception as exc:  # a crash is not a detection
        return MutationOutcome(
            mutation.name,
            mutation.description,
            False,
            f"the mutation raised {type(exc).__name__}: {exc}, "
            "so it could not be judged",
        )
    return MutationOutcome(
        mutation.name,
        mutation.description,
        False,
        f"all {len(layer_scenarios)} {mutation.layer} scenarios still passed",
    )


def run_mutations(
    scenarios: list[dict[str, Any]] | None = None,
    mutations: Sequence[Mutation] | None = None,
) -> list[MutationOutcome]:
    scenarios = load_scenarios() if scenarios is None else scenarios
    mutations = MUTATIONS if mutations is None else mutations
    return [run_mutation(mutation, scenarios) for mutation in mutations]


def evaluate_with_mutations(scenarios: list[dict[str, Any]]) -> SafetyReport:
    """Evaluate the real code, and inject faults only if it passes.

    Mutation results mean nothing when the baseline already fails, so in
    that case no mutation runs and the report keeps the baseline failures.
    """
    report = evaluate(scenarios)
    if report.ok:
        report.mutations = run_mutations(scenarios)
    return report
