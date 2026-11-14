"""Routing accuracy evaluation for the Supervisor router.

Usage: python -m app.evaluate_routing
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.supervisor.routing import SPECIALIST_ORDER, route_text

EVAL_PATH = (
    Path(__file__).resolve().parent.parent / "evals" / "routing_questions.json"
)
VALID_AGENTS = set(SPECIALIST_ORDER)
REQUIRED_KEYS = {"id", "incident", "expected", "category"}


def validate_cases(cases: list[dict]) -> None:
    """Reject malformed or internally inconsistent eval cases."""
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case ids in routing eval set.")

    for case in cases:
        missing = REQUIRED_KEYS - case.keys()
        if missing:
            raise ValueError(f"Case {case.get('id')} is missing {sorted(missing)}.")

        expected = case["expected"]
        unknown = set(expected) - VALID_AGENTS
        if unknown:
            raise ValueError(f"Case {case['id']} has unknown agents {sorted(unknown)}.")

        size = len(set(expected))
        category = case["category"]
        consistent = (
            (category == "out_of_scope" and size == 0)
            or (category == "single" and size == 1)
            or (category == "multi" and size >= 2)
        )
        if not consistent:
            raise ValueError(
                f"Case {case['id']}: category '{category}' does not match "
                f"expected {expected}."
            )


def load_cases(path: Path = EVAL_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    validate_cases(cases)
    return cases


def predict_agents(text: str) -> list[str]:
    """Agents the baseline router selects for this text."""
    decision = route_text(text)
    return list(decision.selected_agents) if decision else []


@dataclass
class AgentMetrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else None


@dataclass
class RoutingReport:
    total: int = 0
    exact_matches: int = 0
    out_of_scope_total: int = 0
    out_of_scope_correct: int = 0
    per_agent: dict[str, AgentMetrics] = field(
        default_factory=lambda: {a: AgentMetrics() for a in SPECIALIST_ORDER}
    )
    misses: list[dict] = field(default_factory=list)

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.total if self.total else 0.0


def evaluate(
    cases: list[dict],
    router: Callable[[str], Sequence[str]] = predict_agents,
) -> RoutingReport:
    report = RoutingReport()

    for case in cases:
        expected = set(case["expected"])
        predicted = set(router(case["incident"]))
        report.total += 1

        if predicted == expected:
            report.exact_matches += 1
        else:
            report.misses.append(
                {
                    "id": case["id"],
                    "incident": case["incident"],
                    "expected": sorted(expected),
                    "predicted": sorted(predicted),
                    "hard": case.get("hard", False),
                }
            )

        if not expected:
            report.out_of_scope_total += 1
            if not predicted:
                report.out_of_scope_correct += 1

        for agent in SPECIALIST_ORDER:
            metrics = report.per_agent[agent]
            if agent in expected and agent in predicted:
                metrics.true_positives += 1
            elif agent in predicted:
                metrics.false_positives += 1
            elif agent in expected:
                metrics.false_negatives += 1

    return report


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_report(report: RoutingReport) -> str:
    lines = [
        "Routing evaluation",
        f"Exact-match accuracy: {report.exact_matches}/{report.total} "
        f"({report.exact_accuracy:.1%})",
        f"Out-of-scope handled: {report.out_of_scope_correct}/"
        f"{report.out_of_scope_total}",
        "",
        f"{'Agent':<15}{'Precision':>10}{'Recall':>10}",
    ]
    for agent in SPECIALIST_ORDER:
        metrics = report.per_agent[agent]
        lines.append(
            f"{agent:<15}{_fmt(metrics.precision):>10}{_fmt(metrics.recall):>10}"
        )

    if report.misses:
        lines += ["", "Misses:"]
        for miss in report.misses:
            tag = " [hard]" if miss["hard"] else ""
            lines.append(
                f"  {miss['id']}{tag}: expected {miss['expected']}, "
                f"got {miss['predicted']}\n    {miss['incident']}"
            )
    return "\n".join(lines)


def main() -> None:
    print(format_report(evaluate(load_cases())))


if __name__ == "__main__":
    main()