"""The single definition of which findings are current.

The Supervisor's correlation and the proposal gate must agree on which
finding represents each specialist, so both call `current_findings`.
Nothing else may re-implement this selection.

Rules, applied per specialist:

- Exact duplicate records (same finding_id, every field equal) collapse
  into one.
- One finding_id on two different records is ambiguous: every specialist
  involved is excluded.
- Two different records from one specialist at the same attempt are
  ambiguous: that specialist is excluded.
- Otherwise the record with the highest attempt is the current one.

An excluded specialist has no current finding at all. No earlier or later
attempt is salvaged, because the Supervisor must not guess which of two
conflicting records is trustworthy. Exclusion fails closed: downstream,
a missing finding can never support a proposal or a correlation.
"""

from collections import Counter
from collections.abc import Iterable

from app.state.models import Finding


def current_findings(findings: Iterable[Finding]) -> dict[str, Finding]:
    """The current, unambiguous finding per specialist, keyed by agent name."""
    by_id: dict[str, list[Finding]] = {}
    for finding in findings:
        by_id.setdefault(finding.finding_id, []).append(finding)

    excluded: set[str] = set()
    unique: list[Finding] = []
    for group in by_id.values():
        if all(item == group[0] for item in group):
            unique.append(group[0])
        else:
            excluded.update(item.agent_name for item in group)

    by_agent: dict[str, list[Finding]] = {}
    for finding in unique:
        by_agent.setdefault(finding.agent_name, []).append(finding)

    current: dict[str, Finding] = {}
    for agent, records in by_agent.items():
        if agent in excluded:
            continue
        per_attempt = Counter(record.attempt for record in records)
        if any(count > 1 for count in per_attempt.values()):
            continue
        current[agent] = max(records, key=lambda record: record.attempt)
    return current
