"""Rule-based Supervisor routing: the deterministic baseline router.

Each specialist has signal rules (regex patterns). An incident is routed
to every specialist whose rules match its description. If no rule
matches, the router returns None and the Supervisor decides what to do.
"""

import re

from app.state.models import IncidentContext, RoutingDecision, SpecialistName

SPECIALIST_ORDER: tuple[SpecialistName, ...] = (
    "kubernetes",
    "observability",
    "security",
)

# (label, regex). Labels appear in the routing reason, so keep them readable.
SIGNAL_RULES: dict[SpecialistName, tuple[tuple[str, str], ...]] = {
    "kubernetes": (
        ("crash loop", r"\bcrash[\s-]?loop(?:[\s-]?back[\s-]?off)?\b"),
        (
            "image pull failure",
            r"\bimage[\s-]?pull(?:[\s-]?back[\s-]?off)?\b|\berrimagepull\b",
        ),
        ("OOM kill", r"\boom[\s-]?kill(?:ed)?\b"),
        ("pod", r"\bpods?\b"),
        (
            "kubernetes object or tool",
            r"\b(?:replicas?|replicaset|statefulset|daemonset|kubelet"
            r"|kubectl|k8s|kubernetes)\b",
        ),
        ("node not ready", r"\bnode[\s-]?not[\s-]?ready\b|\bnotready\b"),
        ("eviction", r"\bevicted\b|\bevictions?\b"),
        ("probe failure", r"\b(?:readiness|liveness) probe\b"),
        ("rollout", r"\brollout\b"),
    ),
    "observability": (
        ("latency", r"\blatency\b|\bp(?:50|90|95|99)\b"),
        ("slowness", r"\bslow(?:er|ness)?\b"),
        (
            "error rate",
            r"\berror rates?\b|\b5xx\b|\bhttp[\s-]?5\d\d\b|\b50[0-4]s?\b",
        ),
        ("timeouts", r"\btime[\s-]?outs?\b|\btimed[\s-]out\b"),
        (
            "resource saturation",
            r"\b(?:cpu|memory|disk)\s(?:usage|utili[sz]ation|spikes?|saturation)\b"
            r"|\bhigh (?:cpu|memory)\b|\bmemory pressure\b",
        ),
        (
            "monitoring tool",
            r"\b(?:metrics?|dashboards?|prometheus|grafana|cloudwatch"
            r"|datadog|alarms?|throughput)\b",
        ),
        (
            "alert or error spike",
            r"\balerts?\s(?:is |are )?firing\b"
            r"|\bspikes? in (?:errors|traffic|requests)\b"
            r"|\belevated (?:errors|latency)\b",
        ),
    ),
    "security": (
        (
            "unauthorized or hostile activity",
            r"\bunauthori[sz]ed\b|\bsuspicious\b|\bcompromised\b|\bmalware\b"
            r"|\bransomware\b|\bintrusion\b|\bexfiltration\b",
        ),
        (
            "vulnerability",
            r"\bvulnerab(?:le|ility|ilities)\b|\bcve-\d{4}-\d+\b",
        ),
        (
            "exposed secret",
            r"\b(?:leaked|exposed|hardcoded)\s+"
            r"(?:secrets?|credentials?|keys?|tokens?|passwords?)\b",
        ),
        (
            "attack pattern",
            r"\bprivilege escalation\b|\bcrypto[\s-]?min(?:er|ing)\b"
            r"|\bbrute[\s-]?force\b|\bport scan(?:ning)?\b",
        ),
        (
            "credential or identity event",
            r"\bfailed logins?\b|\broot login\b|\baccess keys?\b|\biam\b",
        ),
        (
            "exposed resource",
            r"\b(?:public|publicly accessible) (?:s3 )?bucket\b"
            r"|\bsecurity group\b|\b0\.0\.0\.0/0\b",
        ),
        ("anomalous traffic", r"\bunexpected outbound\b|\banomalous\b"),
    ),
}

_COMPILED: dict[SpecialistName, tuple[tuple[str, re.Pattern[str]], ...]] = {
    agent: tuple(
        (label, re.compile(pattern, re.IGNORECASE)) for label, pattern in rules
    )
    for agent, rules in SIGNAL_RULES.items()
}


def match_signals(text: str) -> dict[SpecialistName, list[str]]:
    """Return the labels of the rules that matched, per specialist."""
    signals: dict[SpecialistName, list[str]] = {}
    for agent, rules in _COMPILED.items():
        matched = [label for label, pattern in rules if pattern.search(text)]
        if matched:
            signals[agent] = matched
    return signals


def _decision_from_signals(
    signals: dict[SpecialistName, list[str]],
) -> RoutingDecision | None:
    if not signals:
        return None
    selected = [agent for agent in SPECIALIST_ORDER if agent in signals]
    reason = "; ".join(f"{agent}: {', '.join(signals[agent])}" for agent in selected)
    return RoutingDecision(selected_agents=selected, reason=reason)


def route_text(text: str) -> RoutingDecision | None:
    """Route free text. Returns None when no specialist applies."""
    return _decision_from_signals(match_signals(text))


def route_incident(incident: IncidentContext) -> RoutingDecision | None:
    """Route an incident. A named pod is itself a Kubernetes signal."""
    signals = match_signals(incident.description)
    if incident.pod_name:
        signals.setdefault("kubernetes", []).append(
            "incident targets a specific pod"
        )
    return _decision_from_signals(signals)