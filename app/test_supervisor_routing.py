import pytest

from app.state.models import IncidentContext
from app.supervisor.routing import (
    SIGNAL_RULES,
    SPECIALIST_ORDER,
    match_signals,
    route_incident,
    route_text,
)


def agents(text: str) -> list[str]:
    decision = route_text(text)
    return list(decision.selected_agents) if decision else []


def make_incident(description: str, **overrides) -> IncidentContext:
    data = {
        "incident_id": "inc-001",
        "description": description,
        "environment": "production",
    }
    data.update(overrides)
    return IncidentContext(**data)


def test_crash_loop_routes_to_kubernetes_with_explanation():
    decision = route_text("Pod inventory-api is in CrashLoopBackOff")
    assert decision.selected_agents == ["kubernetes"]
    assert "crash loop" in decision.reason


def test_latency_routes_to_observability():
    assert agents("High p99 latency on the checkout API") == ["observability"]


def test_unauthorized_access_routes_to_security():
    assert agents("Unauthorized access to the admin console") == ["security"]


def test_multi_agent_routing_keeps_a_stable_order():
    decision = route_text("Unauthorized kubectl exec into a production pod")
    assert decision.selected_agents == ["kubernetes", "security"]
    assert "kubernetes:" in decision.reason and "security:" in decision.reason


def test_matching_is_case_insensitive():
    assert agents("CRASHLOOPBACKOFF on the orders pod") == ["kubernetes"]


def test_word_boundaries_prevent_partial_matches():
    assert agents("Our podcast episode is down") == []


def test_out_of_scope_returns_none():
    assert route_text("Failed deployment pipeline on the main branch") is None
    assert match_signals("nothing relevant here") == {}


def test_named_pod_is_a_kubernetes_signal():
    incident = make_incident("Service is unhealthy", pod_name="web-1")
    decision = route_incident(incident)
    assert decision.selected_agents == ["kubernetes"]
    assert "specific pod" in decision.reason


def test_incident_without_signals_routes_nowhere():
    assert route_incident(make_incident("Quarterly cost review")) is None


def test_every_specialist_has_rules_with_unique_labels():
    assert set(SIGNAL_RULES) == set(SPECIALIST_ORDER)
    for rules in SIGNAL_RULES.values():
        labels = [label for label, _ in rules]
        assert len(labels) == len(set(labels))

def test_mixed_crash_loop_and_http_500_routes_to_both_specialists():
    decision = route_text(
        "inventory-api is experiencing a crash loop, "
        "with increased HTTP 500 errors and latency"
    )
    assert decision.selected_agents == ["kubernetes", "observability"]


def test_metrics_signals_route_to_observability_only():
    for text in [
        "High error rate on the login service",
        "HTTP 500 errors from the gateway",
        "Spike in 5xx responses from the checkout API",
        "p95 latency is elevated on payment-api",
        "p99 latency doubled overnight",
        "CPU saturation on the checkout service",
        "Memory pressure on the orders service",
        "Metrics anomaly on the gateway",
    ]:
        assert agents(text) == ["observability"], text


def test_pod_without_a_metrics_signal_does_not_route_to_observability():
    assert agents("Pod payment-api-5d8f is stuck in Pending") == ["kubernetes"]


def test_named_pod_plus_metrics_signal_routes_to_both():
    incident = make_incident(
        "p95 latency is elevated", pod_name="payment-api-5d8f"
    )
    decision = route_incident(incident)
    assert decision.selected_agents == ["kubernetes", "observability"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Spike in failed logins against the checkout-api", ["security"]),
        (
            "IAM role used by payment-api is getting AccessDenied responses",
            ["security"],
        ),
        ("The checkout-api security group allows 0.0.0.0/0 ingress", ["security"]),
        ("Root login detected on the production AWS account", ["security"]),
        (
            "Failed logins and CPU saturation on checkout-api",
            ["observability", "security"],
        ),
        (
            "Exposed credentials found in inventory-api pod logs",
            ["kubernetes", "security"],
        ),
        ("Password reset emails are not being delivered", []),
    ],
)
def test_security_signals_route_without_false_positives(text, expected):
    assert agents(text) == expected

