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