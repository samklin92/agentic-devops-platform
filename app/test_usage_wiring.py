"""Tests for how LLM usage reaches the Supervisor's state."""

import sqlite3
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import app.usage as usage
from app.answer_grounding import evaluate_answer_grounding
from app.contracts.agent_messages import AgentEvidence
from app.generator import DEFAULT_MODEL, generate_answer
from app.state.graph_state import new_platform_state
from app.state.models import Finding, IncidentContext
from app.state.serde import platform_serializer
from app.supervisor.graph import build_supervisor_graph
from app.usage import record_llm_usage, track_usage

CONFIG = {"configurable": {"thread_id": "inc-001"}}
CHUNKS = [{"source": "runbook.md", "section": "Restarts", "content": "Restart it."}]


# ---- the RAG call sites ----

class FakeClient:
    """An Anthropic-shaped client that returns a fixed response."""

    def __init__(self, text="an answer", usage=True, error=None):
        self.text = text
        self.has_usage = usage
        self.error = error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if self.error:
            raise self.error
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
        )
        if self.has_usage:
            response.usage = SimpleNamespace(input_tokens=11, output_tokens=7)
        return response


def test_generate_answer_reports_its_usage_to_an_active_collector():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        result = generate_answer("Why restart?", CHUNKS, FakeClient())
    assert isinstance(result, str)
    (record,) = collector.records
    assert record.model == DEFAULT_MODEL
    assert (record.input_tokens, record.output_tokens) == (11, 7)
    assert record.agent_name == "kubernetes"


def test_generate_answer_is_unchanged_outside_a_collector():
    assert isinstance(generate_answer("Why restart?", CHUNKS, FakeClient()), str)


def test_generate_answer_survives_a_response_without_usage():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        result = generate_answer("Why restart?", CHUNKS, FakeClient(usage=False))
    assert isinstance(result, str)
    assert collector.records == [] and collector.dropped == 1


def test_a_failing_model_call_records_nothing_and_raises_unchanged():
    boom = RuntimeError("the API failed")
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        with pytest.raises(RuntimeError) as raised:
            generate_answer("Why restart?", CHUNKS, FakeClient(error=boom))
    assert raised.value is boom
    assert collector.records == []


def test_grounding_usage_is_recorded_even_when_its_output_is_unusable():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        with pytest.raises(ValueError):
            evaluate_answer_grounding(
                "Why restart?", "Because.", CHUNKS, FakeClient(text="not json")
            )
    (record,) = collector.records
    assert (record.input_tokens, record.output_tokens) == (11, 7)


# ---- the graph ----

def make_incident(**overrides) -> IncidentContext:
    data = {
        "incident_id": "inc-001",
        "description": "Pod inventory-api-7d9f is in CrashLoopBackOff",
        "severity": "high",
        "environment": "production",
        "namespace": "production",
        "pod_name": "inventory-api-7d9f",
        "deployment": "inventory-api",
    }
    data.update(overrides)
    return IncidentContext(**data)


def crash_loop_finding() -> Finding:
    return Finding(
        agent_name="kubernetes",
        summary="Pod is in CrashLoopBackOff with 7 restarts",
        evidence=[
            AgentEvidence(
                source_tool="get_pod_status", observation="CrashLoopBackOff, 7 restarts"
            )
        ],
        evidence_quality="corroborated",
        signals=["crash_loop"],
    )


def metrics_finding() -> Finding:
    return Finding(
        agent_name="observability",
        summary="Latency is elevated",
        evidence=[AgentEvidence(source_tool="get_error_rate", observation="error_rate=1%")],
        evidence_quality="limited",
    )


def reply(input_tokens, output_tokens):
    return SimpleNamespace(
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    )


class LlmInvestigator:
    """A specialist that makes scripted model calls, then returns or raises."""

    def __init__(self, finding=None, error=None, responses=()):
        self.finding = finding
        self.error = error
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, incident):
        self.calls += 1
        for response in self.responses:
            record_llm_usage(response, model="test-model")
        if self.error:
            raise self.error
        return self.finding


def open_saver(path):
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection, serde=platform_serializer()), connection


def run(path, investigators, incident=None):
    saver, connection = open_saver(path)
    try:
        graph = build_supervisor_graph(investigators, saver)
        result = graph.invoke(
            new_platform_state(incident or make_incident()), CONFIG
        )
        return result, graph.get_state(CONFIG)
    finally:
        connection.close()


def resume_approved(path, investigators, action_id):
    saver, connection = open_saver(path)
    try:
        graph = build_supervisor_graph(investigators, saver)
        decision = {"approved": True, "decided_by": "sam", "action_id": action_id}
        return graph.invoke(Command(resume=decision), CONFIG)
    finally:
        connection.close()


def events(values):
    return [entry.event for entry in values["audit_log"]]


def decision_of(snapshot):
    action = snapshot.values["proposed_action"]
    if action is None:
        return None
    return (action.action_type, action.namespace, action.target, action.risk_level)


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "checkpoints.sqlite")


def test_a_specialists_usage_reaches_the_state_attributed_to_it(db_path):
    investigator = LlmInvestigator(
        crash_loop_finding(), responses=[reply(100, 40), reply(5, 5)]
    )
    investigators = {"kubernetes": investigator}
    _, snapshot = run(db_path, investigators)

    records = snapshot.values["usage"]
    assert [(r.agent_name, r.incident_id) for r in records] == [
        ("kubernetes", "inc-001"),
        ("kubernetes", "inc-001"),
    ]
    assert [r.total_tokens for r in records] == [140, 10]

    action_id = snapshot.values["proposed_action"].action_id
    final = resume_approved(db_path, investigators, action_id)
    assert investigator.calls == 1  # no re-run on resume, so no double count
    assert [r.total_tokens for r in final["usage"]] == [140, 10]


def test_two_specialists_are_attributed_separately(db_path):
    investigators = {
        "kubernetes": LlmInvestigator(crash_loop_finding(), responses=[reply(1, 1)]),
        "observability": LlmInvestigator(metrics_finding(), responses=[reply(2, 2)]),
    }
    incident = make_incident(
        description="Pod inventory-api-7d9f is in CrashLoopBackOff and p99 latency is high"
    )
    _, snapshot = run(db_path, investigators, incident)
    totals = {r.agent_name: r.total_tokens for r in snapshot.values["usage"]}
    assert totals == {"kubernetes": 2, "observability": 4}


def test_a_failed_specialists_spend_is_kept(db_path):
    investigator = LlmInvestigator(
        error=RuntimeError("tool failed"), responses=[reply(50, 20)]
    )
    final, _ = run(db_path, {"kubernetes": investigator})
    assert [f.status for f in final["findings"]] == ["failed"]
    assert [r.total_tokens for r in final["usage"]] == [70]
    assert final["proposed_action"] is None


def test_usage_never_changes_the_decision(tmp_path):
    plain = LlmInvestigator(crash_loop_finding())
    heavy = LlmInvestigator(
        crash_loop_finding(), responses=[reply(10**12, 10**12), reply(0, 0)]
    )
    _, without = run(str(tmp_path / "a.sqlite"), {"kubernetes": plain})
    _, with_usage = run(str(tmp_path / "b.sqlite"), {"kubernetes": heavy})
    assert decision_of(without) == decision_of(with_usage) is not None
    assert events(without.values) == events(with_usage.values)
    assert without.next == with_usage.next == ("approval",)


def test_unreadable_usage_is_audited_and_changes_nothing(tmp_path):
    plain = LlmInvestigator(crash_loop_finding())
    broken = LlmInvestigator(crash_loop_finding(), responses=[SimpleNamespace()])
    _, without = run(str(tmp_path / "a.sqlite"), {"kubernetes": plain})
    _, with_broken = run(str(tmp_path / "b.sqlite"), {"kubernetes": broken})

    assert with_broken.values["usage"] == []
    assert "usage_not_recorded" in events(with_broken.values)
    assert decision_of(with_broken) == decision_of(without)
    remaining = [e for e in events(with_broken.values) if e != "usage_not_recorded"]
    assert remaining == events(without.values)
    detail = next(
        e.detail for e in with_broken.values["audit_log"] if e.event == "usage_not_recorded"
    )
    assert "1 call(s) not recorded" in detail


def test_no_model_calls_means_no_usage_and_no_extra_audit_event(db_path):
    _, snapshot = run(db_path, {"kubernetes": LlmInvestigator(crash_loop_finding())})
    assert snapshot.values["usage"] == []
    assert events(snapshot.values) == [
        "alert_received",
        "routed",
        "kubernetes_investigated",
        "action_proposed",
    ]


def test_the_collector_does_not_outlive_the_investigate_node(db_path):
    run(db_path, {"kubernetes": LlmInvestigator(crash_loop_finding(), responses=[reply(1, 1)])})
    assert usage._ACTIVE.get() is None
