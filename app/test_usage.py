"""Tests for LLM usage recording."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

import app.usage as usage
from app.state.models import UsageRecord
from app.usage import (
    estimate_cost_usd,
    load_price_table,
    record_llm_usage,
    track_usage,
)

MODEL = "claude-sonnet-4-6"
PRICES = {
    MODEL: {"input_per_million_usd": 3.0, "output_per_million_usd": 15.0},
}


def response(input_tokens=100, output_tokens=40):
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
    )


class Untouchable:
    """A response that fails if anything reads it."""

    def __getattr__(self, name):
        raise AssertionError(f"the response was read ({name})")


# ---- outside a collector ----

def test_recording_outside_a_collector_does_nothing_and_never_reads_the_response():
    assert record_llm_usage(Untouchable(), model=MODEL) is None


# ---- recording ----

def test_a_response_becomes_a_record_for_the_incident_and_agent():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        record_llm_usage(response(100, 40), model=MODEL)
    (record,) = collector.records
    assert record.incident_id == "inc-1"
    assert record.agent_name == "kubernetes"
    assert record.model == MODEL
    assert (record.input_tokens, record.output_tokens) == (100, 40)
    assert record.total_tokens == 140
    assert collector.dropped == 0


def test_several_calls_are_all_recorded_in_order():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        record_llm_usage(response(1, 2), model=MODEL)
        record_llm_usage(response(3, 4), model=MODEL)
    assert [r.total_tokens for r in collector.records] == [3, 7]


def test_a_record_holds_counts_and_names_only_never_text():
    assert set(UsageRecord.model_fields) == {
        "incident_id",
        "agent_name",
        "model",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "recorded_at",
    }


# ---- cost ----

def test_cost_is_unknown_without_a_price():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        record_llm_usage(response(), model=MODEL)
    assert collector.records[0].estimated_cost_usd is None


def test_cost_is_estimated_from_the_price_table():
    with track_usage("inc-1", "kubernetes", prices=PRICES) as collector:
        record_llm_usage(response(1_000_000, 2_000_000), model=MODEL)
    assert collector.records[0].estimated_cost_usd == pytest.approx(33.0)


def test_a_model_not_in_the_table_has_an_unknown_cost_never_zero():
    with track_usage("inc-1", "kubernetes", prices=PRICES) as collector:
        record_llm_usage(response(), model="some-other-model")
    assert collector.records[0].estimated_cost_usd is None
    assert estimate_cost_usd("some-other-model", 1, 1, PRICES) is None


# ---- a failure to record is counted, never raised ----

@pytest.mark.parametrize(
    "bad",
    [
        SimpleNamespace(),
        SimpleNamespace(usage=None),
        SimpleNamespace(usage=SimpleNamespace(input_tokens=1)),
        SimpleNamespace(usage=SimpleNamespace(input_tokens=-1, output_tokens=1)),
        SimpleNamespace(usage=SimpleNamespace(input_tokens=True, output_tokens=1)),
        SimpleNamespace(usage=SimpleNamespace(input_tokens="5", output_tokens=1)),
        SimpleNamespace(usage=SimpleNamespace(input_tokens=1.5, output_tokens=1)),
        None,
        "not a response",
    ],
)
def test_an_unreadable_response_is_dropped_and_counted(bad):
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        record_llm_usage(bad, model=MODEL)
    assert collector.records == []
    assert collector.dropped == 1


def test_an_empty_model_name_is_dropped_and_counted():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        record_llm_usage(response(), model="")
    assert collector.records == [] and collector.dropped == 1


# ---- the collector's lifetime ----

def test_the_collector_is_removed_when_the_block_ends():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        pass
    record_llm_usage(response(), model=MODEL)
    assert collector.records == []


def test_the_collector_is_removed_even_when_the_block_raises_and_keeps_its_records():
    with pytest.raises(RuntimeError):
        with track_usage("inc-1", "kubernetes", prices={}) as collector:
            record_llm_usage(response(5, 5), model=MODEL)
            raise RuntimeError("the specialist failed")
    assert [r.total_tokens for r in collector.records] == [10]
    record_llm_usage(response(), model=MODEL)
    assert len(collector.records) == 1


def test_nested_blocks_record_to_the_innermost_and_restore_the_outer():
    with track_usage("inc-1", "kubernetes", prices={}) as outer:
        with track_usage("inc-1", "security", prices={}) as inner:
            record_llm_usage(response(1, 1), model=MODEL)
        record_llm_usage(response(2, 2), model=MODEL)
    assert [r.agent_name for r in inner.records] == ["security"]
    assert [r.agent_name for r in outer.records] == ["kubernetes"]


def test_a_new_thread_does_not_see_the_collector():
    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        thread = threading.Thread(
            target=lambda: record_llm_usage(response(), model=MODEL)
        )
        thread.start()
        thread.join()
    assert collector.records == []


def test_asyncio_run_inside_the_block_is_recorded():
    async def call():
        record_llm_usage(response(7, 3), model=MODEL)

    with track_usage("inc-1", "kubernetes", prices={}) as collector:
        asyncio.run(call())
    assert [r.total_tokens for r in collector.records] == [10]


# ---- the price table ----

def test_the_shipped_price_table_loads_and_is_empty_by_default():
    assert load_price_table() == {}


def test_a_missing_price_file_is_an_empty_table(tmp_path):
    assert load_price_table(tmp_path / "absent.json") == {}


def test_a_valid_price_file_loads(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"models": {MODEL: {
        "input_per_million_usd": 3, "output_per_million_usd": 15.5}}}))
    assert load_price_table(path) == {MODEL: {
        "input_per_million_usd": 3.0, "output_per_million_usd": 15.5}}


@pytest.mark.parametrize(
    "content",
    [
        [],
        {"models": []},
        {"models": {MODEL: {}}},
        {"models": {MODEL: {"input_per_million_usd": 1}}},
        {"models": {MODEL: {"input_per_million_usd": -1, "output_per_million_usd": 1}}},
        {"models": {MODEL: {"input_per_million_usd": "3", "output_per_million_usd": 1}}},
        {"models": {MODEL: {"input_per_million_usd": True, "output_per_million_usd": 1}}},
    ],
)
def test_a_malformed_price_file_is_rejected(tmp_path, content):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_price_table(path)


def test_an_unusable_price_file_gives_unknown_costs_and_a_warning_not_a_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "prices.json"
    path.write_text("{not json")
    monkeypatch.setattr(usage, "PRICE_TABLE_PATH", path)
    with track_usage("inc-1", "kubernetes") as collector:
        record_llm_usage(response(), model=MODEL)
    assert collector.records[0].estimated_cost_usd is None
    assert len(collector.warnings) == 1
    assert "price table" in collector.warnings[0]


def test_the_price_table_file_is_used_when_none_is_given(tmp_path, monkeypatch):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"models": {MODEL: {
        "input_per_million_usd": 1, "output_per_million_usd": 1}}}))
    monkeypatch.setattr(usage, "PRICE_TABLE_PATH", path)
    with track_usage("inc-1", "kubernetes") as collector:
        record_llm_usage(response(1_000_000, 0), model=MODEL)
    assert collector.records[0].estimated_cost_usd == pytest.approx(1.0)


# ---- compatibility with the state's aggregation ----

def test_recorded_usage_aggregates_with_summarize_usage():
    from app.state.graph_state import summarize_usage

    with track_usage("inc-1", "kubernetes", prices=PRICES) as collector:
        record_llm_usage(response(1_000_000, 0), model=MODEL)
        record_llm_usage(response(0, 1_000_000), model="unpriced-model")
    summary = summarize_usage(collector.records)["kubernetes"]
    assert summary["calls"] == 2
    assert summary["total_tokens"] == 2_000_000
