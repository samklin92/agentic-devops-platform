"""Tests for the structured crash_loop check."""

import json

import pytest

import app.state.signals as signals
from app.state.signals import (
    CRASH_LOOP_SIGNAL,
    CRASH_LOOP_STATUS,
    crash_loop_from_observations,
)
from app.supervisor import proposals


def pod_observation(status, **extra) -> str:
    return json.dumps(
        {"pod": {"name": "inventory-api", "status": status}, **extra}
    )


def test_the_exact_crash_loop_status_is_detected():
    assert CRASH_LOOP_STATUS == "CrashLoopBackOff"
    assert crash_loop_from_observations([pod_observation("CrashLoopBackOff")])


def test_agreeing_crash_loop_statuses_are_detected():
    observations = [
        pod_observation("CrashLoopBackOff"),
        pod_observation("CrashLoopBackOff", restarts=7),
    ]
    assert crash_loop_from_observations(observations)


def test_no_observations_is_not_a_crash_loop():
    assert not crash_loop_from_observations([])


@pytest.mark.parametrize(
    "status",
    [
        "Running",
        "Pending",
        "Error",
        "",
        "crashloopbackoff",
        "CRASHLOOPBACKOFF",
        "CrashLoopBackOff ",
        "Crash Loop",
    ],
)
def test_any_other_status_is_not_a_crash_loop(status):
    assert not crash_loop_from_observations([pod_observation(status)])


@pytest.mark.parametrize(
    "observation",
    [
        "CrashLoopBackOff",
        "pod is in a crash loop",
        "Pod is Running; no crash loop detected",
        json.dumps({"logs": ["crash loop", "CrashLoopBackOff"]}),
        json.dumps(
            {"events": [{"reason": "BackOff", "message": "CrashLoopBackOff"}]}
        ),
        json.dumps({"deployment": {"status": "CrashLoopBackOff"}}),
        json.dumps({"pod_status": "CrashLoopBackOff"}),
        json.dumps({"status": "CrashLoopBackOff"}),
    ],
)
def test_text_and_other_fields_never_set_the_signal(observation):
    assert not crash_loop_from_observations([observation])


def test_text_beside_a_healthy_pod_never_sets_the_signal():
    observation = pod_observation(
        "Running",
        logs=["crash loop", "CrashLoopBackOff"],
        events=[{"message": "Back-off: CrashLoopBackOff"}],
        deployment={"status": "CrashLoopBackOff"},
        restarts=12,
    )
    assert not crash_loop_from_observations([observation])


def test_disagreeing_statuses_fail_closed_in_either_order():
    crash = pod_observation("CrashLoopBackOff")
    running = pod_observation("Running")
    assert not crash_loop_from_observations([crash, running])
    assert not crash_loop_from_observations([running, crash])


@pytest.mark.parametrize(
    "pod",
    [
        None,
        "CrashLoopBackOff",
        ["CrashLoopBackOff"],
        5,
        {},
        {"name": "inventory-api"},
        {"status": None},
        {"status": 5},
        {"status": ["CrashLoopBackOff"]},
        {"status": {"value": "CrashLoopBackOff"}},
    ],
)
def test_an_invalid_pod_fails_closed_even_beside_a_crash_loop(pod):
    invalid = json.dumps({"pod": pod})
    crash = pod_observation("CrashLoopBackOff")
    assert not crash_loop_from_observations([invalid])
    assert not crash_loop_from_observations([crash, invalid])
    assert not crash_loop_from_observations([invalid, crash])


@pytest.mark.parametrize(
    "ignored",
    [
        "not json",
        "",
        "[1, 2]",
        "null",
        '"CrashLoopBackOff"',
        "{}",
        json.dumps({"deployment": {"status": "Degraded"}}),
        None,
        5,
    ],
)
def test_an_observation_without_a_pod_is_ignored(ignored):
    crash = pod_observation("CrashLoopBackOff")
    assert not crash_loop_from_observations([ignored])
    assert crash_loop_from_observations([crash, ignored])
    assert crash_loop_from_observations([ignored, crash])


def test_the_old_text_pattern_is_gone():
    assert not hasattr(signals, "CRASH_LOOP_PATTERN")
    assert not hasattr(signals, "mentions_crash_loop")


def test_the_gate_names_the_signal_from_the_shared_module_and_reads_no_text():
    # One source of truth for the signal name. The gate requires that
    # structured signal and has no text pattern of its own.
    assert CRASH_LOOP_SIGNAL == "crash_loop"
    assert proposals.CRASH_LOOP_SIGNAL is CRASH_LOOP_SIGNAL
    assert not hasattr(proposals, "_CRASH_LOOP")
    assert not hasattr(proposals, "_finding_text")
