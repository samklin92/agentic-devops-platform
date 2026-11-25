"""Tests for the shared crash-loop detection."""

import pytest

from app.state.signals import CRASH_LOOP_PATTERN, mentions_crash_loop
from app.supervisor import proposals


@pytest.mark.parametrize(
    "text",
    [
        "CrashLoopBackOff",
        "pod is in a crash loop",
        "Crash-Loop detected",
        "CRASHLOOP",
        "get_pod_status: CrashLoopBackOff",
    ],
)
def test_crash_loop_is_detected(text):
    assert mentions_crash_loop(text)


@pytest.mark.parametrize(
    "text",
    ["", "healthy", "the pod crashed once", "it crashed in a loop", "loop"],
)
def test_other_text_is_not_a_crash_loop(text):
    assert not mentions_crash_loop(text)


def test_the_proposal_gate_uses_the_same_pattern_object():
    # One source of truth: the gate and the Kubernetes signal cannot drift.
    assert proposals._CRASH_LOOP is CRASH_LOOP_PATTERN
