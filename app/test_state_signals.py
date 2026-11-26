"""Tests for the shared crash-loop detection."""

import pytest

from app.state.signals import CRASH_LOOP_SIGNAL, mentions_crash_loop
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


def test_the_gate_names_the_signal_from_the_shared_module_and_reads_no_text():
    # One source of truth for the signal name. The gate requires that
    # structured signal and has no text pattern of its own.
    assert CRASH_LOOP_SIGNAL == "crash_loop"
    assert proposals.CRASH_LOOP_SIGNAL is CRASH_LOOP_SIGNAL
    assert not hasattr(proposals, "_CRASH_LOOP")
    assert not hasattr(proposals, "_finding_text")
