"""Shared signal names and the structured crash_loop check.

The Kubernetes adapter derives the `crash_loop` signal from the structured
pod status in successful diagnostic observations. It never reads free text:
summaries, events, logs, deployment data and restart counts cannot set it.
The proposal gate requires the structured signal itself, named once here.
"""

import json
from collections.abc import Iterable

CRASH_LOOP_SIGNAL = "crash_loop"
CRASH_LOOP_STATUS = "CrashLoopBackOff"


def crash_loop_from_observations(observations: Iterable[str]) -> bool:
    """True only if every structured pod status equals CrashLoopBackOff.

    Each observation is a JSON string. One that is not valid JSON, is not an
    object, or has no "pod" key is ignored. A "pod" value that is not an
    object, or whose "status" is missing or not a string, fails closed, as
    do disagreeing statuses. The match is exact and case-sensitive.
    """
    statuses = set()
    for observation in observations:
        try:
            payload = json.loads(observation)
        except (TypeError, ValueError, RecursionError):
            continue
        if not isinstance(payload, dict) or "pod" not in payload:
            continue
        pod = payload["pod"]
        if not isinstance(pod, dict):
            return False
        status = pod.get("status")
        if not isinstance(status, str):
            return False
        statuses.add(status)
    return statuses == {CRASH_LOOP_STATUS}
