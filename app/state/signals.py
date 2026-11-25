"""Shared signal detection.

The crash-loop pattern lives here so the proposal gate and the signal
that Kubernetes findings carry use exactly one definition.
"""

import re

CRASH_LOOP_PATTERN = re.compile(r"\bcrash[\s-]?loop", re.IGNORECASE)


def mentions_crash_loop(text: str) -> bool:
    return CRASH_LOOP_PATTERN.search(text) is not None
