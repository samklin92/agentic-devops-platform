"""Shared signal names and detection.

The Kubernetes adapter derives the `crash_loop` signal from successful
diagnostic output using the pattern below. The proposal gate never reads
text: it requires the structured signal itself, named once here.
"""

import re

CRASH_LOOP_SIGNAL = "crash_loop"
CRASH_LOOP_PATTERN = re.compile(r"\bcrash[\s-]?loop", re.IGNORECASE)


def mentions_crash_loop(text: str) -> bool:
    return CRASH_LOOP_PATTERN.search(text) is not None
