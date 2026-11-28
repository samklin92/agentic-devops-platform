"""LLM usage recording: observation only.

Code that calls a model reports each response with `record_llm_usage`.
When a collector is active (the Supervisor opens one around every
specialist call with `track_usage`) the call becomes a `UsageRecord`
attributed to that incident and agent. Outside a collector it does
nothing and never touches the response, so the RAG code behaves exactly as
it did before.

Rules this module keeps:

- Usage never influences a decision. It is recorded, never read by the
  proposal gate, the approval step or execution.
- A record holds counts, the model name, the agent and an estimated cost.
  It never holds prompt or response text, because that text is untrusted,
  may be sensitive, and the state is checkpointed to SQLite.
- Recording never raises into the caller. A response that cannot be
  recorded is counted in `UsageCollector.dropped`, so a failure is visible
  and never silent.
- A cost is estimated only for a model that appears in the price table.
  Any other model gets `None`, which means "unknown", never zero.

Limits: a model call made from a thread that does not copy the calling
context is not seen by the collector (`asyncio.run` does copy it).
Prompt-cache tokens are not modelled, so a cost is inexact if caching is
ever used.

The price table is `app/llm_prices.json`, maintained by hand and empty by
default:

    {"models": {"<model name>": {"input_per_million_usd": 0.0,
                                  "output_per_million_usd": 0.0}}}
"""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.state.models import UsageRecord

PRICE_TABLE_PATH = Path(__file__).resolve().parent / "llm_prices.json"

PriceTable = Mapping[str, Mapping[str, float]]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


def load_price_table(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load the price table. A missing file is an empty table.

    A malformed file raises ValueError, so a typo is found when the table
    is loaded and not hidden as a missing price.
    """
    path = path or PRICE_TABLE_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        raise ValueError("The price table needs a 'models' object.")
    table: dict[str, dict[str, float]] = {}
    for model, entry in models.items():
        rates = entry if isinstance(entry, dict) else {}
        inputs = _number(rates.get("input_per_million_usd"))
        outputs = _number(rates.get("output_per_million_usd"))
        if inputs is None or outputs is None:
            raise ValueError(
                f"Price entry for '{model}' needs non-negative numbers "
                "input_per_million_usd and output_per_million_usd."
            )
        table[model] = {
            "input_per_million_usd": inputs,
            "output_per_million_usd": outputs,
        }
    return table


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int, prices: PriceTable
) -> float | None:
    """The estimated cost, or None when the model has no price."""
    entry = prices.get(model)
    if entry is None:
        return None
    return (
        input_tokens * entry["input_per_million_usd"]
        + output_tokens * entry["output_per_million_usd"]
    ) / 1_000_000


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("A token count must be a non-negative integer.")
    return value


@dataclass
class UsageCollector:
    """The records one specialist call produced."""

    incident_id: str
    agent_name: str
    prices: PriceTable = field(default_factory=dict)
    records: list[UsageRecord] = field(default_factory=list)
    dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    def record(self, response: Any, model: str) -> None:
        try:
            usage = getattr(response, "usage", None)
            input_tokens = _count(getattr(usage, "input_tokens", None))
            output_tokens = _count(getattr(usage, "output_tokens", None))
            self.records.append(
                UsageRecord(
                    incident_id=self.incident_id,
                    agent_name=self.agent_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=estimate_cost_usd(
                        model, input_tokens, output_tokens, self.prices
                    ),
                )
            )
        except Exception:  # recording must never break the call it observes
            self.dropped += 1


_ACTIVE: ContextVar[UsageCollector | None] = ContextVar(
    "llm_usage_collector", default=None
)


@contextmanager
def track_usage(
    incident_id: str, agent_name: str, prices: PriceTable | None = None
) -> Iterator[UsageCollector]:
    """Collect the usage of every model call made inside the block.

    The collector is restored in a `finally`, so a specialist that raises
    still leaves its records behind and never leaks the collector into the
    next call. A price table that cannot be loaded gives unknown costs and
    a warning, never a failed incident.
    """
    warnings: list[str] = []
    if prices is None:
        try:
            prices = load_price_table()
        except (OSError, ValueError) as exc:
            prices = {}
            warnings.append(f"The price table could not be loaded: {exc}")
    collector = UsageCollector(incident_id, agent_name, prices, warnings=warnings)
    token = _ACTIVE.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE.reset(token)


def record_llm_usage(response: Any, *, model: str) -> None:
    """Report one model response. A no-op outside `track_usage`."""
    collector = _ACTIVE.get()
    if collector is None:
        return
    collector.record(response, model)
