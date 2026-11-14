"""Checkpoint serialization allowlist for the platform state.

LangGraph's checkpointer only deserializes custom types it has been told
about. The allowlist is built from app/state/models.py, so a new record
added there is covered automatically. Nothing outside our own state
records is ever allowed.
"""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

import app.state.models as state_models
from app.contracts.agent_messages import AgentEvidence


def _state_record_classes() -> list[type[BaseModel]]:
    classes = [
        value
        for value in vars(state_models).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == state_models.__name__
    ]
    # Nested inside Finding.evidence, so it is stored in checkpoints too.
    classes.append(AgentEvidence)
    return classes


ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = sorted(
    {(cls.__module__, cls.__name__) for cls in _state_record_classes()}
)


def platform_serializer() -> JsonPlusSerializer:
    """Serializer for every checkpointer used with PlatformState."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)