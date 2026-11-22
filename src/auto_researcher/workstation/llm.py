"""The LLM interface the roles call, and a scripted stand-in for tests.

A role asks for a JSON object matching a Pydantic schema. Retrieved text travels in
``inputs`` (data), never in the role or task (instructions). The Claude client comes in
Phase 2.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    model: str
    training_cutoff: date | None

    def complete_json(self, *, role: str, task: str, inputs: dict[str, Any], schema: type[T]) -> T: ...


Reply = dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]


class ScriptedLLM:
    """Returns prepared replies per role, in order (the last one repeats). A reply is a
    dict or a function of the inputs. Every call is recorded in ``calls``."""

    def __init__(self, replies: dict[str, Reply | list[Reply]], model: str = "scripted",
                 training_cutoff: date | None = None) -> None:
        self.replies = {k: v if isinstance(v, list) else [v] for k, v in replies.items()}
        self.model = model
        self.training_cutoff = training_cutoff
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, *, role: str, task: str, inputs: dict[str, Any], schema: type[T]) -> T:
        if role not in self.replies:
            raise KeyError(f"no scripted reply for role {role!r}")
        n = sum(c["role"] == role for c in self.calls)
        queue = self.replies[role]
        reply = queue[min(n, len(queue) - 1)]
        self.calls.append({"role": role, "task": task, "inputs": inputs})
        return schema.model_validate(reply(inputs) if callable(reply) else reply)
