"""Tool protocols and the as-of guard.

Every tool is read-only and takes the run's ``as_of``. The guard drops anything dated
after it before an agent sees it; the gate checks again afterwards, in case a tool
or an agent lets something through.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, TypeVar

from ..schemas import ContextItem, Document, Metric


class FilingTool(Protocol):
    version: str

    def search(self, ticker: str, query: str, as_of: date, k: int = 5) -> list[Document]: ...


class MetricTool(Protocol):
    version: str

    def metrics(self, ticker: str, names: list[str], as_of: date) -> list[Metric]: ...


class ContextTool(Protocol):
    version: str

    def context(self, ticker: str, names: list[str], as_of: date) -> list[ContextItem]: ...


D = TypeVar("D", Document, Metric, ContextItem)


def available_at(item: Document | Metric | ContextItem) -> date:
    return item.filed if isinstance(item, Document) else item.as_of


def guard(items: list[D], as_of: date) -> tuple[list[D], list[D]]:
    """(items public by ``as_of``, items dated after it)."""
    kept = [i for i in items if available_at(i) <= as_of]
    return kept, [i for i in items if available_at(i) > as_of]
