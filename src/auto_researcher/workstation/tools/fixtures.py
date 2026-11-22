"""Fixture-backed tools for tests and the mocked prototype (no network).

They do not filter by date themselves: the as-of guard and the gate must catch what
a careless tool returns.
"""

from __future__ import annotations

import re
from datetime import date

from ..schemas import ContextItem, Document, Metric

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"what", "which", "does", "with", "from", "that", "this", "were", "have", "there", "their", "about"}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOP}


class FixtureFilings:
    version = "fixture-1"

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def search(self, ticker: str, query: str, as_of: date, k: int = 5) -> list[Document]:
        q = _terms(query)
        scored = [(len(q & _terms(d.text + " " + (d.section or ""))), d) for d in self.documents if d.ticker == ticker]
        return [d for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0][:k]


class FixtureMetrics:
    version = "fixture-1"

    def __init__(self, metrics: list[Metric]) -> None:
        self.by_name = {m.name: m for m in metrics}

    def available(self) -> dict[str, str]:
        return {n: n for n in self.by_name}

    def metrics(self, ticker: str, names: list[str], as_of: date) -> list[Metric]:
        return [self.by_name[n] for n in names if n in self.by_name]


class FixtureContext:
    version = "fixture-1"

    def __init__(self, items: list[ContextItem]) -> None:
        self.by_name = {c.name: c for c in items}

    def available(self) -> dict[str, str]:
        return {n: n for n in self.by_name}

    def context(self, ticker: str, names: list[str], as_of: date) -> list[ContextItem]:
        return [self.by_name[n] for n in names if n in self.by_name]
