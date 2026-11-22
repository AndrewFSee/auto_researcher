"""The workflow state: one object every role reads from, with fixed write permissions.

Each slot has one writer. Flags can only be added, by the gate; nothing removes them.
Every write and tool call is appended to the audit log.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from .schemas import ContextItem, Document, DraftMemo, Flag, Metric, Plan, Request, ResearchNotes

WRITERS: dict[str, str] = {
    "plan": "orchestrator",
    "draft_memo": "orchestrator",
    "research_notes": "filing",
    "retrieval": "filing",
    "bear_notes": "bear",
    "metrics": "data",
    "context": "context",
    "calculations": "calculator",
}
DOCUMENT_WRITERS = {"filing", "bear"}


class WorkflowState:
    def __init__(self, request: Request, versions: dict[str, str] | None = None) -> None:
        self.request = request
        self.versions: dict[str, str] = dict(versions or {})
        self.plan: Plan | None = None
        self.documents: dict[str, Document] = {}
        self.research_notes: ResearchNotes | None = None
        self.retrieval: dict[str, list[str]] = {}   # plan section id -> doc ids found
        self.bear_notes: ResearchNotes | None = None
        self.metrics: list[Metric] = []
        self.context: list[ContextItem] = []
        self.calculations: list[Metric] = []    # the latest draft's calculations, evaluated by code
        self.draft_memo: DraftMemo | None = None
        self.drafts: list[DraftMemo] = []          # every draft, superseded ones included
        self._flags: list[Flag] = []
        self.audit: list[dict[str, Any]] = []

    # -- writes ------------------------------------------------------------------
    def write(self, role: str, slot: str, value: Any) -> None:
        if WRITERS.get(slot) != role:
            raise PermissionError(f"{role!r} may not write {slot!r}")
        setattr(self, slot, value)
        if slot == "draft_memo":
            self.drafts.append(value)
        self.log(role, f"wrote_{slot}")

    def add_documents(self, role: str, docs: list[Document]) -> None:
        if role not in DOCUMENT_WRITERS:
            raise PermissionError(f"{role!r} may not add documents")
        for d in docs:
            self.documents.setdefault(d.doc_id, d)
        self.log(role, "documents", ids=[d.doc_id for d in docs], sha256={d.doc_id: d.sha256 for d in docs})

    def add_flags(self, role: str, flags: list[Flag]) -> None:
        if role != "gate":
            raise PermissionError(f"{role!r} may not add flags")
        self._flags.extend(flags)
        for f in flags:
            self.log("gate", "flag", code=f.code, ref=f.ref, stage=f.stage, message=f.message)

    def need_plan(self) -> Plan:
        if self.plan is None:
            raise RuntimeError("no plan yet")
        return self.plan

    def need_notes(self, slot: str) -> ResearchNotes:
        notes: ResearchNotes | None = {"research_notes": self.research_notes, "bear_notes": self.bear_notes}[slot]
        if notes is None:
            raise RuntimeError(f"{slot} not written yet")
        return notes

    def need_memo(self) -> DraftMemo:
        if self.draft_memo is None:
            raise RuntimeError("no draft yet")
        return self.draft_memo

    @property
    def flags(self) -> tuple[Flag, ...]:
        """Read-only: there is no way to remove a flag."""
        return tuple(self._flags)

    def log(self, actor: str, event: str, **data: Any) -> None:
        self.audit.append({"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "actor": actor,
                           "event": event, **data})

    # -- serialization -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def dump(x: Any) -> Any:
            if isinstance(x, BaseModel):
                return x.model_dump(mode="json")
            if isinstance(x, list):
                return [dump(i) for i in x]
            if isinstance(x, dict):
                return {k: dump(v) for k, v in x.items()}
            return x
        return {"request": dump(self.request), "versions": self.versions, "plan": dump(self.plan),
                "documents": dump(self.documents), "research_notes": dump(self.research_notes), "retrieval": self.retrieval,
                "bear_notes": dump(self.bear_notes), "metrics": dump(self.metrics), "context": dump(self.context),
                "calculations": dump(self.calculations),
                "drafts": dump(self.drafts), "flags": dump(list(self._flags)), "audit": self.audit}
