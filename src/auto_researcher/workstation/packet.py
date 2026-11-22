"""The review packet: what the human sees, and the actions open to them.

Each paragraph comes with the claims, quotes, metrics and context behind it, and the
flags that concern it. A blocking flag removes "approve".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .schemas import Claim, ContextItem, Flag, Metric, Request
from .state import WorkflowState

Action = Literal["approve_for_internal_use", "request_revision", "escalate", "reject"]
ALL_ACTIONS: list[Action] = ["approve_for_internal_use", "request_revision", "escalate", "reject"]


class Quote(BaseModel):
    doc_id: str
    form: str
    filed: str
    section: str | None
    quote: str


class ClaimView(BaseModel):
    id: str
    text: str
    role: str
    stance: str
    quotes: list[Quote]


class ParagraphView(BaseModel):
    ref: str
    section: str
    text: str
    claims: list[ClaimView] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    context: list[ContextItem] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)


class ReviewPacket(BaseModel):
    request: Request
    title: str
    paragraphs: list[ParagraphView]
    omitted_claims: dict[str, str]
    dropped_claims: dict[str, str]
    gaps: list[str]
    flags: list[Flag]
    review_required: bool
    allowed_actions: list[Action]
    versions: dict[str, str]

    @property
    def blocking(self) -> list[Flag]:
        return [f for f in self.flags if f.blocking]


def _view(c: Claim, state: WorkflowState) -> ClaimView:
    quotes = []
    for e in c.evidence:
        d = state.documents[e.doc_id]
        quotes.append(Quote(doc_id=d.doc_id, form=d.form, filed=d.filed.isoformat(), section=d.section, quote=e.quote))
    return ClaimView(id=c.id, text=c.text, role=c.role, stance=c.stance, quotes=quotes)


def build_packet(state: WorkflowState, claims: dict[str, Claim], dropped: dict[str, str]) -> ReviewPacket:
    memo = state.need_memo()
    m_by_id = {m.id: m for m in [*state.metrics, *state.calculations]}
    c_by_id = {c.id: c for c in state.context}
    flags = list(state.flags)
    paragraphs = []
    for i, p in enumerate(memo.paragraphs, 1):
        ref = f"p{i}"
        paragraphs.append(ParagraphView(
            ref=ref, section=p.section, text=p.text,
            claims=[_view(claims[x], state) for x in p.claim_ids if x in claims],
            metrics=[m_by_id[x] for x in p.metric_ids if x in m_by_id],
            context=[c_by_id[x] for x in p.context_ids if x in c_by_id],
            flags=[f for f in flags if f.ref == ref]))
    gaps = [g for notes in (state.research_notes, state.bear_notes) if notes is not None for g in notes.gaps]
    blocking = any(f.blocking for f in flags)
    return ReviewPacket(
        request=state.request, title=memo.title, paragraphs=paragraphs, omitted_claims=memo.omitted(),
        dropped_claims=dropped, gaps=gaps, flags=flags, review_required=blocking,
        allowed_actions=[a for a in ALL_ACTIONS if not (blocking and a == "approve_for_internal_use")],
        versions=state.versions)


def to_markdown(packet: ReviewPacket) -> str:
    r = packet.request
    out = [f"# {packet.title}", "", f"{r.ticker}, as of {r.as_of}, horizon {r.horizon}. *Draft for internal review.*", ""]
    if packet.flags:
        out += ["## Flags", ""]
        out += [f"- {'**BLOCKING** ' if f.blocking else ''}`{f.code}` ({f.ref or '-'}): {f.message}" for f in packet.flags]
        out.append("")
    section = None
    for p in packet.paragraphs:
        if p.section != section:
            section = p.section
            out += [f"## {section}", ""]
        out += [p.text, ""]
        for c in p.claims:
            for q in c.quotes:
                out.append(f"> [{c.id}] {q.form} filed {q.filed}{', ' + q.section if q.section else ''}: \"{q.quote}\"")
        for m in p.metrics:
            out.append(f"> [{m.id}] {m.name} = {m.value} {m.unit} ({m.source}, data {m.as_of}, {m.snapshot_id})")
        for x in p.context:
            rec = x.track_record or "no measured track record"
            out.append(f"> [{x.id}] {x.name} = {x.value} ({x.source}, {x.as_of}; {rec})")
        out.append("")
    if packet.omitted_claims:
        out += ["## Claims left out", ""] + [f"- {k}: {v}" for k, v in packet.omitted_claims.items()] + [""]
    if packet.dropped_claims:
        out += ["## Claims that failed verification", ""] + [f"- {k}: {v}" for k, v in packet.dropped_claims.items()] + [""]
    if packet.gaps:
        out += ["## Not answered by the filings", ""] + [f"- {g}" for g in packet.gaps] + [""]
    out += ["## Actions", "", ", ".join(packet.allowed_actions), ""]
    return "\n".join(out)
