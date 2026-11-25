"""The verifier: reads each analysis paragraph against the evidence it cites, and only that.

The number gate checks figures; nothing in code can check wording. The review of the
2026-10-15 pilot found the remaining errors there: statements no quote supports,
conclusions stronger than the evidence, figures from different dates compared as if
they were one, and distortions read the wrong way. The verifier is a second model call
with independent inputs: the paragraph's text and, for each id it cites, the verified
quotes (never the drafter's wording of a claim), the metric or context value with its
definition and dates, and a calculation's formula and result. It sees no other
paragraph, so a conclusion must stand on what its own paragraph cites.

Advisory: its findings are non-blocking ``unsupported_statement`` flags until its
agreement with a human reviewer has been measured (``scripts/workstation_verify.py``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import LLM
from .schemas import Claim, ContextItem, Document, DraftMemo, Flag, Metric

VERIFY_TASK = (
    "You check an investment memo, one paragraph at a time, against the evidence each paragraph cites. "
    "For every paragraph, judge whether its statements are supported by that evidence alone: the quotes, "
    "the metric and context values (read by their definitions and dates) and the calculations. Report a "
    "problem for: a statement of fact no cited quote or value supports (a named event, term, cause or "
    "attribution); a conclusion stronger than the evidence (a trend from one quarter, a headline about the "
    "horizon from a model with no skill at it, a change claimed without values for both dates); figures "
    "from different dates or periods compared as if they were one, when the paragraph does not say so; a "
    "distortion or comparison read in the wrong direction; arithmetic or logic that does not follow. A "
    "paragraph's opening sentence is its headline: check it on its own against the paragraph's cited "
    "figures and quotes. It is 'over' if those figures do not bear it out or cut against it, or if it "
    "states as fact a cause the evidence shows only in part; a caveat later in the paragraph does not cure "
    "an unqualified headline. A ranking or judgment the figures are consistent with is not a problem. A "
    "statement the paragraph marks as its own inference is a problem only if the inference does not "
    "follow. Wording and style are not problems. Verdict per paragraph: 'right' (no problem), 'over' (some "
    "statement goes beyond the evidence), 'wrong' (a statement contradicts the evidence). Quote the "
    "sentence at fault. Be strict but fair: judge only what the paragraph says."
)

IssueKind = Literal["unsupported", "overreach", "period_mix", "direction", "logic"]


class Issue(BaseModel):
    kind: IssueKind
    sentence: str = Field(description="the sentence at fault, verbatim from the paragraph")
    why: str = Field(description="what the evidence does or does not show")


class ParagraphCheck(BaseModel):
    ref: str = Field(description="the paragraph's ref, p1, p2, ...")
    verdict: Literal["right", "over", "wrong"]
    issues: list[Issue] = Field(default_factory=list)


class Verification(BaseModel):
    paragraphs: list[ParagraphCheck]


def _metric_view(m: Metric) -> dict[str, Any]:
    view = {"id": m.id, "name": m.name, "value": m.value, "unit": m.unit, "data_date": m.as_of.isoformat(),
            "period": m.period, "definition": m.definition}
    if m.snapshot_id.startswith("calc:"):
        view["formula"] = m.source.removeprefix("calculation: ")
    return view


def _context_view(c: ContextItem) -> dict[str, Any]:
    return {"id": c.id, "name": c.name, "value": c.value, "unit": c.unit, "data_date": c.as_of.isoformat(),
            "definition": c.definition, "is_forecast": c.is_forecast, "track_record": c.track_record}


def verify_inputs(memo: DraftMemo, claims: dict[str, Claim], metrics: list[Metric], context: list[ContextItem],
                  documents: dict[str, Document], only: set[str] | None = None) -> list[dict[str, Any]]:
    """Each analysis paragraph (or those whose refs are in ``only``) with its cited
    evidence. Limitations paragraphs cite nothing and state no figures, so they are not
    checked."""
    m_by_id, c_by_id = {m.id: m for m in metrics}, {c.id: c for c in context}
    out = []
    for i, p in enumerate(memo.paragraphs, 1):
        if p.kind == "limitations" or (only is not None and f"p{i}" not in only):
            continue
        quotes = []
        for cid in p.claim_ids:
            for e in claims[cid].evidence if cid in claims else []:
                doc = documents.get(e.doc_id)
                quotes.append({"claim": cid, "quote": e.quote, "form": doc.form if doc else None,
                               "filed": doc.filed.isoformat() if doc else None,
                               "section": doc.section if doc else None})
        cited = [m_by_id[x] for x in p.metric_ids if x in m_by_id]
        # A calculation's inputs (and theirs, for one built on another) are part of what it rests on.
        k = 0
        while k < len(cited):
            if cited[k].snapshot_id.startswith("calc:"):
                cited += [m_by_id[u] for u in cited[k].snapshot_id[5:].split(";")
                          if u in m_by_id and m_by_id[u] not in cited]
            k += 1
        out.append({"ref": f"p{i}", "section": p.section, "text": p.text, "quotes": quotes,
                    "metrics": [_metric_view(m) for m in cited],
                    "context": [_context_view(c_by_id[x]) for x in p.context_ids if x in c_by_id]})
    return out


class Verifier:
    role = "verifier"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, memo: DraftMemo, claims: dict[str, Claim], metrics: list[Metric], context: list[ContextItem],
            documents: dict[str, Document], as_of: str, only: set[str] | None = None) -> Verification:
        paragraphs = verify_inputs(memo, claims, metrics, context, documents, only)
        if not paragraphs:
            return Verification(paragraphs=[])
        return self.llm.complete_json(role="verifier", task=VERIFY_TASK, schema=Verification,
                                      inputs={"as_of": as_of, "paragraphs": paragraphs})


def to_flags(check: Verification, stage: str = "verify") -> list[Flag]:
    return [Flag(code="unsupported_statement", ref=p.ref, stage=stage,
                 message=f"verifier ({p.verdict}, {i.kind}): {i.sentence[:160]!r}: {i.why}")
            for p in check.paragraphs if p.verdict != "right" for i in p.issues]
