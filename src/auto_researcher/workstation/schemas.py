"""Typed objects passed between the workstation's roles (docs/workstation/DESIGN.md).

Specialists return these, not prose; the memo is written last and every paragraph
names the claims, metrics and context items it rests on.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class Request(BaseModel):
    ticker: str
    as_of: date
    question: str
    horizon: str = "12m"


class Section(BaseModel):
    id: str
    title: str
    questions: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    sections: list[Section]
    metrics: list[str] = Field(default_factory=list, description="names of metrics the data analyst computes")
    context: list[str] = Field(default_factory=list, description="names of context tools to read")


class Document(BaseModel):
    """A retrieved passage of a filing. ``filed`` is when it became public."""
    doc_id: str
    ticker: str
    form: str
    filed: date
    section: str | None = None
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class Evidence(BaseModel):
    doc_id: str
    quote: str = Field(description="verbatim text from the document")


class Claim(BaseModel):
    id: str
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    section: str | None = Field(default=None, description="plan section id")
    stance: Literal["supports", "against", "neutral"] = "neutral"
    role: Literal["filing", "bear"] = "filing"


class ResearchNotes(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, description="questions the documents could not answer")


Unit = Literal["ratio", "pct", "usd", "x", "days", "count"]


class Metric(BaseModel):
    """A number a tool computed. ``value`` is in natural units: a 5.1% growth rate is
    0.051 with unit "pct"; 2.4x leverage is 2.4 with unit "x"; dollars are dollars."""
    id: str
    name: str
    value: float | None
    unit: Unit
    as_of: date = Field(description="date of the data the value is computed from (period end, last close)")
    available_date: date | None = Field(default=None, description="when the inputs became public (filing date)")
    period: str | None = None
    source: str
    snapshot_id: str
    max_age_days: int | None = Field(default=None, description="older than this is stale; None: the gate's default")
    definition: str | None = Field(default=None, description="what the number is and how to read it (from the tool)")


class ContextItem(BaseModel):
    """Market backdrop or a companion project's per-stock output."""
    id: str
    name: str
    value: float | str | None
    unit: Unit | None = None
    as_of: date
    source: str
    is_forecast: bool = False
    track_record: str | None = Field(default=None, description="measured out-of-sample skill, if any")
    max_age_days: int | None = Field(default=None, description="older than this is stale; None: the gate's default")
    definition: str | None = Field(default=None, description="what the value is and how to read it (from the tool)")


class Paragraph(BaseModel):
    section: str
    kind: Literal["analysis", "limitations"] = Field(
        default="analysis", description="'limitations': only says what is missing; cites nothing, states no figures")
    text: str
    claim_ids: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    context_ids: list[str] = Field(default_factory=list)


class Calculation(BaseModel):
    id: str = Field(description="c1, c2, ...")
    description: str = Field(description="what it is, e.g. 'buybacks plus dividends over free cash flow'")
    # Calculations may also use other calculations' ids (calc.run_calculations); the text
    # is left unchanged so cached drafts stay valid (the drafter already chains unprompted).
    expression: str = Field(description="formula over metric ids (m3), context ids (x1) and quoted filing numbers "
                                        "(f6[\"21,155\"]), with + - * / ( ), abs, min, max and constants")
    unit: Unit


class Omission(BaseModel):
    claim_id: str
    reason: str


class DraftMemo(BaseModel):
    title: str
    paragraphs: list[Paragraph]
    omitted_claims: list[Omission] = Field(default_factory=list, description="claims left out, each with a reason")
    calculations: list[Calculation] = Field(default_factory=list, description="derived numbers, computed by code")

    def omitted(self) -> dict[str, str]:
        return {o.claim_id: o.reason for o in self.omitted_claims}


class ParagraphFix(BaseModel):
    ref: str = Field(description="the paragraph replaced: p1, p2, ... as numbered in the memo given")
    paragraph: Paragraph


class MemoRepair(BaseModel):
    """Changes to a draft that failed its checks: only the paragraphs named in the
    problems are replaced or dropped; the rest of the memo stays as it was."""
    fixes: list[ParagraphFix] = Field(default_factory=list, description="each named paragraph, rewritten whole")
    drop: list[str] = Field(default_factory=list, description="refs of named paragraphs to remove")
    added: list[Paragraph] = Field(default_factory=list,
                                   description="new analysis paragraphs, e.g. one using a bear-case claim not yet used")
    omitted_claims: list[Omission] = Field(default_factory=list, description="claims to list as omitted, with reasons")
    calculations: list[Calculation] = Field(default_factory=list, description="new calculations, or replacements by id")
    drop_calculations: list[str] = Field(default_factory=list, description="ids of calculations to remove")


FlagCode = Literal["quote_not_found", "uncited_claim", "number_mismatch", "after_as_of", "stale_input",
                   "missing_input", "bear_case_dropped", "skill_not_stated", "possible_injection", "llm_lookahead",
                   "calculation_error", "unsupported_statement"]

BLOCKING: dict[str, bool] = {
    "quote_not_found": False, "uncited_claim": True, "number_mismatch": True, "after_as_of": True,
    "stale_input": False, "missing_input": False, "bear_case_dropped": True, "skill_not_stated": True,
    "possible_injection": True, "llm_lookahead": False, "calculation_error": True,
    "unsupported_statement": False,   # advisory until the verifier is measured against a reviewer
}


class Flag(BaseModel):
    code: FlagCode
    message: str
    ref: str | None = Field(default=None, description="the claim, paragraph, document or input it concerns")
    stage: str = ""

    @property
    def blocking(self) -> bool:
        return BLOCKING[self.code]
