"""Deterministic checks. The model proposes; these decide.

Nothing here calls a model. A failed check adds a flag or drops a claim; it never
edits the claim or the memo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ..llm.moat import verify_quote
from .schemas import Claim, ContextItem, Document, DraftMemo, Flag, FlagCode, Metric, Plan, Request, ResearchNotes, Unit

METRIC_FRESH_DAYS = 120   # fundamentals: a quarter plus the filing lag
CONTEXT_FRESH_DAYS = 7    # market data and companion outputs

_INJECTION = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|prompts?)"
    r"|disregard (?:all |any |the |your )?(?:previous |prior |above )?instructions"
    r"|you are now\b|new instructions\s*:|system prompt|as an ai(?: language)? model"
    r"|<\s*/?\s*(?:system|instructions?)\s*>",
    re.I,
)


# -- documents and claims --------------------------------------------------------------

def check_documents(docs: list[Document], stage: str) -> list[Flag]:
    flags = []
    for d in docs:
        m = _INJECTION.search(d.text)
        if m:
            flags.append(Flag(code="possible_injection", ref=d.doc_id, stage=stage,
                              message=f"instruction-like text in {d.doc_id}: {m.group(0)!r}"))
    return flags


def gate_claims(notes: ResearchNotes, documents: dict[str, Document], as_of: date,
                stage: str) -> tuple[list[Claim], dict[str, str], list[Flag]]:
    """(claims whose every quote is in its cited document, dropped claim id -> reason,
    flags). A claim with any unverified quote is dropped whole: one made-up quote makes
    the rest of the claim suspect."""
    kept, dropped, flags = [], {}, []
    for c in notes.claims:
        problem: tuple[FlagCode, str] | None = ("quote_not_found", "no evidence") if not c.evidence else None
        for e in c.evidence:
            doc = documents.get(e.doc_id)
            if doc is None:
                problem = ("quote_not_found", f"cites {e.doc_id}, which was not retrieved")
            elif doc.filed > as_of:
                problem = ("after_as_of", f"cites {e.doc_id}, filed {doc.filed}, after the as-of date")
            elif not verify_quote(e.quote, doc.text):
                problem = ("quote_not_found", f"quote not found in {e.doc_id}: {e.quote[:80]!r}")
            if problem:
                break
        if problem:
            code, reason = problem
            dropped[c.id] = reason
            flags.append(Flag(code=code, ref=c.id, stage=stage, message=reason))
        else:
            kept.append(c)
    return kept, dropped, flags


# -- inputs ------------------------------------------------------------------------------

def check_inputs(metrics: list[Metric], context: list[ContextItem], as_of: date,
                 metric_days: int = METRIC_FRESH_DAYS, context_days: int = CONTEXT_FRESH_DAYS) -> list[Flag]:
    flags = []
    for item, default in [(m, metric_days) for m in metrics] + [(c, context_days) for c in context]:
        limit = item.max_age_days if item.max_age_days is not None else default
        if item.as_of > as_of:
            flags.append(Flag(code="after_as_of", ref=item.id, stage="inputs",
                              message=f"{item.name} is dated {item.as_of}, after {as_of}"))
        elif (as_of - item.as_of).days > limit:
            flags.append(Flag(code="stale_input", ref=item.id, stage="inputs",
                              message=f"{item.name} is from {item.as_of}, {(as_of - item.as_of).days} days old"))
        if item.value is None:
            flags.append(Flag(code="missing_input", ref=item.id, stage="inputs", message=f"{item.name} has no value"))
    return flags


def check_coverage(plan: Plan, retrieval: dict[str, list[str]], metrics: list[Metric],
                   context: list[ContextItem]) -> list[Flag]:
    flags = [Flag(code="missing_input", ref=s.id, stage="inputs", message=f"no filing text found for {s.title!r}")
             for s in plan.sections if not retrieval.get(s.id)]
    got_m, got_c = {m.name for m in metrics}, {c.name for c in context}
    flags += [Flag(code="missing_input", ref=n, stage="inputs", message=f"metric {n!r} unavailable")
              for n in plan.metrics if n not in got_m]
    flags += [Flag(code="missing_input", ref=n, stage="inputs", message=f"context {n!r} unavailable")
              for n in plan.context if n not in got_c]
    return flags


def check_lookahead(as_of: date, training_cutoff: date | None) -> list[Flag]:
    if training_cutoff is not None and as_of < training_cutoff:
        return [Flag(code="llm_lookahead", stage="inputs",
                     message=f"as-of {as_of} is before the model's training cutoff {training_cutoff}: "
                             "it may know what happened next")]
    return []


# -- numbers -------------------------------------------------------------------------

_NUM = re.compile(
    r"(?<![\w.\-/])(-?)\$?(?>(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?)"
    r"(?:\s*(%|percent\b|x\b|×|bn\b|billion\b|b\b|million\b|mn\b|m\b|thousand\b|k\b))?(?![\w\-/.]\w)",
    re.I,
)
_SCALE = {"bn": 1e9, "billion": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


@dataclass(frozen=True)
class Number:
    text: str
    value: float        # in natural units: 5.1% -> 0.051, $1.2 billion -> 1.2e9
    tol: float          # half a unit of the last digit shown, in the same units
    suffix: str


def numbers_in(text: str) -> list[Number]:
    """Numbers a reader would take as data. Skipped: years (1900-2100), plain integers
    under 10 (counts, ordinals), and numbers inside identifiers (10-K, Q3, 2026-11-20)."""
    out = []
    for m in _NUM.finditer(text):
        sign, whole, frac, suffix = m.group(1), m.group(2), m.group(3) or "", (m.group(4) or "").lower()
        n = float(whole.replace(",", "") + ("." + frac if frac else ""))
        if not suffix and not frac and "," not in whole and (n < 10 or 1900 <= n <= 2100):
            continue
        half = 0.5 * 10 ** -len(frac)
        scale = 0.01 if suffix in ("%", "percent") else _SCALE.get(suffix, 1.0)
        out.append(Number(m.group(0).strip(), (-n if sign else n) * scale, half * scale, suffix))
    return out


def _candidates(value: float, unit: Unit | None) -> list[float]:
    """Ways a value can be written without a suffix: as is, as a percentage, or (for
    dollars) in thousands, millions or billions."""
    c = [value]
    if unit in ("pct", "ratio"):
        c.append(value * 100)
    if unit == "usd":
        c += [value / 1e3, value / 1e6, value / 1e9]
    return c


def _matches(num: Number, values: list[tuple[float, Unit | None]]) -> bool:
    for v, unit in values:
        if not num.suffix:
            cands = _candidates(v, unit)
        elif num.suffix in ("x", "×"):
            cands = [v] if unit in ("x", None, "ratio") else []
        else:
            cands = [v]
        # Signs are compared loosely: prose says "fell 3%" for -0.03.
        if any(abs(abs(num.value) - abs(c)) <= num.tol + 1e-12 for c in cands):
            return True
    return False


# -- the memo --------------------------------------------------------------------------

def check_memo(memo: DraftMemo, request: Request, claims: dict[str, Claim], dropped: dict[str, str],
               metrics: list[Metric], context: list[ContextItem], stage: str = "memo") -> list[Flag]:
    flags: list[Flag] = []
    m_by_id, c_by_id = {m.id: m for m in metrics}, {c.id: c for c in context}
    cited_claims: set[str] = set()
    for i, p in enumerate(memo.paragraphs):
        ref = f"p{i + 1}"
        if not (p.claim_ids or p.metric_ids or p.context_ids):
            flags.append(Flag(code="uncited_claim", ref=ref, stage=stage, message=f"paragraph {i + 1} cites nothing"))
        for cid in p.claim_ids:
            cited_claims.add(cid)
            if cid in dropped:
                flags.append(Flag(code="uncited_claim", ref=ref, stage=stage,
                                  message=f"paragraph {i + 1} rests on claim {cid}, which failed: {dropped[cid]}"))
            elif cid not in claims:
                flags.append(Flag(code="uncited_claim", ref=ref, stage=stage,
                                  message=f"paragraph {i + 1} cites unknown claim {cid}"))
        for mid in p.metric_ids:
            if mid not in m_by_id:
                flags.append(Flag(code="uncited_claim", ref=ref, stage=stage,
                                  message=f"paragraph {i + 1} cites unknown metric {mid}"))
        for xid in p.context_ids:
            item = c_by_id.get(xid)
            if item is None:
                flags.append(Flag(code="uncited_claim", ref=ref, stage=stage,
                                  message=f"paragraph {i + 1} cites unknown context {xid}"))
            elif item.is_forecast and not item.track_record:
                flags.append(Flag(code="skill_not_stated", ref=ref, stage=stage,
                                  message=f"paragraph {i + 1} uses {item.name}, a forecast with no measured track record"))

        # Every number must come from a cited metric or context item, or from the text
        # or quotes of a cited claim.
        values: list[tuple[float, Unit | None]] = []
        for mid in p.metric_ids:
            mv = m_by_id[mid].value if mid in m_by_id else None
            if mv is not None:
                values.append((mv, m_by_id[mid].unit))
        for xid in p.context_ids:
            xv = c_by_id[xid].value if xid in c_by_id else None
            if isinstance(xv, (int, float)):
                values.append((float(xv), c_by_id[xid].unit))
        texts = [s for x in p.claim_ids if x in claims for s in [claims[x].text, *(e.quote for e in claims[x].evidence)]]
        texts += [c_by_id[x].track_record or "" for x in p.context_ids if x in c_by_id]
        sourced = [n.value for s in texts for n in numbers_in(s)]
        values += [(v, None) for v in sourced]
        text = p.text.replace(request.as_of.isoformat(), " ").replace(request.horizon, " ")
        for num in numbers_in(text):
            if not _matches(num, values):
                flags.append(Flag(code="number_mismatch", ref=ref, stage=stage,
                                  message=f"paragraph {i + 1}: {num.text!r} matches no cited metric or source"))

    for cid, claim in claims.items():
        if claim.role == "bear" and cid not in cited_claims:
            if not memo.omitted_claims.get(cid, "").strip():
                flags.append(Flag(code="bear_case_dropped", ref=cid, stage=stage,
                                  message=f"bear-case claim {cid} is neither in the memo nor listed as omitted"))
    return flags
