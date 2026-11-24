"""Repair: fix the paragraphs a failed draft's checks name, instead of redrafting the memo.

In the 2026-10-15 pilots every redraft was triggered by one to three figures, yet a
redraft rewrote the whole memo and roughly doubled its cost. A repair returns only the
named paragraphs, rewritten, plus any calculations, omissions or added paragraphs the
problems need. Code splices them in, and the repaired memo is checked again from
scratch like any draft: the repair cannot clear a flag, only remove its cause.
"""

from __future__ import annotations

from .schemas import DraftMemo, Flag, MemoRepair

REPAIR_TASK = (
    "A draft memo failed the checks listed in problems. Fix exactly those problems and change nothing else. "
    "Return each paragraph a problem names (by its ref, p1, p2, ... as numbered in memo), rewritten whole, "
    "keeping what it says where the evidence supports it; or list its ref in drop if it cannot be supported. "
    "For a figure with no source, cite the metric, claim or calculation that holds it, add a calculation, or "
    "remove the figure. For a failed calculation, return a corrected one with the same id, or drop it and "
    "fix the paragraphs that cite it. For a bear-case claim the memo neither uses nor omits, add a paragraph "
    "that uses it or list it in omitted_claims with a reason. Paragraphs no problem names stay as they are. "
    "The memo's rules, which the fixed paragraphs must follow: "
)   # the orchestrator appends the drafting task


def repairable(flags: list[Flag]) -> set[str]:
    """The paragraph refs the blocking flags name: only these may be replaced or dropped."""
    return {f.ref for f in flags if f.blocking and f.ref and f.ref.startswith("p") and f.ref[1:].isdigit()}


def apply_repair(memo: DraftMemo, repair: MemoRepair, allowed: set[str]) -> tuple[DraftMemo, list[str]]:
    """(the repaired memo, notes on changes refused). Replacements and drops apply only to
    the refs in ``allowed``; added paragraphs go before the limitations paragraphs."""
    refused: list[str] = []
    paras = {f"p{i}": p for i, p in enumerate(memo.paragraphs, 1)}
    for fix in repair.fixes:
        if fix.ref in allowed and fix.ref in paras:
            paras[fix.ref] = fix.paragraph
        else:
            refused.append(f"fix of {fix.ref}: no problem named it")
    for ref in repair.drop:
        if ref in allowed and ref in paras:
            del paras[ref]
        else:
            refused.append(f"drop of {ref}: no problem named it")
    kept = list(paras.values())
    at = next((i for i, p in enumerate(kept) if p.kind == "limitations"), len(kept))
    kept[at:at] = [p.model_copy(update={"kind": "analysis"}) for p in repair.added]

    omitted = {o.claim_id: o for o in memo.omitted_claims}
    omitted.update({o.claim_id: o for o in repair.omitted_claims})
    calcs = {c.id: c for c in memo.calculations if c.id not in set(repair.drop_calculations)}
    calcs.update({c.id: c for c in repair.calculations})
    return memo.model_copy(update={"paragraphs": kept, "omitted_claims": list(omitted.values()),
                                   "calculations": list(calcs.values())}), refused
