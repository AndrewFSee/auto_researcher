"""The workflow: plan, gather, gate the claims, draft, gate the memo, package.

The orchestrator routes, drafts and asks for one redraft when the memo fails its
checks. It cannot clear a flag: the redraft is checked again from scratch, and the
first draft's flags stay in the audit log.
"""

from __future__ import annotations

import hashlib

from . import gates
from .calc import run_calculations
from .agents import BEAR_TASK, FILING_TASK, BearCaseAnalyst, ContextAnalyst, DataAnalyst, FilingAnalyst
from .llm import LLM
from .packet import ReviewPacket, build_packet
from .schemas import SCHEMA_VERSION, Claim, Document, DraftMemo, Flag, Plan, Request
from .state import WorkflowState
from .tools.base import ContextTool, FilingTool, MetricTool

PLAN_TASK = (
    "Turn the request into a plan: memo sections (short ids s1, s2, ...), the questions each must answer "
    "from the company's 10-K and 10-Qs, the metrics to compute and the context items to read. Choose "
    "metrics and context only from the catalogues given, by their exact names; pick what the request "
    "needs, not everything."
)
DRAFT_TASK = (
    "Write the memo from the claims, metrics and context given, and nothing else. Every paragraph lists "
    "the ids it rests on. Every number in the prose must be a cited metric or context value, a number in a "
    "cited claim's quotes, or a calculation: never compute a number yourself. For any derived figure (a sum, "
    "difference, ratio, share, yield or growth rate) add a calculation with a formula over metric ids, "
    "context ids and quoted numbers (claim f6's quoted 21,155 is f6[\"21,155\"]; a table in millions needs "
    "* 1e6), and cite its id (c1, c2, ...) in metric_ids. Use every "
    "bear-case claim or list it in omitted_claims with a reason. When a paragraph uses a forecast, say "
    "what its track record is. Say plainly what is missing, in paragraphs of kind 'limitations', which cite "
    "nothing and state no figures; every other paragraph is 'analysis'. "
    "Read each metric and context item by its definition: a low spread percentile, for example, means credit "
    "is cheap and easy, not tight. A forecast or score whose track record shows no skill at the memo's horizon "
    "may be reported with that caveat, but no conclusion about the horizon may rest on it, in a paragraph's "
    "opening sentence or anywhere else. State as fact only what a quote, metric or context item shows: mark "
    "your own inferences as inferences, and do not name events, terms or causes that no quote mentions. A "
    "claim that something changed (a re-rating, multiple expansion, slowing margins, 'already priced in') "
    "needs values for both dates, such as the _1y_ago metrics; a price return alone does not show it. Compare "
    "figures for the same date or period; when you must combine different ones (year-end debt with June cash, "
    "one quarter with a half-year), say so in the sentence. Each quote shows the form and filing date of its "
    "document: use the latest period the evidence covers. Work out which way a distortion moves a ratio from "
    "its formula: a one-off gain raises earnings, so it lowers P/E. input_flags lists problems found before "
    "drafting: the limitations paragraphs must cover each one. If feedback is given, fix exactly those "
    "problems."
)
# Problems found before drafting that the memo must disclose (the pilot's limitations
# paragraphs left out a missing-filing-text flag the drafter was never shown).
INPUT_FLAGS = ("missing_input", "stale_input", "llm_lookahead")


def prompt_version() -> str:
    return hashlib.sha256("\n".join([PLAN_TASK, DRAFT_TASK, FILING_TASK, BEAR_TASK]).encode()).hexdigest()[:8]


def _input_flags(flags: tuple[Flag, ...], metric_names: dict[str, str]) -> list[str]:
    """The pre-draft flags the memo must disclose, one line each, except that metrics with
    no value (one flag per metric) are listed together."""
    def no_value(f: Flag) -> bool:
        return f.code == "missing_input" and f.ref in metric_names and f.message.endswith("has no value")

    empty = [metric_names[f.ref] for f in flags if no_value(f) and f.ref]
    out = [f"{f.code} ({f.ref}): {f.message}" for f in flags if f.code in INPUT_FLAGS and not no_value(f)]
    if empty:
        out.append(f"missing_input (metrics): no value for {', '.join(empty)}")
    return out


def _claim_inputs(claims: dict[str, Claim], documents: dict[str, Document]) -> list[dict]:
    """Claims as the drafter sees them: each quote with its document's form and filing date."""
    out = []
    for c in claims.values():
        d = c.model_dump(mode="json")
        for e in d["evidence"]:
            doc = documents.get(e["doc_id"])
            if doc is not None:
                e["form"], e["filed"] = doc.form, doc.filed.isoformat()
        out.append(d)
    return out


class Workstation:
    def __init__(self, llm: LLM, filings: FilingTool, metrics: MetricTool, context: ContextTool,
                 max_redrafts: int = 1) -> None:
        self.llm, self.max_redrafts = llm, max_redrafts
        self.filings, self.metrics_tool, self.context_tool = filings, metrics, context
        self.versions = {"schema": SCHEMA_VERSION, "prompts": prompt_version(), "model": llm.model,
                         "filings": filings.version, "metrics": metrics.version, "context": context.version}

    def run(self, request: Request) -> tuple[ReviewPacket, WorkflowState]:
        state = WorkflowState(request, self.versions)
        state.log("orchestrator", "request_received", request=request.model_dump(mode="json"))
        state.add_flags("gate", gates.check_lookahead(request.as_of, self.llm.training_cutoff))

        # Offer only the context items that exist for this stock and date (vol covers 41
        # names, augury 28; snapshots exist only near today).
        offered = self.context_tool.available()
        present = {c.name for c in self.context_tool.context(request.ticker, list(offered), request.as_of)}
        plan = self.llm.complete_json(role="orchestrator_plan", task=PLAN_TASK, schema=Plan,
                                      inputs={"request": request.model_dump(mode="json"),
                                              "metric_catalogue": self.metrics_tool.available(),
                                              "context_catalogue": {k: v for k, v in offered.items() if k in present}})
        state.write("orchestrator", "plan", plan)

        FilingAnalyst(self.llm, self.filings).run(state)
        BearCaseAnalyst(self.llm, self.filings).run(state)
        DataAnalyst(self.metrics_tool).run(state)
        ContextAnalyst(self.context_tool).run(state)

        state.add_flags("gate", gates.check_documents(list(state.documents.values()), stage="documents"))
        notes, bear = state.need_notes("research_notes"), state.need_notes("bear_notes")
        kept_f, dropped_f, fl_f = gates.gate_claims(notes, state.documents, request.as_of, "claims")
        kept_b, dropped_b, fl_b = gates.gate_claims(bear, state.documents, request.as_of, "claims")
        state.add_flags("gate", fl_f + fl_b)
        state.add_flags("gate", gates.check_inputs(state.metrics, state.context, request.as_of))
        state.add_flags("gate", gates.check_coverage(plan, state.retrieval, state.metrics, state.context))
        state.add_flags("gate", gates.check_text_vs_numbers(list(state.documents.values()), state.metrics))
        claims = {c.id: c for c in kept_f + kept_b}
        dropped = {**dropped_f, **dropped_b}

        input_flags = _input_flags(state.flags, {m.id: m.name for m in state.metrics})
        feedback: list[str] = []
        for attempt in range(1, self.max_redrafts + 2):
            memo = self.llm.complete_json(
                role="orchestrator_draft", task=DRAFT_TASK, schema=DraftMemo,
                inputs={"request": request.model_dump(mode="json"), "plan": plan.model_dump(mode="json"),
                        "claims": _claim_inputs(claims, state.documents),
                        "metrics": [m.model_dump(mode="json") for m in state.metrics],
                        "context": [c.model_dump(mode="json") for c in state.context],
                        "gaps": [*notes.gaps, *bear.gaps],
                        "input_flags": input_flags,
                        "feedback": feedback})
            state.write("orchestrator", "draft_memo", memo)
            calcs, errors = run_calculations(memo.calculations, state.metrics, state.context, claims, request.as_of)
            state.write("calculator", "calculations", calcs)
            memo_flags = [Flag(code="calculation_error", ref=cid, stage=f"memo_draft_{attempt}",
                                     message=f"calculation {cid} failed: {err}") for cid, err in errors.items()]
            memo_flags += gates.check_memo(memo, request, claims, dropped, [*state.metrics, *calcs], state.context,
                                           stage=f"memo_draft_{attempt}", documents=state.documents)
            if not any(f.blocking for f in memo_flags) or attempt > self.max_redrafts:
                break
            state.log("gate", "draft_failed", draft=attempt,
                      flags=[f.model_dump(mode="json") for f in memo_flags])
            feedback = [f"{f.code} ({f.ref}): {f.message}" for f in memo_flags if f.blocking]
        state.add_flags("gate", memo_flags)
        calls = getattr(self.llm, "calls", None)
        if calls and isinstance(calls[0], dict) and "usage" in calls[0]:
            state.log("orchestrator", "llm_usage", calls=calls, cost=round(sum(c["cost"] or 0 for c in calls), 4))
        state.log("orchestrator", "packet_built", drafts=len(state.drafts))
        return build_packet(state, claims, dropped), state
