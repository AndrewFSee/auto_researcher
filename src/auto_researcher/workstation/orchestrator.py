"""The workflow: plan, gather, gate the claims, draft, gate the memo, package.

The orchestrator routes, drafts and asks for one redraft when the memo fails its
checks. It cannot clear a flag: the redraft is checked again from scratch, and the
first draft's flags stay in the audit log.
"""

from __future__ import annotations

import hashlib

from . import gates
from .agents import BEAR_TASK, FILING_TASK, BearCaseAnalyst, ContextAnalyst, DataAnalyst, FilingAnalyst
from .llm import LLM
from .packet import ReviewPacket, build_packet
from .schemas import SCHEMA_VERSION, DraftMemo, Plan, Request
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
    "the ids it rests on; every number must come from a cited metric, context item or claim. Use every "
    "bear-case claim or list it in omitted_claims with a reason. When a paragraph uses a forecast, say "
    "what its track record is. Say plainly what is missing, in paragraphs of kind 'limitations', which cite "
    "nothing and state no figures; every other paragraph is 'analysis'. If feedback is given, fix exactly "
    "those problems."
)


def prompt_version() -> str:
    return hashlib.sha256("\n".join([PLAN_TASK, DRAFT_TASK, FILING_TASK, BEAR_TASK]).encode()).hexdigest()[:8]


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

        feedback: list[str] = []
        for attempt in range(1, self.max_redrafts + 2):
            memo = self.llm.complete_json(
                role="orchestrator_draft", task=DRAFT_TASK, schema=DraftMemo,
                inputs={"request": request.model_dump(mode="json"), "plan": plan.model_dump(mode="json"),
                        "claims": [c.model_dump(mode="json") for c in claims.values()],
                        "metrics": [m.model_dump(mode="json") for m in state.metrics],
                        "context": [c.model_dump(mode="json") for c in state.context],
                        "gaps": [*notes.gaps, *bear.gaps],
                        "feedback": feedback})
            state.write("orchestrator", "draft_memo", memo)
            memo_flags = gates.check_memo(memo, request, claims, dropped, state.metrics, state.context,
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
