# Research workstation: design

Status: Phase 1 (mocked prototype). Written 2026-11-21.

The workstation answers a request like *"Prepare a memo on ACME as of 2026-11-20:
earnings quality, leverage, downside risks"* with a **review packet**: a memo whose
every material sentence links to a verified quote from a filing or to a number a
tool computed, plus the flags a reviewer must clear before using it. It drafts and
assembles research; it never trades or recommends a position size.

It follows the governed-workstation pattern in *Large Language Models in Finance*
(Noguer i Alonso), sections 5.1.3-5.1.5, 5.3.5, 5.4.11-5.4.13, 5.7, 7.1.4 and 7.4.4,
and adds things the book does not have: an as-of date on every run, tools that
state their own measured skill, and memos that can be scored later.

## Principles

1. **The model proposes; code disposes.** LLM output is a proposal. Deterministic
   gates (schema, quote matching, number reconciliation, as-of, staleness) decide
   whether it enters the memo. A failed check rejects or escalates; it never
   "repairs" the claim. No path bypasses the gates.
2. **Point in time.** Every run has an `as_of` date. Agents see only what was
   public by then (filings by filing date, prices to the close of `as_of`, published
   companion data by its own date). Code enforces this, not the prompt: filing
   searches span years, so later filings are dropped before an agent sees them (and
   the drop is logged); a metric or context tool asked for `as_of` that returns later
   data is broken, so that raises a blocking flag. Each tool also declares how old
   its data may be (`max_age_days`: monthly macro data, daily prices).
3. **Structured before prose.** Specialists return typed objects (claims, evidence,
   metrics, flags). The prose memo is written last, from those objects, and each
   paragraph lists the claim ids it rests on.
4. **Numbers come from code.** A metric is computed by a tool from data, with a
   snapshot id. The LLM may choose which metrics to show; any number in the memo
   must match a metric.
5. **Tools carry their track record.** A tool whose output is a forecast or a signal
   declares its measured skill (from this repo's research log or the companion
   project's own walk-forward results). The memo must state it when it leans on one;
   it cannot present a signal tested as null here as an edge.
6. **Tool output is data, never instructions.** Retrieved text goes in the user turn,
   clearly delimited. Instruction-like text inside a filing raises a flag.
7. **The orchestrator has no override authority.** It can ask a specialist to redo
   work. It cannot clear a flag, drop a disagreement or invent missing evidence.
8. **Everything is recorded.** Each run writes its state, every tool call, the
   documents and snapshots used (with hashes) and a version vector: model, prompt,
   tool, schema and corpus versions.

## Roles

| Role | What it does | LLM? | Writes |
|---|---|---|---|
| Orchestrator | Parses the request into a plan (sections, questions, data needs); routes; asks for reruns; writes the memo from typed outputs | Claude Opus 5.5 | `plan`, `draft_memo` |
| Filing analyst | Retrieves passages from the company's 10-K/10-Q/8-K filings as of the date; returns claims, each with verbatim quotes | Claude Sonnet 5.5 (local Qwen3-8B fine-tune for numerical questions) | `research_notes` |
| Bear-case analyst | Looks for disconfirming evidence on its own (risk factors, auditor and accounting changes, going-concern language, litigation, customer concentration, covenant terms) | Claude Sonnet 5.5 | `bear_notes` |
| Data analyst | Computes point-in-time fundamentals, valuation, price and risk metrics | No (code) | `metrics` |
| Context analyst | Reads the market backdrop and per-stock companion outputs (below) with their track records | No (code); LLM only to phrase | `context` |
| Evidence gate | Deterministic checks on every claim and on the memo | No | `flags` |
| Human reviewer | Approves, requests a revision, or rejects; the decision is logged | You | `decision` |

The bear-case analyst is a separate researcher with its own retrieval, not a critic
re-reading the draft: per section 5.4.12 a critic is worth having only where it has
independent evidence. Verification is the evidence gate's job.

## Tools

Read-only. Each takes `as_of` and returns data plus a snapshot id.

**From this repo (measured):**

| Tool | Source | Measured |
|---|---|---|
| `filings.list` / `filings.search` | sentiq's EDGAR lake (2018+) and `data/customers/early_10k` (2009-2017); sections via sentiq's locator; `llm/retrieval.py` BM25 + re-ranker | right table found 69%; answers right 77.6% end to end |
| `filings.ask_number` | FinQA fine-tune (local) | 95.3% within 1% given the right page |
| `fundamentals.asof` | `data/sec_fundamentals.py` (first-reported XBRL, usable after filing) | causality-tested |
| `prices.history` | TickerLake OHLCV (survivorship-free) or yfinance; not sentiq's Tradier files, which are a rolling window | - |
| `moat.tags` | `llm/moat.py` | 85% agreement with blind review; **has lookahead** for past dates |
| `customers.links` | `data/customers/` links 2009-2026 | 85% correct on a blind check |
| `research_log.findings` | `docs/results/` | the verdict of every study, e.g. earnings drift not tradeable after 2001 |

**From the companion projects, read from market-compass's published data
(`market-compass/data/<module>/`, with `manifest.json` timestamps):**

| Tool | Project | Gives | Note |
|---|---|---|---|
| `macro.regime` | macro-regime-nowcaster | recession probability, five macro factors; walk-forward history since 1967 | point-in-time history exists |
| `credit.stress` | credit-spread-analyzer | stress-regime probability, spreads, 5-day forecast; history since 1992 | |
| `vol.surface` | stochastic-vol-surface | ATM vol by tenor, skew, term structure and their percentiles; 41 underlyings | latest snapshot only on the hub |
| `augury.distribution` | augury | 1-63 day return bands, VaR/ES, P(up) with track record; 28 tickers | |
| `style.scores` | TickerLake | value and growth scores, ratios; S&P 500 | latest snapshot |
| `buybacks.status` | TickerLake | repurchases, authorization, blackout window | latest snapshot |
| `revisions.names` | TickerLake | 30/90-day EPS-estimate revisions | history starts when collection began |
| `tail_risk.scenarios` | tail-risk-scenario-engine | VaR/CVaR and stress scenarios for a portfolio | later: "what if this name is added" |

Hub tools that hold only today's snapshot can serve runs dated today only; for an
earlier `as_of` they return nothing rather than today's values. Later candidates:
haruspex event probabilities (macro event risk for the horizon), proteus-finance
break detection on the stock's returns.

## Workflow

1. **Plan.** The orchestrator turns the request into a `Plan`: sections, questions
   for each, metrics needed, and which context tools apply.
2. **Gather** (in parallel, each specialist gets only its slice of state): filing
   analyst, bear-case analyst, data analyst, context analyst.
3. **Gate the claims.** Every quote is matched against the stored filing text;
   unmatched quotes drop the claim, and the drop is logged.
4. **Draft.** The orchestrator writes the memo from the surviving claims, metrics and
   context. Each paragraph lists its claim and metric ids.
5. **Gate the memo.** Each paragraph's claims exist and passed; every number in the
   prose matches a cited metric; nothing is dated after `as_of`; stale inputs are
   named; bear-case claims are not dropped (each is either in the memo or listed as
   omitted with a reason); claims leaning on a tool state its track record.
6. **Packet.** Memo, claims with evidence, metrics, context, flags, audit log, version
   vector, and the allowed actions. Blocking flags disable "approve".
7. **Review.** You approve, request a revision (with notes, which become a rerun) or
   reject. The decision is stored with the run.

## Flags

| Code | Blocking | Raised when |
|---|---|---|
| `quote_not_found` | no (claim dropped) | a quote is not in the cited document |
| `uncited_claim` | yes | a memo paragraph cites nothing, or cites a claim that failed verification or an input that was never fetched |
| `number_mismatch` | yes | a number in the prose matches no cited metric |
| `after_as_of` | yes | evidence or data is dated after `as_of` |
| `stale_input` | no | data older than its freshness limit |
| `missing_input` | no | a planned input could not be fetched |
| `bear_case_dropped` | yes | a bear-case claim neither appears nor is listed as omitted |
| `skill_not_stated` | yes | a paragraph uses a forecast or signal that has no measured track record (the packet prints the record next to every paragraph that cites one) |
| `possible_injection` | yes | retrieved text contains instruction-like content |
| `llm_lookahead` | no | `as_of` is before the model's training cutoff (the model may know what happened) |

## Storage

`data/workstation/runs/<run_id>/`: `state.json` (the full workflow state, including the
packet), `audit.jsonl` (one line per event and tool call), and `decision.json` once
reviewed. Runs are written once; a revision is a new run that names its parent.

## Evaluation

* **Fixture suite (Phase 1, pytest):** normal request; stale market data; missing
  filing; injection inside a filing; a quote the model made up; a number in the prose
  that matches no metric (and the one redraft that fixes it); a dropped bear-case
  claim; data dated after `as_of`; a forecast with no track record; the lookahead
  flag; write permissions. Each
  case states the expected flags and whether review is required. Every model, prompt,
  tool or schema change reruns it.
* **Hand check (Phase 3):** about ten companies, reviewed by you with the moat
  tagger's protocol: correctness of each claim, usefulness, what is missing.
* **Point-in-time scoring (Phase 4):** memos as of past dates, scored on what
  happened next (earnings, drawdowns, the flagged risks). For an `as_of` before the
  model's training cutoff, Claude may already know what happened next, so those memos
  are flagged `llm_lookahead`; a fair test needs anonymized text or the local model,
  as for the moat tags.

## Phases

1. **Mocked prototype** (this commit): schemas, state with write permissions, gates,
   fixture tools, a scripted LLM, the orchestrator, the review packet, run storage
   and the fixture suite. No network, no API calls.
2. **Real read-only tools:** filings, fundamentals, prices, research log, hub data;
   the Claude client (sentiq's key); a CLI `python scripts/workstation.py memo ACME
   --as-of 2026-11-20`.
3. **Pilot:** ten memos, your blind review, and a review page in the Streamlit app.
4. **Extensions:** point-in-time scoring; peer comparison (several tickers); an MCP
   server exposing the tools so Claude Code and the desktop app can use them;
   study memos (option B: hypothesis, pre-registration, test) on the same skeleton.

## Layout

```
src/auto_researcher/workstation/
  schemas.py        typed objects: Request, Plan, Evidence, Claim, Metric, ContextItem,
                    DraftMemo, Flag, ReviewPacket
  state.py          WorkflowState: per-role write permissions, audit log, version vector
  llm.py            LLM protocol; ScriptedLLM for tests (Claude client in Phase 2)
  tools/base.py     tool protocols, as-of enforcement, snapshot ids
  tools/fixtures.py fixture-backed tools
  agents.py         filing, bear-case, data and context analysts
  gates.py          deterministic checks -> flags
  orchestrator.py   the workflow
  packet.py         review packet and allowed actions
  runs.py           run storage
tests/test_workstation.py   the fixture suite
```
