# Research workstation: design

Status: Phase 2 (real read-only tools and the Claude client), not yet run live. Written 2026-11-21.

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
   must be a cited metric or context value, a number in a cited claim's verified
   quotes, or a calculation. A derived figure (a sum, ratio, share, yield) is a
   calculation the draft declares as a formula over metric ids, context ids and
   quoted numbers (`(f10["132.4"] - f10["91.3"]) * 1e9`); code evaluates it
   (`calc.py`, a whitelist of arithmetic parsed with `ast`) and it is cited like a
   metric, with its formula shown to the reviewer. A claim's own wording is the
   model's and never counts as a source.
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
| `calculation_error` | yes | a declared calculation does not evaluate (unknown id, a quoted number not in the claim, a form not allowed) |

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
2. **Real read-only tools:** filings (`tools/filings.py`: sentiq's lake, 2018+, latest
   10-K and the 10-Qs after it, BM25 over located sections), fundamentals and prices
   (`tools/market.py`: TickerLake company facts through `sec_fundamentals`' point-in-time
   table, TickerLake OHLCV; 31 metrics), hub context (`tools/hub.py`: 18 items from
   market-compass, histories as of the date, snapshots only near today, track records
   from published scores or the project README with its source), the Claude client
   (`claude.py`: Opus 5.5 plans and drafts, Sonnet 5.5 reads filings; streamed, strict
   JSON schemas, cost per call) and `scripts/workstation.py`. Still to come: the research
   log as a tool, filings before 2018, 8-K earnings releases.
3. **Pilot:** ten memos, your blind review, and a review page in the Streamlit app.
   Batch mode (`batch.py`, `scripts/workstation.py batch`): each company's workflow runs
   in its own thread; whenever all of them wait on Claude, their calls go out as one
   Message Batch (half price). Replies are cached by request hash and batches in flight
   are recorded, so an interrupted pilot reruns at no extra cost. First pilot, as of
   2026-10-15 (the filing store's last date), 10 companies: $4.69, five batch rounds of
   about two minutes. Its blocking flags were mostly the number check's own false alarms
   (written dates, metric windows, model codes, trillions, figures from tables stated in
   millions, "what is missing" paragraphs); those are fixed, and memo paragraphs now have
   a kind (`analysis` or `limitations`; a limitations paragraph cites nothing and states
   no figures). What the check still catches in first drafts is real: figures the model
   computed itself, and figures cited without the claim or metric that holds them.

   **Your review of the ten final memos** (the review page, 2026-11-23): of 181
   analysis paragraphs, 166 right, 15 overstated, none wrong; 3 memos useful as they
   stand, 7 needing edits, none not useful. A note on a "right" mark means a borderline
   right that could have been "overstated": 55 of them. No arithmetic error survived the
   calculator. What remains is reasoning beyond the evidence:
   * "benign backdrop" headlines resting on the macro and credit models the same
     paragraph says have no 12-month skill, and a 2nd-percentile Baa spread read as
     "tight credit" (nine of ten memos);
   * "re-rating" or "already priced in" from a 12-month return, with no past multiple;
   * figures from different dates or periods compared as if they were the same date
     (year-end debt with June cash, a half-year item in a quarter's bridge);
   * statements no quote supports (named probes, "no minimum commitment");
   * a few reversed directions (a one-off gain said to inflate P/E);
   * the volatility premium compared with 63-day realized vol, though it uses 21-day;
   * a missing-filing-text flag left out of the "what is missing" paragraphs.

   Two of these were gaps in what the drafter was given: it saw each metric and
   context item only by its short name, and never saw the flags raised before drafting.
   Fixes, none of which change the plan's or the analysts' inputs (so their cached
   replies still serve a rerun):
   * every metric and context item carries a `definition`: the catalogue line plus how
     to read it (`READINGS` in `tools/hub.py` and `tools/market.py`), such as which
     direction of the spread percentile means easy credit, the premium's 21-day window,
     and the fact that the nowcast is not a forecast;
   * when the plan asks for a margin or a valuation multiple, the data analyst also
     gets its value from a year earlier (`<name>_1y_ago`, computed only from data
     public then);
   * each quote reaches the drafter with its document's form and filing date;
   * `input_flags`: the pre-draft flags, which the "what is missing" paragraphs must
     cover (metrics with no value are listed together);
   * drafting rules: read values by their definitions; a model with no skill at the
     horizon cannot carry a conclusion about it; mark inferences as inferences; a change
     needs both values; say so when combining dates or periods; work out the direction
     of a distortion from the formula.

   The rerun with these fixes (10 memos, $4.57 spent, drafts only) turned up two more
   problems. First, TickerLake's closes are split-adjusted back in time while filed share
   counts are not. Year-ago market caps, and so NFLX's year-ago P/E, came out 10 times too
   small, and its share-count change came out at +880%; both memos were given the latter.
   Share counts are now put on the closes' split basis (`TickerLakeMetrics.on_price_basis`).
   Second, one JPM redraft spent its 32k output tokens, most of them on thinking, and was
   cut off; drafts now get 64k (`DRAFT_MAX_TOKENS`).

   **Verifier** (`verify.py`, `scripts/workstation_verify.py`). A Sonnet 5.5 call reads
   each analysis paragraph against only what it cites: the verified quotes (never a
   claim's wording), values with their definitions and dates, and calculations with
   their inputs. It returns a verdict per paragraph (right, over or wrong) and the
   sentences at fault. Measured on the ten memos you reviewed, at $0.61 for all ten,
   it flagged:
   * 13 of your 15 overstated paragraphs;
   * 24 of 55 borderline ones (right with a note);
   * 22 of 111 clean ones, mostly strict "no cited quote says this" findings.

   Its four "wrong" verdicts are paragraphs you marked right whose notes name the same
   error (a reversed P/E direction, and "tight credit" three times). It is advisory: it
   raises `unsupported_statement` flags, which do not block, and it is not yet a step of
   the workflow.

   **Draft A/B**, judged by the verifier. Plans and filing reads are shared, from cache.
   Costs are per memo at batch prices, including the cached calls.

   | | words | cost | blocked | paragraphs flagged | issues per memo |
   |---|---|---|---|---|---|
   | Original (Opus) | 2,208 | $0.38 | 0 | 33% | 7.2 |
   | Opus, review fixes | 2,408 | $0.48 | 1 | 17% | 3.4 |
   | Opus, review fixes, about 12 paragraphs | 2,044 | $0.48 | 0 | 12% | 1.6 |
   | Sonnet, medium, about 12 paragraphs | 1,035 | $0.21 | 5 | 27% | 3.7 |

   With the length target, Opus writes fewer but longer paragraphs, so it costs no less.
   The cost lever is the redraft. Memos passing on the first draft cost about $0.20; the
   five that were redrafted cost about $0.47 each. Every redraft was triggered by one to
   three figures, or in one case by a calculation that used another calculation's result,
   which the calculator did not allow.

   **Repair instead of redraft** (`repair.py`, now the default). A failed draft goes back
   with its problems. The model returns only the paragraphs they name, rewritten, and any
   calculations, omissions or added paragraphs the problems need. Code splices these in,
   refuses changes to paragraphs no problem names, and checks the repaired memo again from
   scratch. Calculations may now use other calculations (`c8 = c7 / m3`). The drafter is
   not told; it already did so unprompted.

   The run below used the same cached first drafts as the 12-paragraph run, so it differs
   only in how failures were fixed. NEE's draft now passes with no fix, because its chained
   calculation evaluates. The other four failures were repaired at about $0.075 each,
   against about $0.27 for a redraft. All ten memos pass.

   | 12 paragraphs | cost per memo | paragraphs flagged | issues per memo |
   |---|---|---|---|
   | redraft | $0.48 | 12% | 1.6 |
   | repair | $0.38 | 15% | 1.9 |

   The small difference in flags is in paragraphs a redraft rewrites but a repair keeps:
   NEE is 4 of 12 unrepaired against 2 of 12 redrafted. The first draft is now most of
   the cost, about $0.21 a memo.

   **The verifier as a workflow step** (`Workstation(verify=True)`, on by default in
   `scripts/workstation.py`; `--no-verify` skips it):
   1. Once a memo passes the gates, the verifier reads it.
   2. The paragraphs it faults go to the same repair, once.
   3. The repaired memo is checked against the gates again. If any gate now blocks, the
      memo that passed is kept, so the verifier cannot make a memo fail its gates.
   4. Only the paragraphs that changed are verified again.
   5. What remains becomes advisory `unsupported_statement` flags on the packet.

   On the 2026-10-15 pilot, from the same cached drafts: the verifier faulted 17 of 116
   analysis paragraphs, and after one repair 4 remain. No repair was reverted, and all
   ten memos pass. The step costs about $0.12 a memo (verify, repair, re-verify), so a memo
   costs about $0.50 at batch prices.

   The verifier grades the paragraphs it had repaired, so "4 remain" is not an
   independent measure. A reviewer's marks on these memos are the test.

   A caveat: the verifier reads the same definitions the fixed drafter now gets, so part
   of the drop may be the two agreeing rather than the memo being better. Sonnet's five
   blocked memos are real number-gate catches: figures that appear only in a claim's
   wording or in an uncited claim's quote, and a ratio it computed itself.
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
