"""
Research workstation: a cited, gated memo on one company as of a date, for human review
(docs/workstation/DESIGN.md).

    python scripts/workstation.py memo WMT --as-of 2026-11-20 --dry-run     # tools only, no API calls
    python scripts/workstation.py memo WMT --as-of 2026-11-20               # full run (Claude)
    python scripts/workstation.py memo WMT --question "Earnings quality and leverage"
    python scripts/workstation.py batch AAPL JPM WMT --as-of 2026-11-20     # several memos, batch API (half price)
    python scripts/workstation.py runs                                      # list runs and decisions
    python scripts/workstation.py decide <run_id> approve_for_internal_use --notes "..."

A run is stored under data/workstation/runs/<run_id>/ (state, packet, audit log and the
memo as Markdown). Claude calls use ANTHROPIC_API_KEY from --env-file (sentiq's .env by
default); the key is never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.workstation import Request, Workstation  # noqa: E402
from auto_researcher.workstation.runs import RUNS, record_decision, save_run  # noqa: E402
from auto_researcher.workstation.schemas import BLOCKING  # noqa: E402
from auto_researcher.workstation.tools.filings import LakeFilings  # noqa: E402
from auto_researcher.workstation.tools.hub import CATALOGUE as CONTEXT, HubContext  # noqa: E402
from auto_researcher.workstation.tools.market import CATALOGUE as METRICS, TickerLakeMetrics  # noqa: E402

logger = logging.getLogger("workstation")
MODELS_BY_NAME = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}
# About 12 paragraphs: on the 2026-10-15 pilot the verifier flagged 12% of paragraphs at
# this length against 17% without a target, at the same cost (docs/workstation/DESIGN.md).
PARAGRAPHS = 12
PARAGRAPHS_HELP = f"length target for the memo, in paragraphs (default {PARAGRAPHS}; 0: none)"
NO_REPAIR_HELP = "redraft a failed memo whole instead of repairing the paragraphs its checks name"
ENV_FILE = Path(r"C:\Users\andre\projects\sentiq\.env")
DEFAULT_QUESTION = ("Pre-investment memo: business and competitive position, earnings quality, balance sheet and "
                    "leverage, valuation, and the main downside risks over the horizon. Flag anything a reviewer "
                    "should check before relying on it.")


def dry_run(req: Request) -> None:
    filings, metrics, hub = LakeFilings(), TickerLakeMetrics(), HubContext()
    docs = filings.documents(req.ticker, req.as_of)
    print(f"{req.ticker} as of {req.as_of}: {len(docs)} filing passages from "
          f"{sorted({(d.form, d.filed.isoformat()) for d in docs})}")
    for m in metrics.metrics(req.ticker, list(METRICS), req.as_of):
        print(f"  metric  {m.name:26s} {m.value!s:>22.22} {m.unit:5s} data {m.as_of}  public {m.available_date}")
    for c in hub.context(req.ticker, list(CONTEXT), req.as_of):
        print(f"  context {c.name:32s} {str(c.value):>22.22} {c.as_of}  {'forecast' if c.is_forecast else ''}")
    for q in ("risk factors", "customer concentration", "liquidity and capital resources"):
        hits = filings.search(req.ticker, q, req.as_of, 3)
        print(f"  search {q!r}: {[d.doc_id for d in hits]}")


def cmd_memo(args) -> None:
    req = Request(ticker=args.ticker.upper(), as_of=date.fromisoformat(args.as_of), question=args.question,
                  horizon=args.horizon)
    if args.dry_run:
        dry_run(req)
        return
    from auto_researcher.workstation.claude import ClaudeLLM

    llm = ClaudeLLM(_client(args.env_file), effort=args.effort)
    ws = Workstation(llm, LakeFilings(), TickerLakeMetrics(), HubContext(), paragraphs=args.paragraphs or None,
                     repair=not args.no_repair)
    try:
        packet, state = ws.run(req)
    except Exception:
        print(f"run failed after {len(llm.calls)} Claude calls costing ${llm.cost:.4f}: "
              f"{[(c['role'], c['stop_reason'], c['cost']) for c in llm.calls]}")
        raise
    folder = save_run(state, packet, root=args.runs)
    blocking = [f for f in packet.flags if f.blocking]
    print(f"run {folder.name}: {len(packet.paragraphs)} paragraphs, {len(packet.flags)} flags "
          f"({len(blocking)} blocking), cost ${llm.cost:.2f}")
    for f in packet.flags:
        print(f"  {'BLOCKING ' if f.blocking else ''}{f.code} ({f.ref}): {f.message}")
    print(f"memo: {folder / 'memo.md'}")
    print(f"actions: {', '.join(packet.allowed_actions)}")


def _client(env_file: Path):
    import anthropic
    from dotenv import dotenv_values

    key = dotenv_values(env_file).get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(f"no ANTHROPIC_API_KEY in {env_file}")
    return anthropic.Anthropic(api_key=key, max_retries=4)


def cmd_batch(args) -> None:
    """Several memos through the Message Batches API (half price; each stage is one batch
    for all companies, so expect four or five batches of a few minutes to an hour each)."""
    from auto_researcher.workstation.batch import Batcher, BatchLLM, ReplyCache

    as_of = date.fromisoformat(args.as_of)
    tickers = [t.upper() for t in args.tickers]
    batcher = Batcher(_client(args.env_file), ReplyCache(), poll_seconds=args.poll)
    filings, metrics, hub = LakeFilings(), TickerLakeMetrics(), HubContext()
    draft = "orchestrator_draft"
    llms = {t: BatchLLM(batcher, effort=args.effort,
                        models={draft: MODELS_BY_NAME[args.draft_model]} if args.draft_model else None,
                        efforts={draft: args.draft_effort} if args.draft_effort else None) for t in tickers}

    def job(t: str):
        def run():
            ws = Workstation(llms[t], filings, metrics, hub, paragraphs=args.paragraphs or None,
                             repair=not args.no_repair)
            packet, state = ws.run(Request(ticker=t, as_of=as_of, question=args.question, horizon=args.horizon))
            return packet, save_run(state, packet, root=args.runs)
        return run

    results = batcher.run([job(t) for t in tickers])
    rows = []
    for t, res in zip(tickers, results):
        cost = llms[t].cost
        if isinstance(res, Exception):
            rows.append({"ticker": t, "error": str(res), "cost": cost})
            print(f"{t:6s} FAILED ({res}) cost ${cost:.2f}")
            continue
        packet, folder = res
        blocking = [f.code for f in packet.flags if f.blocking]
        rows.append({"ticker": t, "run": folder.name, "paragraphs": len(packet.paragraphs), "flags": len(packet.flags),
                     "blocking": blocking, "drafts": sum(1 for c in llms[t].calls if c["role"] == "orchestrator_draft"),
                     "repairs": sum(1 for c in llms[t].calls if c["role"] == "orchestrator_repair"),
                     "cost": cost})
        print(f"{t:6s} {folder.name}  {len(packet.paragraphs):2d} paragraphs  {len(packet.flags)} flags "
              f"({len(blocking)} blocking{': ' + ', '.join(blocking) if blocking else ''})  ${cost:.2f}")
    total = sum(r["cost"] for r in rows)
    print(f"total ${total:.2f} at batch prices over {len(batcher.batches)} batches: "
          f"{[(b['requests'], b['seconds']) for b in batcher.batches]}")
    out = Path(args.runs).parent / "pilots" / f"{as_of}_{len(tickers)}{'_' + args.label if args.label else ''}.json"
    if out.exists():   # an earlier pilot's summary (its review page reads it): keep it
        out = out.with_name(f"{out.stem}_{datetime.now():%Y%m%d%H%M}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"as_of": as_of.isoformat(), "question": args.question, "memos": rows,
                               "batches": batcher.batches, "cost": total}, indent=1), encoding="utf-8")
    print(f"summary: {out}")


def cmd_runs(args) -> None:
    root = Path(args.runs)
    for folder in sorted(root.glob("*")) if root.exists() else []:
        packet = json.loads((folder / "packet.json").read_text(encoding="utf-8"))
        dec = folder / "decision.json"
        status = json.loads(dec.read_text(encoding="utf-8"))["action"] if dec.exists() else "awaiting review"
        n_block = sum(1 for f in packet["flags"] if BLOCKING[f["code"]])
        print(f"{folder.name}  {len(packet['flags'])} flags ({n_block} blocking)  {status}")


def cmd_decide(args) -> None:
    d = record_decision(Path(args.runs) / args.run_id, args.action, args.notes)
    print(f"{args.run_id}: {d['action']}")


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--runs", type=Path, default=RUNS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("memo")
    m.add_argument("ticker")
    m.add_argument("--as-of", default=date.today().isoformat())
    m.add_argument("--question", default=DEFAULT_QUESTION)
    m.add_argument("--horizon", default="12m")
    m.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    m.add_argument("--paragraphs", type=int, default=PARAGRAPHS, help=PARAGRAPHS_HELP)
    m.add_argument("--no-repair", action="store_true", help=NO_REPAIR_HELP)
    m.add_argument("--env-file", type=Path, default=ENV_FILE)
    m.add_argument("--dry-run", action="store_true", help="show what the tools return; no API calls")
    m.set_defaults(fn=cmd_memo)
    b = sub.add_parser("batch", help="several memos through the Message Batches API (half price, slower)")
    b.add_argument("tickers", nargs="+")
    b.add_argument("--as-of", default=date.today().isoformat())
    b.add_argument("--question", default=DEFAULT_QUESTION)
    b.add_argument("--horizon", default="12m")
    b.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    b.add_argument("--poll", type=float, default=30.0, help="seconds between batch status checks")
    b.add_argument("--label", default="", help="suffix for the pilot summary's file name")
    b.add_argument("--draft-model", choices=sorted(MODELS_BY_NAME), help="model for the draft only")
    b.add_argument("--draft-effort", choices=["low", "medium", "high"], help="effort for the draft only")
    b.add_argument("--paragraphs", type=int, default=PARAGRAPHS, help=PARAGRAPHS_HELP)
    b.add_argument("--no-repair", action="store_true", help=NO_REPAIR_HELP)
    b.add_argument("--env-file", type=Path, default=ENV_FILE)
    b.set_defaults(fn=cmd_batch)
    sub.add_parser("runs").set_defaults(fn=cmd_runs)
    d = sub.add_parser("decide")
    d.add_argument("run_id")
    d.add_argument("action", choices=["approve_for_internal_use", "request_revision", "escalate", "reject"])
    d.add_argument("--notes", default="")
    d.set_defaults(fn=cmd_decide)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
