"""
Run the verifier on a pilot's saved memos and, given a reviewer's marks, measure how
often it agrees with them.

    python scripts/workstation_verify.py data/workstation/pilots/2026-10-15_10.json \
        --marks data/workstation/review/marks/2026-10-15_10

Each memo's final draft is checked through the Message Batches API (Sonnet 5.5; half
price; replies cached, so a rerun costs nothing). Results go to
data/workstation/verify/<run>.json (or <label>/<run>.json). Saved runs are never changed. Runs made before
metrics and context items carried definitions get them from the tools' catalogues, by
name: they are fixed text, not data of the run.

Marks are the review page's documents (collection ``reviews``, one JSON file per run).
A paragraph marked right with a note counts as borderline: the reviewer's notes on
right marks flag paragraphs that could have been marked overstated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.workstation import gates  # noqa: E402
from auto_researcher.workstation.batch import Batcher, BatchLLM, ReplyCache  # noqa: E402
from auto_researcher.workstation.schemas import (  # noqa: E402
    ContextItem, Document, DraftMemo, Metric, Request, ResearchNotes)
from auto_researcher.workstation.tools import hub, market  # noqa: E402
from auto_researcher.workstation.verify import Verification, Verifier  # noqa: E402

RUNS = ROOT / "data" / "workstation" / "runs"
OUT = ROOT / "data" / "workstation" / "verify"
ENV_FILE = ROOT.parent / "sentiq" / ".env"   # sibling project


def _definition_metric(name: str) -> str | None:
    base = name.removesuffix("_1y_ago")
    if base not in market.CATALOGUE:
        return None
    text = " ".join(filter(None, [market.CATALOGUE[base][1] + ".", market.READINGS.get(base)]))
    return text + (" Computed with only the data public a year before the as-of date." if base != name else "")


def _definition_context(name: str) -> str | None:
    if name not in hub.CATALOGUE:
        return None
    return " ".join(filter(None, [hub.CATALOGUE[name] + ".", hub.READINGS.get(name)]))


def load(run: str) -> dict:
    """A saved run's final memo and the evidence it may cite."""
    s = json.loads((RUNS / run / "state.json").read_text(encoding="utf-8"))
    req = Request.model_validate(s["request"])
    docs = {k: Document.model_validate(v) for k, v in s["documents"].items()}
    claims = {}
    for slot in ("research_notes", "bear_notes"):
        kept, _, _ = gates.gate_claims(ResearchNotes.model_validate(s[slot]), docs, req.as_of, "claims")
        claims.update({c.id: c for c in kept})
    metrics = [Metric.model_validate(m) for m in [*s["metrics"], *s["calculations"]]]
    metrics = [m if m.definition or m.snapshot_id.startswith("calc:")
               else m.model_copy(update={"definition": _definition_metric(m.name)}) for m in metrics]
    context = [ContextItem.model_validate(c) for c in s["context"]]
    context = [c if c.definition else c.model_copy(update={"definition": _definition_context(c.name)})
               for c in context]
    return {"request": req, "memo": DraftMemo.model_validate(s["drafts"][-1]), "claims": claims,
            "metrics": metrics, "context": context, "documents": docs}


def human_label(mark: dict | None) -> str | None:
    if not mark or not mark.get("verdict"):
        return None
    if mark["verdict"] == "right":
        return "borderline" if (mark.get("note") or "").strip() else "right"
    return mark["verdict"]


def agreement(results: dict[str, Verification], marks_dir: Path) -> None:
    table: Counter = Counter()
    misses = []
    for run, check in results.items():
        f = marks_dir / f"{run}.json"
        if not f.exists():
            continue
        marks = json.loads(f.read_text(encoding="utf-8")).get("paragraphs", {})
        for p in check.paragraphs:
            h = human_label(marks.get(p.ref))
            if h is None:
                continue
            table[(h, p.verdict)] += 1
            if h == "over" and p.verdict == "right":
                misses.append((run, p.ref, marks[p.ref].get("note", "")))
    humans = ["right", "borderline", "unsure", "over", "wrong"]
    print("\nreviewer (rows) vs verifier (columns)")
    print(f"{'':12s}{'right':>8s}{'over':>8s}{'wrong':>8s}")
    for h in humans:
        row = [table[(h, v)] for v in ("right", "over", "wrong")]
        if sum(row):
            print(f"{h:12s}" + "".join(f"{n:8d}" for n in row))

    def flagged(h: str) -> str:
        n = sum(table[(h, v)] for v in ("right", "over", "wrong"))
        k = table[(h, "over")] + table[(h, "wrong")]
        return f"{k}/{n}" + (f" ({k / n:.0%})" if n else "")
    print(f"\nflagged by the verifier: over {flagged('over')}, borderline {flagged('borderline')}, "
          f"clean right {flagged('right')}")
    for run, ref, note in misses:
        print(f"  missed {run} {ref}: {note[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("pilot", type=Path)
    ap.add_argument("--marks", type=Path, help="folder of the review page's mark documents, one per run")
    ap.add_argument("--label", help="write to data/workstation/verify/<label>/, keeping earlier results")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--env-file", type=Path, default=ENV_FILE)
    args = ap.parse_args()

    from workstation import _client  # the same key handling as the memo script

    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    runs = [r["run"] for r in pilot["memos"] if r.get("run")]
    batcher = Batcher(_client(args.env_file), ReplyCache(), poll_seconds=args.poll)
    llms = {run: BatchLLM(batcher, effort=args.effort) for run in runs}

    def job(run: str):
        def go() -> Verification:
            x = load(run)
            return Verifier(llms[run]).run(x["memo"], x["claims"], x["metrics"], x["context"], x["documents"],
                                           x["request"].as_of.isoformat())
        return go

    got = batcher.run([job(r) for r in runs])
    out = OUT / args.label if args.label else OUT
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Verification] = {}
    for run, res in zip(runs, got):
        cost = llms[run].cost
        if isinstance(res, Exception):
            print(f"{run}: FAILED ({res}) ${cost:.3f}")
            continue
        results[run] = res
        (out / f"{run}.json").write_text(json.dumps({"run": run, "calls": llms[run].calls,
                                                     **res.model_dump(mode="json")}, indent=1), encoding="utf-8")
        v = Counter(p.verdict for p in res.paragraphs)
        print(f"{run}: {dict(v)}  ${cost:.3f}")
    print(f"total ${sum(llm.cost for llm in llms.values()):.2f} at batch prices")
    if args.marks:
        agreement(results, args.marks)


if __name__ == "__main__":
    main()
