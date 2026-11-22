"""
Build the review page for a workstation pilot: every memo with its evidence, and a
verdict per paragraph and per memo that the reviewer marks on the published page.

    python scripts/workstation_review.py data/workstation/pilots/2026-10-15_10.json

Writes data/workstation/review/<pilot>.html from the template
src/auto_researcher/workstation/review_template.html. The page stores its marks in the
artifact's database (collection ``reviews``, one document per run), where Claude reads
them back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.workstation.batch import BATCH_DISCOUNT  # noqa: E402
from auto_researcher.workstation.claude import call_cost  # noqa: E402
from auto_researcher.workstation.schemas import BLOCKING  # noqa: E402

TEMPLATE = ROOT / "src" / "auto_researcher" / "workstation" / "review_template.html"
RUNS = ROOT / "data" / "workstation" / "runs"
OUT = ROOT / "data" / "workstation" / "review"


def memo_data(row: dict) -> dict:
    folder = RUNS / row["run"]
    packet = json.loads((folder / "packet.json").read_text(encoding="utf-8"))
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    kinds = [p.get("kind", "analysis") for p in state["drafts"][-1]["paragraphs"]]
    # What the memo's calls cost when they were made (a rerun served from the cache logs
    # them at no cost), at batch prices.
    audit = [json.loads(line) for line in (folder / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = next((e["calls"] for e in audit if e["event"] == "llm_usage"), [])
    cost = sum((call_cost(c["model"], c["usage"]) or 0) * BATCH_DISCOUNT for c in calls if c.get("usage"))
    paragraphs = []
    for p, kind in zip(packet["paragraphs"], kinds):
        paragraphs.append({
            "ref": p["ref"], "section": p["section"], "kind": kind, "text": p["text"],
            "claims": [{"id": c["id"], "role": c["role"], "text": c["text"], "quotes": c["quotes"]} for c in p["claims"]],
            "metrics": [{k: m.get(k) for k in ("id", "name", "value", "unit", "source", "as_of", "available_date")}
                        for m in p["metrics"]],
            "context": [{k: x.get(k) for k in ("id", "name", "value", "unit", "source", "as_of", "track_record")}
                        for x in p["context"]],
            "flags": [{"code": f["code"], "message": f["message"]} for f in p["flags"]],
        })
    return {"ticker": row["ticker"], "run": row["run"], "title": packet["title"], "cost": round(cost, 4),
            "drafts": row.get("drafts", len(state["drafts"])), "paragraphs": paragraphs,
            "flags": [{"code": f["code"], "ref": f.get("ref"), "message": f["message"], "blocking": BLOCKING[f["code"]]}
                      for f in packet["flags"]],
            "omitted": packet["omitted_claims"], "dropped": packet["dropped_claims"], "gaps": packet["gaps"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("pilot", type=Path)
    ap.add_argument("--spent", type=float, help="total spent on the pilot's runs, if more than these memos cost")
    args = ap.parse_args()
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    memos = [memo_data(r) for r in pilot["memos"] if r.get("run")]
    cost = sum(m["cost"] for m in memos)
    data = {"as_of": pilot["as_of"], "cost": round(cost, 2), "spent": args.spent, "memos": memos}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/", blob)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.pilot.stem}.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out} ({len(html) / 1e3:.0f} kB, {len(memos)} memos, "
          f"{sum(len(m['paragraphs']) for m in memos)} paragraphs)")


if __name__ == "__main__":
    main()
