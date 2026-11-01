"""
Re-rank BM25's top tables with an LLM, then answer from the table it picks.

Within a 10-K, BM25's first table is the right one for about 45% of the filing
questions, but the right one is in its top 10 far more often
(``docs/results/filing_rag.md``). Here a general model (``--rerank-model``) sees the
question and BM25's top ``--k`` tables, each as a summary without numbers (the text
before it, column headers, row labels; ``retrieval.table_summary``), and names the
one that holds the numbers. If the reply names no candidate, BM25's first is kept.
The fine-tuned model then answers from the chosen table (``--model``), with the same
oracle and BM25 runs as ``filing_rag_eval.py`` (shared run files, so answers already
computed there are reused).

The prompt was written and checked on ``dev`` (10-Ks filed in 2024, from
``build_filing_qa.py --filed-from 2024-01-01 --filed-to 2024-12-31``) before running
the test sets.

Example::

    python scripts/filing_rerank_eval.py --sets dev --model qwen3-finqa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filing_rag_eval import DATA, answer, gold_key, load_all_chunks  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.llm.retrieval import BM25, parse_choice, ranking, rerank_messages, tokenize  # noqa: E402

logger = logging.getLogger("filing_rerank_eval")


# Written after the test-set run; the thinking numbers are from the dev run
# (--sets dev --think), which these test tables do not include.
FINDINGS = [
    "## Findings", "",
    "- The re-ranker was frozen on dev (commit c305a4c) before either test set was run. On test it "
    "lifts the right table from 44.9% to 55.5% and end-to-end accuracy from 53.4% to 65.8%; on the "
    "paraphrased questions, 44.5% to 49.0% and 50.2% to 59.0%. About 30 points remain to the oracle page.",
    "- Retrieval is no longer the main limit: the right table is among BM25's top 10 for about 90% of "
    "questions. The pick is: given the right table among the 10, the re-ranker chooses it 61% of the "
    "time (55% paraphrased).",
    "- The gain is smaller on paraphrased questions (+4.5 points on the pick, against +10.6). The "
    "summaries show only the row labels that share a word with the question, so rewording likely "
    "hides the rows that would identify the table.",
    "- Thinking, dev only (all 447 questions): right table 66.7% against 54.4% without thinking. "
    "83 questions are right only with thinking and 28 only without (sign test p = 2e-7). It costs "
    "31.2 s per question against 2.7 s. 4.5% of thinking replies are empty after about 90 s, likely "
    "the 4,096-token limit running out mid-thought; those keep BM25's first table.",
    "- Thinking has not been run on the test sets: about 17 hours for both on this GPU.",
]


def candidates(records: list[dict], chunks: dict[str, list], k: int) -> dict[str, list[str]]:
    """BM25's top k table keys per question (ranked over all chunks, tables kept)."""
    out: dict[str, list[str]] = {}
    by_filing: dict[str, list[dict]] = {}
    for r in records:
        by_filing.setdefault(r["meta"]["accession"], []).append(r)
    for acc, recs in by_filing.items():
        ch = chunks[acc]
        bm25 = BM25([tokenize(c.text) for c in ch])
        for r in recs:
            order = ranking(bm25.scores(tokenize(r["qa"]["question"])))
            out[r["id"]] = [ch[j].key for j in order if ch[j].kind == "table"][:k]
    return out


def rerank(url: str, model: str, question: str, pages: list, think: bool, retries: int = 5) -> dict:
    body = {"model": model, "stream": False, "think": think, "messages": rerank_messages(question, pages),
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 4096 if think else 8}}
    for attempt in range(1, retries + 1):  # CUDA OOM on a shared laptop GPU: retry after a pause
        start = time.perf_counter()
        try:
            resp = requests.post(f"{url}/api/chat", json=body, timeout=900)
            resp.raise_for_status()
            break
        except (requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(exc.response, "status_code", None)
            if attempt == retries or (status is not None and status < 500):
                raise
            logger.warning("%s (attempt %d); retrying in %ds", exc, attempt, 15 * attempt)
            time.sleep(15 * attempt)
    reply = resp.json()["message"]["content"].strip()
    return {"reply": reply, "seconds": time.perf_counter() - start}


def run_rerank(records: list[dict], chunks: dict[str, list], cands: dict[str, list[str]], args,
               set_name: str) -> dict[str, dict]:
    runs = DATA / "rerank_runs"
    runs.mkdir(exist_ok=True)
    tag = f"{args.rerank_model.replace(':', '_')}__k{args.k}{'__think' if args.think else ''}"
    path = runs / f"{set_name}__{tag}.jsonl"
    done = {}
    if path.exists():
        done = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    out = {}
    with path.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(records, 1):
            if r["id"] in done:
                out[r["id"]] = done[r["id"]]
                continue
            keys = cands[r["id"]]
            pages = {c.key: c.page for c in chunks[r["meta"]["accession"]] if c.kind == "table"}
            res = rerank(args.url, args.rerank_model, r["qa"]["question"], [pages[k] for k in keys], args.think)
            choice = parse_choice(res["reply"], len(keys))
            rec = {"id": r["id"], "candidates": keys, "choice": choice,
                   "chosen": keys[choice if choice is not None else 0], **res}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            out[r["id"]] = rec
            if i % 100 == 0:
                logger.info("  %s rerank %d/%d", set_name, i, len(records))
    return out


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["dev"])
    ap.add_argument("--rerank-model", default="qwen3:8b")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--think", action="store_true", help="let the re-ranker think first (slower)")
    ap.add_argument("--model", default=None, help="answer end to end with this Ollama model")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "filing_rerank.md")
    args = ap.parse_args()

    method = f"rerank:{args.rerank_model}:k{args.k}{':think' if args.think else ''}"
    pct = "{:.1%}".format
    lines = [
        "# Re-ranking BM25's top tables with an LLM (RAG step 2)", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/filing_rerank_eval.py`. "
        f"Re-ranker `{args.rerank_model}`{' (thinking)' if args.think else ''}, top {args.k} BM25 "
        f"tables shown as summaries without numbers.", "",
        "| Questions | Gold in candidates | BM25 first table | Re-ranker's pick | Pick, gold in candidates "
        "| No valid reply | Seconds / question |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    answers = {}
    for name in args.sets:
        records = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))[: args.limit]
        chunks = load_all_chunks({r["meta"]["accession"] for r in records}, args.workers)
        cands = candidates(records, chunks, args.k)
        picks = run_rerank(records, chunks, cands, args, name)
        gold = {r["id"]: gold_key(r["id"]) for r in records}
        inside = [i for i in gold if gold[i] in cands[i]]
        lines.append(
            f"| {name} ({len(records)}) | {pct(len(inside) / len(gold))} | "
            f"{pct(np.mean([cands[i][0] == gold[i] for i in gold]))} | "
            f"{pct(np.mean([picks[i]['chosen'] == gold[i] for i in gold]))} | "
            f"{pct(np.mean([picks[i]['chosen'] == gold[i] for i in inside]))} | "
            f"{pct(np.mean([picks[i]['choice'] is None for i in gold]))} | "
            f"{np.mean([picks[i]['seconds'] for i in gold]):.1f} |")
        if args.model:
            ranks = {i: {"bm25": {"top_table": cands[i][0]}, method: {"top_table": picks[i]["chosen"]}}
                     for i in gold}
            answers[name] = answer(records, ranks, chunks, ["bm25", method], args.model, DATA / "rag_runs", name)
    if answers:
        lines += ["", f"## End to end (`{args.model}`, within 1%)", "",
                  "| Questions | Page from | Right page | Within 1% |", "| --- | --- | ---: | ---: |"]
        for name, res in answers.items():
            for m, recs in res.items():
                lines.append(f"| {name} | {m} | {pct(np.mean([r['page'] == gold_key(r['id']) for r in recs]))} | "
                             f"{pct(np.mean([r['tolerant'] for r in recs]))} |")
    if "test" in args.sets:
        lines += ["", *FINDINGS]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
