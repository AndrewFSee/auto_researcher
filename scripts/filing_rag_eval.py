"""
Retrieval within a filing, then answering: how much does finding the page cost?

For each filing question (``build_filing_qa.py``; optionally its rephrased version from
``paraphrase_filing_questions.py``) the filing is split into chunks (every table with
numbers, tagged or not, plus narrative text; ``filings.parse_chunks``) and ranked by
each retriever (``llm.retrieval``): BM25, embeddings, and their hybrid. The gold chunk
is the table the question was generated from.

1. Retrieval: recall@k and mean reciprocal rank per question set and retriever.
2. End to end (``--model``): the fine-tuned model answers from the top-ranked table
   page. ``oracle`` hands it the gold page instead (the ceiling). When a retriever's
   top page is the gold page the prompt is identical to the oracle's, so the oracle's
   answer is reused and only the misses are asked again.

Example::

    python scripts/filing_rag_eval.py --embed-models nomic-embed-text bge-m3 --model qwen3-finqa
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_filing_qa import _header_from_json  # noqa: E402
from finqa_baseline import ask, score  # noqa: E402

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.llm.filings import load_chunks  # noqa: E402
from auto_researcher.llm.finqa import FinQAExample  # noqa: E402
from auto_researcher.llm.retrieval import (  # noqa: E402
    BM25,
    OllamaEmbedder,
    filing_chunks,
    ranking,
    rrf,
    table_summary,
    tokenize,
)

logger = logging.getLogger("filing_rag_eval")
DATA = ROOT / "data" / "filingqa"
KS = (1, 3, 5, 10)
FINDINGS = [
    "", "## Findings (2026-10-31; 526 10-Ks filed 2025-26, model `qwen3-finqa`)", "",
    "* Finding the page is now the bottleneck. Handed the right page, the fine-tuned model "
    "answers 95.3% of the questions; answering from BM25's top table it answers 53.4%. "
    "BM25's top table is the right one for 45% of questions; when it is not, the model still "
    "answers 20% of them, from another table holding the same figures (a statement and its "
    "note, or a continuation page).",
    "* Keyword search beats embeddings by a wide margin within a filing: recall@1 40% vs "
    "15-16% and recall@10 85% vs 41-43% (nomic-embed-text, bge-m3). Counting tables only, "
    "embeddings pick the right one 21-23% of the time against BM25's 45%. A 10-K has about "
    "96 tables in the same vocabulary (net income, depreciation, cash); what tells them "
    "apart is the exact row label, which BM25 matches and a general-purpose embedding blurs. "
    "Embedding a whole table, numbers included (`dense-page`), is worse still: one row "
    "is lost in a grid of numbers and short text passages on the same topic outrank it. "
    "Embedding a number-free summary instead was decided after 10 questions and not tuned "
    "further.",
    "* The hybrid is worse than BM25 alone (recall@1 25-27%, end to end 39%): rank fusion "
    "assumes both rankings are informative, and here the dense one mostly adds noise.",
    "* Rewording costs BM25 little on average (recall@1 40.3% to 39.9%) because most rewrites "
    "keep the line-item name (79% of its words on average). On the 77 rewrites that keep "
    "less than half, BM25's recall@1 falls from 31% to 21%, and embeddings still do not "
    "catch up (12-18%).",
    "* With the right page, rephrased questions score 92.4% against 95.3% for the same "
    "questions as generated (38 lost, 10 gained, p = 6e-5). This mixes rewrites that "
    "drifted to a nearby row (\"increase (decrease) in cash\" became \"cash\") with real "
    "sensitivity to wording (\"average annual operating income\"), so it is an upper bound "
    "on what rewording costs the model.",
    "", "## Caveats", "",
    "* Questions are templated and generated from one table each; real questions can need "
    "two tables, the text, or several filings.",
    "* The gold page is the table the question was generated from; another table with the "
    "same figures counts as a retrieval miss (the end-to-end score credits it).",
    "* Rephrasings come from a local 8B model; 2 to 4 of 50 spot-checked drift to a "
    "broader or nearby item (\"non-cash additions to PP&E\" became \"capex\").",
    "* Nothing was tuned on these questions beyond the table summary above; tuning (BM25 "
    "fields, re-ranking, a fine-tuned embedder) should use 10-Ks filed before 2025.",
]
LABEL = re.compile(r"(?:change in|average|portion of .+ in \d{4} was) (.+?)(?: from \d{4}| ?\?)")


def gold_key(record_id: str) -> str:
    return "t" + re.search(r"/t(\d+)-\d+$", record_id).group(1)


def _parse(args: tuple[dict, Path]) -> tuple[str, int]:
    header_json, cache_dir = args
    header = _header_from_json(header_json)
    path = cache_dir / f"{header.accession}.pkl"
    if not path.exists():
        pages, text = load_chunks(header)
        path.write_bytes(pickle.dumps(filing_chunks(pages, text)))
    return header.accession, 0


def load_all_chunks(accessions: set[str], workers: int) -> dict[str, list]:
    cache_dir = DATA / "chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = {h["accession"]: h for h in json.loads((DATA / "filing_index.json").read_text(encoding="utf-8"))}
    todo = [(index[a], cache_dir) for a in sorted(accessions) if not (cache_dir / f"{a}.pkl").exists()]
    if todo:
        logger.info("chunking %d filings", len(todo))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_parse, todo, chunksize=8))
    return {a: pickle.loads((cache_dir / f"{a}.pkl").read_bytes()) for a in accessions}


def rank_all(records: list[dict], chunks: dict[str, list], embed_models: list[str]) -> dict[str, dict]:
    """Per question: the gold chunk's rank (0-based, None if absent) and top keys, per retriever."""
    by_filing: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_filing[r["meta"]["accession"]].append(r)
    embedders = {m: OllamaEmbedder(m, cache_dir=DATA / "embeddings") for m in embed_models}
    out: dict[str, dict] = {}
    for n, (acc, recs) in enumerate(sorted(by_filing.items()), 1):
        ch = chunks[acc]
        keys = [c.key for c in ch]
        bm25 = BM25([tokenize(c.text) for c in ch])
        questions = [r["qa"]["question"] for r in recs]
        rankings: dict[str, list[list[int]]] = {"bm25": [ranking(bm25.scores(tokenize(q))) for q in questions]}
        tables = [i for i, c in enumerate(ch) if c.kind == "table"]
        texts = [i for i, c in enumerate(ch) if c.kind == "text"]
        for m, emb in embedders.items():
            qv = emb.queries(questions)
            text_vecs = emb.documents([ch[i].text for i in texts], cache_key=f"{acc}.text")
            for name, table_text in (("dense", lambda c: table_summary(c.page)), ("dense-page", lambda c: c.text)):
                docs = np.zeros((len(ch), text_vecs.shape[1] if texts else qv.shape[1]), dtype=np.float32)
                docs[texts] = text_vecs
                docs[tables] = emb.documents([table_text(ch[i]) for i in tables], cache_key=f"{acc}.{name}")
                rankings[f"{name}:{m}"] = [ranking(docs @ v) for v in qv]
            rankings[f"hybrid:{m}"] = [rrf([b, d]) for b, d in zip(rankings["bm25"], rankings[f"dense:{m}"])]
        for i, r in enumerate(recs):
            gold = gold_key(r["id"])
            out[r["id"]] = {"n_chunks": len(ch), "n_tables": sum(c.kind == "table" for c in ch)}
            for method, rs in rankings.items():
                order = [keys[j] for j in rs[i]]
                top_table = next((k for k in order if k.startswith("t")), None)
                out[r["id"]][method] = {"rank": order.index(gold) if gold in order else None,
                                        "top": order[:5], "top_table": top_table}
        if n % 100 == 0:
            logger.info("  ranked %d/%d filings", n, len(by_filing))
    return out


def retrieval_metrics(ranks: dict[str, dict], methods: list[str]) -> dict[str, dict]:
    out = {}
    for m in methods:
        rs = [v[m]["rank"] for v in ranks.values()]
        out[m] = {f"recall@{k}": float(np.mean([r is not None and r < k for r in rs])) for k in KS}
        out[m]["mrr"] = float(np.mean([1 / (r + 1) if r is not None else 0 for r in rs]))
    return out


def answer(records: list[dict], ranks: dict[str, dict], chunks: dict[str, list], methods: list[str],
           model: str, runs_dir: Path, set_name: str) -> dict[str, list[dict]]:
    """End-to-end answers per method; the oracle first, its answers reused on retrieval hits."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    oracle: dict[str, dict] = {}
    for method in ["oracle", *methods]:
        path = runs_dir / f"{set_name}__{method.replace(':', '_')}__{model.replace(':', '_')}.jsonl"
        done = {}
        if path.exists():
            done = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        recs_out = []
        with path.open("a", encoding="utf-8") as fh:
            for i, r in enumerate(records, 1):
                if r["id"] in done:
                    recs_out.append(done[r["id"]])
                    continue
                gold = gold_key(r["id"])
                key = gold if method == "oracle" else ranks[r["id"]][method]["top_table"]
                if method != "oracle" and key == gold:
                    rec = {**oracle[r["id"]], "page": key, "reused": True}
                else:
                    page = next(c.page for c in chunks[r["meta"]["accession"]] if c.key == key)
                    ex = FinQAExample(r["id"], r["qa"]["question"], page.pre_text, page.post_text,
                                      page.table, r["qa"]["program"], r["qa"]["exe_ans"])
                    out = ask("http://localhost:11434", model, "program", ex, 8192)
                    rec = {"id": r["id"], "gold": r["qa"]["exe_ans"], "page": key, "reused": False,
                           **out, **score("program", ex, out["response"])}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                recs_out.append(rec)
                if i % 100 == 0:
                    logger.info("  %s %s %d/%d", set_name, method, i, len(records))
        if method == "oracle":
            oracle = {r["id"]: r for r in recs_out}
        results[method] = recs_out
    return results


def write_report(sets: dict[str, dict], out: Path) -> None:
    pct = "{:.1%}".format
    lines = [
        "# Retrieval within a filing (RAG step 1)", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/filing_rag_eval.py`.", "",
        "Each filing question (FinQA-style, auto-graded from inline XBRL; "
        "[filing_qa.md](filing_qa.md)) is asked of its own 10-K: the filing is split into every "
        "table with numbers (tagged or not) plus narrative text chunks, and each retriever ranks "
        "all of them. The gold chunk is the table the question came from. *Rephrased* questions "
        "were rewritten by a local model to read like an analyst's, with checks (same years, "
        "same calculation, judged to ask for the same number) and a manual spot-check.", "",
        "Retrievers: `bm25` on the full chunk text; `dense` embeds text chunks as they are and "
        "tables as a summary without numbers (caption, column headers, row labels); "
        "`dense-page` embeds each table page's full text, numbers included; `hybrid` is "
        "reciprocal rank fusion (k = 60) of `bm25` and `dense`. Embedding models run in Ollama.", "",
        "## Retrieval", "",
        "| Questions | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in sets.items():
        for m, v in s["retrieval"].items():
            lines.append(f"| {name} ({s['n']}) | {m} | " + " | ".join(pct(v[f'recall@{k}']) for k in KS)
                         + f" | {v['mrr']:.3f} |")
    chunks = next(iter(sets.values()))["chunk_stats"]
    lines += ["", f"A filing has {chunks['tables']:.0f} table chunks and {chunks['text']:.0f} text "
              f"chunks on average (median {chunks['median_total']:.0f} in all).", ""]
    if any(s.get("answers") for s in sets.values()):
        lines += ["## End to end: answer from the top-ranked table page", "",
                  "`oracle` is handed the gold page. Within 1% (scale-tolerant), as in the filing "
                  "QA report.", "",
                  "| Questions | Page from | Right page | Within 1% | Exact | Failed "
                  "| Within 1%, right page | Within 1%, wrong page |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, s in sets.items():
            for m, recs in (s.get("answers") or {}).items():
                n = len(recs)
                hit = [r for r in recs if r["page"] == gold_key(r["id"])]
                miss = [r for r in recs if r["page"] != gold_key(r["id"])]
                acc = (lambda rs: f"{pct(sum(r['tolerant'] for r in rs) / len(rs))} ({len(rs)})"
                       if rs else "-")
                lines.append(f"| {name} | {m} | {pct(len(hit) / n)} | {pct(sum(r['tolerant'] for r in recs) / n)} | "
                             f"{pct(sum(r['exact'] for r in recs) / n)} | "
                             f"{pct(sum(r['error'] is not None for r in recs) / n)} | {acc(hit)} | {acc(miss)} |")
    lines += FINDINGS
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["test", "test_paraphrased"])
    ap.add_argument("--embed-models", nargs="*", default=["nomic-embed-text"])
    ap.add_argument("--model", default=None, help="answer end to end with this Ollama model")
    ap.add_argument("--e2e-methods", nargs="*", default=None,
                    help="retrievers to answer with (default: bm25 and every hybrid)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "filing_rag.md")
    args = ap.parse_args()

    sets = {}
    for name in args.sets:
        path = DATA / f"{name}.json"
        if not path.exists():
            logger.warning("no %s; skipped", path)
            continue
        sets[name] = json.loads(path.read_text(encoding="utf-8"))[: args.limit]
    chunks = load_all_chunks({r["meta"]["accession"] for recs in sets.values() for r in recs}, args.workers)
    methods = ["bm25"] + [f"{k}:{m}" for m in args.embed_models for k in ("dense-page", "dense", "hybrid")]
    report: dict[str, dict] = {}
    all_ranks: dict[str, dict] = {}
    for name, records in sets.items():
        ranks = all_ranks[name] = rank_all(records, chunks, args.embed_models)
        (DATA / "rag").mkdir(exist_ok=True)
        (DATA / "rag" / f"ranks_{name}.json").write_text(json.dumps(ranks), encoding="utf-8")
        per_filing = [chunks[a] for a in {r["meta"]["accession"] for r in records}]
        report[name] = {
            "n": len(records), "retrieval": retrieval_metrics(ranks, methods),
            "chunk_stats": {"tables": np.mean([sum(c.kind == "table" for c in ch) for ch in per_filing]),
                            "text": np.mean([sum(c.kind == "text" for c in ch) for ch in per_filing]),
                            "median_total": float(np.median([len(ch) for ch in per_filing]))},
        }
        if args.model:
            e2e = args.e2e_methods if args.e2e_methods is not None else \
                ["bm25"] + [f"hybrid:{m}" for m in args.embed_models]
            report[name]["answers"] = answer(records, ranks, chunks, e2e, args.model,
                                             DATA / "rag_runs", name)
    if "test" in all_ranks and "test_paraphrased" in all_ranks:
        # The rewrites that changed the line-item wording most, next to the same questions as generated
        low = {r["id"] for r in sets["test_paraphrased"] if r["meta"]["label_overlap"] < 0.5}
        for name in ("test", "test_paraphrased"):
            sub = {k: v for k, v in all_ranks[name].items() if k in low}
            report[f"{name}, label overlap < 50%"] = {"n": len(sub), "retrieval": retrieval_metrics(sub, methods)}
    write_report(report, args.out)
    json_out = {k: {kk: vv for kk, vv in v.items() if kk != "answers"} for k, v in report.items()}
    args.out.with_suffix(".json").write_text(json.dumps(json_out, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
