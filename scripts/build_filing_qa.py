"""
Build an auto-graded FinQA-style question set from 10-K / 10-Q filings.

Reads the filings the sentiq project stored (``<raw-dir>/<accession>/``, read-only),
turns each table with inline-XBRL facts into a FinQA-style page
(``auto_researcher.llm.filings``) and samples questions whose gold programs and answers
come from the tagged facts (``auto_researcher.llm.filing_questions``). The output is in
FinQA's JSON format, so ``scripts/finqa_baseline.py --data-dir <dir> --split <name>``
scores models on it unchanged.

The filing headers are indexed once into ``--index`` (the raw store is on a USB hard
drive, where reading 18k small headers takes minutes); ``--refresh-index`` rebuilds it.

Example::

    python scripts/build_filing_qa.py --forms 10-K --filed-from 2025-01-01 --per-filing 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.llm.filing_questions import (  # noqa: E402
    TEMPLATES,
    filing_questions,
    finqa_record,
    page_questions,
)
from auto_researcher.llm.filings import FilingHeader, load_filing, read_filing_header  # noqa: E402

logger = logging.getLogger("build_filing_qa")

DEFAULT_RAW_DIR = os.environ.get("SENTIQ_RAW_DIR", "D:/sentiq-data/raw/edgar_periodic")


def _header_to_json(h: FilingHeader) -> dict:
    return {"accession": h.accession, "form": h.form, "company": h.company, "cik": h.cik,
            "filed": h.filed.isoformat(), "period": h.period.isoformat() if h.period else None,
            "document": str(h.document)}


def _header_from_json(d: dict) -> FilingHeader:
    return FilingHeader(d["accession"], d["form"], d["company"], d["cik"], date.fromisoformat(d["filed"]),
                        date.fromisoformat(d["period"]) if d["period"] else None, Path(d["document"]))


def load_index(raw_dir: Path, index_path: Path, refresh: bool) -> list[FilingHeader]:
    if index_path.exists() and not refresh:
        return [_header_from_json(d) for d in json.loads(index_path.read_text(encoding="utf-8"))]
    folders = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    logger.info("indexing %d filing folders in %s (one-off)", len(folders), raw_dir)
    headers = []
    for i, folder in enumerate(folders, 1):
        h = read_filing_header(folder)
        if h is not None:
            headers.append(h)
        if i % 2000 == 0:
            logger.info("  %d/%d", i, len(folders))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps([_header_to_json(h) for h in headers]), encoding="utf-8")
    return headers


def process(header: FilingHeader, per_filing: int, seed: int) -> tuple[list[dict], Counter]:
    """Questions for one filing; runs in a worker process."""
    stats: Counter = Counter()
    try:
        pages = load_filing(header)
    except Exception as exc:  # a malformed document must not stop the build
        stats["parse_errors"] += 1
        logger.warning("%s: %r", header.accession, exc)
        return [], stats
    stats["filings"] += 1
    stats["pages"] += len(pages)
    candidates = [q for p in pages for q in page_questions(p)]
    stats["candidates"] += len(candidates)
    if not candidates:
        stats["filings_without_questions"] += 1
    questions = filing_questions(header, pages, per_filing, seed)
    for q in questions:
        stats[q.template] += 1
    return [finqa_record(header, q, n) for n, q in enumerate(questions)], stats


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path(DEFAULT_RAW_DIR))
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "filingqa" / "filing_index.json")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--forms", nargs="+", default=["10-K"])
    parser.add_argument("--filed-from", type=date.fromisoformat, default=date(2025, 1, 1))
    parser.add_argument("--filed-to", type=date.fromisoformat, default=None)
    parser.add_argument("--per-filing", type=int, default=3)
    parser.add_argument("--limit-filings", type=int, default=None, help="random sample of N filings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "filingqa" / "test.json")
    args = parser.parse_args()

    headers = load_index(args.raw_dir, args.index, args.refresh_index)
    chosen = [h for h in headers if h.form in args.forms and h.filed >= args.filed_from
              and (args.filed_to is None or h.filed <= args.filed_to)]
    if args.limit_filings:
        chosen = random.Random(args.seed).sample(chosen, min(args.limit_filings, len(chosen)))
    logger.info("%d of %d indexed filings match forms %s filed from %s", len(chosen), len(headers),
                args.forms, args.filed_from)

    records: list[dict] = []
    stats: Counter = Counter()
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, h, args.per_filing, args.seed) for h in chosen]
        for i, fut in enumerate(as_completed(futures), 1):
            recs, s = fut.result()
            records += recs
            stats += s
            if i % 100 == 0 or i == len(futures):
                logger.info("  %d/%d filings, %d questions (%.0fs)", i, len(futures), len(records),
                            time.perf_counter() - start)
    records.sort(key=lambda r: (r["meta"]["filed"], r["id"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    summary = {
        "built": time.strftime("%Y-%m-%d %H:%M"), "raw_dir": str(args.raw_dir), "forms": args.forms,
        "filed_from": args.filed_from.isoformat(),
        "filed_to": args.filed_to.isoformat() if args.filed_to else None,
        "per_filing": args.per_filing, "seed": args.seed, "filings_selected": len(chosen),
        "questions": len(records), "companies": len({r["meta"]["cik"] for r in records}),
        **{k: stats[k] for k in ("filings", "parse_errors", "filings_without_questions", "pages", "candidates")},
        "by_template": {t: stats[t] for t in TEMPLATES},
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %d questions to %s", len(records), args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
