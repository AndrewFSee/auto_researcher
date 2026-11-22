"""Filing passages from sentiq's EDGAR lake (read-only), searched with BM25.

For an as-of date the tool reads the company's latest 10-K filed by then and the 10-Qs
filed after it. Each located section (10-K items 1, 1A, 7, 7A; 10-Q MD&A and
risk-factor updates; sentiq's ``locate_sections`` offsets in ``meta.sections``) is cut
into passages of about ``PASSAGE_CHARS`` on line breaks. A filing whose sections were
not located (many banks cross-reference their MD&A) is searched whole.

The lake starts in 2018: earlier as-of dates find nothing and are flagged missing.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from ...llm.retrieval import BM25, tokenize
from ..schemas import Document

LAKE = Path(r"C:\Users\andre\projects\sentiq\data\lake\items\src=edgar_periodic")
PASSAGE_CHARS = 2500
LOOKBACK_DAYS = 500   # enough to reach the last 10-K from any date
SECTION_TITLES = {
    "10-K": {"1": "Item 1 Business", "1A": "Item 1A Risk Factors", "7": "Item 7 MD&A", "7A": "Item 7A Market Risk"},
    "10-Q": {"I-2": "Part I Item 2 MD&A", "II-1A": "Part II Item 1A Risk Factors"},
}


def passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Consecutive pieces of about ``size`` characters, cut at line breaks (a line longer
    than ``size`` is cut at a space)."""
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > size:
            cut = line.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:cut])
            line = line[cut:]
        if len(cur) + len(line) > size and cur:
            out.append(cur)
            cur = ""
        cur += line
    if cur.strip():
        out.append(cur)
    return [" ".join(p.split()) for p in out if p.strip()]


class LakeFilings:
    version = "sentiq-lake-1"

    def __init__(self, lake: Path = LAKE) -> None:
        self.lake = lake
        self._cache: dict[tuple[str, date], tuple[list[Document], BM25]] = {}

    def filings(self, ticker: str, as_of: date) -> list[dict]:
        """Rows (meta + body) for the latest 10-K filed by ``as_of`` and the 10-Qs after it."""
        d = ds.dataset(self.lake, format="parquet", partitioning="hive")
        start = (as_of - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m")
        flt = (ds.field("ym") >= start) & (ds.field("ym") <= as_of.strftime("%Y-%m"))
        heads = d.to_table(filter=flt, columns=["item_id", "tickers", "meta"])
        keep = [i for i, t in enumerate(heads["tickers"].to_pylist()) if t and ticker in t]
        rows = []
        for i in keep:
            m = json.loads(heads["meta"][i].as_py())
            if m.get("form") in ("10-K", "10-Q") and m["filing_date"] <= as_of.isoformat():
                rows.append({"item_id": heads["item_id"][i].as_py(), **m})
        tenks = [r for r in rows if r["form"] == "10-K"]
        if not tenks:
            return []
        last = max(tenks, key=lambda r: r["filing_date"])
        chosen = [last] + [r for r in rows if r["form"] == "10-Q" and r["filing_date"] > last["filing_date"]]
        ids = pa.array([r["item_id"] for r in chosen])
        bodies = d.to_table(filter=flt & pc.is_in(ds.field("item_id"), value_set=ids), columns=["item_id", "body_text"])
        text = dict(zip(bodies["item_id"].to_pylist(), bodies["body_text"].to_pylist()))
        return [{**r, "text": text.get(r["item_id"], "")} for r in chosen]

    def documents(self, ticker: str, as_of: date) -> list[Document]:
        docs = []
        for f in self.filings(ticker, as_of):
            filed = date.fromisoformat(f["filing_date"])
            spans = f.get("sections") or {}
            titles = SECTION_TITLES.get(f["form"], {})
            parts = [(k, titles.get(k, k), f["text"][a:b]) for k, (a, b) in sorted(spans.items())] \
                or [("full", None, f["text"])]
            for key, title, body in parts:
                for n, p in enumerate(passages(body), 1):
                    docs.append(Document(doc_id=f"{f['form']}:{f['accession']}#{key}.{n}", ticker=ticker,
                                         form=f["form"], filed=filed, section=title, text=p))
        return docs

    def search(self, ticker: str, query: str, as_of: date, k: int = 5) -> list[Document]:
        key = (ticker, as_of)
        if key not in self._cache:
            docs = self.documents(ticker, as_of)
            self._cache[key] = (docs, BM25([tokenize(f"{d.section or ''} {d.text}") for d in docs]))
        docs, index = self._cache[key]
        if not docs:
            return []
        scores = index.scores(tokenize(query))
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        return [docs[i] for i in order[:k] if scores[i] > 0]
