"""
Retrieval over one filing's chunks: keyword (BM25), embeddings (Ollama) and hybrid.

A filing is split by ``auto_researcher.llm.filings.parse_chunks`` into table pages
(every table with numbers, tagged or not) and narrative text chunks. Within one
filing there are a few hundred chunks, so search is exhaustive and in memory; a
vector database is only needed once the search spans many filings.

* ``BM25``: Okapi BM25 over lower-cased word tokens, with plurals folded
  ("revenues" = "revenue") and years kept as tokens.
* ``OllamaEmbedder``: dense embeddings from an Ollama embedding model, cosine
  similarity. ``nomic-embed-text`` needs task prefixes, which are added. A table is
  embedded as ``table_summary`` (caption, column headers, row labels, no numbers):
  one vector for a whole statement's grid of numbers buries the row a question is
  about, and short text passages on the same topic outrank it.
* ``rrf``: reciprocal rank fusion of several rankings (Cormack et al., 2009), the
  usual way to combine keyword and embedding search without tuning weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from auto_researcher.llm.filings import Page, TextChunk

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "from", "and", "or", "by", "at", "as",
    "is", "was", "were", "be", "been", "what", "how", "much", "did", "does", "do", "which",
    "that", "this", "with", "its", "it", "their", "than", "between", "during", "year", "years",
}
# Prefixes some embedding models were trained with (query vs document)
PREFIXES = {"nomic-embed-text": ("search_query: ", "search_document: ")}


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of a filing."""

    key: str            # "t<table index>" or "x<after_table>.<part>"
    kind: str           # "table" or "text"
    text: str           # what is indexed and embedded
    page: Page | None   # the table page (None for text chunks)


def page_text(page: Page) -> str:
    """A table page as plain text: text before, the table's rows, text after."""
    rows = "\n".join(" | ".join(r) for r in page.table)
    return "\n".join([" ".join(page.pre_text), rows, " ".join(page.post_text)]).strip()


def table_summary(page: Page, caption_chars: int = 300) -> str:
    """A table without its numbers: the end of the text before it, the headers, the row labels."""
    caption = " ".join(page.pre_text)[-caption_chars:]
    header = " ".join(c for row in page.table[:page.n_header_rows] for c in row if c.strip())
    labels = "; ".join(row[0] for row in page.table[page.n_header_rows:] if row and row[0].strip())
    return "\n".join(x for x in (caption, header, labels) if x)


def filing_chunks(pages: Sequence[Page], text: Sequence[TextChunk]) -> list[Chunk]:
    chunks = [Chunk(f"t{p.index}", "table", page_text(p), p) for p in pages]
    chunks += [Chunk(f"x{t.after_table}.{t.part}", "text", " ".join(t.sentences), None) for t in text]
    return chunks


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+|(?:19|20)\d{2}", text.lower().replace("&", " and "))
    out = []
    for t in tokens:
        if t in STOPWORDS or len(t) < 2:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # fold plurals
        out.append(t)
    return out


class BM25:
    """Okapi BM25 (k1 = 1.5, b = 0.75)."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / max(self.avg, 1e-9))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([d.get(t, 0) for d in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


class OllamaEmbedder:
    """Embeddings from an Ollama model, L2-normalized, with an on-disk cache of documents."""

    def __init__(self, model: str, url: str = "http://localhost:11434", num_ctx: int = 4096,
                 cache_dir: Path | None = None, batch: int = 16, workers: int = 16):
        self.model, self.url, self.num_ctx, self.batch, self.workers = model, url, num_ctx, batch, workers
        self.query_prefix, self.doc_prefix = PREFIXES.get(model.split(":")[0], ("", ""))
        self.cache_dir = Path(cache_dir) / model.replace(":", "_") if cache_dir else None

    def _post(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(f"{self.url}/api/embed", timeout=600, json={
            "model": self.model, "input": texts, "truncate": True, "options": {"num_ctx": self.num_ctx}})
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _embed(self, texts: list[str]) -> np.ndarray:
        # Ollama embeds one request's inputs one after another; concurrent requests run in
        # parallel slots (about 7x faster with 16 on an RTX laptop GPU)
        batches = [texts[i:i + self.batch] for i in range(0, len(texts), self.batch)]
        with ThreadPoolExecutor(max(1, min(self.workers, len(batches)))) as pool:
            vecs = [v for b in pool.map(self._post, batches) for v in b]
        arr = np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.query_prefix + t for t in texts])

    def documents(self, texts: list[str], cache_key: str | None = None) -> np.ndarray:
        """Embed documents; with ``cache_key`` the result is stored and reused (checked by a hash)."""
        digest = hashlib.sha1(json.dumps(texts).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{cache_key}.npz" if self.cache_dir and cache_key else None
        if path is not None and path.exists():
            cached = np.load(path)
            if str(cached["digest"]) == digest:
                return cached["vecs"]
        vecs = self._embed([self.doc_prefix + t for t in texts])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, vecs=vecs, digest=digest)
        return vecs


RERANK_SYSTEM = """\
You pick the table in a company's annual report that holds the numbers a question \
needs. Each candidate shows the text just before the table, its column headers and \
the row labels that share a word with the question (with the table's row count); the \
numbers are left out. Pick the table with a row for the line item \
the question asks about and columns for the years it asks about. If several tables \
qualify, pick the one where that line item is reported in full (a financial statement \
or the note that breaks it down) rather than one that mentions it in passing. Reply \
with the candidate's number only."""


def focused_summary(page: Page, query: Sequence[str], max_rows: int = 15, caption_chars: int = 200) -> str:
    """``table_summary`` limited to the rows that share a word with the query (in table order).

    A financial statement has 40+ rows, so a summary cut to length can drop the very row
    a question asks about; showing the matching rows keeps it. With fewer than 5 matches
    the first rows fill in, and the row count is given.
    """
    q = set(query)
    body = [row[0] for row in page.table[page.n_header_rows:] if row and row[0].strip()]
    hits = [j for j, label in enumerate(body) if q & set(tokenize(label))]
    keep = sorted(set(hits[:max_rows]) | set(range(min(len(body), max(0, 5 - len(hits))))))
    labels = "; ".join(body[j] for j in keep)
    if len(keep) < len(body):
        labels += f" ({len(body)} rows in all)"
    caption = " ".join(page.pre_text)[-caption_chars:]
    header = " ".join(c for row in page.table[:page.n_header_rows] for c in row if c.strip())
    return "\n".join(x for x in (caption, header, labels) if x)


def rerank_messages(question: str, candidates: Sequence[Page]) -> list[dict]:
    """A listwise re-ranking prompt: the question and numbered table summaries."""
    query = tokenize(question)
    blocks = [f"[{i}]\n{focused_summary(p, query)}" for i, p in enumerate(candidates, 1)]
    return [{"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\n" + "\n\n".join(blocks)}]


def parse_choice(reply: str, k: int) -> int | None:
    """The 0-based candidate a re-ranking reply names, or None if it names none of 1..k."""
    m = re.search(r"\d+", reply)
    return int(m.group()) - 1 if m and 1 <= int(m.group()) <= k else None


def ranking(scores: np.ndarray) -> list[int]:
    """Indices from best to worst (stable for ties)."""
    return list(np.argsort(-scores, kind="stable"))


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    score: dict[int, float] = {}
    for r in rankings:
        for pos, doc in enumerate(r):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda d: (-score[d], d))
