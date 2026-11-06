"""Tests for within-filing retrieval: tokenizer, BM25, fusion and the embedding cache."""

from __future__ import annotations

import numpy as np

from auto_researcher.llm.filings import Page, TextChunk
from auto_researcher.llm.retrieval import (
    BM25,
    OllamaEmbedder,
    filing_chunks,
    focused_summary,
    page_text,
    parse_choice,
    ranking,
    rerank_messages,
    rrf,
    table_summary,
    tokenize,
)


def _page(index: int, label: str, pre: str = "") -> Page:
    table = (("", "2025", "2024"), (label, "100", "90"))
    return Page(index, (pre,) if pre else (), table, (), 1, ())


def test_tokenize_folds_plurals_drops_stopwords_keeps_years():
    assert tokenize("What was the change in Total Revenues from 2024 to 2025?") == [
        "change", "total", "revenue", "2024", "2025"]
    assert tokenize("R&D expenses") == ["expense"]  # single letters dropped, & -> and (a stopword)
    assert tokenize("gross margin, net of taxes") == ["gross", "margin", "net", "taxe"]


def test_bm25_prefers_the_rarer_matching_term():
    docs = [tokenize("net sales products services"), tokenize("net income per share"),
            tokenize("net sales by region americas europe")]
    bm = BM25(docs)
    order = ranking(bm.scores(tokenize("net sales americas")))
    assert order[0] == 2 and order[-1] == 1
    assert bm.scores(tokenize("unrelated words"))[0] == 0


def test_rrf_rewards_agreement():
    assert rrf([[0, 1, 2], [1, 0, 2]])[:2] == [0, 1]  # tie broken by id
    assert rrf([[2, 0, 1], [2, 1, 0]])[0] == 2


def test_filing_chunks_keys_and_text():
    pages = [_page(3, "net sales", pre="consolidated statements of operations")]
    text = [TextChunk(0, 0, ("risk factors apply .", "more text ."))]
    chunks = filing_chunks(pages, text)
    assert [c.key for c in chunks] == ["t3", "x0.0"]
    assert [c.kind for c in chunks] == ["table", "text"]
    assert page_text(pages[0]).startswith("consolidated statements of operations\n | 2025 | 2024\nnet sales")
    assert chunks[1].page is None and chunks[1].text == "risk factors apply . more text ."


def test_table_summary_keeps_caption_headers_and_labels_not_numbers():
    page = Page(0, ("see notes .", "consolidated statements of cash flows ( in millions )"),
                (("", "2025", "2024"), ("net income", "$ 100", "$ 90"), ("depreciation", "10", "9"), ("", "1", "2")),
                (), 1, ())
    assert table_summary(page) == ("see notes . consolidated statements of cash flows ( in millions )\n"
                                   "2025 2024\nnet income; depreciation")
    assert table_summary(page, caption_chars=20).startswith("flows ( in millions )"[-20:])


def test_embedder_adds_prefixes_and_caches_documents(tmp_path, monkeypatch):
    calls = []

    def fake(self, texts):
        calls.append(list(texts))
        arr = np.array([[len(t), 1.0] for t in texts], dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    monkeypatch.setattr(OllamaEmbedder, "_embed", fake)
    emb = OllamaEmbedder("nomic-embed-text", cache_dir=tmp_path)
    a = emb.documents(["doc one", "doc two"], cache_key="ACC")
    b = emb.documents(["doc one", "doc two"], cache_key="ACC")
    assert np.allclose(a, b) and len(calls) == 1  # second call served from the cache
    assert calls[0] == ["search_document: doc one", "search_document: doc two"]
    emb.documents(["doc one", "changed"], cache_key="ACC")  # different texts: recomputed
    assert len(calls) == 2
    emb.queries(["q"])
    assert calls[-1] == ["search_query: q"]
    assert OllamaEmbedder("bge-m3").query_prefix == ""


def test_rerank_prompt_numbers_candidates_and_parse_choice_is_strict():
    msgs = rerank_messages("how much did revenue change?", [_page(0, "net sales"), _page(5, "net income")])
    user = msgs[1]["content"]
    assert user.startswith("Question: how much did revenue change?\n\n[1]\n")
    assert "[2]" in user and "net income" in user and "100" not in user  # no numbers
    assert parse_choice("3", 10) == 2
    assert parse_choice("[2] the income statement", 10) == 1
    assert parse_choice("11", 10) is None and parse_choice("none", 10) is None


def test_focused_summary_shows_matching_rows_and_the_row_count():
    rows = tuple((f"line {i}", "1", "2") for i in range(30)) + (("interest income", "5", "6"),)
    page = Page(0, ("statements of operations",), (("", "2025", "2024"),) + rows, (), 1, ())
    s = focused_summary(page, tokenize("average interest income 2023 to 2025"))
    assert s.splitlines()[-1] == "line 0; line 1; line 2; line 3; interest income (31 rows in all)"
    few = focused_summary(page, tokenize("unrelated"))
    assert few.splitlines()[-1] == "line 0; line 1; line 2; line 3; line 4 (31 rows in all)"
