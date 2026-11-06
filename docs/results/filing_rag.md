# Retrieval within a filing (RAG step 1)

Generated 2026-10-31 17:51 by `scripts/filing_rag_eval.py`.

Each filing question (FinQA-style, auto-graded from inline XBRL; [filing_qa.md](filing_qa.md)) is asked of its own 10-K: the filing is split into every table with numbers (tagged or not) plus narrative text chunks, and each retriever ranks all of them. The gold chunk is the table the question came from. *Rephrased* questions were rewritten by a local model to read like an analyst's, with checks (same years, same calculation, judged to ask for the same number) and a manual spot-check.

Retrievers: `bm25` on the full chunk text; `dense` embeds text chunks as they are and tables as a summary without numbers (caption, column headers, row labels); `dense-page` embeds each table page's full text, numbers included; `hybrid` is reciprocal rank fusion (k = 60) of `bm25` and `dense`. Embedding models run in Ollama.

## Retrieval

| Questions | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| test (980) | bm25 | 40.3% | 62.9% | 71.9% | 84.9% | 0.551 |
| test (980) | dense-page:nomic-embed-text | 9.5% | 16.9% | 21.2% | 28.8% | 0.162 |
| test (980) | dense:nomic-embed-text | 15.0% | 26.7% | 31.4% | 41.0% | 0.242 |
| test (980) | hybrid:nomic-embed-text | 26.5% | 40.8% | 49.3% | 63.0% | 0.383 |
| test (980) | dense-page:bge-m3 | 11.8% | 20.6% | 27.2% | 38.2% | 0.204 |
| test (980) | dense:bge-m3 | 15.6% | 27.3% | 32.2% | 43.0% | 0.250 |
| test (980) | hybrid:bge-m3 | 24.9% | 43.3% | 52.9% | 68.4% | 0.388 |
| test_paraphrased (957) | bm25 | 39.9% | 61.1% | 70.4% | 82.0% | 0.540 |
| test_paraphrased (957) | dense-page:nomic-embed-text | 9.1% | 16.3% | 22.0% | 28.0% | 0.158 |
| test_paraphrased (957) | dense:nomic-embed-text | 13.1% | 23.7% | 28.8% | 38.6% | 0.219 |
| test_paraphrased (957) | hybrid:nomic-embed-text | 24.6% | 39.5% | 46.9% | 60.8% | 0.363 |
| test_paraphrased (957) | dense-page:bge-m3 | 12.1% | 21.8% | 27.7% | 38.5% | 0.208 |
| test_paraphrased (957) | dense:bge-m3 | 15.2% | 26.5% | 31.8% | 42.7% | 0.245 |
| test_paraphrased (957) | hybrid:bge-m3 | 25.6% | 42.3% | 52.0% | 66.8% | 0.387 |
| test, label overlap < 50% (77) | bm25 | 31.2% | 59.7% | 67.5% | 77.9% | 0.479 |
| test, label overlap < 50% (77) | dense-page:nomic-embed-text | 11.7% | 15.6% | 20.8% | 29.9% | 0.170 |
| test, label overlap < 50% (77) | dense:nomic-embed-text | 13.0% | 27.3% | 32.5% | 44.2% | 0.235 |
| test, label overlap < 50% (77) | hybrid:nomic-embed-text | 26.0% | 40.3% | 48.1% | 64.9% | 0.382 |
| test, label overlap < 50% (77) | dense-page:bge-m3 | 7.8% | 26.0% | 28.6% | 40.3% | 0.193 |
| test, label overlap < 50% (77) | dense:bge-m3 | 16.9% | 24.7% | 31.2% | 51.9% | 0.258 |
| test, label overlap < 50% (77) | hybrid:bge-m3 | 24.7% | 46.8% | 58.4% | 64.9% | 0.393 |
| test_paraphrased, label overlap < 50% (77) | bm25 | 20.8% | 36.4% | 46.8% | 62.3% | 0.336 |
| test_paraphrased, label overlap < 50% (77) | dense-page:nomic-embed-text | 9.1% | 15.6% | 15.6% | 22.1% | 0.146 |
| test_paraphrased, label overlap < 50% (77) | dense:nomic-embed-text | 11.7% | 15.6% | 18.2% | 26.0% | 0.176 |
| test_paraphrased, label overlap < 50% (77) | hybrid:nomic-embed-text | 11.7% | 26.0% | 35.1% | 48.1% | 0.239 |
| test_paraphrased, label overlap < 50% (77) | dense-page:bge-m3 | 9.1% | 18.2% | 23.4% | 33.8% | 0.176 |
| test_paraphrased, label overlap < 50% (77) | dense:bge-m3 | 11.7% | 24.7% | 28.6% | 39.0% | 0.216 |
| test_paraphrased, label overlap < 50% (77) | hybrid:bge-m3 | 18.2% | 33.8% | 39.0% | 57.1% | 0.305 |

A filing has 96 table chunks and 321 text chunks on average (median 382 in all).

## End to end: answer from the top-ranked table page

`oracle` is handed the gold page. Within 1% (scale-tolerant), as in the filing QA report.

| Questions | Page from | Right page | Within 1% | Exact | Failed | Within 1%, right page | Within 1%, wrong page |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | oracle | 100.0% | 95.3% | 95.2% | 0.1% | 95.3% (980) | - |
| test | bm25 | 44.9% | 53.4% | 52.9% | 1.3% | 94.8% (440) | 19.6% (540) |
| test | hybrid:bge-m3 | 29.7% | 39.2% | 38.5% | 1.3% | 94.8% (291) | 15.7% (689) |
| test_paraphrased | oracle | 100.0% | 92.4% | 92.1% | 0.0% | 92.4% (957) | - |
| test_paraphrased | bm25 | 44.5% | 50.2% | 49.6% | 0.7% | 93.0% (426) | 15.8% (531) |
| test_paraphrased | hybrid:bge-m3 | 30.0% | 39.4% | 38.7% | 0.4% | 93.4% (287) | 16.3% (670) |

## Findings (2026-10-31; 526 10-Ks filed 2025-26, model `qwen3-finqa`)

* Finding the page is now the bottleneck. Handed the right page, the fine-tuned model answers 95.3% of the questions; answering from BM25's top table it answers 53.4%. BM25's top table is the right one for 45% of questions; when it is not, the model still answers 20% of them, from another table holding the same figures (a statement and its note, or a continuation page).
* Keyword search beats embeddings by a wide margin within a filing: recall@1 40% vs 15-16% and recall@10 85% vs 41-43% (nomic-embed-text, bge-m3). Counting tables only, embeddings pick the right one 21-23% of the time against BM25's 45%. A 10-K has about 96 tables in the same vocabulary (net income, depreciation, cash); what tells them apart is the exact row label, which BM25 matches and a general-purpose embedding blurs. Embedding a whole table, numbers included (`dense-page`), is worse still: one row is lost in a grid of numbers and short text passages on the same topic outrank it. Embedding a number-free summary instead was decided after 10 questions and not tuned further.
* The hybrid is worse than BM25 alone (recall@1 25-27%, end to end 39%): rank fusion assumes both rankings are informative, and here the dense one mostly adds noise.
* Rewording costs BM25 little on average (recall@1 40.3% to 39.9%) because most rewrites keep the line-item name (79% of its words on average). On the 77 rewrites that keep less than half, BM25's recall@1 falls from 31% to 21%, and embeddings still do not catch up (12-18%).
* With the right page, rephrased questions score 92.4% against 95.3% for the same questions as generated (38 lost, 10 gained, p = 6e-5). This mixes rewrites that drifted to a nearby row ("increase (decrease) in cash" became "cash") with real sensitivity to wording ("average annual operating income"), so it is an upper bound on what rewording costs the model.

## Caveats

* Questions are templated and generated from one table each; real questions can need two tables, the text, or several filings.
* The gold page is the table the question was generated from; another table with the same figures counts as a retrieval miss (the end-to-end score credits it).
* Rephrasings come from a local 8B model; 2 to 4 of 50 spot-checked drift to a broader or nearby item ("non-cash additions to PP&E" became "capex").
* Nothing was tuned on these questions beyond the table summary above; tuning (BM25 fields, re-ranking, a fine-tuned embedder) should use 10-Ks filed before 2025.
