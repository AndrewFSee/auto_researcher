# Re-ranking BM25's top tables with an LLM (RAG step 2)

Generated 2026-11-04 07:48 by `scripts/filing_rerank_eval.py`. Re-ranker `qwen3:8b` (thinking), top 10 BM25 tables shown as summaries without numbers.

| Questions | Gold in candidates | BM25 first table | Re-ranker's pick | Pick, gold in candidates | No valid reply | Seconds / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test (980) | 90.7% | 44.9% | 69.3% | 76.4% | 3.2% | 25.9 |
| test_paraphrased (957) | 88.8% | 44.5% | 63.7% | 71.8% | 4.4% | 27.6 |

## End to end (`qwen3-finqa`, within 1%)

| Questions | Page from | Right page | Within 1% |
| --- | --- | ---: | ---: |
| test | oracle | 100.0% | 95.3% |
| test | bm25 | 44.9% | 53.4% |
| test | rerank:qwen3:8b:k10:think | 69.3% | 77.6% |
| test_paraphrased | oracle | 100.0% | 92.4% |
| test_paraphrased | bm25 | 44.5% | 50.2% |
| test_paraphrased | rerank:qwen3:8b:k10:think | 63.7% | 72.8% |

## Findings

- Thinking was chosen on dev (66.7% right table against 54.4%) before either test set was run with it. The re-ranker's prompt and the 4,096-token limit are unchanged from commit c305a4c.
- On test, thinking lifts the right table from 55.5% to 69.3% and end-to-end accuracy from 65.8% to 77.6%. On the paraphrased questions, 49.0% to 63.7% and 59.0% to 72.8%. About 18 points remain to the oracle page (20 paraphrased), against about 30 without thinking.
- Paired by question: on test, 174 questions get the right table only with thinking and 39 only without (sign test p = 2e-21); end to end, 143 against 28 (p = 9e-20). Paraphrased: 184 against 43 (p = 6e-22); end to end, 155 against 23 (p = 3e-25).
- Thinking closes most of the paraphrase gap: without it the re-ranker added 4.5 points over BM25 on paraphrased questions against 10.6 on test; with it, 19.2 against 24.4.
- Given the right table among the 10, the re-ranker now chooses it 76% of the time (72% paraphrased), against 61% (55%).
- Cost: 26 to 28 s per question against about 3 s, 7 hours per test set on this GPU. 3.2% of replies (4.2% paraphrased) are empty after about 90 s, likely the token limit running out mid-thought; those keep BM25's first table.
