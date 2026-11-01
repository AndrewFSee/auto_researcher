# Re-ranking BM25's top tables with an LLM (RAG step 2)

Generated 2026-10-31 20:32 by `scripts/filing_rerank_eval.py`. Re-ranker `qwen3:8b`, top 10 BM25 tables shown as summaries without numbers.

| Questions | Gold in candidates | BM25 first table | Re-ranker's pick | Pick, gold in candidates | No valid reply | Seconds / question |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test (980) | 90.7% | 44.9% | 55.5% | 61.2% | 0.3% | 2.7 |
| test_paraphrased (957) | 88.8% | 44.5% | 49.0% | 55.2% | 0.6% | 2.8 |

## End to end (`qwen3-finqa`, within 1%)

| Questions | Page from | Right page | Within 1% |
| --- | --- | ---: | ---: |
| test | oracle | 100.0% | 95.3% |
| test | bm25 | 44.9% | 53.4% |
| test | rerank:qwen3:8b:k10 | 55.5% | 65.8% |
| test_paraphrased | oracle | 100.0% | 92.4% |
| test_paraphrased | bm25 | 44.5% | 50.2% |
| test_paraphrased | rerank:qwen3:8b:k10 | 49.0% | 59.0% |

## Findings

- The re-ranker was frozen on dev (commit c305a4c) before either test set was run. On test it lifts the right table from 44.9% to 55.5% and end-to-end accuracy from 53.4% to 65.8%; on the paraphrased questions, 44.5% to 49.0% and 50.2% to 59.0%. About 30 points remain to the oracle page.
- Retrieval is no longer the main limit: the right table is among BM25's top 10 for about 90% of questions. The pick is: given the right table among the 10, the re-ranker chooses it 61% of the time (55% paraphrased).
- The gain is smaller on paraphrased questions (+4.5 points on the pick, against +10.6). The summaries show only the row labels that share a word with the question, so rewording likely hides the rows that would identify the table.
- Thinking, dev only (all 447 questions): right table 66.7% against 54.4% without thinking. 83 questions are right only with thinking and 28 only without (sign test p = 2e-7). It costs 31.2 s per question against 2.7 s. 4.5% of thinking replies are empty after about 90 s, likely the 4,096-token limit running out mid-thought; those keep BM25's first table.
- Thinking has not been run on the test sets: about 17 hours for both on this GPU.
