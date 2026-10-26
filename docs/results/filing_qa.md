# Filing QA: FinQA-style questions on recent 10-K filings

Generated 2026-10-26 03:01 by `scripts/finqa_baseline.py`; questions built by
`scripts/build_filing_qa.py` (980 questions from 996 10-K filings of
526 companies filed from 2025-01-01, at most 1 per filing).

The filings are S&P 500 annual reports filed after the base model's training data,
read from the sentiq project's EDGAR store. Each table with inline-XBRL facts becomes
a FinQA-style page (text before, table, text after, in FinQA's formatting), and a
question is asked only where the tags fix the answer: each number is one tagged fact
on the page, its fiscal year comes from the XBRL context and is the year shown above
its column, and the numbers compared share concept, unit and dimensions. Gold
programs use the numbers as displayed and run on the same executor that scores the
models. Same prompts, options and scoring as the FinQA evaluation
(`docs/results/finqa_baseline.md`).

* **Exact**: official execution accuracy (answer to 5 decimals).
* **Scale**: also accepts a x100 mismatch (14.5 vs 0.145). FinQA's labels mix
  ratios and percentages (about 10% of test programs multiply by 100), so this
  is the fairer number for comparing prompts.
* **Within 1%**: scale-tolerant and within 1% (the book's `validate_calculation`
  default); used for `direct`, where the model rounds.
* **Failed**: no parseable answer, or the program did not execute.

| Model | Split | Mode | N | Exact | Scale | Within 1% | Failed | s/question | Output tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-finqa | test | program | 980 | 95.2% | 95.2% | 95.3% | 0.1% | 3.5 | 23 |
| qwen3_8b | test | direct | 980 | 39.2% | 60.1% | 90.5% | 0.0% | 6.7 | 176 |
| qwen3_8b | test | program | 980 | 90.2% | 90.4% | 90.7% | 0.9% | 3.2 | 19 |

## By question type (scale-tolerant accuracy)

| Model | Mode | 1 step | 2 steps | 3+ steps | Table op |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3-finqa | program | 96.6% (n=465) | 97.3% (n=259) | 90.6% (n=256) |  |
| qwen3_8b | direct | 59.1% (n=465) | 47.1% (n=259) | 75.0% (n=256) |  |
| qwen3_8b | program | 86.2% (n=465) | 93.4% (n=259) | 94.9% (n=256) |  |

## Paired comparison (within 1%, same questions)

McNemar exact test on the questions exactly one of the two gets right.

| A | B | Both right | Only A | Only B | Neither | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-finqa program | qwen3_8b direct | 859 | 75 | 28 | 18 | 4e-06 |
| qwen3-finqa program | qwen3_8b program | 858 | 76 | 31 | 15 | 2e-05 |
| qwen3_8b direct | qwen3_8b program | 822 | 65 | 67 | 26 | 0.9 |

## Why program answers miss (share of all questions)

| Model | Mode | no usable program | sign flipped | right numbers, wrong formula | some right numbers | wrong numbers | yes/no wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-finqa | program | 0.1% | 0.0% | 2.1% | 1.3% | 1.1% | 0.0% |
| qwen3_8b | program | 0.9% | 0.7% | 2.2% | 3.5% | 1.9% | 0.0% |

"Sign flipped" is a subtraction in the wrong direction; "right numbers, wrong
formula" means every number in the gold program appears in the model's program.

## By question template (within 1%)

| Model | Mode | average | change | pct_change | share |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3-finqa | program | 90.6% (n=256) | 97.2% (n=212) | 97.3% (n=259) | 96.4% (n=253) |
| qwen3_8b | direct | 85.5% (n=256) | 90.1% (n=212) | 94.6% (n=259) | 91.7% (n=253) |
| qwen3_8b | program | 94.9% (n=256) | 84.0% (n=212) | 93.4% (n=259) | 89.3% (n=253) |

## Findings (980 questions, 10-Ks filed 2025-26)

* The FinQA fine-tune carries over to filings the model has never seen: 95.3% within
  1% against 90.7% for the untuned model writing programs (76 vs 31 discordant
  questions, p = 2e-5) and 90.5% for its free-form answers (75 vs 28, p = 4e-6).
  No answer fails to run (0.9% before). The gain is smaller than on FinQA (+5 points
  vs +20) because these questions are easier: the untuned model already gets 90%.
* The untuned model's main error here is reading "change" as a percentage change
  (right subtraction, then a division: 20 of its 34 misses on `change`); the
  fine-tune learned FinQA's convention that a change is a difference (6 misses).
* Fine-tuning made one thing worse: three-year averages (90.6% vs 94.9% untuned).
  15 of its 24 misses are the same broken program, `add(a, b), add(c, #0),
  add(#1, const_3), divide(#2, const_2)` (add 3, then halve), which comes from FinQA's
  own labels: 34 of the 599 'average' questions in FinQA's training set are labelled
  that way (8 in dev, 2 in test, so FinQA's test score barely shows it). The model
  learned the label noise along with the format. Repairing those labels and
  retraining is the obvious fix.
* Free-form answers and programs tie for the untuned model (p = 0.9), unlike on
  FinQA, where free-form was 12 points better: with simple questions, fluency in the
  FinQA language matters less.

## Caveats

* Templated questions (change, percentage change, three-year average, portion of a
  total) are more uniform than FinQA's: they test reading the right cells of a real
  filing and combining them correctly, not open-ended analysis or facts stated only
  in the text.
* The model is handed the page that holds the answer; finding that page in a full
  10-K (retrieval) is not tested here.
* Questions use the table's own row labels, which can be terse ("iphone").
* Pages are formatted like FinQA's, which the fine-tuned model was trained on.
