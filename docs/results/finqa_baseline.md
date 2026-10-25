# FinQA baseline (before fine-tuning)

Generated 2026-10-25 05:40 by `scripts/finqa_baseline.py`.

FinQA (Chen et al., 2021) asks numerical questions about one page of an S&P 500
annual report. The model sees the whole page (text and table, not just the gold
facts) and, in the program modes, writes a FinQA program that is executed by
`auto_researcher.llm.finqa` (the executor reproduces 100% of gold answers on dev
and test from the gold programs). Greedy decoding, thinking disabled.

* **Exact**: official execution accuracy (answer to 5 decimals).
* **Scale**: also accepts a x100 mismatch (14.5 vs 0.145). FinQA's labels mix
  ratios and percentages (about 10% of test programs multiply by 100), so this
  is the fairer number for comparing prompts.
* **Within 1%**: scale-tolerant and within 1% (the book's `validate_calculation`
  default); used for `direct`, where the model rounds.
* **Failed**: no parseable answer, or the program did not execute.

Published reference points on the test set (execution accuracy): FinQANet 61.2%,
APOLLO ensemble 71.1%, human experts 91.2%, general crowd 50.7%.

| Model | Split | Mode | N | Exact | Scale | Within 1% | Failed | s/question | Output tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_8b | test | cot_program | 1147 | 58.7% | 64.5% | 65.6% | 3.2% | 6.4 | 106 |
| qwen3_8b | test | direct | 1147 | 23.2% | 40.2% | 71.4% | 0.1% | 9.2 | 178 |
| qwen3_8b | test | program | 1147 | 53.6% | 58.1% | 59.2% | 4.7% | 3.5 | 24 |

## By question type (scale-tolerant accuracy)

| Model | Mode | 1 step | 2 steps | 3+ steps | Table op |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3_8b | cot_program | 68.3% (n=616) | 60.8% (n=408) | 48.8% (n=84) | 76.9% (n=39) |
| qwen3_8b | direct | 41.6% (n=616) | 35.0% (n=408) | 35.7% (n=84) | 82.1% (n=39) |
| qwen3_8b | program | 60.2% (n=616) | 59.1% (n=408) | 26.2% (n=84) | 82.1% (n=39) |

## Paired comparison (within 1%, same questions)

| Model | A | B | Both right | Only A | Only B | Neither |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen3_8b | cot_program | direct | 701 | 51 | 118 | 277 |
| qwen3_8b | cot_program | program | 606 | 146 | 73 | 322 |
| qwen3_8b | direct | program | 614 | 205 | 65 | 263 |

## Why program answers miss (share of all questions)

| Model | Mode | no usable program | sign flipped | right numbers, wrong formula | some right numbers | wrong numbers | yes/no wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_8b | cot_program | 3.2% | 3.7% | 9.2% | 9.7% | 7.8% | 0.8% |
| qwen3_8b | program | 4.7% | 2.9% | 12.4% | 11.5% | 9.0% | 0.3% |

"Sign flipped" is a subtraction in the wrong direction; "right numbers, wrong
formula" means every number in the gold program appears in the model's program.

## Findings (qwen3:8b, before fine-tuning)

* Writing the program alone (`program`) scores 58% scale-tolerant, roughly the
  general-crowd level and below FinQANet (61%). Reasoning first (`cot_program`)
  adds about 6 points, mostly on 3+ step questions.
* Free-form reasoning (`direct`) is right within 1% more often (71%) than either
  program mode, and still 70% within 0.5%, so this is not a tolerance artefact.
  In the questions only `direct` gets right, the reasoning in `cot_program` is
  usually correct and the program is not: subtraction arguments reversed, steps
  in the wrong order, a bare number instead of a program. The model can reason
  about the page but is not fluent in the FinQA language.
* That is the gap supervised fine-tuning on FinQA's 6,251 training programs
  targets. A fine-tuned program model has to beat `direct` (71% within 1%), not
  just `program`, to justify itself on accuracy; the executed, auditable trace is
  the reason to prefer programs at equal accuracy.
* Exact-match accuracy for `direct` (23%) is not meaningful: the model rounds.

## Caveats

* The FinQA test set has been public since 2021 and may be in the model's
  pre-training data; scores on unseen filings may be lower.
* FinQA hands the model the one relevant page. On a full 10-K a retrieval step
  must find that page first, which adds its own errors.
* A correct number from a wrong program still counts as correct here; program
  accuracy (matching the reference program) is not measured.
