# FinQA: baseline and QLoRA fine-tune

Generated 2026-10-25 17:43 by `scripts/finqa_baseline.py`.

FinQA (Chen et al., 2021) asks numerical questions about one page of an S&P 500
annual report. The model sees the whole page (text and table, not just the gold
facts) and, in the program modes, writes a FinQA program that is executed by
`auto_researcher.llm.finqa` (the executor reproduces 100% of gold answers on dev
and test from the gold programs). Greedy decoding, thinking disabled, all models
served by Ollama at Q4_K_M on the laptop GPU. `qwen3_8b` is the untuned model;
`qwen3-finqa` is it after QLoRA fine-tuning (see below).

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
| qwen3-finqa | test | program | 1147 | 76.5% | 78.4% | 79.0% | 0.0% | 3.7 | 21 |
| qwen3_8b | test | cot_program | 1147 | 58.7% | 64.5% | 65.6% | 3.2% | 6.4 | 106 |
| qwen3_8b | test | direct | 1147 | 23.2% | 40.2% | 71.4% | 0.1% | 9.2 | 178 |
| qwen3_8b | test | program | 1147 | 53.6% | 58.1% | 59.2% | 4.7% | 3.5 | 24 |

## By question type (scale-tolerant accuracy)

| Model | Mode | 1 step | 2 steps | 3+ steps | Table op |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3-finqa | program | 81.3% (n=616) | 75.5% (n=408) | 63.1% (n=84) | 94.9% (n=39) |
| qwen3_8b | cot_program | 68.3% (n=616) | 60.8% (n=408) | 48.8% (n=84) | 76.9% (n=39) |
| qwen3_8b | direct | 41.6% (n=616) | 35.0% (n=408) | 35.7% (n=84) | 82.1% (n=39) |
| qwen3_8b | program | 60.2% (n=616) | 59.1% (n=408) | 26.2% (n=84) | 82.1% (n=39) |

## Paired comparison (within 1%, same questions)

McNemar exact test on the questions exactly one of the two gets right.

| A | B | Both right | Only A | Only B | Neither | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-finqa program | qwen3_8b cot_program | 715 | 191 | 37 | 204 | 3e-26 |
| qwen3-finqa program | qwen3_8b direct | 762 | 144 | 57 | 184 | 7e-10 |
| qwen3-finqa program | qwen3_8b program | 652 | 254 | 27 | 214 | 2e-47 |
| qwen3_8b cot_program | qwen3_8b direct | 701 | 51 | 118 | 277 | 3e-07 |
| qwen3_8b cot_program | qwen3_8b program | 606 | 146 | 73 | 322 | 9e-07 |
| qwen3_8b direct | qwen3_8b program | 614 | 205 | 65 | 263 | 5e-18 |

## Why program answers miss (share of all questions)

| Model | Mode | no usable program | sign flipped | right numbers, wrong formula | some right numbers | wrong numbers | yes/no wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-finqa | program | 0.0% | 1.6% | 8.9% | 8.0% | 2.1% | 0.4% |
| qwen3_8b | cot_program | 3.2% | 3.7% | 9.2% | 9.7% | 7.8% | 0.8% |
| qwen3_8b | program | 4.7% | 2.9% | 12.4% | 11.5% | 9.0% | 0.3% |

"Sign flipped" is a subtraction in the wrong direction; "right numbers, wrong
formula" means every number in the gold program appears in the model's program.

## Fine-tune: `qwen3-8b-finqa`

Trained with `notebooks/finqa_qlora_colab.ipynb` on an NVIDIA A100-SXM4-80GB
(3.3 h): `Qwen/Qwen3-8B` in 4-bit NF4,
LoRA r=16, alpha=32, dropout=0.05 on q,k,v,o,gate,up,down,
learning rate 0.0002 (cosine), effective batch 16,
2 epochs (782 steps) on all 6,251 training questions, loss on
the program only. Prompts are rendered exactly as Ollama renders them. Checkpoints
scored on 300 dev questions; the best (`step782`) was merged into the bf16 base,
quantized to Q4_K_M and served with `qwen3:8b`'s own Modelfile.
Code at commit `4aa1aa3954`; prompt sha256 `71e761a062a8`.

| Dev (4-bit training model) | Epoch | Dev loss | Exact | Within 1% | Failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| untuned | 0 |  | 52.3% | 57.0% | 6.0% |
| step196 | 0.50 | 0.0825 | 73.0% | 75.3% | 0.3% |
| step392 | 1.00 | 0.0730 | 73.3% | 76.3% | 0.7% |
| step588 | 1.50 | 0.0715 | 75.0% | 77.7% | 0.0% |
| step782 | 2.00 | 0.0684 | 77.0% | 79.7% | 0.0% |

## Findings

Fine-tuning (`qwen3-finqa`):

* QLoRA on the 6,251 training programs lifts execution accuracy from 53.6% to
  76.5% (within 1%: 59.2% to 79.0%) with the same prompt, and no answer fails to
  parse or execute (4.7% before). It also beats the untuned model's free-form
  answers (71.4% within 1%; 144 questions only the fine-tune gets right vs 57,
  p < 1e-9), so programs now win on accuracy as well as auditability.
* The gain is largest where the untuned model was weakest: 3+ step questions
  (26% to 63%). Wrong numbers fall from 9.0% to 2.1% of questions; formula
  errors with the right numbers (8.9%) are now the main miss: unit scaling
  (thousands vs millions), the wrong base year in a change, and some gold labels
  that are themselves wrong. One miss reproduces a nonsensical pattern found in
  FinQA's own labels: fine-tuning learns the dataset's noise along with its format.
* 76.5% execution accuracy is above the published FinQANet (61.2%) and APOLLO
  (71.1%) results and below human experts (91.2%). The contamination caveat below
  applies to that comparison, not to the before/after one: both runs share the
  same base model.

Untuned model (`qwen3_8b`):

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
* One training run (one seed). The checkpoint was chosen on 300 dev questions;
  the test set was used once, for the numbers above.
* Exact reruns can differ by a question or two: GPU arithmetic varies slightly
  when Ollama reuses a cached prompt prefix.
