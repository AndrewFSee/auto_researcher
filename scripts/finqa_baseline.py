"""
FinQA baseline for a local model served by Ollama.

Measures how well an off-the-shelf model answers FinQA questions before any
fine-tuning, in three modes:

* ``program``: the model replies with a FinQA program only, which is executed
  (``auto_researcher.llm.finqa``). This is the output a fine-tuned model would
  be trained to produce, so it is the number fine-tuning has to beat.
* ``cot_program``: the model reasons briefly, then writes the program.
* ``direct``: the model reasons and states the number itself; nothing is
  executed. The gap to the program modes shows what deterministic execution buys.

Every response is appended to ``<runs-dir>/<model>__<split>__<mode>.jsonl`` so a
run can be stopped and resumed; the report is rebuilt from all run files.

Caveat: the FinQA test set has been public since 2021 and may be in the model's
pre-training data, so these scores can overstate accuracy on unseen filings.

Example::

    python scripts/finqa_baseline.py --model qwen3:8b --modes program cot_program direct
    python scripts/finqa_baseline.py --report-only
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.llm.finqa import (  # noqa: E402
    PROGRAM_LANGUAGE,
    PROGRAM_PROMPT,
    TABLE_OPERATIONS,
    FinQAExample,
    ProgramError,
    answers_match,
    build_messages,
    execute_program,
    extract_answer,
    extract_program,
    load_finqa,
)

logger = logging.getLogger("finqa_baseline")

MODES = {
    "program": (PROGRAM_PROMPT, 256),
    "cot_program": (
        PROGRAM_LANGUAGE + "\nFirst explain briefly which numbers you need and how to combine "
        "them (a few sentences). Then end with a final line:\nProgram: <program>",
        1024,
    ),
    "direct": (
        "You answer questions about one page of a company's annual report. Work the "
        "answer out step by step, then end with a final line:\nAnswer: <number>\n"
        "Give a plain number (a % sign is fine for percentages), or yes/no for a "
        "yes/no question.",
        1024,
    ),
}


def run_file(runs_dir: Path, model: str, split: str, mode: str) -> Path:
    return runs_dir / f"{model.replace(':', '_').replace('/', '_')}__{split}__{mode}.jsonl"


def ask(url: str, model: str, mode: str, ex: FinQAExample, num_ctx: int, retries: int = 5) -> dict:
    system, max_tokens = MODES[mode]
    body = {
        "model": model,
        "messages": build_messages(ex, system),
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": max_tokens},
    }
    # Ollama answers a CUDA out-of-memory with a 500 and unloads the model, so
    # a retry after a pause (it reloads) usually succeeds on a shared laptop GPU.
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            resp = requests.post(f"{url}/api/chat", json=body, timeout=600)
            resp.raise_for_status()
            break
        except (requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(exc.response, "status_code", None)
            if attempt == retries or (status is not None and status < 500):
                raise
            logger.warning("%s on %s (attempt %d); retrying in %ds", exc, ex.id, attempt, 15 * attempt)
            time.sleep(15 * attempt)
    out = resp.json()
    return {
        "response": out["message"]["content"],
        "seconds": time.perf_counter() - start,
        "prompt_tokens": out.get("prompt_eval_count"),
        "output_tokens": out.get("eval_count"),
    }


def score(mode: str, ex: FinQAExample, response: str) -> dict:
    """Parse and execute one response; never raises."""
    rec: dict = {"program": None, "pred": None, "error": None}
    if mode == "direct":
        rec["pred"] = extract_answer(response)
        if rec["pred"] is None:
            rec["error"] = "no answer line"
    else:
        rec["program"] = extract_program(response)
        if rec["program"] is None:
            rec["error"] = "no program line"
        else:
            try:
                rec["pred"] = execute_program(rec["program"], ex.table)
            except ProgramError as exc:
                rec["error"] = f"exec: {exc}"
    for m in ("exact", "scale", "tolerant"):
        rec[m] = answers_match(rec["pred"], ex.answer, m)
    return rec


def evaluate(args: argparse.Namespace) -> None:
    examples = load_finqa(args.data_dir / f"{args.split}.json")[: args.limit]
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        path = run_file(args.runs_dir, args.model, args.split, mode)
        done = set()
        if path.exists():
            done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}
        todo = [ex for ex in examples if ex.id not in done]
        logger.info("%s %s: %d done, %d to go", args.model, mode, len(done), len(todo))
        with path.open("a", encoding="utf-8") as fh:
            for i, ex in enumerate(todo, 1):
                out = ask(args.url, args.model, mode, ex, args.num_ctx)
                rec = {"id": ex.id, "gold": ex.answer, "gold_program": ex.program, **out,
                       **score(mode, ex, out["response"])}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(todo):
                    logger.info("  %s %d/%d (%.1fs/question)", mode, i, len(todo), out["seconds"])


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def bucket(program: str) -> str:
    if any(op in program for op in TABLE_OPERATIONS):
        return "table op"
    steps = program.count("(")
    return "1 step" if steps == 1 else "2 steps" if steps == 2 else "3+ steps"


MISS_TYPES = (
    "no usable program",
    "sign flipped",
    "right numbers, wrong formula",
    "some right numbers",
    "wrong numbers",
    "yes/no wrong",
)


def classify_miss(rec: dict) -> str | None:
    """Why a program-mode answer missed (within 1%), or None if it did not."""
    if rec["tolerant"]:
        return None
    if rec["error"]:
        return "no usable program"
    pred, gold = rec["pred"], rec["gold"]
    if isinstance(gold, str) or isinstance(pred, str):
        return "yes/no wrong"
    if gold and any(math.isclose(c, -gold, rel_tol=0.01) for c in (pred, pred / 100, pred * 100)):
        return "sign flipped"
    gold_nums = set(re.findall(r"(?<![_#\d.])\d+\.?\d*", rec["gold_program"]))
    pred_nums = set(re.findall(r"(?<![_#\d.])\d+\.?\d*", rec["program"]))
    if gold_nums and gold_nums <= pred_nums:
        return "right numbers, wrong formula"
    return "some right numbers" if gold_nums & pred_nums else "wrong numbers"


def summarise(path: Path) -> dict:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    model, split, mode = path.stem.split("__")
    n = len(recs)

    def rate(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    buckets = {}
    for name in ("1 step", "2 steps", "3+ steps", "table op"):
        rows = [r for r in recs if bucket(r["gold_program"]) == name]
        buckets[name] = {"n": len(rows), "exact": rate(rows, "exact"), "scale": rate(rows, "scale")}
    misses = {}
    if mode != "direct":
        kinds = [classify_miss(r) for r in recs]
        misses = {k: kinds.count(k) / n for k in MISS_TYPES}
    return {
        "model": model, "split": split, "mode": mode, "n": n,
        "correct_ids": sorted(r["id"] for r in recs if r["tolerant"]),
        "miss_types": misses,
        "exact": rate(recs, "exact"), "scale": rate(recs, "scale"), "tolerant": rate(recs, "tolerant"),
        "failed": sum(r["error"] is not None for r in recs) / n,
        "seconds": sum(r["seconds"] for r in recs) / n,
        "output_tokens": sum(r["output_tokens"] or 0 for r in recs) / n,
        "buckets": buckets,
        "errors": [r for r in recs if r["error"]][:5],
        "misses": [r for r in recs if not r["scale"] and not r["error"]][:5],
    }


def finetune_section(models_dir: Path) -> list[str]:
    """Training details for each fine-tuned model, from the run_info.json the notebook writes."""
    pct = "{:.1%}".format
    lines: list[str] = []
    for path in sorted(models_dir.glob("*/run_info.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("smoke"):
            continue
        h = info["hyperparameters"]
        evals = {f"step{e['step']}": e for e in info["log_history"] if "eval_loss" in e}
        lines += [
            f"## Fine-tune: `{path.parent.name}`",
            "",
            f"Trained with `notebooks/finqa_qlora_colab.ipynb` on an {info['versions']['gpu']}",
            f"({info['log_history'][-1].get('train_runtime', 0) / 3600:.1f} h): `{info['base_model']}` in 4-bit NF4,",
            f"LoRA r={h['lora_r']}, alpha={h['lora_alpha']}, dropout={h['lora_dropout']} on {h['target_modules']},",
            f"learning rate {h['lr']} ({h['scheduler']}), effective batch {h['batch'] * h['grad_accum']},",
            f"{h['epochs']} epochs ({h['max_steps']} steps) on all {info['n_train']:,} training questions, loss on",
            "the program only. Prompts are rendered exactly as Ollama renders them. Checkpoints",
            f"scored on {info['dev_results']['base']['n']} dev questions; the best "
            f"(`{info['best_checkpoint']}`) was merged into the bf16 base,",
            "quantized to Q4_K_M and served with `qwen3:8b`'s own Modelfile.",
            f"Code at commit `{info['repo_commit'][:10]}`; prompt sha256 `{info['prompt_sha256'][:12]}`.",
            "",
            "| Dev (4-bit training model) | Epoch | Dev loss | Exact | Within 1% | Failed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        rows = [("untuned", info["dev_results"]["base"], None)]
        rows += [(k, v, evals.get(k)) for k, v in info["dev_results"]["checkpoints"].items()]
        for name, r, ev in rows:
            epoch = f"{ev['epoch']:.2f}" if ev else "0"
            loss = f"{ev['eval_loss']:.4f}" if ev else ""
            lines.append(f"| {name} | {epoch} | {loss} | {pct(r['exact'])} | {pct(r['tolerant'])} | "
                         f"{pct(r['failed'])} |")
        lines.append("")
    return lines


SCORES = [
    "* **Exact**: official execution accuracy (answer to 5 decimals).",
    "* **Scale**: also accepts a x100 mismatch (14.5 vs 0.145). FinQA's labels mix",
    "  ratios and percentages (about 10% of test programs multiply by 100), so this",
    "  is the fairer number for comparing prompts.",
    "* **Within 1%**: scale-tolerant and within 1% (the book's `validate_calculation`",
    "  default); used for `direct`, where the model rounds.",
    "* **Failed**: no parseable answer, or the program did not execute.",
    "",
]


def filings_intro(data_file: Path) -> list[str]:
    summary_path = data_file.with_suffix(".summary.json")
    s = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return [
        "# Filing QA: FinQA-style questions on recent 10-K filings",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/finqa_baseline.py`; questions built by",
        f"`scripts/build_filing_qa.py` ({s.get('questions', '?')} questions from {s.get('filings', '?')} "
        f"{'/'.join(s.get('forms', ['10-K']))} filings of",
        f"{s.get('companies', '?')} companies filed from {s.get('filed_from', '?')}, at most "
        f"{s.get('per_filing', '?')} per filing).",
        "",
        "The filings are S&P 500 annual reports filed after the base model's training data,",
        "read from the sentiq project's EDGAR store. Each table with inline-XBRL facts becomes",
        "a FinQA-style page (text before, table, text after, in FinQA's formatting), and a",
        "question is asked only where the tags fix the answer: each number is one tagged fact",
        "on the page, its fiscal year comes from the XBRL context and is the year shown above",
        "its column, and the numbers compared share concept, unit and dimensions. Gold",
        "programs use the numbers as displayed and run on the same executor that scores the",
        "models. Same prompts, options and scoring as the FinQA evaluation",
        "(`docs/results/finqa_baseline.md`).",
        "",
        *SCORES,
    ]


def filings_closing(data_file: Path, summaries: list[dict], runs_dir: Path) -> list[str]:
    pct = "{:.1%}".format
    templates = {r["id"]: r["meta"]["template"]
                 for r in json.loads(data_file.read_text(encoding="utf-8"))}
    names = sorted(set(templates.values()))
    lines = ["## By question template (within 1%)", "",
             "| Model | Mode | " + " | ".join(names) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in names) + " |"]
    for s in summaries:
        path = runs_dir / f"{s['model']}__{s['split']}__{s['mode']}.jsonl"
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        cells = []
        for name in names:
            rows = [r for r in recs if templates.get(r["id"]) == name]
            cells.append(f"{pct(sum(r['tolerant'] for r in rows) / len(rows))} (n={len(rows)})" if rows else "")
        lines.append(f"| {s['model']} | {s['mode']} | " + " | ".join(cells) + " |")
    return lines + [
        "",
        "## Caveats",
        "",
        "* Templated questions (change, percentage change, three-year average, portion of a",
        "  total) are more uniform than FinQA's: they test reading the right cells of a real",
        "  filing and combining them correctly, not open-ended analysis or facts stated only",
        "  in the text.",
        "* The model is handed the page that holds the answer; finding that page in a full",
        "  10-K (retrieval) is not tested here.",
        "* Questions use the table's own row labels, which can be terse (\"iphone\").",
        "* Pages are formatted like FinQA's, which the fine-tuned model was trained on.",
        "",
    ]


def finqa_intro() -> list[str]:
    return [
        "# FinQA: baseline and QLoRA fine-tune",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/finqa_baseline.py`.",
        "",
        "FinQA (Chen et al., 2021) asks numerical questions about one page of an S&P 500",
        "annual report. The model sees the whole page (text and table, not just the gold",
        "facts) and, in the program modes, writes a FinQA program that is executed by",
        "`auto_researcher.llm.finqa` (the executor reproduces 100% of gold answers on dev",
        "and test from the gold programs). Greedy decoding, thinking disabled, all models",
        "served by Ollama at Q4_K_M on the laptop GPU. `qwen3_8b` is the untuned model;",
        "`qwen3-finqa` is it after QLoRA fine-tuning (see below).",
        "",
        *SCORES,
        "Published reference points on the test set (execution accuracy): FinQANet 61.2%,",
        "APOLLO ensemble 71.1%, human experts 91.2%, general crowd 50.7%.",
        "",
    ]


def write_report(runs_dir: Path, report: Path, models_dir: Path, kind: str = "finqa",
                 data_file: Path | None = None) -> None:
    summaries = [summarise(p) for p in sorted(runs_dir.glob("*.jsonl"))]
    if not summaries:
        raise SystemExit(f"no run files in {runs_dir}")
    report.with_suffix(".json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    pct = "{:.1%}".format
    lines = finqa_intro() if kind == "finqa" else filings_intro(data_file)
    lines += [
        "| Model | Split | Mode | N | Exact | Scale | Within 1% | Failed | s/question | Output tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['split']} | {s['mode']} | {s['n']} | {pct(s['exact'])} | "
            f"{pct(s['scale'])} | {pct(s['tolerant'])} | {pct(s['failed'])} | "
            f"{s['seconds']:.1f} | {s['output_tokens']:.0f} |"
        )
    lines += ["", "## By question type (scale-tolerant accuracy)", "",
              "| Model | Mode | 1 step | 2 steps | 3+ steps | Table op |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        b = s["buckets"]
        cells = " | ".join(f"{pct(b[k]['scale'])} (n={b[k]['n']})" for k in ("1 step", "2 steps", "3+ steps", "table op"))
        lines.append(f"| {s['model']} | {s['mode']} | {cells} |")
    lines += ["", "## Paired comparison (within 1%, same questions)", "",
              "McNemar exact test on the questions exactly one of the two gets right.", "",
              "| A | B | Both right | Only A | Only B | Neither | McNemar p |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for a, b in itertools.combinations(summaries, 2):
        if a["split"] != b["split"] or a["n"] != b["n"]:
            continue
        ra, rb = set(a["correct_ids"]), set(b["correct_ids"])
        only_a, only_b = len(ra - rb), len(rb - ra)
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        lines.append(
            f"| {a['model']} {a['mode']} | {b['model']} {b['mode']} | {len(ra & rb)} | {only_a} | "
            f"{only_b} | {a['n'] - len(ra | rb)} | {p:.1g} |"
        )
    lines += ["", "## Why program answers miss (share of all questions)", "",
              "| Model | Mode | " + " | ".join(MISS_TYPES) + " |",
              "| --- | --- | " + " | ".join("---:" for _ in MISS_TYPES) + " |"]
    for s in summaries:
        if s["miss_types"]:
            lines.append(f"| {s['model']} | {s['mode']} | "
                         + " | ".join(pct(s["miss_types"][k]) for k in MISS_TYPES) + " |")
    lines += [
        "",
        "\"Sign flipped\" is a subtraction in the wrong direction; \"right numbers, wrong",
        "formula\" means every number in the gold program appears in the model's program.",
        "",
    ]
    lines += (finqa_closing(models_dir) if kind == "finqa"
              else filings_closing(data_file, summaries, runs_dir))
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", report)


def finqa_closing(models_dir: Path) -> list[str]:
    return [
        *finetune_section(models_dir),
        "## Findings",
        "",
        "Fine-tuning (`qwen3-finqa`):",
        "",
        "* QLoRA on the 6,251 training programs lifts execution accuracy from 53.6% to",
        "  76.5% (within 1%: 59.2% to 79.0%) with the same prompt, and no answer fails to",
        "  parse or execute (4.7% before). It also beats the untuned model's free-form",
        "  answers (71.4% within 1%; 144 questions only the fine-tune gets right vs 57,",
        "  p < 1e-9), so programs now win on accuracy as well as auditability.",
        "* The gain is largest where the untuned model was weakest: 3+ step questions",
        "  (26% to 63%). Wrong numbers fall from 9.0% to 2.1% of questions; formula",
        "  errors with the right numbers (8.9%) are now the main miss: unit scaling",
        "  (thousands vs millions), the wrong base year in a change, and some gold labels",
        "  that are themselves wrong. One miss reproduces a nonsensical pattern found in",
        "  FinQA's own labels: fine-tuning learns the dataset's noise along with its format.",
        "* 76.5% execution accuracy is above the published FinQANet (61.2%) and APOLLO",
        "  (71.1%) results and below human experts (91.2%). The contamination caveat below",
        "  applies to that comparison, not to the before/after one: both runs share the",
        "  same base model.",
        "",
        "Untuned model (`qwen3_8b`):",
        "",
        "* Writing the program alone (`program`) scores 58% scale-tolerant, roughly the",
        "  general-crowd level and below FinQANet (61%). Reasoning first (`cot_program`)",
        "  adds about 6 points, mostly on 3+ step questions.",
        "* Free-form reasoning (`direct`) is right within 1% more often (71%) than either",
        "  program mode, and still 70% within 0.5%, so this is not a tolerance artefact.",
        "  In the questions only `direct` gets right, the reasoning in `cot_program` is",
        "  usually correct and the program is not: subtraction arguments reversed, steps",
        "  in the wrong order, a bare number instead of a program. The model can reason",
        "  about the page but is not fluent in the FinQA language.",
        "* That is the gap supervised fine-tuning on FinQA's 6,251 training programs",
        "  targets. A fine-tuned program model has to beat `direct` (71% within 1%), not",
        "  just `program`, to justify itself on accuracy; the executed, auditable trace is",
        "  the reason to prefer programs at equal accuracy.",
        "* Exact-match accuracy for `direct` (23%) is not meaningful: the model rounds.",
        "",
        "## Caveats",
        "",
        "* The FinQA test set has been public since 2021 and may be in the model's",
        "  pre-training data; scores on unseen filings may be lower.",
        "* FinQA hands the model the one relevant page. On a full 10-K a retrieval step",
        "  must find that page first, which adds its own errors.",
        "* A correct number from a wrong program still counts as correct here; program",
        "  accuracy (matching the reference program) is not measured.",
        "* One training run (one seed). The checkpoint was chosen on 300 dev questions;",
        "  the test set was used once, for the numbers above.",
        "* Exact reruns can differ by a question or two: GPU arithmetic varies slightly",
        "  when Ollama reuses a cached prompt prefix.",
        "",
    ]


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=["program"])
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=None, help="first N questions only")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "finqa")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "data" / "finqa" / "runs")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "results" / "finqa_baseline.md")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "data" / "finqa" / "models",
                        help="fine-tuned models with the notebook's run_info.json")
    parser.add_argument("--kind", choices=["finqa", "filings"], default="finqa",
                        help="which report to write: FinQA, or questions from build_filing_qa.py")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        evaluate(args)
    write_report(args.runs_dir, args.report, args.models_dir, args.kind,
                 args.data_dir / f"{args.split}.json")


if __name__ == "__main__":
    main()
