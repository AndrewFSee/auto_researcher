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


def write_report(runs_dir: Path, report: Path) -> None:
    summaries = [summarise(p) for p in sorted(runs_dir.glob("*.jsonl"))]
    if not summaries:
        raise SystemExit(f"no run files in {runs_dir}")
    report.with_suffix(".json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    pct = "{:.1%}".format
    lines = [
        "# FinQA baseline (before fine-tuning)",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/finqa_baseline.py`.",
        "",
        "FinQA (Chen et al., 2021) asks numerical questions about one page of an S&P 500",
        "annual report. The model sees the whole page (text and table, not just the gold",
        "facts) and, in the program modes, writes a FinQA program that is executed by",
        "`auto_researcher.llm.finqa` (the executor reproduces 100% of gold answers on dev",
        "and test from the gold programs). Greedy decoding, thinking disabled.",
        "",
        "* **Exact**: official execution accuracy (answer to 5 decimals).",
        "* **Scale**: also accepts a x100 mismatch (14.5 vs 0.145). FinQA's labels mix",
        "  ratios and percentages (about 10% of test programs multiply by 100), so this",
        "  is the fairer number for comparing prompts.",
        "* **Within 1%**: scale-tolerant and within 1% (the book's `validate_calculation`",
        "  default); used for `direct`, where the model rounds.",
        "* **Failed**: no parseable answer, or the program did not execute.",
        "",
        "Published reference points on the test set (execution accuracy): FinQANet 61.2%,",
        "APOLLO ensemble 71.1%, human experts 91.2%, general crowd 50.7%.",
        "",
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
              "| Model | A | B | Both right | Only A | Only B | Neither |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for a, b in itertools.combinations(summaries, 2):
        if (a["model"], a["split"]) != (b["model"], b["split"]):
            continue
        ra, rb = set(a["correct_ids"]), set(b["correct_ids"])
        lines.append(
            f"| {a['model']} | {a['mode']} | {b['mode']} | {len(ra & rb)} | {len(ra - rb)} | "
            f"{len(rb - ra)} | {a['n'] - len(ra | rb)} |"
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
        "## Findings (qwen3:8b, before fine-tuning)",
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
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", report)


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
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        evaluate(args)
    write_report(args.runs_dir, args.report)


if __name__ == "__main__":
    main()
