"""
Rephrase the auto-generated filing questions the way an analyst would ask them.

``build_filing_qa.py`` writes questions that repeat the table's row label word for
word ("what was the change in total net sales from 2024 to 2025?"), which makes
keyword retrieval look better than it would on real questions. This script asks a
local model (Ollama) to rephrase each one without copying the label, then checks the
rewrite mechanically: same years and no new numbers, different wording, and the same
kind of calculation (a percentage change still says "percent", a plain change does
not, an average still says "average", a share still asks for a portion). A failed
rewrite is retried once; if it fails again the question is left out and counted.

Gold programs and answers are unchanged; the original question is kept in
``meta.original_question``.

Example::

    python scripts/paraphrase_filing_questions.py --model qwen3:8b
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.console import use_utf8_output  # noqa: E402

logger = logging.getLogger("paraphrase_filing_questions")

SYSTEM = """\
You rewrite questions about a company's annual report so they read the way a \
financial analyst would ask them, without repeating the line-item name word for word.

Rules:
- Keep every year, and the same kind of calculation: a change (a difference), a \
percentage change, an average, or a portion of a total.
- Keep the line item's meaning exactly. You may use a standard synonym (net income \
for net earnings, revenue for net sales, R&D for research and development, buybacks \
for share repurchases, capex for capital expenditures), reorder words, or drop filler.
- Never generalize (do not turn "depreciation and amortization" into "non-cash \
expenses") and never add qualifiers ("per employee", "per share", "operating") that \
are not in the question. If there is no natural synonym, keep the term.
- Do not add numbers or company names. Reply with the rewritten question only.

Examples:
what was the change in net earnings from 2022 to 2023?
-> how much did net income change between 2022 and 2023?
what was the percentage change in research and development from 2024 to 2025?
-> by what percent did r&d spending grow from 2024 to 2025?
what portion of total revenues in 2024 was services?
-> what share of 2024 revenue came from services?
what was the average common stock repurchased from 2022 to 2024?
-> what was the average annual value of share buybacks from 2022 to 2024?"""

JUDGE = """\
Do these two questions about the same annual report ask for exactly the same number \
(the same line item, years and calculation)? A synonym for the line item is fine; a \
broader, narrower or different item is not. Answer yes or no.

A: {a}
B: {b}"""

KIND_WORDS = {
    "pct_change": re.compile(r"percent|%|growth rate|rate of change"),
    "average": re.compile(r"average|mean"),
    "share": re.compile(r"portion|share|percent|%|fraction|proportion|what part"),
}
STOP = {"what", "was", "the", "in", "from", "to", "of", "and", "a", "an", "total", "net", "for", "by"}


def label_overlap(original: str, rewrite: str) -> float:
    """Share of the original's content words that the rewrite repeats."""
    words = {w for w in re.findall(r"[a-z]+", original.lower()) if w not in STOP}
    if not words:
        return 0.0
    return len(words & set(re.findall(r"[a-z]+", rewrite.lower()))) / len(words)


def check(original: str, rewrite: str, template: str) -> str | None:
    """Why a rewrite is unusable, or None if it passes."""
    if not rewrite or rewrite.lower() == original.lower():
        return "unchanged"
    years_o = set(re.findall(r"\b(?:19|20)\d{2}\b", original))
    years_r = set(re.findall(r"\b(?:19|20)\d{2}\b", rewrite))
    if years_o != years_r:
        return "years changed"
    numbers_r = set(re.findall(r"\d+(?:\.\d+)?", rewrite)) - years_r
    if numbers_r - set(re.findall(r"\d+(?:\.\d+)?", original)):
        return "numbers added"
    if template in KIND_WORDS and not KIND_WORDS[template].search(rewrite):
        return "calculation lost"
    if template == "change" and re.search(r"percent|%|rate of change|growth rate", rewrite):
        return "became a percentage"
    return None


def _chat(url: str, model: str, messages: list[dict], seed: int, temperature: float,
          max_tokens: int) -> str:
    body = {"model": model, "stream": False, "think": False, "messages": messages,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens}}
    resp = requests.post(f"{url}/api/chat", json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def rephrase(url: str, model: str, question: str, seed: int, temperature: float) -> str:
    text = _chat(url, model, [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": question}], seed, temperature, 80)
    text = text.strip('"').strip()
    text = text.splitlines()[0] if text else ""
    return re.sub(r"\s+", " ", text.removeprefix("->").strip()).lower()


def same_question(url: str, model: str, a: str, b: str) -> bool:
    """The model's judgement that two questions ask for the same number (a second filter)."""
    reply = _chat(url, model, [{"role": "user", "content": JUDGE.format(a=a, b=b)}], 0, 0.0, 5)
    return reply.lower().startswith("yes")


def main() -> None:
    use_utf8_output()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "filingqa" / "test.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "filingqa" / "test_paraphrased.json")
    args = ap.parse_args()

    records = json.loads(args.data.read_text(encoding="utf-8"))
    done = {}
    if args.out.exists():  # resume
        done = {r["id"]: r for r in json.loads(args.out.read_text(encoding="utf-8"))}
    out, failures = [], Counter()
    for i, rec in enumerate(records, 1):
        if rec["id"] in done:
            out.append(done[rec["id"]])
            continue
        original, template = rec["qa"]["question"], rec["meta"]["template"]
        rewrite, why = "", "not tried"
        for attempt, temperature in ((0, 0.0), (1, 0.7)):
            rewrite = rephrase(args.url, args.model, original, seed=attempt, temperature=temperature)
            why = check(original, rewrite, template)
            if why is None and not same_question(args.url, args.model, original, rewrite):
                why = "judged different"
            if why is None:
                break
        if why is not None:
            failures[why] += 1
            continue
        new = json.loads(json.dumps(rec))
        new["qa"]["question"] = rewrite
        new["meta"]["original_question"] = original
        new["meta"]["label_overlap"] = round(label_overlap(original, rewrite), 3)
        out.append(new)
        if i % 100 == 0:
            logger.info("%d/%d (%d kept)", i, len(records), len(out))
            args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    overlap = sum(r["meta"]["label_overlap"] for r in out) / max(len(out), 1)
    summary = {"model": args.model, "questions": len(records), "kept": len(out),
               "dropped": dict(failures), "mean_label_overlap": round(overlap, 3)}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
