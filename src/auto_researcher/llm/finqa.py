"""
FinQA: numerical reasoning over earnings-report tables and text.

FinQA (Chen et al., 2021, https://github.com/czyssrs/FinQA) pairs a question
with one page of an S&P 500 annual report (text before the table, the table,
text after it) and a reference program in a small DSL::

    subtract(5829, 5735), divide(#0, 5735)

``#n`` refers to the result of step n, ``const_100`` etc. are constants and
``table_*`` ops aggregate a table row by name. The model is asked for the
program; the program is executed here, so the arithmetic is deterministic and
every answer has an auditable trace.

The executor follows the official ``evaluate.py`` (number parsing, ``exp`` as
power, results rounded to 5 decimals) so ``execution accuracy`` is comparable
with published results. ``answers_match`` adds two looser scores used for
free-form answers, where a model may write 14.5 for a gold ratio of 0.145.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

OPERATIONS = ("add", "subtract", "multiply", "divide", "exp", "greater")
TABLE_OPERATIONS = ("table_max", "table_min", "table_sum", "table_average")


@dataclass(frozen=True)
class FinQAExample:
    """One FinQA question with its report page and gold answer."""

    id: str
    question: str
    pre_text: tuple[str, ...]
    post_text: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]
    program: str
    answer: float | str  # gold execution result ("yes"/"no" for greater)


def load_finqa(path: str | Path) -> list[FinQAExample]:
    """Load one split (``train.json``, ``dev.json`` or ``test.json``)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FinQAExample(
            id=r["id"],
            question=r["qa"]["question"],
            pre_text=tuple(r["pre_text"]),
            post_text=tuple(r["post_text"]),
            table=tuple(tuple(row) for row in r["table"]),
            program=r["qa"]["program"],
            answer=r["qa"]["exe_ans"],
        )
        for r in rows
    ]


def format_document(ex: FinQAExample) -> str:
    """The report page as plain text: pre-text, a pipe table, post-text."""
    table = "\n".join(" | ".join(cell.strip() for cell in row) for row in ex.table)
    return "\n".join(
        [" ".join(ex.pre_text).strip(), "", "Table:", table, "", " ".join(ex.post_text).strip()]
    ).strip()


# --------------------------------------------------------------------------- #
# Prompts (shared by the baseline script and fine-tuning, so both see the same text)
# --------------------------------------------------------------------------- #

PROGRAM_LANGUAGE = """\
You answer questions about one page of a company's annual report by writing a \
short program in the FinQA language. The program is executed to get the answer.

Operations (two arguments each):
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)
  exp(a, b) = a to the power b
  greater(a, b) = "yes" if a > b else "no"
Table operations take a row name copied exactly from the table, then the word none:
  table_sum(row, none)  table_average(row, none)  table_max(row, none)  table_min(row, none)

Arguments are numbers copied from the page (no units, $ signs or thousands \
separators), constants (const_1, const_2, ..., const_100, const_1000, \
const_1000000, const_m1 for -1), or #n for the result of step n, counting from 0. \
Separate steps with ", ".
Give percentages and growth rates as ratios (0.25, not 25): do not multiply by 100.

Examples:
Q: what was the percentage change in net revenue from 2014 to 2015?
Facts: net revenue was $ 5735 million in 2014 and $ 5829 million in 2015.
Program: subtract(5829, 5735), divide(#0, 5735)

Q: what portion of total debt is due within one year?
Facts: current portion of long-term debt 1,200; total debt 8,000.
Program: divide(1200, 8000)

Q: what was the average provision for income taxes over the three years?
Facts: table row "provision for income taxes | $ 829 | $ 480 | $ 104".
Program: table_average(provision for income taxes, none)

Q: what was the total grant-date value of the shares granted, in millions?
Facts: 607 thousand shares were granted at a fair value of $ 18.13 per share.
Program: multiply(607, 18.13), divide(#0, const_1000)

Q: did the company's five-year cumulative return exceed the s&p 500's?
Facts: $ 100 invested grew to $ 286.61 in the company and $ 198.09 in the s&p 500.
Program: greater(286.61, 198.09)

Q: what was the compound annual growth rate of the index over the five years?
Facts: the index grew from 100 to 208.14 over five years.
Program: divide(208.14, const_100), divide(const_1, const_5), exp(#0, #1), subtract(#2, const_1)
"""

PROGRAM_PROMPT = PROGRAM_LANGUAGE + "\nReply with one line only:\nProgram: <program>"


def build_messages(ex: FinQAExample, system: str = PROGRAM_PROMPT) -> list[dict[str, str]]:
    """Chat messages for one question: the system prompt, then the page and question."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Report page:\n{format_document(ex)}\n\nQuestion: {ex.question}"},
    ]


# --------------------------------------------------------------------------- #
# Program parsing and execution
# --------------------------------------------------------------------------- #


class ProgramError(ValueError):
    """The program could not be parsed or executed."""


_STEP_START = re.compile(r"\s*,?\s*([a-z_]+)\s*\(")


def str_to_num(text: str) -> float | None:
    """Parse a program argument or table cell the way the official evaluator does."""
    text = text.replace(",", "").replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        pass
    if "%" in text:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            return None
    if text.startswith("const_"):
        value = text[len("const_"):]
        try:
            return -1.0 if value == "m1" else float(value)
        except ValueError:
            return None
    return None


def parse_program(program: str) -> list[tuple[str, list[str]]]:
    """Split ``op(a, b), op(#0, c)`` into ``[(op, [a, b]), ...]``.

    Parentheses inside arguments (table row names such as ``net income ( loss )``)
    are kept: only the parentheses that close each step end it.
    """
    steps: list[tuple[str, list[str]]] = []
    i, n = 0, len(program)
    while i < n:
        match = _STEP_START.match(program, i)
        if not match:
            if program[i:].strip(" ,\n"):
                raise ProgramError(f"unparseable program text: {program[i:]!r}")
            break
        op, depth, j = match.group(1), 1, match.end()
        while j < n and depth:
            depth += {"(": 1, ")": -1}.get(program[j], 0)
            j += 1
        if depth:
            raise ProgramError(f"unbalanced parentheses in {program!r}")
        body = program[match.end(): j - 1]
        if op in TABLE_OPERATIONS:
            name, _, _ = body.rpartition(",")  # second argument is always "none"
            args = [(name or body).strip()]
        else:
            # FinQA separates arguments with ", "; a bare comma is then a
            # thousands separator (``add(1,000, 1)``).
            sep = r",\s+" if re.search(r",\s", body) else ","
            args = [a.strip() for a in re.split(sep, body)]
        steps.append((op, args))
        i = j
    if not steps:
        raise ProgramError("empty program")
    return steps


def _normalise(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _table_row(table: tuple[tuple[str, ...], ...], name: str) -> list[float]:
    for row in table:
        if row and _normalise(row[0]) == _normalise(name):
            values = [str_to_num(cell.split("(")[0]) for cell in row[1:]]
            if any(v is None for v in values) or not values:
                raise ProgramError(f"non-numeric cell in table row {name!r}")
            return values  # type: ignore[return-value]
    raise ProgramError(f"no table row named {name!r}")


def execute_program(program: str, table: tuple[tuple[str, ...], ...] = ()) -> float | str:
    """Run a FinQA program; returns a float rounded to 5 decimals, or "yes"/"no"."""
    results: list[float | str] = []
    for op, args in parse_program(program):
        if op in TABLE_OPERATIONS:
            values = _table_row(table, args[0])
            result: float | str = {
                "table_max": max, "table_min": min, "table_sum": sum,
                "table_average": lambda v: sum(v) / len(v),
            }[op](values)
        elif op in OPERATIONS:
            if len(args) != 2:
                raise ProgramError(f"{op} takes 2 arguments, got {args}")
            nums: list[float] = []
            for arg in args:
                if arg.startswith("#"):
                    try:
                        prior = results[int(arg[1:])]
                    except (ValueError, IndexError) as exc:
                        raise ProgramError(f"bad step reference {arg}") from exc
                    if isinstance(prior, str):
                        raise ProgramError(f"step {arg} is not numeric")
                    nums.append(prior)
                else:
                    num = str_to_num(arg)
                    if num is None:
                        raise ProgramError(f"not a number: {arg!r}")
                    nums.append(num)
            a, b = nums
            try:
                if op == "add":
                    result = a + b
                elif op == "subtract":
                    result = a - b
                elif op == "multiply":
                    result = a * b
                elif op == "divide":
                    result = a / b
                elif op == "exp":
                    result = a ** b
                else:
                    result = "yes" if a > b else "no"
            except (ZeroDivisionError, OverflowError) as exc:
                raise ProgramError(f"{op}({a}, {b}) failed: {exc}") from exc
            if isinstance(result, complex):
                raise ProgramError(f"{op}({a}, {b}) is complex")
        else:
            raise ProgramError(f"unknown operation {op!r}")
        results.append(result)
    final = results[-1]
    return final if isinstance(final, str) else round(final, 5)


# --------------------------------------------------------------------------- #
# Model output parsing and scoring
# --------------------------------------------------------------------------- #

_PROGRAM_LINE = re.compile(r"program\s*:\s*(.+)", re.IGNORECASE)
_ANSWER_LINE = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\(?\$?\s*-?\d[\d,]*\.?\d*\)?\s*%?")


def extract_program(response: str) -> str | None:
    """The program from the last ``Program:`` line (or the whole reply if bare)."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).replace("`", "")
    lines = _PROGRAM_LINE.findall(response)
    if lines:
        return lines[-1].strip()
    stripped = response.strip()
    return stripped if re.match(r"^[a-z_]+\(", stripped) else None


def extract_answer(response: str) -> float | str | None:
    """The value on the last ``Answer:`` line: a number (``%`` kept as a percent), or yes/no."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    lines = _ANSWER_LINE.findall(response)
    if not lines:
        return None
    text = lines[-1].strip().lower()
    if text.startswith(("yes", "no")):
        return "yes" if text.startswith("yes") else "no"
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = match.group(0)
    negative = raw.strip().startswith("-") or "(" in raw
    value = str_to_num(re.sub(r"[()%\s-]", "", raw))
    if value is None:
        return None
    return -value if negative else value


def answers_match(pred: float | str | None, gold: float | str, mode: str = "exact") -> bool:
    """Compare a predicted answer with the gold execution result.

    ``exact``: official execution accuracy (5-decimal rounding, tiny float slack).
    ``scale``: also accepts a factor-of-100 mismatch (14.5 vs 0.145), the
    percent-versus-ratio ambiguity that FinQA's own labels do not resolve.
    ``tolerant``: ``scale`` with a 1% relative tolerance (the book's
    ``validate_calculation`` default), for free-form answers that round.
    """
    if pred is None:
        return False
    if isinstance(gold, str) or isinstance(pred, str):
        return str(pred).lower() == str(gold).lower()
    if not math.isfinite(pred):
        return False
    rel = 0.01 if mode == "tolerant" else 1e-4
    candidates = [pred] if mode == "exact" else [pred, pred / 100.0, pred * 100.0]
    for c in candidates:
        if math.isclose(gold, 0.0, abs_tol=1e-5):
            if math.isclose(c, 0.0, abs_tol=1e-5):
                return True
        elif math.isclose(round(c, 5), gold, rel_tol=rel, abs_tol=1e-5):
            return True
    return False
