"""
Auto-graded FinQA-style questions from the inline-XBRL facts on filing pages.

A question is only generated when the tags make its answer unambiguous:

* every number comes from one tagged fact, in a cell of the rendered table;
* each fact covers a fiscal year (an instant, or a duration of about a year), and the
  header above its column shows exactly one year, within a year of the period end
  (fiscal years ending in January or February are labelled by the year they start in);
  that header year is the one used in the question, so its years are the ones a
  reader sees;
* the numbers compared across years share concept, unit and dimension members;
* the row label is specific (not "total" or "other") and unique within the table.

The gold program is written in the FinQA language over the numbers as displayed (the
table's own units), and the answer is computed by the same executor that scores the
model, as FinQA's gold answers are.

Templates (FinQA's most common question types):

* ``pct_change``: "what was the percentage change in {row} from {y1} to {y2}?"
* ``change``:     "what was the change in {row} from {y1} to {y2}?"
* ``average``:    "what was the average {row} from {y1} to {y3}?"
* ``share``:      "what portion of {total row} in {y} was {row}?"
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from auto_researcher.llm.filings import Fact, FilingHeader, Page
from auto_researcher.llm.finqa import ProgramError, execute_program

TEMPLATES = ("pct_change", "change", "average", "share")
MAX_PAGE_CHARS = 9000  # the longest FinQA page; keeps prompts inside the 8k context
GENERIC_LABELS = {"total", "totals", "other", "net", "subtotal", "total other", "other, net",
                  "other , net", "balance", "amount", "amounts", "thereafter", "total amount",
                  "total amounts", "total net", "total , net", "net amount", "value", "total value"}
TOTAL_LABEL = re.compile(r"^total\b")
# Revenue lines: any other row as a portion of them is meaningful (a margin, a cost share)
REVENUE_LABEL = re.compile(r"^(net sales|(total )?(net )?(revenues?|sales))$")


@dataclass(frozen=True)
class Question:
    template: str
    question: str
    program: str
    answer: float
    page: Page
    row: int
    facts: tuple[Fact, ...]


def clean_label(label: str) -> str:
    label = label.rstrip(" :")
    label = re.sub(r"(\s*\(\s*(\d|[a-z])\s*\))+\s*$", "", label)  # footnote markers "( 1 )"
    label = re.sub(r"^(less|add|plus|deduct)\s*:?\s+", "", label)    # "less : treasury stock"
    return label.rstrip(" :").strip()


def fact_year(fact: Fact) -> int | None:
    """The fiscal year shown above the fact's column, if unambiguous."""
    if not fact.context.annual:
        return None
    years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", fact.column_header)}
    if len(years) != 1:
        return None
    year = years.pop()
    end = fact.context.end
    if not (end.year - 1 <= year <= end.year):
        return None
    if year == end.year - 1 and end.month > 3:  # header a year behind a period ending mid-year
        return None
    return year


def _value(fact: Fact) -> str:
    return ("-" if fact.negative and fact.magnitude != "0" else "") + fact.magnitude


def _row_series(page: Page) -> dict[int, list[tuple[int, Fact]]]:
    """Per labelled row: (year, fact) pairs, consecutive-comparable, sorted by year."""
    labels = Counter(clean_label(r[0]) for r in page.table if r and r[0])
    by_row: dict[int, list[Fact]] = defaultdict(list)
    for fact in page.facts:
        by_row[fact.row].append(fact)
    series = {}
    for row, facts in by_row.items():
        label = clean_label(page.table[row][0])
        if (len(label) < 4 or label in GENERIC_LABELS or labels[label] > 1
                or not re.search(r"[a-z]{3}", label)):
            continue
        dated = [(fact_year(f), f) for f in facts]
        dated = [(y, f) for y, f in dated if y is not None]
        if not dated or len({y for y, _ in dated}) != len(dated):
            # two values for one year (assets and liabilities columns, segments side by
            # side): a question naming only the row and year could mean either
            continue
        keys = {(f.concept, f.unit, f.context.dims) for _, f in dated}
        if len(keys) != 1:
            continue
        series[row] = sorted(dated, key=lambda yf: yf[0])
    return series


def page_questions(page: Page) -> list[Question]:
    """Every question the templates can ask about one page."""
    if page.chars > MAX_PAGE_CHARS:
        return []
    out: list[Question] = []
    series = _row_series(page)
    for row, dated in series.items():
        label = clean_label(page.table[row][0])
        # Rates and percentages: "percentage change" would be ambiguous (relative or points)
        percent_row = ("%" in label or "percent" in label or "rate" in label.split()
                       or dated[0][1].unit.lower() in ("pure", "percent", "number")
                       or any("%" in page.table[f.row][f.cell] for _, f in dated))
        # A row shown entirely in parentheses (repurchases, expenses in a cash-flow
        # statement) reads naturally either signed or as magnitudes, so questions whose
        # answer would flip sign between the readings are not asked of it.
        all_negative = all(f.negative for _, f in dated)
        for (y1, f1), (y2, f2) in zip(dated, dated[1:]):
            if y2 != y1 + 1:
                continue
            v1, v2 = _value(f1), _value(f2)
            if float(v1) == float(v2):  # no change: a question that tests nothing
                continue
            if not all_negative:
                out.append(Question("change", f"what was the change in {label} from {y1} to {y2}?",
                                    f"subtract({v2}, {v1})", 0.0, page, row, (f1, f2)))
            # percentage change is the same under both readings, but not across a sign flip
            if not percent_row and float(v1) != 0 and f1.negative == f2.negative:
                out.append(Question("pct_change",
                                    f"what was the percentage change in {label} from {y1} to {y2}?",
                                    f"subtract({v2}, {v1}), divide(#0, {v1})", 0.0, page, row, (f1, f2)))
        for (y1, f1), (y2, f2), (y3, f3) in zip(dated, dated[1:], dated[2:]):
            if y3 == y1 + 2 and not percent_row and not all_negative:
                v1, v2, v3 = _value(f1), _value(f2), _value(f3)
                out.append(Question("average", f"what was the average {label} from {y1} to {y3}?",
                                    f"add({v1}, {v2}), add(#0, {v3}), divide(#1, const_3)", 0.0,
                                    page, row, (f1, f2, f3)))
    # share: a row as a portion of a total row, same column (same period), same unit.
    # For a "total ..." row the part must be one of its components: above it, with no
    # other total row in between ("accounts payable" of "total current liabilities", not
    # "current assets" of "total liabilities"). Revenue rows accept any other row.
    def label_of(row: int) -> str:
        return clean_label(page.table[row][0]) if page.table[row] else ""

    total_rows = [r for r in range(len(page.table)) if TOTAL_LABEL.match(label_of(r))]
    totals = {row: d for row, d in series.items()
              if TOTAL_LABEL.match(label_of(row)) or REVENUE_LABEL.match(label_of(row))}
    for t_row, t_dated in totals.items():
        t_label = label_of(t_row)
        revenue = bool(REVENUE_LABEL.match(t_label))
        prev_total = max((r for r in total_rows if r < t_row), default=-1)
        for year, tf in t_dated:
            if tf.negative or float(tf.magnitude) == 0:
                continue
            for row, dated in series.items():
                if row == t_row or (not revenue and not prev_total < row < t_row):
                    continue
                if label_of(row) in t_label:  # "operating lease liabilities" of "total operating
                    continue                  # lease liabilities": a current portion, unnamed
                for y, f in dated:
                    if (y == year and f.unit == tf.unit and f.context.end == tf.context.end
                            and not f.negative and 0 < float(f.magnitude) < float(tf.magnitude)):
                        label = clean_label(page.table[row][0])
                        out.append(Question("share", f"what portion of {t_label} in {year} was {label}?",
                                            f"divide({_value(f)}, {_value(tf)})", 0.0, page, row, (f, tf)))
    return [q for q in (_with_answer(q) for q in out) if q is not None]


def _with_answer(q: Question) -> Question | None:
    """Execute the gold program; also check every number is on the page as shown."""
    for f in q.facts:
        if _value(f).lstrip("-") not in q.page.table[f.row][f.cell]:
            return None
    try:
        answer = execute_program(q.program, q.page.table)
    except ProgramError:
        return None
    if isinstance(answer, str) or not math.isfinite(answer):
        return None
    return Question(q.template, q.question, q.program, answer, q.page, q.row, q.facts)


def filing_questions(header: FilingHeader, pages: list[Page], per_filing: int, seed: int = 0) -> list[Question]:
    """Sample up to ``per_filing`` questions, spread over templates, rows and pages."""
    candidates = [q for p in pages for q in page_questions(p)]
    rng = random.Random(f"{seed}:{header.accession}")
    by_template: dict[str, list[Question]] = defaultdict(list)
    for q in candidates:
        by_template[q.template].append(q)
    for qs in by_template.values():
        rng.shuffle(qs)
    order = list(TEMPLATES)
    rng.shuffle(order)
    picked: list[Question] = []
    used_rows: set[tuple[int, int]] = set()
    used_pages: Counter = Counter()
    allow_used_rows = False
    while len(picked) < per_filing and any(by_template.values()):
        progressed = False
        for template in order:  # round-robin over templates
            qs = by_template[template]
            if not qs or len(picked) >= per_filing:
                continue
            # rows not asked about yet first, then the least-used page
            qs.sort(key=lambda q: ((q.page.index, q.row) in used_rows, used_pages[q.page.index]))
            if (qs[0].page.index, qs[0].row) in used_rows and not allow_used_rows:
                continue
            q = qs.pop(0)
            picked.append(q)
            used_rows.add((q.page.index, q.row))
            used_pages[q.page.index] += 1
            progressed = True
        if not progressed:  # only rows already asked about remain
            allow_used_rows = True
    return picked


def finqa_record(header: FilingHeader, q: Question, n: int) -> dict:
    """A question in FinQA's JSON format (readable by ``auto_researcher.llm.finqa.load_finqa``)."""
    year = header.period.year if header.period else header.filed.year
    return {
        "id": f"{header.cik}/{year}/{header.accession}/t{q.page.index}-{n}",
        "filename": f"{header.accession}/{header.document.name}",
        "pre_text": list(q.page.pre_text),
        "post_text": list(q.page.post_text),
        "table": [list(r) for r in q.page.table],
        "qa": {"question": q.question, "program": q.program, "exe_ans": q.answer, "answer": ""},
        "meta": {
            "company": header.company, "cik": header.cik, "form": header.form,
            "filed": header.filed.isoformat(),
            "period": header.period.isoformat() if header.period else None,
            "accession": header.accession, "template": q.template,
            "concepts": sorted({f.concept for f in q.facts}),
            "periods": [[f.context.start.isoformat() if f.context.start else None,
                         f.context.end.isoformat()] for f in q.facts],
            "dims": [list(map(list, f.context.dims)) for f in q.facts],
        },
    }
