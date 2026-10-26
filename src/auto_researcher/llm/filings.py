"""
FinQA-style pages from 10-K / 10-Q filings (inline XBRL HTML).

A filing's primary document is split into text and tables. Every table that holds
inline-XBRL facts becomes a FinQA-style page: the text before it, the table as rows of
cells, and the text after it. Pages are formatted the way FinQA formats its own (lower
case, punctuation spaced, no thousands separators, negatives as ``-123 ( 123 )``),
because that is what the fine-tuned model was trained on.

Each tagged number keeps its XBRL context (the period it covers, any dimension members)
and the text of the header cells above its column, located on the table's column grid
(``colspan`` expanded). ``auto_researcher.llm.filing_questions`` turns these into
questions with gold programs.

The SEC documents are read as they are stored by the sentiq project
(``<raw_dir>/<accession>/``: the gzipped primary document, ``-index.htm.gz`` and
``.hdr.sgml.gz``); nothing is written there.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import lxml.html
from lxml import etree

PRE_CHARS = 1500   # FinQA pages: median 1458 characters of text before the table
POST_CHARS = 1700  # and 1751 after it
BLOCK_TAGS = {"p", "div", "br", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
PREFIX_CELLS = {"$", "(", "$(", "($", "€", "£"}
SUFFIX_CELLS = {")", "%", ")%", "%)", "pts", "bps"}
DASHES = {"—", "–", "-", "−", "— ", "-0-"}
_NUMBER = re.compile(
    r"^(?P<cur>[$€£])?\s*(?P<open>\()?\s*(?:[$€£])?\s*(?P<sign>[-−–])?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<close>\))?\s*(?P<pct>%)?\s*\)?$"
)
_YEAR = re.compile(r"^(19|20)\d{2}$")
# Zero-width characters: many filers fill spacer cells with U+200B, which would
# otherwise make empty cells look non-empty.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


@dataclass(frozen=True)
class FilingHeader:
    """What the SEC header says about a filing, plus where its primary document is."""

    accession: str
    form: str
    company: str
    cik: str
    filed: date
    period: date | None
    document: Path


@dataclass(frozen=True)
class Context:
    """An XBRL context: the period a fact covers and its dimension members."""

    start: date | None  # None for an instant (balance-sheet date)
    end: date
    dims: tuple[tuple[str, str], ...]

    @property
    def annual(self) -> bool:
        """An instant, or a duration of about a year (not a quarter or year-to-date)."""
        return self.start is None or 330 <= (self.end - self.start).days <= 380


@dataclass(frozen=True)
class Fact:
    """A tagged number on a page, located in the rendered table."""

    concept: str
    context: Context
    unit: str
    magnitude: str      # as displayed, without thousands separators ("391035", "3.88")
    negative: bool      # as displayed (parentheses or a minus sign in the cell)
    row: int            # index into Page.table
    cell: int           # index into Page.table[row]
    column_header: str  # text of the header cells above this fact's column


@dataclass(frozen=True)
class Page:
    """One table of a filing with its surrounding text, in FinQA's format."""

    index: int  # position of the table in the document
    pre_text: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]
    post_text: tuple[str, ...]
    n_header_rows: int
    facts: tuple[Fact, ...]

    @property
    def chars(self) -> int:
        cells = sum(len(c) + 3 for row in self.table for c in row)
        return cells + sum(len(s) + 1 for s in self.pre_text + self.post_text)


# --------------------------------------------------------------------------- #
# Filing headers
# --------------------------------------------------------------------------- #


def _sgml_date(value: str | None) -> date | None:
    if not value or not re.fullmatch(r"\d{8}", value):
        return None
    return date(int(value[:4]), int(value[4:6]), int(value[6:]))


def read_filing_header(folder: Path) -> FilingHeader | None:
    """Parse ``<accession>.hdr.sgml.gz``; None if the folder is not a complete filing."""
    headers = list(folder.glob("*.hdr.sgml.gz"))
    docs = [p for p in folder.glob("*.htm*.gz") if not p.name.endswith("-index.htm.gz")]
    if not headers or not docs:
        return None
    text = gzip.open(headers[0], "rt", encoding="utf-8", errors="replace").read()
    fields: dict[str, str] = {}
    for tag, value in re.findall(r"^<([A-Z-]+)>(.+)$", text, re.MULTILINE):
        fields.setdefault(tag, value.strip())  # first occurrence: the filer, not a later party
    filed = _sgml_date(fields.get("FILING-DATE"))
    if filed is None:
        return None
    return FilingHeader(
        accession=fields.get("ACCESSION-NUMBER", folder.name),
        form=fields.get("TYPE", fields.get("FORM-TYPE", "")),
        company=fields.get("CONFORMED-NAME", ""),
        cik=fields.get("CIK", ""),
        filed=filed,
        period=_sgml_date(fields.get("PERIOD")),
        document=max(docs, key=lambda p: p.stat().st_size),
    )


# --------------------------------------------------------------------------- #
# Text formatting
# --------------------------------------------------------------------------- #


def finqa_text(text: str) -> str:
    """Format text the way FinQA's pages are formatted."""
    text = (text.translate(_INVISIBLE).replace("\xa0", " ").replace("’", "'").replace("‘", "'").replace("“", '"')
            .replace("”", '"').replace("—", "-").replace("–", "-").replace("−", "-"))
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # thousands separators
    text = re.sub(r"([,;:()$%?!\"])", r" \1 ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " . ", text)  # periods, but not decimal points
    text = re.sub(r"'s\b", " 's", text)
    return re.sub(r"\s+", " ", text).strip()


def render_cell(text: str) -> str:
    """A table cell in FinQA's format: ``$ 18086``, ``$ -23158 ( 23158 )``, ``5.2 %``."""
    text = re.sub(r"\s+", " ", text.translate(_INVISIBLE).replace("\xa0", " ")).strip()
    if text in DASHES:
        return "-"
    m = _NUMBER.match(text)
    if not m:
        return finqa_text(text)
    num = m["num"].replace(",", "")
    if m["open"]:
        value = f"-{num} ( {num} )"
    elif m["sign"]:
        value = f"-{num}"
    else:
        value = num
    return ("$ " if m["cur"] else "") + value + (" %" if m["pct"] else "")


def _is_number(text: str) -> bool:
    return bool(_NUMBER.match(text.strip())) or text.strip() in DASHES


def _sentences(block: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z(\"'])", block)
    out = []
    for part in parts:
        s = finqa_text(part)
        if len(s) > 3 and not re.fullmatch(r"[\d\s.|-]+", s):  # page numbers, rules
            out.append(s)
    return out


def _take(sentences: list[str], limit: int, from_end: bool) -> tuple[str, ...]:
    picked, total = [], 0
    for s in (reversed(sentences) if from_end else sentences):
        if total + len(s) > limit and picked:
            break
        picked.append(s[:limit])
        total += len(s) + 1
    return tuple(reversed(picked) if from_end else picked)


# --------------------------------------------------------------------------- #
# Document parsing
# --------------------------------------------------------------------------- #


def _local(tag: str) -> str:
    return tag.rsplit(":", 1)[-1]


def _parse_date(text: str | None) -> date | None:
    try:
        return date.fromisoformat((text or "").strip()[:10])
    except ValueError:
        return None


def _contexts(doc: lxml.html.HtmlElement) -> dict[str, Context]:
    contexts: dict[str, Context] = {}
    for el in doc.iter("xbrli:context"):
        start = end = None
        dims = []
        for child in el.iter():
            if not isinstance(child.tag, str):
                continue
            name = _local(child.tag)
            if name == "startdate":
                start = _parse_date(child.text)
            elif name in ("enddate", "instant"):
                end = _parse_date(child.text)
            elif name in ("explicitmember", "typedmember"):
                dims.append((child.get("dimension", ""), (child.text or "").strip()))
        if end is not None and el.get("id"):
            contexts[el.get("id")] = Context(start, end, tuple(sorted(dims)))
    return contexts


def _text_segments(doc: lxml.html.HtmlElement, tables: list) -> list[list[str]]:
    """Sentences between consecutive top-level tables (len(tables) + 1 segments)."""
    table_ids = {id(t) for t in tables}
    segments: list[list[str]] = [[]]
    buf: list[str] = []

    def flush() -> None:
        for block in "".join(buf).split("\n"):
            block = block.strip()
            if block:
                segments[-1].extend(_sentences(block))
        buf.clear()

    walker = etree.iterwalk(doc, events=("start", "end"))
    for event, el in walker:
        if not isinstance(el.tag, str):  # comments, processing instructions
            if event == "end" and el.tail:
                buf.append(el.tail)
            continue
        if event == "start":
            if id(el) in table_ids:
                flush()
                segments.append([])
                walker.skip_subtree()
                continue
            if el.tag in BLOCK_TAGS:
                buf.append("\n")
            if el.text:
                buf.append(el.text)
        else:
            if el.tag in BLOCK_TAGS:
                buf.append("\n")
            if el.tail:
                buf.append(el.tail)
    flush()
    return segments


def _cell_text(td) -> str:
    text = " ".join(td.itertext()).translate(_INVISIBLE).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_table(table, contexts: dict[str, Context]):
    """Rendered rows, number of header rows and facts of one HTML table."""
    raw_rows = []
    for tr in table.iter("tr"):
        if next(tr.iterancestors("table")) is not table:  # a nested table's row
            continue
        col, cells = 0, []
        for td in tr:
            if not isinstance(td.tag, str) or td.tag not in ("td", "th"):
                continue
            try:
                span = max(1, int(td.get("colspan", "1")))
            except ValueError:
                span = 1
            cells.append((_cell_text(td), col, span, list(td.iter("ix:nonfraction"))))
            col += span
        raw_rows.append(cells)

    first_fact_row = next((i for i, cells in enumerate(raw_rows) if any(c[3] for c in cells)), None)
    if first_fact_row is None:
        return (), 0, ()
    header_cells = [c for cells in raw_rows[:first_fact_row] for c in cells
                    if c[0] and (not _is_number(c[0]) or _YEAR.match(c[0]))]

    def column_header(col: int) -> str:
        return " ".join(t for t, start, span, _ in header_cells if start <= col < start + span)

    rendered: list[tuple[str, ...]] = []
    facts: list[Fact] = []
    n_header = 0
    for r, cells in enumerate(raw_rows):
        merged: list[list] = []  # [text, grid column, ix elements]
        prefix = ""
        for text, col, _, ix in cells:
            if not text:
                continue
            if text in PREFIX_CELLS:
                prefix += text
                continue
            if text in SUFFIX_CELLS and merged:
                merged[-1][0] += text
                continue
            merged.append([prefix + text, col, ix])
            prefix = ""
        if not merged:
            continue
        # A row whose label column is blank (column headers, unlabelled values) gets an
        # empty first cell, as in FinQA, so its cells line up with the labelled rows.
        offset = 1 if merged[0][1] > 0 or _is_number(merged[0][0]) else 0
        row_cells = ([""] if offset else []) + [render_cell(m[0]) for m in merged]
        rendered.append(tuple(row_cells))
        if r < first_fact_row:
            n_header += 1
        for k, (_, col, ix) in enumerate(merged):
            if len(ix) != 1:  # untagged, or several facts in one cell: ambiguous
                continue
            fact = _fact(ix[0], contexts, len(rendered) - 1, k + offset, row_cells[k + offset],
                         column_header(col))
            if fact is not None:
                facts.append(fact)
    return tuple(rendered), n_header, tuple(facts)


def _fact(el, contexts, row, cell, rendered_cell, header) -> Fact | None:
    context = contexts.get(el.get("contextref", ""))
    if context is None:
        return None
    shown = (el.text_content() or "").strip()
    if "fixed-zero" in el.get("format", "") or shown in DASHES:
        magnitude = "0"
    else:
        shown = shown.replace(",", "").replace(" ", "")
        if not re.fullmatch(r"\d+(?:\.\d+)?", shown):
            return None
        magnitude = shown
    negative = rendered_cell.startswith("-") or rendered_cell.startswith("$ -")
    return Fact(
        concept=el.get("name", ""),
        context=context,
        unit=el.get("unitref", ""),
        magnitude=magnitude,
        negative=negative,
        row=row,
        cell=cell,
        column_header=finqa_text(header),
    )


def parse_filing(html: bytes) -> list[Page]:
    """FinQA-style pages for every table of a filing that has inline-XBRL facts."""
    doc = lxml.html.fromstring(html, parser=lxml.html.HTMLParser(huge_tree=True))
    contexts = _contexts(doc)
    for el in list(doc.iter("ix:header")):  # hidden: contexts, units, hidden facts
        el.drop_tree()
    for el in [e for e in doc.iter() if isinstance(e.tag, str)
               and "display:none" in (e.get("style") or "").replace(" ", "").lower()]:
        el.drop_tree()
    tables = [t for t in doc.iter("table") if next(t.iterancestors("table"), None) is None]
    segments = _text_segments(doc, tables)
    pages = []
    for i, table in enumerate(tables):
        rows, n_header, facts = _parse_table(table, contexts)
        if not facts:
            continue
        pages.append(Page(
            index=i,
            pre_text=_take(segments[i], PRE_CHARS, from_end=True),
            table=rows,
            post_text=_take(segments[i + 1], POST_CHARS, from_end=False),
            n_header_rows=n_header,
            facts=facts,
        ))
    return pages


def load_filing(header: FilingHeader) -> list[Page]:
    with gzip.open(header.document, "rb") as fh:
        return parse_filing(fh.read())
