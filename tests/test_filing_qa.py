"""Tests for the filing page converter and the auto-graded question generator."""

from __future__ import annotations

import gzip
import json
import re
from datetime import date

import pytest

from auto_researcher.llm.filing_questions import (
    clean_label,
    fact_year,
    filing_questions,
    finqa_record,
    page_questions,
)
from auto_researcher.llm.filings import (
    finqa_text,
    load_filing,
    parse_filing,
    read_filing_header,
    render_cell,
)
from auto_researcher.llm.finqa import answers_match, execute_program, load_finqa


def _ctx(cid: str, start: str | None, end: str, member: str | None = None) -> str:
    seg = (f'<xbrli:segment><xbrldi:explicitMember dimension="srt:ProductOrServiceAxis">{member}'
           "</xbrldi:explicitMember></xbrli:segment>") if member else ""
    period = (f"<xbrli:startDate>{start}</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>"
              if start else f"<xbrli:instant>{end}</xbrli:instant>")
    return (f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="x">1</xbrli:identifier>'
            f"{seg}</xbrli:entity><xbrli:period>{period}</xbrli:period></xbrli:context>")


def _fact(concept: str, ctx: str, text: str, sign: str = "") -> str:
    sign_attr = f' sign="{sign}"' if sign else ""
    return (f'<ix:nonFraction unitRef="usd" contextRef="{ctx}" decimals="-6" name="us-gaap:{concept}"'
            f' scale="6" format="ixt:num-dot-decimal"{sign_attr}>{text}</ix:nonFraction>')


def _row(label: str, *cells: str) -> str:
    return f"<tr><td>{label}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


CONTEXTS = "".join([
    _ctx("fy25", "2024-09-29", "2025-09-27"), _ctx("fy24", "2023-10-01", "2024-09-28"),
    _ctx("fy23", "2022-09-25", "2023-09-30"), _ctx("q4", "2025-06-29", "2025-09-27"),
    _ctx("fy25p", "2024-09-29", "2025-09-27", "us-gaap:ProductMember"),
    _ctx("fy24p", "2023-10-01", "2024-09-28", "us-gaap:ProductMember"),
    _ctx("i25", None, "2025-09-27"), _ctx("i24", None, "2024-09-28"),
])


def _bs_row(label: str, concept: str, v25: str, v24: str) -> str:
    return _row(label, "", _fact(concept, "i25", v25), "", _fact(concept, "i24", v24))


BALANCE_SHEET = "".join([
    '<tr><td></td><td colspan="2">2025</td><td colspan="2">2024</td></tr>',
    _bs_row("Current assets", "AssetsCurrent", "300", "250"),
    _bs_row("Total assets", "Assets", "900", "800"),
    _bs_row("Accounts payable", "AP", "60", "50"),
    _bs_row("Accrued liabilities", "Accrued", "40", "30"),
    _bs_row("Total current liabilities", "LiabCurrent", "100", "80"),
    _bs_row("Long-term debt", "LTD", "200", "150"),
    _bs_row("Total liabilities", "Liabilities", "300", "230"),
    # assets and liabilities columns side by side: "deferred taxes in 2025" is ambiguous
    _row("Deferred taxes", _fact("DTA", "i25", "10"), _fact("DTL", "i25", "20"),
         _fact("DTA", "i24", "11"), _fact("DTL", "i24", "21")),
])

HTML = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<div style="display: none"><ix:header><ix:resources>{CONTEXTS}</ix:resources></ix:header></div>
<p>Apple Inc.</p>
<p>CONSOLIDATED STATEMENTS OF OPERATIONS</p>
<p>(In millions, except per-share amounts)</p>
<table>
<tr><td></td><td colspan="2">Years ended</td><td colspan="2"></td><td colspan="2"></td></tr>
<tr><td></td><td colspan="2">September 27, 2025</td><td colspan="2">September 28, 2024</td>
    <td colspan="2">September 30, 2023</td></tr>
{_row("Products", "$", _fact("Revenue", "fy25p", "300"), "$", _fact("Revenue", "fy24p", "250"), "", "")}
{_row("Net sales (1)", "$", _fact("Revenues", "fy25", "1,000"), "$", _fact("Revenues", "fy24", "800"),
      "$", _fact("Revenues", "fy23", "500"))}
{_row("Research and development", "", _fact("RD", "fy25", "120"), "", _fact("RD", "fy24", "100"),
      "", _fact("RD", "fy23", "90"))}
{_row("Common stock repurchased", "(", _fact("Buyback", "fy25", "40", "-") + ")", "(",
      _fact("Buyback", "fy24", "30", "-") + ")", "(", _fact("Buyback", "fy23", "20", "-") + ")")}
{_row("Fourth quarter sales", "", _fact("Revenues", "q4", "260"), "", "", "", "")}
{_row("Effective tax rate", "​", _fact("Rate", "fy25", "15.2") + "%", "​",
      _fact("Rate", "fy24", "16.1") + "%", "", "")}
{_row("Dividends declared", "", _fact("Div", "fy25", "50"), "", _fact("Div", "fy24", "50"), "", "")}
{_row("Total", "", _fact("Tot", "fy25", "900"), "", _fact("Tot", "fy24", "700"), "", "")}
</table>
<p>See accompanying Notes to Consolidated Financial Statements.</p>
<p>CONSOLIDATED BALANCE SHEETS</p>
<table>{BALANCE_SHEET}</table>
<table><tr><td>Signatures</td><td>Tim Cook</td></tr></table>
</body></html>""".encode()


@pytest.fixture(scope="module")
def pages():
    pages = parse_filing(HTML)
    assert len(pages) == 2  # the signature table has no facts
    return pages


@pytest.fixture(scope="module")
def page(pages):
    return pages[0]


def test_finqa_text_and_cells_follow_finqa_formatting():
    assert finqa_text("Net income was $1,234.5 million, up 5%.") == "net income was $ 1234.5 million , up 5 % ."
    assert finqa_text("Apple's 3.5 billion (approx.)") == "apple 's 3.5 billion ( approx . )"
    assert render_cell("(1,234)") == "-1234 ( 1234 )"
    assert render_cell("$(1,234)") == "$ -1234 ( 1234 )"
    assert render_cell("$1,000") == "$ 1000"
    assert render_cell("5.2%") == "5.2 %"
    assert render_cell("—") == "-"
    assert render_cell("Net Sales") == "net sales"


def test_page_table_is_rendered_like_finqa(page):
    assert page.table[1] == ("", "september 27 , 2025", "september 28 , 2024", "september 30 , 2023")
    assert page.table[3] == ("net sales ( 1 )", "$ 1000", "$ 800", "$ 500")
    assert page.table[5] == ("common stock repurchased", "-40 ( 40 )", "-30 ( 30 )", "-20 ( 20 )")
    assert page.table[7] == ("effective tax rate", "15.2 %", "16.1 %")  # zero-width spacers dropped
    assert page.n_header_rows == 2
    assert page.pre_text[-2:] == ("consolidated statements of operations",
                                  "( in millions , except per-share amounts )")
    assert page.post_text[0] == "see accompanying notes to consolidated financial statements ."
    assert not any("xbrli" in s or "2024-09-29" in s for s in page.pre_text)  # hidden header dropped


def test_facts_keep_context_column_header_and_sign(page):
    by_value = {f.magnitude: f for f in page.facts}
    net25 = by_value["1000"]
    assert net25.context.start == date(2024, 9, 29) and net25.context.annual
    assert "2025" in net25.column_header and "2024" not in net25.column_header
    assert page.table[net25.row][net25.cell] == "$ 1000"
    assert by_value["40"].negative and not net25.negative
    assert not by_value["260"].context.annual  # a quarter
    assert by_value["300"].context.dims == (("srt:ProductOrServiceAxis", "us-gaap:ProductMember"),)


def test_fact_year_needs_one_header_year_matching_the_period(page):
    years = {f.magnitude: fact_year(f) for f in page.facts}
    assert years["1000"] == 2025 and years["800"] == 2024 and years["500"] == 2023
    assert years["260"] is None


def test_questions_are_unambiguous_and_answer_from_the_page(page):
    qs = {(q.template, q.question): q for q in page_questions(page)}
    q = qs[("pct_change", "what was the percentage change in net sales from 2024 to 2025?")]
    assert q.program == "subtract(1000, 800), divide(#0, 800)" and q.answer == 0.25
    assert ("average", "what was the average research and development from 2023 to 2025?") in qs
    assert qs[("share", "what portion of net sales in 2025 was research and development?")].answer == 0.12
    # all-negative row: change / average would flip sign between readings; % change is safe
    assert not any(t in ("change", "average") and "repurchased" in text for t, text in qs)
    assert ("pct_change", "what was the percentage change in common stock repurchased from 2024 to 2025?") in qs
    assert not any("fourth quarter" in text for _, text in qs)  # quarterly fact
    # a rate: change in points is fine, "percentage change" of a percentage is ambiguous
    assert ("change", "what was the change in effective tax rate from 2024 to 2025?") in qs
    assert not any(t == "pct_change" and "tax rate" in text for t, text in qs)
    assert not any("dividends declared" in text for _, text in qs if _ != "share")  # 50 -> 50
    assert not any(text.endswith("was total?") or " in total " in text for _, text in qs)  # generic label
    for q in qs.values():  # every gold program runs on the page and reproduces its answer
        assert answers_match(execute_program(q.program, page.table), q.answer)


def test_share_questions_only_use_components_of_a_total(pages):
    sheet = pages[1]
    assert sheet.pre_text[-1] == "consolidated balance sheets"
    shares = {q.question for q in page_questions(sheet) if q.template == "share"}
    assert "what portion of total current liabilities in 2025 was accounts payable?" in shares
    assert "what portion of total liabilities in 2025 was long-term debt?" in shares
    assert "what portion of total assets in 2024 was current assets?" in shares
    # not components: a sub-total sits in between, or the row is below the total
    assert not any("of total liabilities" in s and "accounts payable" in s for s in shares)
    assert not any("of total liabilities" in s and "current assets" in s for s in shares)
    assert not any("of total assets" in s and "liabilities" in s.split(" was ")[1] for s in shares)


def test_rows_with_two_values_per_year_are_skipped(pages):
    assert not any("deferred taxes" in q.question for q in page_questions(pages[1]))


def test_clean_label_strips_footnotes_and_operators():
    assert clean_label("net sales ( 1 )") == "net sales"
    assert clean_label("other income ( a ) ( 2 ) :") == "other income"
    assert clean_label("less : treasury stock") == "treasury stock"


def test_header_index_records_and_finqa_round_trip(tmp_path):
    folder = tmp_path / "0000320193-25-000079"
    folder.mkdir()
    with gzip.open(folder / "0000320193-25-000079.hdr.sgml.gz", "wt") as fh:
        fh.write("<ACCESSION-NUMBER>0000320193-25-000079\n<TYPE>10-K\n<PERIOD>20250927\n"
                 "<FILING-DATE>20251031\n<CONFORMED-NAME>Apple Inc.\n<CIK>0000320193\n")
    with gzip.open(folder / "0000320193-25-000079-index.htm.gz", "wb") as fh:
        fh.write(b"<html></html>")
    with gzip.open(folder / "aapl-20250927.htm.gz", "wb") as fh:
        fh.write(HTML)
    header = read_filing_header(folder)
    assert (header.form, header.company, header.filed, header.period) == (
        "10-K", "Apple Inc.", date(2025, 10, 31), date(2025, 9, 27))
    assert header.document.name == "aapl-20250927.htm.gz"

    questions = filing_questions(header, load_filing(header), per_filing=4, seed=0)
    assert len(questions) == 4
    assert len({q.template for q in questions}) >= 3  # spread over templates
    path = tmp_path / "test.json"
    path.write_text(json.dumps([finqa_record(header, q, n) for n, q in enumerate(questions)]))
    loaded = load_finqa(path)
    assert [e.question for e in loaded] == [q.question for q in questions]
    assert all(answers_match(execute_program(e.program, e.table), e.answer) for e in loaded)
    assert all(re.fullmatch(r"0000320193/2025/0000320193-25-000079/t\d+-\d+", e.id) for e in loaded)
