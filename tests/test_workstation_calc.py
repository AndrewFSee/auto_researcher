"""The calculator: derived numbers are computed by code from cited ids and quoted numbers,
and a memo may state a derived figure only through a calculation."""

from __future__ import annotations

from datetime import date

import pytest
from test_workstation import AS_OF, FILING, GOOD, MACRO, METRICS, draft, para, run

from auto_researcher.workstation.calc import CalcError, evaluate, run_calculations
from auto_researcher.workstation.schemas import Calculation, Claim

CLAIMS = {"f6": Claim(id="f6", text="Net income rose to about $21bn.", evidence=[
    {"doc_id": "d", "quote": "Net income 21,155 14,987 41"},
    {"doc_id": "d", "quote": "Included a $4.6 billion net gain related to Visa shares"}])}
M = {m.id: m for m in METRICS}
X = {MACRO.id: MACRO}


def test_formulas_use_ids_quoted_numbers_and_constants():
    assert evaluate("m3 * 2", M, X, CLAIMS) == (pytest.approx(4.8), ["m3"])
    v, used = evaluate('f6["21,155"] * 1e6 - f6["$4.6 billion"]', M, X, CLAIMS)
    assert v == pytest.approx(21_155e6 - 4.6e9) and used == ['f6[21,155]', 'f6[$4.6 billion]']
    assert evaluate("max(m1, x1) / 4", M, X, CLAIMS)[0] == pytest.approx(0.08 / 4)
    # Bare digits match the quote's digits and come back as written; the formula scales them.
    assert evaluate('f6["4.6"] * 1e9', M, X, CLAIMS)[0] == pytest.approx(4.6e9)
    with pytest.raises(CalcError, match="not in claim"):
        evaluate('f6["$4.6 million"]', M, X, CLAIMS)   # a unit, if written, must match the quote's


@pytest.mark.parametrize("expr, why", [
    ("m99 + 1", "unknown id"), ('f6["22,000"]', "not in claim f6"), ('f7["21,155"]', "unknown claim"),
    ("m1 / (m2 - m2)", "division by zero"), ("__import__('os')", "not allowed"), ("m1.real", "not allowed"),
    ("m1 ** 2", "not allowed"), ("m1 +", "cannot parse"),
])
def test_anything_else_is_an_error(expr, why):
    with pytest.raises(CalcError, match=why):
        evaluate(expr, M, X, CLAIMS)


def test_results_are_metrics_dated_by_their_inputs():
    calcs = [Calculation(id="c1", description="leverage doubled", expression="m3 * 2", unit="x"),
             Calculation(id="c2", description="bad", expression="m99", unit="x")]
    out, errors = run_calculations(calcs, METRICS, [MACRO], CLAIMS, AS_OF)
    assert [(m.id, m.value, m.as_of) for m in out] == [("c1", pytest.approx(4.8), date(2026, 9, 30))]
    assert out[0].source == "calculation: m3 * 2" and set(errors) == {"c2"}


def _calc_draft(paragraphs, calcs):
    return {**draft(paragraphs), "calculations": calcs}


def test_a_derived_figure_needs_a_calculation():
    derived = para("Earnings quality", "Cash conversion was 84%, a gap of $98 million to net income.", ["f1"], ["m2"])
    packet, *_ = run(drafts=[draft([derived, *GOOD[1:]])], max_redrafts=0)
    assert [f.code for f in packet.flags] == ["number_mismatch"]   # $98 million appears nowhere

    calc = {"id": "c1", "description": "net income minus operating cash flow",
            "expression": 'f1["$610 million"] - f1["$512 million"]', "unit": "usd"}
    cited = {**derived, "metric_ids": ["m2", "c1"]}
    packet, state, _ = run(drafts=[_calc_draft([cited, *GOOD[1:]], [calc])], max_redrafts=0)
    assert [f.code for f in packet.flags] == []
    assert packet.paragraphs[0].metrics[-1].source.startswith("calculation:")
    assert state.calculations[0].value == pytest.approx(98e6)

    # The calculation's verified inputs may be stated too, without citing their claim.
    inputs = {**cited, "claim_ids": [], "text": "Net income was $610 million, $98 million above cash flow."}
    packet, *_ = run(drafts=[_calc_draft([inputs, *GOOD[1:]], [calc])], max_redrafts=0)
    assert [f.code for f in packet.flags] == []


def test_a_failed_calculation_blocks_and_is_sent_back():
    bad = {"id": "c1", "description": "x", "expression": 'f1["$700 million"] - 1', "unit": "usd"}
    packet, state, llm = run(drafts=[_calc_draft(GOOD, [bad]), draft()])
    assert [f.code for f in packet.flags] == [] and len(state.drafts) == 2
    second = [c for c in llm.calls if c["role"] == "orchestrator_draft"][1]
    assert any("calculation_error" in f for f in second["inputs"]["feedback"])
    packet, *_ = run(drafts=[_calc_draft(GOOD, [bad])], max_redrafts=0)
    assert [f.code for f in packet.flags] == ["calculation_error"] and packet.review_required


def test_numbers_in_a_claims_own_wording_do_not_count():
    worded = {**FILING, "claims": [{**FILING["claims"][0], "text": "Operating cash flow trailed by $98 million."},
                                   FILING["claims"][1]]}
    p = para("Earnings quality", "Operating cash flow trailed net income by $98 million.", ["f1"])
    packet, *_ = run(filing=worded, drafts=[draft([p, *GOOD[1:]])], max_redrafts=0)
    assert [f.code for f in packet.flags] == ["number_mismatch"]
