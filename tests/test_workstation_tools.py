"""Tests for the workstation's real tools on small synthetic copies of their data sources
(sentiq's lake, TickerLake, market-compass) and for the Claude client with a fake API."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from auto_researcher.workstation.claude import PREAMBLE, ClaudeLLM, LLMError, render_inputs, strict_schema
from auto_researcher.workstation.schemas import DraftMemo, Plan, ResearchNotes
from auto_researcher.workstation.tools.filings import LakeFilings, passages
from auto_researcher.workstation.tools.hub import HubContext
from auto_researcher.workstation.tools.market import TickerLakeMetrics


# -- Claude client ------------------------------------------------------------------------

def _no_refs_and_closed(node):
    if isinstance(node, dict):
        assert "$ref" not in node and "default" not in node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
        for v in node.values():
            _no_refs_and_closed(v)
    elif isinstance(node, list):
        for v in node:
            _no_refs_and_closed(v)


@pytest.mark.parametrize("model", [Plan, ResearchNotes, DraftMemo])
def test_strict_schema_inlines_references_and_closes_objects(model):
    s = strict_schema(model)
    _no_refs_and_closed(s)
    assert "$defs" not in s
    assert set(s["properties"]) == set(model.model_fields)  # field names (even "title") survive
    if model is Plan:
        assert set(s["properties"]["sections"]["items"]["properties"]) == {"id", "title", "questions"}


def test_documents_are_tagged_as_data():
    text = render_inputs({"request": {"ticker": "ACME"}, "documents": [
        {"doc_id": "d1", "form": "10-K", "filed": "2026-02-20", "section": "Item 1A", "text": "Risk text."}]})
    assert '<document doc_id="d1" form="10-K" filed="2026-02-20" section="Item 1A">\nRisk text.\n</document>' in text
    assert text.index("Inputs (JSON)") < text.index("<document")


class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class _FakeClient:
    def __init__(self, replies):
        self.replies, self.params = list(replies), []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **params):
        self.params.append(params)
        text, stop = self.replies.pop(0)
        usage = SimpleNamespace(input_tokens=1000, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                                output_tokens=500)
        return _FakeStream(SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
                                           usage=usage, model=params["model"]))


def test_claude_client_routes_roles_validates_and_costs():
    plan = {"sections": [{"id": "s1", "title": "Leverage", "questions": ["Debt?"]}], "metrics": ["net_debt"],
            "context": []}
    client = _FakeClient([(json.dumps(plan), "end_turn"), ("{}", "max_tokens")])
    llm = ClaudeLLM(client)
    got = llm.complete_json(role="orchestrator_plan", task="Plan it.", inputs={"x": 1}, schema=Plan)
    assert got.sections[0].title == "Leverage"
    p = client.params[0]
    assert p["model"] == "claude-opus-5-5" and p["thinking"] == {"type": "adaptive"}
    assert p["system"][0]["text"].startswith(PREAMBLE) and p["system"][0]["text"].endswith("Plan it.")
    assert p["output_config"]["format"]["schema"] == strict_schema(Plan)
    assert llm.calls[0]["cost"] == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    with pytest.raises(LLMError):
        llm.complete_json(role="filing_analyst", task="t", inputs={}, schema=ResearchNotes)
    assert client.params[1]["model"] == "claude-sonnet-5-5"
    assert llm.training_cutoff == date(2026, 7, 1)


# -- filings ------------------------------------------------------------------------------

def test_passages_cut_at_lines_and_keep_all_text():
    text = "\n".join(f"Line {i} " + "word " * 40 for i in range(60))
    ps = passages(text, size=1000)
    assert all(len(p) <= 1000 for p in ps) and len(ps) > 10
    assert " ".join(ps) == " ".join(text.split())


def _lake(tmp_path):
    rows = []

    def add(ym, accession, form, filed, body, sections, ticker="ACME"):
        meta = {"accession": accession, "form": form, "filing_date": filed, "cik": 1, "company": "ACME CORP",
                "sections": sections}
        rows.append({"ym": ym, "item_id": accession, "tickers": [ticker], "meta": json.dumps(meta), "body_text": body,
                     "title": f"ACME {form}"})

    risk_old = "Item 1A Risk Factors\n" + "Our old risk is weather. " * 100
    add("2025-02", "old-10k", "10-K", "2025-02-20", risk_old, {"1A": [0, len(risk_old)]})
    mda = "Item 7 MD&A\n" + "Liquidity and capital resources are ample. " * 80
    risk = "Item 1A\n" + "Our largest customer accounted for 18% of net sales. " * 60
    add("2026-02", "new-10k", "10-K", "2026-02-20", risk + mda, {"1A": [0, len(risk)], "7": [len(risk), len(risk + mda)]})
    q = "Item 2 Discussion\n" + "Net sales rose in the quarter on volume. " * 60
    add("2026-05", "q1", "10-Q", "2026-05-01", q, {"I-2": [0, len(q)]})
    add("2026-05", "q-late", "10-Q", "2026-05-30", q, {"I-2": [0, len(q)]})
    add("2026-05", "other", "10-K", "2026-05-02", risk, {"1A": [0, len(risk)]}, ticker="OTHER")
    for ym in sorted({r["ym"] for r in rows}):
        part = tmp_path / f"ym={ym}"
        part.mkdir()
        pq.write_table(pa.Table.from_pylist([{k: v for k, v in r.items() if k != "ym"} for r in rows if r["ym"] == ym]),
                       part / "part-0.parquet")
    return tmp_path


def test_lake_filings_reads_the_latest_10k_and_later_10qs_only(tmp_path):
    tool = LakeFilings(_lake(tmp_path))
    docs = tool.documents("ACME", date(2026, 5, 15))
    assert {d.doc_id.split("#")[0] for d in docs} == {"10-K:new-10k", "10-Q:q1"}
    assert {d.section for d in docs} == {"Item 1A Risk Factors", "Item 7 MD&A", "Part I Item 2 MD&A"}
    hits = tool.search("ACME", "largest customer percent of net sales", date(2026, 5, 15), k=2)
    assert hits and all("largest customer" in d.text for d in hits)
    assert tool.documents("ACME", date(2024, 12, 31)) == []


# -- TickerLake metrics ---------------------------------------------------------------------

def _facts():
    rows = []

    def fact(concept, value, end, filed, start=None, taxonomy="us-gaap", unit="USD", form="10-K"):
        rows.append({"cik": "1", "symbol": "ACME", "taxonomy": taxonomy, "concept": concept, "unit": unit,
                     "value": value, "start_date": start, "end_date": end, "fiscal_year": 0.0, "fiscal_period": "FY",
                     "form_type": form, "filed_date": filed, "accession_number": f"a-{filed}", "frame": None,
                     "source": "sec_edgar"})

    for year, filed, scale in ((2024, "2025-02-20", 1.0), (2025, "2026-02-20", 1.1)):
        s, e = f"{year}-01-01", f"{year}-12-31"
        fact("Revenues", 1000e6 * scale, e, filed, s)
        fact("NetIncomeLoss", 100e6 * scale, e, filed, s)
        fact("NetCashProvidedByUsedInOperatingActivities", 80e6 * scale, e, filed, s)
        fact("OperatingIncomeLoss", 150e6 * scale, e, filed, s)
        fact("DepreciationDepletionAndAmortization", 50e6 * scale, e, filed, s)
        fact("Assets", 2000e6 * scale, e, filed)
        fact("LongTermDebtNoncurrent", 400e6 * scale, e, filed)
        fact("DebtCurrent", 100e6 * scale, e, filed)
        fact("CashAndCashEquivalentsAtCarryingValue", 100e6 * scale, e, filed)
        fact("EntityCommonStockSharesOutstanding", 10e6, e, filed, taxonomy="dei", unit="shares")
    return pd.DataFrame(rows)


def _tickerlake(tmp_path):
    (tmp_path / "filings_facts").mkdir()
    _facts().to_parquet(tmp_path / "filings_facts" / "sym_ACME.parquet")
    days = pd.bdate_range("2025-01-01", "2026-03-31")
    rng = np.random.default_rng(1)
    m = rng.normal(0.0004, 0.01, len(days))
    spy = 100 * np.cumprod(1 + m)
    acme = 50 * np.cumprod(1 + 2 * m)
    frame = pd.concat([pd.DataFrame({"symbol": s, "date": days.strftime("%Y-%m-%d"), "close": p, "adj_close": p})
                       for s, p in (("SPY", spy), ("ACME", acme))])
    frame["y"], frame["m"] = pd.to_datetime(frame["date"]).dt.year, pd.to_datetime(frame["date"]).dt.month
    for (y, mo), g in frame.groupby(["y", "m"]):
        d = tmp_path / "ohlcv" / f"year={y}" / f"month={mo:02d}"
        d.mkdir(parents=True)
        g.drop(columns=["y", "m"]).to_parquet(d / "part.parquet", index=False)
    return tmp_path


def test_metrics_use_only_filed_data_and_compute_from_code(tmp_path):
    tool = TickerLakeMetrics(_tickerlake(tmp_path))
    before = tool.compute("ACME", date(2026, 1, 15))   # FY2025 10-K not yet filed; FY2024 is 380 days old
    after = tool.compute("ACME", date(2026, 3, 31))
    assert before["revenue_ttm"][0] == 1000e6 and before["revenue_ttm"][2] == date(2025, 2, 20)
    assert after["revenue_ttm"][0] == pytest.approx(1100e6) and after["revenue_ttm"][1] == date(2025, 12, 31)
    assert after["revenue_growth"][0] == pytest.approx(0.10)
    assert after["net_debt_to_ebitda"][0] == pytest.approx((550e6 - 110e6) / 220e6)
    assert after["cfo_to_net_income"][0] == pytest.approx(0.8)
    assert after["beta_252d"][0] == pytest.approx(2.0, abs=0.02)
    assert after["market_cap"][0] == pytest.approx(after["price"][0] * 10e6)
    # Short-term borrowings count on top of current long-term debt when DebtCurrent is not tagged.
    facts = _facts()
    facts = facts[facts["concept"] != "DebtCurrent"]
    extra = facts[facts["concept"] == "LongTermDebtNoncurrent"].assign(concept="LongTermDebtCurrent", value=30e6)
    stb = extra.assign(concept="ShortTermBorrowings", value=70e6)
    pd.concat([facts, extra, stb]).to_parquet(tmp_path / "filings_facts" / "sym_ACME.parquet")
    assert TickerLakeMetrics(tmp_path).compute("ACME", date(2026, 3, 31))["total_debt"][0] == pytest.approx(440e6 + 100e6)
    got = tool.metrics("ACME", ["revenue_growth", "gross_margin", "not_a_metric"], date(2026, 3, 31))
    assert [(m.name, m.value is None) for m in got] == [("revenue_growth", False), ("gross_margin", True),
                                                        ("gross_margin_1y_ago", True)]
    assert got[0].available_date == date(2026, 2, 20) and got[0].max_age_days == 200
    assert got[0].definition.startswith("TTM revenue vs the TTM a year earlier")


def test_margins_and_multiples_come_with_their_values_a_year_earlier(tmp_path):
    from auto_researcher.workstation.gates import check_inputs
    tool = TickerLakeMetrics(_tickerlake(tmp_path))
    as_of = date(2026, 3, 31)
    got = {m.name: m for m in tool.metrics("ACME", ["operating_margin", "pe_ttm", "revenue_ttm"], as_of)}
    assert set(got) == {"operating_margin", "pe_ttm", "revenue_ttm", "operating_margin_1y_ago", "pe_ttm_1y_ago"}
    then = tool.compute("ACME", date(2025, 3, 31))
    assert got["pe_ttm_1y_ago"].value == pytest.approx(then["pe_ttm"][0])
    assert got["operating_margin_1y_ago"].value == pytest.approx(0.15) and got["operating_margin"].value == pytest.approx(0.15)
    assert got["operating_margin_1y_ago"].available_date == date(2025, 2, 20)
    assert "a year before the as-of date" in got["pe_ttm_1y_ago"].definition
    assert "lowers P/E" in got["pe_ttm"].definition
    assert got["pe_ttm"].period.startswith("price 2026-03-31, fundamentals to 2025-12-31")
    assert sorted(m.id for m in got.values()) == ["m1", "m2", "m3", "m4", "m5"]
    assert check_inputs(list(got.values()), [], as_of) == []   # a year old by design, not stale


# -- market-compass context ------------------------------------------------------------------

def _hub(tmp_path):
    (tmp_path / "macro").mkdir()
    pd.DataFrame({"p_recession": [0.1, 0.2, 0.3]},
                 index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"])).to_parquet(
        tmp_path / "macro" / "walk_forward.parquet")
    (tmp_path / "macro" / "summary.json").write_text(json.dumps({
        "run_at": "2026-04-20T18:00:00", "nowcast": {"recession_probability": 0.35,
                                                     "signal_as_of": {"ensemble": "2026-03-31"}}}))
    (tmp_path / "style").mkdir()
    pd.DataFrame({"date": pd.to_datetime(["2026-02-27", "2026-03-31"]), "symbol": "ACME", "value": [0.5, 0.7],
                  "growth": [-0.1, -0.2]}).to_parquet(tmp_path / "style" / "stock_history.parquet")
    (tmp_path / "style" / "summary.json").write_text(json.dumps({"backtest": {
        "start": "2012-12-31", "end": "2026-03-31",
        "stock": {"value": {"1m": {"mean": 0.004, "t": 0.24, "n": 166}}, "growth": {"1m": {"mean": 0.004, "t": 0.39,
                                                                                           "n": 166}}}}}))
    return tmp_path


def test_hub_context_uses_history_with_lags_and_refuses_old_snapshots(tmp_path):
    hub = HubContext(_hub(tmp_path))
    today = {c.name: c for c in hub.context("ACME", ["recession_probability", "value_score"], date(2026, 4, 21))}
    assert today["recession_probability"].value == 0.35 and today["recession_probability"].as_of == date(2026, 3, 31)
    assert "AUC 0.936" in today["recession_probability"].track_record
    assert today["value_score"].value == 0.7 and "t 0.24" in today["value_score"].track_record
    assert "not a forecast for the memo's horizon" in today["recession_probability"].definition
    assert today["value_score"].definition.startswith("TickerLake style: value score")
    # An earlier date gets the walk-forward history, lagged for publication, not today's nowcast.
    past = {c.name: c for c in hub.context("ACME", ["recession_probability", "value_score"], date(2026, 3, 20))}
    assert past["recession_probability"].value == 0.1 and past["recession_probability"].as_of == date(2026, 1, 31)
    assert past["value_score"].value == 0.5
    assert hub.context("ACME", ["p_up_21d", "eps_revisions_30d"], date(2026, 4, 21)) == []  # no snapshots here
