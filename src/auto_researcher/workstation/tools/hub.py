"""Market backdrop and per-stock companion outputs, read from market-compass's published
data (``market-compass/data/<module>/``). Read-only: the hub's nightly job runs each
companion project's own publisher; this never runs or imports those projects.

Two kinds of input:

* **Histories** (macro walk-forward, credit stress and spreads, vol history, style
  scores): the value on or before the as-of date.
* **Snapshots** (augury forecasts, vol surface percentiles, buybacks, estimate
  revisions): today's values only. They are returned only for an as-of date within
  ``SNAPSHOT_DAYS`` of the snapshot, never for an earlier date.

Track records come from each project's own published scores where it publishes them
(augury, stochastic-vol-surface, the style backtest), else from its README, quoted with
the source (``README_RECORDS``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..schemas import ContextItem

HUB = Path(r"C:\Users\andre\projects\market-compass\data")
SNAPSHOT_DAYS = 3
MACRO_LAG_DAYS = 45   # a month's macro panel row is built after that month's data is released

README_RECORDS = {
    "macro": ("macro-regime-nowcaster README, 717-month walk-forward 1967-2026: real-time recession detection "
              "AUC 0.936, Brier 0.0625; 12 months ahead AUC 0.526 (chance); no skill for forward equity returns "
              "or drawdowns"),
    "credit_stress": ("credit-spread-analyzer README: hidden Markov model refitted each January on past data; "
                      "switched on in Aug 1998, early 2000 and Dec 2007, late for COVID (23 Mar 2020); measured as a "
                      "risk overlay (max drawdown -18% vs -55% since 1993), not as a return signal"),
    "credit_forecast": ("credit-spread-analyzer README: the 5-day Baa spread forecast ranked outcomes correctly in "
                        "24 of 27 years, including all 8 holdout years 2019-2026; out-of-sample R² 4-11%"),
}

CATALOGUE: dict[str, str] = {
    "recession_probability": "macro-regime-nowcaster: probability the US economy is in recession now",
    "credit_stress_probability": "credit-spread-analyzer: probability credit markets are in a stress regime",
    "baa_spread_bps": "Moody's Baa corporate spread over Treasuries, bps",
    "baa_spread_percentile": "Baa spread percentile since 1990",
    "credit_spread_forecast_5d_bps": "credit-spread-analyzer: forecast change in the Baa spread over 5 days, bps",
    "implied_vol_30d": "stochastic-vol-surface: 30-day at-the-money implied volatility of the stock (41 names)",
    "implied_vol_30d_percentile_1y": "30-day implied vol percentile over the past year (snapshot)",
    "skew_rr25_30d": "30-day 25-delta risk reversal (negative: puts dearer than calls)",
    "vol_risk_premium_30d": "30-day implied minus realized volatility",
    "vol_forecast_21d": "stochastic-vol-surface: forecast volatility over 21 trading days",
    "return_band_90_21d_low": "augury: 5th percentile of the 21-day return (28 tickers, snapshot)",
    "return_band_90_21d_high": "augury: 95th percentile of the 21-day return",
    "p_up_21d": "augury: probability the price is higher in 21 trading days",
    "value_score": "TickerLake style: value score (z, S&P 500)",
    "growth_score": "TickerLake style: growth score (z, S&P 500)",
    "buybacks_remaining_authorization": "TickerLake: remaining buyback authorization, USD (snapshot)",
    "buyback_blackout": "TickerLake: estimated buyback blackout window (snapshot)",
    "eps_revisions_30d": "TickerLake: 30-day change in the next fiscal year's consensus EPS estimate (snapshot)",
}

# How to read each value, given to the drafter with the item (``ContextItem.definition``).
# The 2026-10-15 pilot's review found memos reading a 2nd-percentile spread as "tight
# credit" and comparing the vol premium with 63-day realized vol. The plan sees only
# CATALOGUE (to choose items), so these notes don't change the plan's inputs.
READINGS: dict[str, str] = {
    "recession_probability": ("A nowcast of whether the US is in recession now, not a forecast for the memo's "
                              "horizon: its 12-month-ahead skill is chance. Its as_of is the date of the data, "
                              "which can lag the memo's date by weeks."),
    "credit_stress_probability": ("The probability that credit is in a stress regime now, not a forecast; it "
                                  "switched on late in March 2020."),
    "baa_spread_bps": "Higher means costlier corporate credit; lower means cheaper and easier credit.",
    "baa_spread_percentile": ("Share of days since 1990 with a spread at or below today's. A low percentile means "
                              "spreads are narrow: credit is cheap and easy to get (loose, not tight, credit "
                              "conditions). Narrow spreads leave more room to widen than to narrow, but the level "
                              "is not a forecast."),
    "credit_spread_forecast_5d_bps": "Positive means the spread is forecast to widen over the next 5 trading days.",
    "implied_vol_30d": "Annualized, in volatility points.",
    "implied_vol_30d_percentile_1y": "Share of the past year's days with implied vol at or below today's.",
    "skew_rr25_30d": ("25-delta call implied vol minus 25-delta put implied vol. Negative means puts are priced above "
                      "calls. It compares puts with calls, not implied with realized volatility."),
    "vol_risk_premium_30d": ("30-day at-the-money implied vol minus the stock's 21-trading-day close-to-close "
                             "realized vol, in volatility points. It is not comparable with a 63-day realized "
                             "volatility metric."),
    "vol_forecast_21d": "Annualized volatility forecast for the next 21 trading days.",
    "return_band_90_21d_low": "As a return from the last price; negative means a fall.",
    "return_band_90_21d_high": "As a return from the last price.",
    "p_up_21d": "Augury does not score its direction calls: there is no measured skill behind this number.",
    "value_score": ("Describes the stock's style relative to the S&P 500 (0 is average); it has shown no skill at "
                    "predicting returns."),
    "growth_score": ("Describes the stock's style relative to the S&P 500 (0 is average); it has shown no skill at "
                     "predicting returns."),
    "buyback_blackout": "Estimated from the company's reporting calendar, not disclosed by the company.",
    "eps_revisions_30d": "Positive means analysts raised the estimate over the past 30 days.",
}


class HubContext:
    version = "market-compass-1"

    def __init__(self, root: Path = HUB) -> None:
        self.root = root
        self._cache: dict[str, Any] = {}

    def available(self) -> dict[str, str]:
        return dict(CATALOGUE)

    # -- loading ----------------------------------------------------------------------
    def _pq(self, rel: str) -> pd.DataFrame:
        if rel not in self._cache:
            p = self.root / rel
            self._cache[rel] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        df: pd.DataFrame = self._cache[rel]
        return df

    def _json(self, rel: str) -> dict:
        if rel not in self._cache:
            p = self.root / rel
            self._cache[rel] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        d: dict = self._cache[rel]
        return d

    @staticmethod
    def _on_or_before(df: pd.DataFrame, as_of: date, col: str, max_days: int) -> tuple[float, date] | None:
        s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        s = s[s.index <= pd.Timestamp(as_of)]
        if s.empty or (pd.Timestamp(as_of) - s.index[-1]).days > max_days:
            return None
        return float(s.iloc[-1]), s.index[-1].date()

    def _snapshot_ok(self, snap: date, as_of: date) -> bool:
        return snap <= as_of <= snap + timedelta(days=SNAPSHOT_DAYS)

    # -- items ------------------------------------------------------------------------
    def context(self, ticker: str, names: list[str], as_of: date) -> list[ContextItem]:
        out = []
        for i, name in enumerate(names, 1):
            fn: Callable[[str, date], dict | None] | None = getattr(self, f"_{name}", None)
            got = fn(ticker, as_of) if fn and name in CATALOGUE else None
            if got:
                definition = " ".join(filter(None, [CATALOGUE[name] + ".", READINGS.get(name)]))
                out.append(ContextItem(id=f"x{i}", name=name, definition=definition, **got))
        return out

    def _recession_probability(self, ticker: str, as_of: date) -> dict | None:
        s = self._json("macro/summary.json")
        now = s.get("nowcast", {})
        sig = now.get("signal_as_of", {}).get("ensemble")
        run = s.get("run_at", "")[:10]
        base = {"unit": "ratio", "source": "macro-regime-nowcaster via market-compass", "is_forecast": True,
                "track_record": README_RECORDS["macro"], "max_age_days": 80}
        if sig and run and self._snapshot_ok(date.fromisoformat(run), as_of):
            return {**base, "value": now["recession_probability"], "as_of": date.fromisoformat(sig)}
        wf = self._pq("macro/walk_forward.parquet")
        if wf.empty:
            return None
        rows = wf[wf.index + pd.Timedelta(days=MACRO_LAG_DAYS) <= pd.Timestamp(as_of)]["p_recession"].dropna()
        if rows.empty:
            return None
        return {**base, "value": float(rows.iloc[-1]), "as_of": rows.index[-1].date()}

    def _credit_stress_probability(self, ticker: str, as_of: date) -> dict | None:
        got = self._on_or_before(self._pq("credit/stress.parquet"), as_of, "stress_prob", 7)
        return got and {"value": got[0], "as_of": got[1], "unit": "ratio", "is_forecast": True,
                        "source": "credit-spread-analyzer via market-compass", "max_age_days": 7,
                        "track_record": README_RECORDS["credit_stress"]}

    def _baa_spread_bps(self, ticker: str, as_of: date) -> dict | None:
        got = self._on_or_before(self._pq("credit/spreads_bps.parquet"), as_of, "baa_spread", 7)
        return got and {"value": got[0], "as_of": got[1], "source": "credit-spread-analyzer (FRED) via market-compass",
                        "max_age_days": 7}

    def _baa_spread_percentile(self, ticker: str, as_of: date) -> dict | None:
        s = self._pq("credit/spreads_bps.parquet")
        if "baa_spread" not in s.columns:
            return None
        h = s["baa_spread"].dropna()
        h = h[h.index <= pd.Timestamp(as_of)]
        if h.empty or (pd.Timestamp(as_of) - h.index[-1]).days > 7:
            return None
        return {"value": float((h <= h.iloc[-1]).mean()), "unit": "ratio", "as_of": h.index[-1].date(),
                "source": f"computed from the Baa spread history since {h.index[0].date()}", "max_age_days": 7}

    def _credit_spread_forecast_5d_bps(self, ticker: str, as_of: date) -> dict | None:
        got = self._on_or_before(self._pq("credit/forecast.parquet"), as_of, "forecast_bps", 7)
        return got and {"value": got[0], "as_of": got[1], "is_forecast": True, "max_age_days": 7,
                        "source": "credit-spread-analyzer via market-compass",
                        "track_record": README_RECORDS["credit_forecast"]}

    def _vol_history(self, ticker: str, as_of: date, col: str) -> tuple[float, date] | None:
        h = self._pq("vol/history.parquet")
        if h.empty:
            return None
        g = h[(h["ticker"] == ticker)].set_index("date").sort_index()
        return self._on_or_before(g, as_of, col, 5)

    def _implied_vol_30d(self, ticker: str, as_of: date) -> dict | None:
        got = self._vol_history(ticker, as_of, "atm_30d")
        return got and {"value": got[0], "as_of": got[1], "unit": "pct", "max_age_days": 5,
                        "source": "stochastic-vol-surface via market-compass"}

    def _skew_rr25_30d(self, ticker: str, as_of: date) -> dict | None:
        got = self._vol_history(ticker, as_of, "rr25_30d")
        return got and {"value": got[0], "as_of": got[1], "unit": "pct", "max_age_days": 5,
                        "source": "stochastic-vol-surface via market-compass"}

    def _vol_risk_premium_30d(self, ticker: str, as_of: date) -> dict | None:
        got = self._vol_history(ticker, as_of, "vrp_30d")
        return got and {"value": got[0], "as_of": got[1], "unit": "pct", "max_age_days": 5,
                        "source": "stochastic-vol-surface via market-compass"}

    def _implied_vol_30d_percentile_1y(self, ticker: str, as_of: date) -> dict | None:
        o = self._pq("vol/overview.parquet")
        row = o[o["ticker"] == ticker] if len(o) else o
        if row.empty or not self._snapshot_ok(row["date"].iloc[0].date(), as_of):
            return None
        return {"value": float(row["atm_30d_pct252"].iloc[0]), "unit": "ratio", "as_of": row["date"].iloc[0].date(),
                "source": "stochastic-vol-surface via market-compass", "max_age_days": 5}

    def _vol_forecast_21d(self, ticker: str, as_of: date) -> dict | None:
        f = self._pq("vol/forecasts.parquet")
        if f.empty:
            return None
        g = f[(f["ticker"] == ticker) & (f["horizon"] == 21)].set_index("date").sort_index()
        got = self._on_or_before(g, as_of, "forecast_vol", 5)
        if not got:
            return None
        rec = self._json("vol/summary.json").get("track_record_21d", {}).get(ticker)
        track = (f"stochastic-vol-surface published record: 21-day forecast RMSE {rec['rmse_forecast']:.1f} vol "
                 f"points vs {rec['rmse_implied']:.1f} for implied vol over {rec['n']} forecasts; 80% band covered "
                 f"{rec['coverage80']:.0%}") if rec else None
        return {"value": got[0], "as_of": got[1], "unit": "pct", "is_forecast": True, "track_record": track,
                "source": "stochastic-vol-surface via market-compass", "max_age_days": 5}

    def _augury(self, ticker: str, as_of: date) -> tuple[pd.Series, dict] | None:
        s = self._json("augury/summary.json")
        f = self._pq("augury/forecast.parquet")
        if f.empty or not s.get("as_of"):
            return None
        snap = date.fromisoformat(s["as_of"][:10])
        row = f[(f["ticker"] == ticker) & (f["horizon"] == 21)]
        if row.empty or not self._snapshot_ok(snap, as_of):
            return None
        return row.iloc[0], s.get("scores", {}).get(ticker, {})

    def _augury_item(self, ticker: str, as_of: date, value: Callable[[pd.Series], float], track: str | None) -> dict | None:
        got = self._augury(ticker, as_of)
        if not got:
            return None
        row, _ = got
        return {"value": float(value(row)), "unit": "pct", "as_of": row["as_of"].date(), "is_forecast": True,
                "track_record": track, "source": f"augury ({row['model']}) via market-compass", "max_age_days": 3}

    def _augury_record(self, ticker: str) -> str | None:
        sc = self._json("augury/summary.json").get("scores", {}).get(ticker)
        if not sc:
            return None
        since = str(sc.get("since", ""))[:10]
        return (f"augury published record since {since} ({sc['n']} days): the 21-day 90% band covered "
                f"{sc['coverage_90']['21']:.0%} of outcomes")

    def _return_band_90_21d_low(self, ticker: str, as_of: date) -> dict | None:
        return self._augury_item(ticker, as_of, lambda r: r["lo_90"] / r["last_price"] - 1,
                                 self._augury_record(ticker))

    def _return_band_90_21d_high(self, ticker: str, as_of: date) -> dict | None:
        return self._augury_item(ticker, as_of, lambda r: r["hi_90"] / r["last_price"] - 1,
                                 self._augury_record(ticker))

    def _p_up_21d(self, ticker: str, as_of: date) -> dict | None:
        # Augury scores its bands and VaR, not the direction call: no track record.
        return self._augury_item(ticker, as_of, lambda r: r["p_up"], None)

    def _style(self, ticker: str, as_of: date, col: str) -> dict | None:
        h = self._pq("style/stock_history.parquet")
        if h.empty:
            return None
        g = h[h["symbol"] == ticker].set_index("date").sort_index()
        got = self._on_or_before(g, as_of, col, 40)
        if not got:
            return None
        bt = self._json("style/summary.json").get("backtest", {}).get("stock", {}).get(col.split("_")[0], {}).get("1m")
        track = (f"market-compass style backtest {self._json('style/summary.json')['backtest']['start'][:4]}-"
                 f"{self._json('style/summary.json')['backtest']['end'][:4]}: rank correlation with next month's "
                 f"return {bt['mean']:+.3f} (t {bt['t']:.2f}, {bt['n']} months): no measured return skill") if bt else None
        return {"value": got[0], "as_of": got[1], "track_record": track, "max_age_days": 40,
                "source": "TickerLake style scores via market-compass"}

    def _value_score(self, ticker: str, as_of: date) -> dict | None:
        return self._style(ticker, as_of, "value")

    def _growth_score(self, ticker: str, as_of: date) -> dict | None:
        return self._style(ticker, as_of, "growth")

    def _buyback_row(self, ticker: str, as_of: date) -> pd.Series | None:
        s = self._json("buybacks/summary.json")
        b = self._pq("buybacks/stocks.parquet")
        if b.empty or not s.get("as_of") or not self._snapshot_ok(date.fromisoformat(s["as_of"]), as_of):
            return None
        row = b[b["symbol"] == ticker]
        return None if row.empty else row.iloc[0]

    def _buybacks_remaining_authorization(self, ticker: str, as_of: date) -> dict | None:
        row = self._buyback_row(ticker, as_of)
        if row is None or pd.isna(row["remaining_auth"]):
            return None
        return {"value": float(row["remaining_auth"]), "unit": "usd", "as_of": row["remaining_as_of"].date(),
                "source": "TickerLake buybacks (SEC filings) via market-compass", "max_age_days": 200}

    def _buyback_blackout(self, ticker: str, as_of: date) -> dict | None:
        row = self._buyback_row(ticker, as_of)
        if row is None:
            return None
        snap = date.fromisoformat(self._json("buybacks/summary.json")["as_of"])
        value = (f"in an estimated blackout until {row['reopens'].date()}" if row["blackout"] and pd.notna(row["reopens"])
                 else "not in an estimated blackout")
        return {"value": value, "as_of": snap, "source": "TickerLake buybacks via market-compass (estimated windows)",
                "max_age_days": 3}

    def _eps_revisions_30d(self, ticker: str, as_of: date) -> dict | None:
        s = self._json("revisions/summary.json")
        r = self._pq("revisions/revisions_names.parquet")
        if r.empty or not s.get("snapshot_date") or not self._snapshot_ok(date.fromisoformat(s["snapshot_date"]), as_of):
            return None
        row = r[r["symbol"] == ticker]
        if row.empty or pd.isna(row["rev_30d"].iloc[0]):
            return None
        # Published in percent (5.0 is +5%).
        return {"value": float(row["rev_30d"].iloc[0]) / 100, "unit": "pct", "as_of": date.fromisoformat(s["snapshot_date"]),
                "source": f"TickerLake Yahoo consensus snapshots since {s.get('first_snapshot')} via market-compass",
                "track_record": f"not measured: snapshots began {s.get('first_snapshot')}", "max_age_days": 3}
