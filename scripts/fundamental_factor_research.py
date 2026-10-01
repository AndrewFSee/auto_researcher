"""
Round 3: pre-registered tests of fundamental factors and an ML valuation gap.

The protocol is fixed in ``docs/results/fundamental_factors_protocol.md``
(written before any result): six published factors and their composite
(Part A), and a valuation gap from an ML forecast of growth and margins vs. a
naive forecast and the plain reverse DCF (Part B). Every candidate is run once
on the development period (through 2022) and once on the holdout (2023 on),
on identical (date, stock) rows within each part.

Example::

    python scripts/fundamental_factor_research.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_fundamentals import MISMATCH_PATH, PIT_PATH, load_close_and_splits  # noqa: E402
from feature_research import load_ohlcv, sp500_constituents  # noqa: E402
from reverse_dcf import CAPTIVE_FINANCE  # noqa: E402

from auto_researcher.backtest.baselines import FeatureScoreModel  # noqa: E402
from auto_researcher.backtest.metrics import compute_ic_stats  # noqa: E402
from auto_researcher.backtest.walk_forward import (  # noqa: E402
    WalkForwardConfig,
    WalkForwardResult,
    overlap_lag,
    run_walk_forward,
)
from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.features.fundamental_factors import (  # noqa: E402
    FACTOR_SIGNS,
    composite_score,
    factor_panel,
    growth_features,
    growth_labels,
    intrinsic_ev,
)
from auto_researcher.models.growth_forecast import (  # noqa: E402
    baseline_forecasts,
    walk_forward_forecasts,
)

logger = logging.getLogger("fundamental_factor_research")

BENCHMARK = "SPY"
PRICE_START = "2009-01-01"
PANEL_START = "2011-01-03"
GAP_START = "2016-01-01"
TRAIN_START = "2012-06-30"
DEV_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"
HORIZONS = (21, 63)
PRIMARY = 21
STEP = 21
TOP_K = 50
N_TRIALS = 20
T_THRESHOLD = 2.0

PART_A = {**FACTOR_SIGNS, "composite": 1}
PART_B = {"valuation_gap_ml": 1, "valuation_gap_naive": 1, "reverse_dcf": 1, "composite": 1}
PART_B_DESC = {
    "valuation_gap_ml": "intrinsic EV from the ML growth and margin forecasts / EV",
    "valuation_gap_naive": "intrinsic EV from the naive forecasts (B2 growth, M2 margin) / EV",
    "reverse_dcf": "reverse DCF implied growth (scored as FCF-to-firm / EV, same ranking)",
    "composite": "Part A composite on the same rows (comparison, not a Part B hypothesis)",
}


def run(feats: pd.DataFrame, col: str, sign: int, prices: pd.DataFrame, horizon: int,
        **window) -> WalkForwardResult:
    # Test dates fall every STEP trading days from the first feature date, which
    # matches the feature grid when min_train_dates + horizon + lag is a multiple of STEP.
    min_train = (-(horizon + 1)) % STEP + STEP
    cfg = WalkForwardConfig(horizon=horizon, rebalance_every=STEP, execution_lag=1,
                            train_window=252, min_train_dates=min_train, top_k=TOP_K,
                            cost_bps=10.0, n_trials=N_TRIALS, n_random_paths=500, **window)
    return run_walk_forward(feats[[col]], prices, lambda: FeatureScoreModel(col, sign), cfg,
                            benchmark=BENCHMARK, name=col)


def run_part(feats: pd.DataFrame, signs: dict[str, int], prices: pd.DataFrame) -> dict:
    out: dict = {}
    for h in HORIZONS:
        for period, window in (("dev", {"end": DEV_END}), ("holdout", {"start": HOLDOUT_START})):
            for col, sign in signs.items():
                r = run(feats, col, sign, prices, h, **window)
                out[(col, h, period)] = r
                s = r.summary()
                logger.info("%-22s h=%-3d %-7s IC %+.4f (t %+.2f, n=%d)  IR vs EW %+.2f", col, h, period,
                            s["ic_mean"], s["ic_t_nw"], s["n_periods"], s.get("net_ir_vs_equal_weight", np.nan))
    return out


def paired_t(a: WalkForwardResult, b: WalkForwardResult) -> tuple[float, float]:
    diff = (a.ic - b.ic).dropna()
    lag = overlap_lag(a.config.horizon, a.config.rebalance_every)
    st = compute_ic_stats(diff.to_numpy(), horizon_days=lag + 1)
    return float(st["mean"]), float(st["t_stat_nw"])


ROWS = [
    ("n_periods", "Periods", "{:.0f}"),
    ("ic_mean", "Mean IC", "{:+.4f}"),
    ("ic_t_nw", "IC t (NW)", "{:+.2f}"),
    ("ic_hit_rate", "IC > 0", "{:.0%}"),
    ("net_active_vs_equal_weight_ann", "Top-50 net vs EW / yr", "{:+.2%}"),
    ("net_ir_vs_equal_weight", "IR vs EW", "{:+.2f}"),
    ("net_ir_vs_equal_weight_deflated_prob", "P(IR > 0), deflated", "{:.2f}"),
    ("gross_sharpe_percentile_vs_random", "Pctl vs random top-50", "{:.0%}"),
]


def results_table(results: dict, cols: list[str], h: int, period: str) -> list[str]:
    lines = ["| Metric | " + " | ".join(f"`{c}`" for c in cols) + " |",
             "| --- |" + " ---: |" * len(cols)]
    sums = {c: results[(c, h, period)].summary() for c in cols}
    for key, label, fmt in ROWS:
        cells = []
        for c in cols:
            v = sums[c].get(key)
            cells.append("n/a" if v is None or not np.isfinite(v) else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def verdicts(results: dict, signs: dict[str, int]) -> dict[str, dict]:
    """Protocol rule: dev IC (h=21) |t| >= 2 (IC already signed) and holdout IC > 0."""
    out = {}
    for col in signs:
        dev = results[(col, PRIMARY, "dev")].summary()
        hold = results[(col, PRIMARY, "holdout")].summary()
        supported = dev["ic_t_nw"] >= T_THRESHOLD and dev["ic_mean"] > 0 and hold["ic_mean"] > 0
        out[col] = {"dev_ic": dev["ic_mean"], "dev_t": dev["ic_t_nw"], "holdout_ic": hold["ic_mean"],
                    "holdout_t": hold["ic_t_nw"], "supported": bool(supported)}
    return out


def forecast_accuracy(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-date rank correlation and MAE of each forecast against realized values."""
    specs = {"growth": ("y_growth", ["growth_ml", "growth_b1", "growth_b2"]),
             "margin": ("y_margin", ["margin_ml", "margin_m1", "margin_m2"])}
    rows, paired = [], {}
    for target, (label, cols) in specs.items():
        d = pred.dropna(subset=[label, *cols])
        by_date = d.groupby(level="date")
        corr = pd.DataFrame({
            c: by_date.apply(lambda g, c=c, label=label: g[c].corr(g[label], method="spearman"))
            for c in cols})
        for c in cols:
            rows.append({"target": target, "forecast": c, "dates": len(corr), "rows": len(d),
                         "mean_rank_corr": corr[c].mean(), "mae": (d[c] - d[label]).abs().mean()})
        base = cols[2]
        # Quarterly dates with 3-year labels overlap 12 quarters.
        st = compute_ic_stats((corr[cols[0]] - corr[base]).to_numpy(), horizon_days=12)
        paired[target] = {"vs": base, "mean_diff": float(st["mean"]), "t_nw": float(st["t_stat_nw"])}
    return pd.DataFrame(rows), paired


def membership(df: pd.DataFrame, added: pd.Series) -> pd.DataFrame:
    dates = df.index.get_level_values("date")
    start = df.index.get_level_values("symbol").map(added).to_numpy()
    return df[pd.isna(start) | (dates.to_numpy() >= start)]


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "research_cache"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "fundamental_factors"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("auto_researcher.models.growth_forecast").setLevel(logging.INFO)
    warnings.filterwarnings("ignore")
    cache = Path(args.cache_dir)

    # ---------------- universe and data ----------------
    table = pd.read_parquet(PIT_PATH)
    members = sp500_constituents().set_index("ticker")
    mismatch = set(json.loads(MISMATCH_PATH.read_text())) if MISMATCH_PATH.exists() else set()
    names = sorted(t for t in members.index if t in set(table["symbol"])
                   and members.loc[t, "sector"] != "Financials"
                   and t not in CAPTIVE_FINANCE and t not in mismatch)
    returns_px, _ = load_ohlcv(names + [BENCHMARK], cache, start=PRICE_START)
    names = [t for t in names if t in returns_px.columns]
    close, splits = load_close_and_splits(sorted(table["symbol"].unique()), max_age_days=30)
    cal = returns_px.index
    data_end = cal[-1]
    grid = cal[cal.searchsorted(pd.Timestamp(PANEL_START))::STEP]
    sectors = members["sector"]
    added = members["date_added"]
    logger.info("Universe: %d stocks; %d signal dates %s to %s", len(names), len(grid),
                grid[0].date(), grid[-1].date())

    # ---------------- Part A ----------------
    panel = membership(factor_panel(table, close, splits, grid, names), added)
    feats_a = panel[list(FACTOR_SIGNS)].dropna()
    feats_a = feats_a.assign(composite=composite_score(feats_a))
    feats_a.index = feats_a.index.set_names(["date", "ticker"])
    logger.info("Part A rows: %d (%d dates, median %d stocks per date)", len(feats_a),
                feats_a.index.get_level_values(0).nunique(),
                feats_a.groupby(level=0).size().median())
    res_a = run_part(feats_a, PART_A, returns_px)

    # ---------------- Part B: forecasts ----------------
    qdates = pd.DatetimeIndex(pd.Series(cal, index=cal).groupby(cal.to_period("Q")).max().to_numpy())
    qdates = qdates[qdates >= pd.Timestamp(TRAIN_START)]
    train = membership(growth_features(table, qdates, names, sectors), added)
    train = train.join(growth_labels(table, qdates, names, data_end))
    gap_dates = grid[grid >= pd.Timestamp(GAP_START)]
    gx = membership(growth_features(table, gap_dates, names, sectors), added)
    refits = list(pd.date_range(GAP_START, data_end, freq="YS"))
    ml_q = walk_forward_forecasts(train, train, refits)
    ml_g = walk_forward_forecasts(train, gx, refits)

    pred_q = train.join(ml_q, how="inner").join(baseline_forecasts(train))
    acc, paired_fc = forecast_accuracy(pred_q)
    logger.info("Forecast accuracy:\n%s", acc.round(4).to_string())

    # ---------------- Part B: valuation gaps ----------------
    gb = gx.join(ml_g, how="inner").join(baseline_forecasts(gx))
    gb.index = gb.index.set_names(["date", "ticker"])
    ev = panel["enterprise_value"].rename_axis(["date", "ticker"]).reindex(gb.index)
    fcff = panel["fcff"].rename_axis(["date", "ticker"]).reindex(gb.index)
    fcff_ev = panel["fcff_ev"].rename_axis(["date", "ticker"]).reindex(gb.index)
    pos_ev = ev.where(ev > 0)
    gaps = pd.DataFrame({
        "valuation_gap_ml": intrinsic_ev(gb["revenue"], gb["fcff_margin"], gb["growth_ml"],
                                         gb["margin_ml"]) / pos_ev,
        "valuation_gap_naive": intrinsic_ev(gb["revenue"], gb["fcff_margin"], gb["growth_b2"],
                                            gb["margin_m2"]) / pos_ev,
        "reverse_dcf": fcff_ev.where(fcff > 0),
    }, index=gb.index)
    feats_b = feats_a[["composite"]].join(gaps, how="inner").dropna()
    logger.info("Part B rows: %d (%d dates, median %d stocks per date)", len(feats_b),
                feats_b.index.get_level_values(0).nunique(),
                feats_b.groupby(level=0).size().median())
    res_b = run_part(feats_b, PART_B, returns_px)

    # ---------------- verdicts ----------------
    ver_a = verdicts(res_a, PART_A)
    ver_b = verdicts(res_b, PART_B)
    ml_vs = {other: paired_t(res_b[("valuation_gap_ml", PRIMARY, "dev")], res_b[(other, PRIMARY, "dev")])
             for other in ("valuation_gap_naive", "reverse_dcf")}
    hold = {c: res_b[(c, PRIMARY, "holdout")].summary()["ic_mean"] for c in PART_B}
    ml_adds = (all(d > 0 and t >= T_THRESHOLD for d, t in ml_vs.values())
               and hold["valuation_gap_ml"] > 0
               and hold["valuation_gap_ml"] >= max(hold["valuation_gap_naive"], hold["reverse_dcf"]))
    fc_useful = paired_fc["growth"]["mean_diff"] > 0 and paired_fc["growth"]["t_nw"] >= T_THRESHOLD

    # ---------------- report ----------------
    def fmt_ver(v: dict) -> str:
        return (f"dev IC {v['dev_ic']:+.4f} (t {v['dev_t']:+.2f}), holdout IC {v['holdout_ic']:+.4f} "
                f"(t {v['holdout_t']:+.2f}): **{'supported' if v['supported'] else 'not supported'}**")

    a_cols, b_cols = list(PART_A), list(PART_B)
    md = [
        "# Round 3: fundamental factors and an ML valuation gap", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/fundamental_factor_research.py`, following "
        "the pre-registered [protocol](fundamental_factors_protocol.md) without changes. Signs are "
        "applied before scoring, so a positive IC always means the factor worked as published. "
        f"Deflated probabilities assume {N_TRIALS} trials.", "",
        "Run log: the first complete run crashed while writing this report (a naming bug in the "
        "report code, after all results were computed). The fix touched only report formatting; "
        "the rerun reproduced every logged statistic exactly.", "",
        f"Universe: {len(names)} current S&P 500 members (from their add dates; no financials or "
        "captive-finance industrials). Survivorship bias remains.", "",
        "## Verdicts (21-day horizon)", "",
        *[f"* `{c}` (expected sign {'+' if s > 0 else '−'}): {fmt_ver(ver_a[c])}" for c, s in PART_A.items()],
        "",
        f"* ML growth forecast useful (beats B2 on revenue-growth rank correlation, paired t ≥ 2): "
        f"**{'yes' if fc_useful else 'no'}** (difference {paired_fc['growth']['mean_diff']:+.3f}, "
        f"t {paired_fc['growth']['t_nw']:+.2f}).",
        f"* ML valuation gap adds value over the naive gap and the reverse DCF: "
        f"**{'yes' if ml_adds else 'no'}** (dev IC difference vs naive "
        f"{ml_vs['valuation_gap_naive'][0]:+.4f}, t {ml_vs['valuation_gap_naive'][1]:+.2f}; vs reverse DCF "
        f"{ml_vs['reverse_dcf'][0]:+.4f}, t {ml_vs['reverse_dcf'][1]:+.2f}).", "",
        f"## Part A: six factors ({feats_a.index.get_level_values(0).nunique()} dates, median "
        f"{int(feats_a.groupby(level=0).size().median())} stocks)", "",
    ]
    for h in HORIZONS:
        tag = "primary" if h == PRIMARY else "secondary"
        md += [f"### {h}-day horizon ({tag}), development (through {DEV_END})", "",
               *results_table(res_a, a_cols, h, "dev"), "",
               f"### {h}-day horizon ({tag}), holdout (from {HOLDOUT_START})", "",
               *results_table(res_a, a_cols, h, "holdout"), ""]
    yearly = pd.DataFrame({c: res_a[(c, PRIMARY, "dev")].ic.groupby(
        res_a[(c, PRIMARY, "dev")].ic.index.year).mean() for c in a_cols})
    md += ["### Mean IC by year (21-day, development)", "",
           "| Year | " + " | ".join(f"`{c}`" for c in a_cols) + " |", "| --- |" + " ---: |" * len(a_cols)]
    md += [f"| {y} | " + " | ".join(f"{v:+.3f}" for v in r) + " |" for y, r in yearly.iterrows()]
    md += ["", "## Part B: growth forecast", "",
           "Out-of-sample forecasts at quarter-ends from 2016 whose 3-year outcome is known. "
           "Rank correlation is per date, averaged.", "",
           "| Target | Forecast | Dates | Rows | Mean rank corr. | MAE |",
           "| --- | --- | ---: | ---: | ---: | ---: |"]
    md += [f"| {r.target} | `{r.forecast}` | {r.dates} | {r.rows:,} | {r.mean_rank_corr:+.3f} | {r.mae:.3f} |"
           for r in acc.itertuples()]
    md += ["", f"Paired difference ML − B2 (growth): {paired_fc['growth']['mean_diff']:+.3f} "
           f"(t {paired_fc['growth']['t_nw']:+.2f}); ML − M2 (margin): "
           f"{paired_fc['margin']['mean_diff']:+.3f} (t {paired_fc['margin']['t_nw']:+.2f}).", "",
           f"## Part B: valuation gap ({feats_b.index.get_level_values(0).nunique()} dates, median "
           f"{int(feats_b.groupby(level=0).size().median())} stocks; positive FCF only)", "",
           *[f"* `{c}`: {d}" for c, d in PART_B_DESC.items()], ""]
    for h in HORIZONS:
        tag = "primary" if h == PRIMARY else "secondary"
        md += [f"### {h}-day horizon ({tag}), development", "", *results_table(res_b, b_cols, h, "dev"), "",
               f"### {h}-day horizon ({tag}), holdout", "", *results_table(res_b, b_cols, h, "holdout"), ""]
    md += ["### Part B verdicts", "", *[f"* `{c}`: {fmt_ver(ver_b[c])}" for c in PART_B], ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (float, np.floating)):
            return float(obj) if np.isfinite(obj) else None
        return obj

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"), "n_trials": N_TRIALS,
        "universe_size": len(names),
        "part_a": {f"{c}|h{h}|{p}": r.summary() for (c, h, p), r in res_a.items()},
        "part_b": {f"{c}|h{h}|{p}": r.summary() for (c, h, p), r in res_b.items()},
        "verdicts_a": ver_a, "verdicts_b": ver_b,
        "forecast_accuracy": acc.to_dict(orient="records"), "forecast_paired": paired_fc,
        "ml_gap_vs": {k: {"mean_diff": d, "t_nw": t} for k, (d, t) in ml_vs.items()},
        "ml_gap_adds_value": bool(ml_adds), "forecast_useful": bool(fc_useful),
    }
    out.with_suffix(".json").write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    logger.info("Wrote %s.md/.json", out)


if __name__ == "__main__":
    main()
