"""
Do better features give the ML ranker an edge? A pre-registered test.

Protocol (fixed in code before any result was seen):

1. Universes. ``broad``: current S&P 500 constituents, each included only from
   the date it was added to the index (so future winners are not in the
   universe before they became large; companies that later left the index are
   still missing). ``narrow``: the 103 large caps used in the audit.
2. Candidates. Five model/feature combinations (``CANDIDATES``): the audit's
   price features vs. those plus published factors (``features.alpha_factors``),
   XGBoost vs. a linear IC-weighted model, raw vs. sector-neutral targets.
3. Development period: test dates up to ``DEV_END``. Each universe's candidate
   with the highest development-period mean IC is selected.
4. Holdout: test dates from ``HOLDOUT_START``, run once for the selected
   candidate plus the reference model and a momentum baseline.
5. Deflation: ``n_trials`` counts every configuration evaluated here plus the
   six models in the original audit.

Example::

    python scripts/feature_research.py                    # downloads ~500 tickers
    python scripts/feature_research.py --cache-dir data/research_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from auto_researcher.backtest.baselines import FeatureScoreModel, LinearICModel  # noqa: E402
from auto_researcher.backtest.walk_forward import (  # noqa: E402
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)
from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.features.alpha_factors import compute_alpha_factors  # noqa: E402
from auto_researcher.models.xgb_ranking_model import (  # noqa: E402
    XGBRegressionConfig,
    XGBRegressionModel,
)
from auto_researcher.screening import build_feature_panel  # noqa: E402

logger = logging.getLogger("feature_research")

BENCHMARK = "SPY"
DATA_START = "2014-01-01"
DEV_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"
PRIOR_AUDIT_TRIALS = 6
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _xgb() -> XGBRegressionModel:
    return XGBRegressionModel(
        XGBRegressionConfig(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            early_stopping_rounds=None,
        )
    )


@dataclass(frozen=True)
class Candidate:
    features: str  # "base" or "extended"
    factory: Callable[[], object]
    target: str
    description: str


CANDIDATES = {
    "base_xgb": Candidate("base", _xgb, "rank", "Audit model: price features, XGBoost"),
    "ext_xgb": Candidate("extended", _xgb, "rank", "+ published factors, XGBoost"),
    "ext_xgb_sn": Candidate("extended", _xgb, "group_rank", "+ factors, XGBoost, sector-neutral target"),
    "ext_linear": Candidate("extended", LinearICModel, "rank", "+ factors, linear IC-weighted"),
    "ext_linear_sn": Candidate("extended", LinearICModel, "group_rank",
                               "+ factors, linear IC-weighted, sector-neutral target"),
}
REFERENCE = "base_xgb"
MOMENTUM = Candidate("extended", lambda: FeatureScoreModel("mom_12_1"), "rank",
                     "12-1 momentum, no model")


# =============================================================================
# Data
# =============================================================================


def sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 members with GICS sector and the date each was added."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(WIKI_SP500, timeout=30,
                        headers={"User-Agent": "auto-researcher research script"})
    resp.raise_for_status()
    table = BeautifulSoup(resp.text, "html.parser").find("table", {"id": "constituents"})
    rows = [[c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")[1:]]
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["symbol", "name", "sector", "sub_industry", "hq", "date_added"])
    df["ticker"] = df["symbol"].str.replace(".", "-", regex=False)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    return df[["ticker", "sector", "sub_industry", "date_added"]]


def load_ohlcv(
    tickers: list[str], cache_dir: Path, start: str = DATA_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted close and volume from yfinance, cached as one parquet file."""
    import yfinance as yf

    key = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    path = cache_dir / f"ohlcv_{key}_{start}.parquet"
    if path.exists():
        raw = pd.read_parquet(path)
    else:
        frames = []
        for i in range(0, len(tickers), 100):
            batch = tickers[i : i + 100]
            for attempt in range(3):
                try:
                    data = yf.download(batch, start=start, auto_adjust=True,
                                       progress=False, threads=True)
                    frames.append(data[["Close", "Volume"]])
                    break
                except Exception as exc:  # rate limits, transient network errors
                    logger.warning("batch %d attempt %d failed: %s", i // 100, attempt + 1, exc)
                    time.sleep(5 * (attempt + 1))
        raw = pd.concat(frames, axis=1)
        # yfinance occasionally drops a ticker from a batch; retry those alone.
        got = set(raw["Close"].dropna(axis=1, how="all").columns)
        for ticker in sorted(set(tickers) - got):
            try:
                one = yf.download([ticker], start=start, auto_adjust=True, progress=False)
                if not one.empty:
                    raw = raw.drop(columns=[c for c in raw.columns if c[1] == ticker])
                    raw = pd.concat([raw, one[["Close", "Volume"]]], axis=1)
            except Exception as exc:
                logger.warning("retry for %s failed: %s", ticker, exc)
        cache_dir.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(path)
    close = raw["Close"].dropna(axis=1, how="all")
    volume = raw["Volume"].reindex(columns=close.columns)
    return close.sort_index(), volume.sort_index()


# =============================================================================
# Runs
# =============================================================================


def run(name: str, cand: Candidate, feats: dict[str, pd.DataFrame], prices: pd.DataFrame,
        sectors: pd.Series, top_k: int, n_trials: int, **window) -> WalkForwardResult:
    cfg = WalkForwardConfig(
        horizon=21, rebalance_every=21, execution_lag=1, train_window=504,
        min_train_dates=252, top_k=top_k, cost_bps=10.0, target=cand.target,
        n_trials=n_trials, n_random_paths=500, **window,
    )
    t0 = time.time()
    result = run_walk_forward(feats[cand.features], prices, cand.factory, cfg,
                              benchmark=BENCHMARK, name=name, groups=sectors)
    s = result.summary()
    logger.info("%-26s IC %+.4f (t %+.2f, n=%d)  IR vs EW %+.2f  [%.0fs]", name, s["ic_mean"],
                s["ic_t_nw"], s["n_periods"], s.get("net_ir_vs_equal_weight", np.nan),
                time.time() - t0)
    return result


ROWS = [
    ("n_periods", "Periods", "{:.0f}"),
    ("ic_mean", "Mean IC", "{:+.4f}"),
    ("ic_t_nw", "IC t-stat (NW)", "{:+.2f}"),
    ("ic_hit_rate", "IC > 0", "{:.0%}"),
    ("net_sharpe", "Top-k net Sharpe", "{:.2f}"),
    ("equal_weight_sharpe", "Equal-weight Sharpe", "{:.2f}"),
    ("net_active_vs_equal_weight_ann", "Net active vs EW / yr", "{:+.2%}"),
    ("net_ir_vs_equal_weight", "IR vs EW", "{:+.2f}"),
    ("net_ir_vs_equal_weight_deflated_prob", "P(IR > 0), deflated", "{:.2f}"),
    ("gross_sharpe_percentile_vs_random", "Pctl vs random top-k", "{:.0%}"),
    ("avg_turnover", "Turnover / rebalance", "{:.0%}"),
]


def table(results: dict[str, WalkForwardResult]) -> str:
    names = list(results)
    lines = ["| Metric | " + " | ".join(f"`{n}`" for n in names) + " |",
             "| --- | " + " | ".join("---:" for _ in names) + " |"]
    sums = {n: r.summary() for n, r in results.items()}
    for key, label, fmt in ROWS:
        cells = []
        for n in names:
            v = sums[n].get(key)
            cells.append("n/a" if v is None or not np.isfinite(v) else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "research_cache"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "feature_research"))
    ap.add_argument("--universes", default="narrow,broad")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "numexpr", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    members = sp500_constituents()
    sectors = members.set_index("ticker")["sector"]
    added = members.set_index("ticker")["date_added"]
    cached = pd.read_parquet(ROOT / "data" / "price_cache" / "prices_2015-01-01_2020-01-01.parquet") \
        if (ROOT / "data" / "price_cache" / "prices_2015-01-01_2020-01-01.parquet").exists() else None
    narrow_names = sorted(set(cached.columns.get_level_values(1)) - {BENCHMARK}) if cached is not None else []

    close, volume = load_ohlcv(sorted(set(members["ticker"]) | set(narrow_names) | {BENCHMARK}),
                               Path(args.cache_dir))
    logger.info("Prices: %d tickers, %s to %s", close.shape[1], close.index.min().date(),
                close.index.max().date())

    universes = {}
    for u in [x.strip() for x in args.universes.split(",") if x.strip()]:
        names = narrow_names if u == "narrow" else sorted(members["ticker"])
        names = [t for t in names if t in close.columns and close[t].notna().sum() > 300]
        universes[u] = names

    n_trials = PRIOR_AUDIT_TRIALS + len(universes) * (len(CANDIDATES) + 1)
    report = {"dev": {}, "holdout": {}, "selected": {}, "factor_ic": {}, "universe_size": {}}
    md_sections: list[str] = []

    for u, names in universes.items():
        px = close[names + [BENCHMARK]]
        vol = volume[names + [BENCHMARK]]
        base = build_feature_panel(px, benchmark=BENCHMARK)
        alpha = compute_alpha_factors(px, vol, benchmark=BENCHMARK, sectors=sectors)
        extended = base.join(alpha, how="inner").dropna()
        # Same rows for every feature set (the factors need more history), so
        # candidates differ only in their columns and share test dates.
        feats = {"base": base.loc[extended.index], "extended": extended}
        # Index membership: a stock enters the universe on the date it joined
        # the S&P 500 (unknown dates: always a member).
        for k, f in feats.items():
            dates = f.index.get_level_values("date")
            start = f.index.get_level_values("ticker").map(added).to_numpy()
            keep = pd.isna(start) | (dates.to_numpy() >= start)
            feats[k] = f[keep]
        top_k = 10 if u == "narrow" else 50
        report["universe_size"][u] = len(names)
        logger.info("=== %s universe: %d stocks, top-%d portfolios ===", u, len(names), top_k)

        dev = {name: run(f"{u}:{name}", c, feats, px, sectors, top_k, n_trials, end=DEV_END)
               for name, c in CANDIDATES.items()}
        dev["momentum_12_1"] = run(f"{u}:momentum_12_1", MOMENTUM, feats, px, sectors, top_k,
                                   n_trials, end=DEV_END)
        selected = max(CANDIDATES, key=lambda n: dev[n].summary()["ic_mean"])
        logger.info("%s: selected %s on the development period", u, selected)

        # Single-factor diagnostics on the development period (not used for selection).
        factor_ic = {}
        cfg = WalkForwardConfig(horizon=21, rebalance_every=21, execution_lag=1, train_window=504,
                                min_train_dates=252, top_k=top_k, n_random_paths=0, end=DEV_END)
        for col in alpha.columns:
            r = run_walk_forward(feats["extended"], px, lambda c=col: FeatureScoreModel(c), cfg,
                                 benchmark=BENCHMARK, name=col)
            s = r.summary()
            factor_ic[col] = {"ic_mean": s["ic_mean"], "ic_t_nw": s["ic_t_nw"]}

        holdout_names = list(dict.fromkeys([selected, REFERENCE]))
        holdout = {name: run(f"{u}:{name}:holdout", CANDIDATES[name], feats, px, sectors, top_k,
                             n_trials, start=HOLDOUT_START) for name in holdout_names}
        holdout["momentum_12_1"] = run(f"{u}:momentum_12_1:holdout", MOMENTUM, feats, px, sectors,
                                       top_k, n_trials, start=HOLDOUT_START)

        report["dev"][u] = {n: r.summary() for n, r in dev.items()}
        report["holdout"][u] = {n: r.summary() for n, r in holdout.items()}
        report["selected"][u] = selected
        report["factor_ic"][u] = factor_ic

        fac_lines = ["| Factor | Mean IC | t (NW) |", "| --- | ---: | ---: |"]
        for col, v in sorted(factor_ic.items(), key=lambda kv: -kv[1]["ic_mean"]):
            fac_lines.append(f"| `{col}` | {v['ic_mean']:+.4f} | {v['ic_t_nw']:+.2f} |")
        md_sections += [
            f"## {u.capitalize()} universe ({len(names)} stocks, top-{top_k} portfolios)", "",
            f"### Development period (test dates through {DEV_END})", "", table(dev), "",
            f"Selected by development-period mean IC: **`{selected}`**.", "",
            f"### Holdout (test dates from {HOLDOUT_START}, run once)", "", table(holdout), "",
            "### Single-factor ICs, development period (diagnostic only)", "", *fac_lines, "",
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Feature research: does the ML ranker gain an edge?", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/feature_research.py`.", "",
        "Pre-registered protocol (see the script docstring): five candidates per universe, "
        f"selection by mean IC on test dates through {DEV_END}, then one holdout run from "
        f"{HOLDOUT_START}. Purged walk-forward, 21-day labels, next-close execution, "
        f"504-day rolling training window, 10 bps costs. Deflated probabilities assume "
        f"{n_trials} configurations were tried. Every candidate uses the same (date, ticker) "
        "rows, so they differ only in features, model and target.", "",
        "Candidates:", "", *[f"* `{n}`: {c.description}" for n, c in CANDIDATES.items()],
        "* `momentum_12_1`: 12-1 momentum, no model (baseline)", "",
        "Universes use current constituents (survivorship-biased). In the broad universe each "
        "stock enters on its S&P 500 add date, which removes the bias from future additions "
        "but not from companies that later left the index.", "",
    ]
    out.with_suffix(".md").write_text("\n".join(header + md_sections), encoding="utf-8")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    report.update({"generated": datetime.now().isoformat(timespec="seconds"), "n_trials": n_trials})
    out.with_suffix(".json").write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    logger.info("Wrote %s.md/.json", out)


if __name__ == "__main__":
    main()
