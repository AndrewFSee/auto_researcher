"""
Do earnings-surprise features give the ML ranker an edge? Pre-registered test, round 2.

Same protocol as ``scripts/feature_research.py`` (round 1): fixed candidates,
selection by mean IC on test dates through ``DEV_END``, one holdout run from
``HOLDOUT_START``, deflation over every configuration tried in both rounds.

New inputs (``features.earnings_events``), all point-in-time:

* ``sue``: time-series standardized unexpected earnings (Bernard & Thomas 1989)
  from DefeatBeta's quarterly EPS history; no analyst estimates are needed.
* ``earnings_ann_return``: the stock's return around the announcement in
  excess of SPY, a strong drift predictor (Brandt et al. 2008).
* ``eps_beat_streak``: year-over-year EPS increases in the last four quarters.
* ``days_since_ann``: recency of the last announcement.

Announcement dates are inferred from SEC 8-K/10-Q filings and become usable
two trading days after the filing date.

Candidates (round 1's broad winner is the reference):

* ``ext_linear``: round-1 features (price + 13 factors), linear IC-weighted
* ``earn_linear``: + earnings features, linear IC-weighted
* ``earn_linear_sn``: + earnings features, linear, sector-neutral target
* ``earn_xgb``: + earnings features, XGBoost
* ``earn_only_linear``: earnings features only, linear

Example::

    python scripts/earnings_feature_research.py --cache-dir data/research_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_research import (  # noqa: E402
    BENCHMARK,
    DEV_END,
    HOLDOUT_START,
    PRIOR_AUDIT_TRIALS,
    Candidate,
    _xgb,
    load_ohlcv,
    run,
    sp500_constituents,
    table,
)

from auto_researcher.backtest.baselines import FeatureScoreModel, LinearICModel  # noqa: E402
from auto_researcher.backtest.walk_forward import WalkForwardConfig, run_walk_forward  # noqa: E402
from auto_researcher.console import use_utf8_output  # noqa: E402
from auto_researcher.features.alpha_factors import compute_alpha_factors  # noqa: E402
from auto_researcher.features.earnings_events import (  # noqa: E402
    EARNINGS_FEATURES,
    announcement_dates,
    attach_surprises,
    earnings_feature_panel,
    time_series_sue,
)
from auto_researcher.screening import build_feature_panel  # noqa: E402

logger = logging.getLogger("earnings_feature_research")

DEFEATBETA = "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/US/"
ROUND1_TRIALS = 2 * 6  # two universes x (five candidates + momentum) in round 1

CANDIDATES = {
    "ext_linear": Candidate("extended", LinearICModel, "rank", "Round-1 reference: price + 13 factors, linear"),
    "earn_linear": Candidate("earnings", LinearICModel, "rank", "+ earnings features, linear"),
    "earn_linear_sn": Candidate("earnings", LinearICModel, "group_rank",
                                "+ earnings features, linear, sector-neutral target"),
    "earn_xgb": Candidate("earnings", _xgb, "rank", "+ earnings features, XGBoost"),
    "earn_only_linear": Candidate("earnings_only", LinearICModel, "rank", "Earnings features only, linear"),
}
REFERENCE = "ext_linear"
SUE_ONLY = Candidate("earnings", lambda: FeatureScoreModel("sue"), "rank", "SUE alone, no model")


def defeatbeta_table(name: str, cache_dir: Path, **read_kwargs) -> pd.DataFrame:
    """Download (once) and read a DefeatBeta US parquet file."""
    import requests

    path = cache_dir / f"defeatbeta_{name}.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with requests.get(DEFEATBETA + name + ".parquet", stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
    return pd.read_parquet(path, **read_kwargs)


def build_events(cache_dir: Path) -> pd.DataFrame:
    filings = defeatbeta_table(
        "stock_sec_filing", cache_dir,
        columns=["symbol", "form_type", "filing_date", "report_date"],
        filters=[("form_type", "in", ["8-K", "10-Q", "10-K"])],
    )
    eps = defeatbeta_table("stock_tailing_eps", cache_dir).rename(columns={"report_date": "period_end"})
    return attach_surprises(announcement_dates(filings), time_series_sue(eps))


def event_time_check(events: pd.DataFrame, close: pd.DataFrame, names: list[str]) -> list[str]:
    """
    Event-time view of the same surprise: quarterly cross-sectional IC of SUE
    against the reaction and the subsequent drift (market-adjusted), measured
    from the availability date. Separates "the signal is weak" from "the
    monthly panel dilutes it".
    """
    from scipy import stats

    cal, spy = close.index, close[BENCHMARK].to_numpy()
    ev = events[events["symbol"].isin(names) & (events["announce_date"] >= "2014-06-01")]
    rows = []
    for e in ev.itertuples(index=False):
        d = cal.searchsorted(pd.Timestamp(e.announce_date))
        a = d + 2
        if e.symbol not in close.columns or d < 1 or a + 61 >= len(cal):
            continue
        p = close[e.symbol].to_numpy()

        def excess(i: int, j: int, p: np.ndarray = p) -> float:
            ok = np.isfinite(p[i]) and np.isfinite(p[j])
            return (p[j] / p[i] - 1) - (spy[j] / spy[i] - 1) if ok else np.nan

        rows.append((e.announce_date, e.sue, excess(d - 1, a), excess(a, a + 20),
                     excess(a, a + 40), excess(a, a + 60)))
    df = pd.DataFrame(rows, columns=["date", "sue", "reaction", "drift_20d", "drift_40d", "drift_60d"])
    df["q"] = pd.to_datetime(df["date"]).dt.to_period("Q")
    lines = [
        "## Event-time check (broad universe, 2014 onward)", "",
        f"{len(df):,} announcements, {df['q'].nunique()} quarters. Mean quarterly Spearman IC of "
        "SUE against market-adjusted returns from the availability date; the reaction runs from "
        "the close before the filing to availability.", "",
        "| Window | Mean quarterly IC | t | Top - bottom decile |", "| --- | ---: | ---: | ---: |",
    ]
    hi, lo = df["sue"].quantile(0.9), df["sue"].quantile(0.1)
    for col in ["reaction", "drift_20d", "drift_40d", "drift_60d"]:
        per = df.groupby("q").apply(
            lambda g, c=col: stats.spearmanr(g["sue"], g[c], nan_policy="omit").statistic
            if len(g) > 30 else np.nan, include_groups=False).dropna()
        t = per.mean() / per.std() * np.sqrt(len(per))
        spread = df.loc[df["sue"] >= hi, col].mean() - df.loc[df["sue"] <= lo, col].mean()
        lines.append(f"| `{col}` | {per.mean():+.4f} | {t:+.2f} | {spread:+.2%} |")
    return lines + [""]


def main(argv: list[str] | None = None) -> None:
    use_utf8_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "research_cache"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "earnings_feature_research"))
    ap.add_argument("--universes", default="narrow,broad")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("auto_researcher", "numexpr", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    cache = Path(args.cache_dir)

    members = sp500_constituents()
    sectors = members.set_index("ticker")["sector"]
    added = members.set_index("ticker")["date_added"]
    cached_path = ROOT / "data" / "price_cache" / "prices_2015-01-01_2020-01-01.parquet"
    narrow_names = (sorted(set(pd.read_parquet(cached_path).columns.get_level_values(1)) - {BENCHMARK})
                    if cached_path.exists() else [])
    close, volume = load_ohlcv(sorted(set(members["ticker"]) | set(narrow_names) | {BENCHMARK}), cache)

    t0 = time.time()
    events = build_events(cache)
    events = events[events["sue"].notna()]
    logger.info("Earnings events with SUE: %d (%d symbols) [%.0fs]", len(events),
                events["symbol"].nunique(), time.time() - t0)

    n_trials = PRIOR_AUDIT_TRIALS + ROUND1_TRIALS + 2 * (len(CANDIDATES) + 1)
    report: dict = {"dev": {}, "holdout": {}, "selected": {}, "factor_ic": {}, "universe_size": {}}
    sections: list[str] = []

    for u in [x.strip() for x in args.universes.split(",") if x.strip()]:
        names = narrow_names if u == "narrow" else sorted(members["ticker"])
        names = [t for t in names if t in close.columns and close[t].notna().sum() > 300]
        px, vol = close[names + [BENCHMARK]], volume[names + [BENCHMARK]]

        base = build_feature_panel(px, benchmark=BENCHMARK)
        alpha = compute_alpha_factors(px, vol, benchmark=BENCHMARK, sectors=sectors)
        earn = earnings_feature_panel(events[events["symbol"].isin(names)], px, benchmark=BENCHMARK)
        extended = base.join(alpha, how="inner").dropna()
        # Same rows for every feature set; earnings features are 0 (neutral) when absent.
        earn = earn.reindex(extended.index).fillna(0.0)
        feats = {
            "extended": extended,
            "earnings": extended.join(earn),
            "earnings_only": earn,
        }
        for k, f in feats.items():
            dates = f.index.get_level_values("date")
            start = f.index.get_level_values("ticker").map(added).to_numpy()
            feats[k] = f[pd.isna(start) | (dates.to_numpy() >= start)]

        coverage = float((feats["earnings"]["sue"] != 0).mean())
        top_k = 10 if u == "narrow" else 50
        report["universe_size"][u] = len(names)
        logger.info("=== %s: %d stocks, top-%d; rows with a live surprise: %.0f%% ===",
                    u, len(names), top_k, 100 * coverage)

        dev = {n: run(f"{u}:{n}", c, feats, px, sectors, top_k, n_trials, end=DEV_END)
               for n, c in CANDIDATES.items()}
        dev["sue_only"] = run(f"{u}:sue_only", SUE_ONLY, feats, px, sectors, top_k, n_trials, end=DEV_END)
        eligible = [n for n in CANDIDATES if n != REFERENCE]
        selected = max(eligible, key=lambda n: dev[n].summary()["ic_mean"])
        logger.info("%s: selected %s", u, selected)

        cfg = WalkForwardConfig(horizon=21, rebalance_every=21, execution_lag=1, train_window=504,
                                min_train_dates=252, top_k=top_k, n_random_paths=0, end=DEV_END)
        factor_ic = {}
        for col in EARNINGS_FEATURES:
            s = run_walk_forward(feats["earnings"], px, lambda c=col: FeatureScoreModel(c), cfg,
                                 benchmark=BENCHMARK, name=col).summary()
            factor_ic[col] = {"ic_mean": s["ic_mean"], "ic_t_nw": s["ic_t_nw"]}

        holdout = {n: run(f"{u}:{n}:holdout", CANDIDATES[n], feats, px, sectors, top_k, n_trials,
                          start=HOLDOUT_START) for n in dict.fromkeys([selected, REFERENCE])}
        holdout["sue_only"] = run(f"{u}:sue_only:holdout", SUE_ONLY, feats, px, sectors, top_k,
                                  n_trials, start=HOLDOUT_START)

        report["dev"][u] = {n: r.summary() for n, r in dev.items()}
        report["holdout"][u] = {n: r.summary() for n, r in holdout.items()}
        report["selected"][u] = selected
        report["factor_ic"][u] = factor_ic
        fac = ["| Feature | Mean IC | t (NW) |", "| --- | ---: | ---: |"] + [
            f"| `{k}` | {v['ic_mean']:+.4f} | {v['ic_t_nw']:+.2f} |" for k, v in factor_ic.items()]
        sections += [
            f"## {u.capitalize()} universe ({len(names)} stocks, top-{top_k} portfolios)", "",
            f"Rows with a live (non-stale) surprise: {coverage:.0%}.", "",
            f"### Development period (test dates through {DEV_END})", "", table(dev), "",
            f"Selected by development-period mean IC (excluding the reference): **`{selected}`**.", "",
            f"### Holdout (test dates from {HOLDOUT_START}, run once)", "", table(holdout), "",
            "### Single-feature ICs, development period (diagnostic only)", "", *fac, "",
        ]

    if "broad" in args.universes:
        broad = [t for t in sorted(members["ticker"]) if t in close.columns]
        sections += event_time_check(events, close, broad)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Earnings-surprise features: round 2 of the feature research", "",
        f"Generated {datetime.now():%Y-%m-%d} by `scripts/earnings_feature_research.py`.", "",
        "Pre-registered protocol as in [feature_research.md](feature_research.md): fixed candidates, "
        f"selection by mean IC on test dates through {DEV_END}, one holdout run from {HOLDOUT_START}. "
        f"Deflated probabilities assume {n_trials} configurations (audit + both rounds).", "",
        "Earnings inputs are point-in-time: time-series SUE from quarterly EPS history (no analyst "
        "estimates), announcement dates inferred from SEC 8-K/10-Q filings and usable two trading "
        "days after filing. EPS histories are a current vintage, so rare restatements are a "
        "residual look-ahead risk.", "",
        "Candidates:", "", *[f"* `{n}`: {c.description}" for n, c in CANDIDATES.items()],
        "* `sue_only`: SUE alone, no model (baseline)", "",
    ]
    out.with_suffix(".md").write_text("\n".join(header + sections), encoding="utf-8")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        return None if isinstance(obj, float) and not np.isfinite(obj) else obj

    report.update({"generated": datetime.now().isoformat(timespec="seconds"), "n_trials": n_trials})
    out.with_suffix(".json").write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    logger.info("Wrote %s.md/.json", out)


if __name__ == "__main__":
    main()
