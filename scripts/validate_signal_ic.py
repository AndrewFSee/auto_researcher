"""
Validate a scalar alpha signal against forward returns.

Use this script to replace hardcoded IC claims (e.g. PEAD's +0.152,
Early Adopter's +0.36) with honest, walk-forward statistics.

Two input modes
---------------

**Cross-sectional mode** (``--signal`` + ``--prices``):

    A parquet / CSV with either a MultiIndex ``(date, ticker)`` and a
    single column, or columns ``['date', 'ticker', 'score']``. Forward
    returns are computed from a prices CSV at ``--horizon`` days.
    Reports per-date Spearman IC, Newey-West-adjusted t-stat, and
    deflated Sharpe.

**Event mode** (``--dataset NAME`` or ``--event-parquet``):

    A parquet where each row already carries (signal, realized
    forward return, event date) — the format produced by the project's
    pre-computed PEAD / Early-Adopter backtests. We compute pooled
    Spearman IC, plus walk-forward IC by calendar year and report
    Newey-West-adjusted t-stat across folds.

Pre-registered datasets
-----------------------

    pead_base         data/pead_backtest_results.parquet, signal=sue, ret40d
    pead_enhanced     same, |SUE|>=0.20 only — replicates the +0.152 claim
    pead_enhanced_60d same, 60-day horizon

Outputs
-------
JSON to ``results/validate_signal_ic_<name>.json``. When using
``--all-presets``, also writes a roll-up summary to
``results/validate_signal_ic_summary.json``.

Usage
-----

    # Re-measure all the hardcoded claims at once.
    python scripts/validate_signal_ic.py --all-presets

    # Just one preset.
    python scripts/validate_signal_ic.py --dataset pead_enhanced

    # Custom event-style parquet.
    python scripts/validate_signal_ic.py --event-parquet data/some_signal.parquet \\
        --signal-col score --return-col fwd60d --date-col timestamp \\
        --horizon-days 60 --label custom_signal

    # Cross-sectional mode (original behavior).
    python scripts/validate_signal_ic.py --signal results/early_adopter.parquet \\
        --prices data/prices.csv --name early_adopter --horizon 21
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
DEFAULT_PRICES = PROJECT_ROOT / "data" / "prices.csv"
OUT_DIR = PROJECT_ROOT / "results" / "signal_ic"
EVENT_OUT_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Event-mode dataset registry
# ---------------------------------------------------------------------------

@dataclass
class EventDatasetSpec:
    """Maps a friendly --dataset name to a concrete (file, cols, filter)."""

    parquet: Path
    signal_col: str
    return_col: str
    date_col: str
    horizon_days: int
    note: str
    filter_expr: str | None = None  # pandas.query() expression


PRESETS: dict[str, EventDatasetSpec] = {
    "pead_base": EventDatasetSpec(
        parquet=PROJECT_ROOT / "data" / "pead_backtest_results.parquet",
        signal_col="sue",
        return_col="ret40d",
        date_col="quarter_date",
        horizon_days=40,
        note="All earnings events; signal=SUE, horizon=40 trading days.",
    ),
    "pead_enhanced": EventDatasetSpec(
        parquet=PROJECT_ROOT / "data" / "pead_backtest_results.parquet",
        signal_col="sue",
        return_col="ret40d",
        date_col="quarter_date",
        horizon_days=40,
        note="Big-surprise events (|SUE| >= 0.20); replicates the +0.152 claim.",
        filter_expr="abs(sue) >= 0.20",
    ),
    "pead_enhanced_60d": EventDatasetSpec(
        parquet=PROJECT_ROOT / "data" / "pead_backtest_results.parquet",
        signal_col="sue",
        return_col="ret60d",
        date_col="quarter_date",
        horizon_days=60,
        note="Big-surprise events only (|SUE| >= 0.20), 60-day horizon.",
        filter_expr="abs(sue) >= 0.20",
    ),
}


# Hardcoded IC claims in the codebase — used for verdict text only.
EVENT_CLAIMS: dict[str, dict] = {
    "pead_base":         {"ic": 0.049, "p": 0.029, "n": 1948, "source": "pead_enhanced.py:79"},
    "pead_enhanced":     {"ic": 0.152, "p": 0.006, "n": 334,  "source": "pead_enhanced.py:88"},
    "pead_enhanced_60d": {"ic": 0.138, "p": 0.012, "n": 334,  "source": "pead_enhanced.py:89"},
}


def load_signal(path: Path) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if isinstance(df.index, pd.MultiIndex) and df.shape[1] == 1:
        df = df.reset_index()
        df.columns = ["date", "ticker", "score"]

    required = {"date", "ticker", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Signal file {path} missing required columns: {missing}. "
            "Expected long-format with date/ticker/score."
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["score"]).sort_values(["date", "ticker"])
    return df


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    return prices


def compute_forward_returns(
    prices: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
    """
    Forward total return from close(t) to close(t+horizon). Returns a panel
    with the same index/columns as ``prices``. The last ``horizon_days`` rows
    are NaN by construction.
    """
    return prices.shift(-horizon_days) / prices - 1.0


def merge_signal_with_returns(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame
) -> pd.DataFrame:
    """
    Long-format merge of ``(date, ticker, score)`` with realized forward
    returns. Drops rows where fwd_ret is NaN (at the tail of the series).
    """
    fwd_long = fwd_ret.stack().rename("fwd_ret").reset_index()
    fwd_long.columns = ["date", "ticker", "fwd_ret"]
    merged = signal.merge(fwd_long, on=["date", "ticker"], how="inner")
    return merged.dropna(subset=["score", "fwd_ret"])


def spearman_per_date(merged: pd.DataFrame) -> pd.Series:
    """Per-date cross-sectional Spearman IC."""
    from scipy.stats import spearmanr

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 5:
            return np.nan
        ic, _ = spearmanr(group["score"], group["fwd_ret"])
        return ic if np.isfinite(ic) else np.nan

    return merged.groupby("date").apply(_ic).dropna()


def newey_west_tstat(series: pd.Series, lag: int) -> tuple[float, float]:
    """
    Newey-West-adjusted t-stat of ``series.mean()`` under autocorrelation.
    When ``lag`` <= 0 this reduces to the plain t-stat ``mean / (std/sqrt(N))``.

    Returns (t_stat, two_sided_p).
    """
    from scipy.stats import norm

    x = series.dropna().to_numpy()
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")

    x_bar = x.mean()
    u = x - x_bar
    gamma0 = np.dot(u, u) / n
    lrv = gamma0
    for k in range(1, max(int(lag), 0) + 1):
        if k >= n:
            break
        weight = 1.0 - k / (lag + 1)
        cov = np.dot(u[k:], u[:-k]) / n
        lrv += 2.0 * weight * cov

    lrv = max(lrv, 1e-12)
    t = x_bar / math.sqrt(lrv / n)
    p = 2.0 * (1.0 - norm.cdf(abs(t)))
    return float(t), float(p)


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    """
    Bailey / López de Prado deflated Sharpe ratio.

    Penalizes the observed Sharpe for the selection bias from trying many
    strategies. Returns the probability that the deflated Sharpe > 0.
    """
    from scipy.stats import norm

    if n_trials <= 1 or n_obs < 2:
        return float("nan")

    # Expected max Sharpe under N(0,1) after ``n_trials`` draws.
    euler_mascheroni = 0.5772156649
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    expected_max = (1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2

    # Approximate Sharpe variance under normal returns, ignoring skew/kurt.
    sharpe_var = (1.0 + 0.5 * sharpe ** 2) / (n_obs - 1.0)
    sharpe_sd = math.sqrt(max(sharpe_var, 1e-12))

    deflated = (sharpe - expected_max * sharpe_sd) / sharpe_sd
    return float(norm.cdf(deflated))


def save_rolling_ic_plot(
    ic_series: pd.Series, out_path: Path, window: int = 252
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping rolling IC plot")
        return

    rolling = ic_series.rolling(window, min_periods=window // 4).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling.index, rolling.values, color="C0", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"Rolling {window}-day mean IC")
    ax.set_ylabel("IC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def validate(
    signal_df: pd.DataFrame,
    prices: pd.DataFrame,
    name: str,
    horizon_days: int,
    n_trials: int,
    out_dir: Path,
) -> dict:
    fwd_ret = compute_forward_returns(prices, horizon_days)
    merged = merge_signal_with_returns(signal_df, fwd_ret)
    if merged.empty:
        raise RuntimeError(
            f"No overlap between signal ({len(signal_df)} rows) and forward "
            f"returns. Check date alignment and ticker overlap."
        )

    ic_series = spearman_per_date(merged)
    n_dates = int(len(ic_series))
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std(ddof=1))
    pct_pos = float((ic_series > 0).mean())

    # Newey-West with lag = horizon_days-1 (IC samples autocorrelate because
    # forward returns overlap across consecutive dates).
    t_stat, p_val = newey_west_tstat(ic_series, lag=max(horizon_days - 1, 0))

    ic_sharpe = mean_ic / std_ic if std_ic > 0 else float("nan")
    ic_sharpe_annualized = ic_sharpe * math.sqrt(252) if np.isfinite(ic_sharpe) else float("nan")
    dsr = deflated_sharpe(ic_sharpe, n_trials, n_dates)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{name}_rolling.png"
    save_rolling_ic_plot(ic_series, plot_path)

    report = {
        "name": name,
        "n_observations": int(len(merged)),
        "n_dates": n_dates,
        "horizon_days": int(horizon_days),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "pct_ic_positive": pct_pos,
        "t_stat_newey_west": t_stat,
        "p_value_two_sided": p_val,
        "ic_sharpe": ic_sharpe,
        "ic_sharpe_annualized": ic_sharpe_annualized,
        "deflated_sharpe_prob_positive": dsr,
        "n_trials_for_deflation": int(n_trials),
        "rolling_ic_plot": str(plot_path),
    }

    report_path = out_dir / f"{name}.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info("IC validation for '%s' written to %s", name, report_path)
    print(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Event-mode: signal + realized return already aligned per row
# ---------------------------------------------------------------------------

ALLOW_PERIOD_END_DATES = False  # set by --allow-period-end-dates (demonstration only)


def _load_event(spec: EventDatasetSpec) -> pd.DataFrame:
    from auto_researcher.validation.event_dates import assert_announcement_dates

    df = pd.read_parquet(spec.parquet)
    df = df.dropna(subset=[spec.signal_col, spec.return_col, spec.date_col]).copy()
    df[spec.date_col] = pd.to_datetime(df[spec.date_col])
    if not ALLOW_PERIOD_END_DATES:
        # The PEAD presets below are keyed on fiscal quarter ends; their
        # forward returns contain the announcement. This refuses them.
        assert_announcement_dates(df[spec.date_col], name=f"{spec.parquet.name}:{spec.date_col}")
    if spec.filter_expr:
        before = len(df)
        df = df.query(spec.filter_expr).copy()
        print(f"  filter '{spec.filter_expr}': {before:,} -> {len(df):,} rows")
    return df.sort_values(spec.date_col).reset_index(drop=True)


def _per_year_ic(
    df: pd.DataFrame, signal_col: str, return_col: str, date_col: str
) -> pd.DataFrame:
    """Spearman IC per calendar year — one walk-forward fold per year."""
    from scipy.stats import spearmanr

    rows = []
    for year, grp in df.groupby(df[date_col].dt.year):
        if len(grp) < 30:
            continue
        ic, p = spearmanr(grp[signal_col], grp[return_col])
        rows.append({
            "year": int(year),
            "n": int(len(grp)),
            "ic": float(ic) if np.isfinite(ic) else np.nan,
            "p_within_year": float(p) if np.isfinite(p) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _ls_decile_spread(df: pd.DataFrame, signal_col: str, return_col: str) -> dict:
    """Top-decile minus bottom-decile mean return on the pooled sample."""
    n = len(df)
    if n < 50:
        return {"n_top": 0, "n_bot": 0, "mean_top": None, "mean_bot": None,
                "spread": None, "spread_bps": None}
    top_q = df[signal_col].quantile(0.90)
    bot_q = df[signal_col].quantile(0.10)
    top = df[df[signal_col] >= top_q][return_col]
    bot = df[df[signal_col] <= bot_q][return_col]
    spread = float(top.mean() - bot.mean())
    return {
        "n_top": int(len(top)),
        "n_bot": int(len(bot)),
        "mean_top": float(top.mean()),
        "mean_bot": float(bot.mean()),
        "spread": spread,
        "spread_bps": float(1e4 * spread),
    }


def validate_event(spec: EventDatasetSpec, label: str) -> dict:
    from scipy.stats import spearmanr, t as t_dist

    print(f"\n=== validating: {label} ===")
    print(f"  parquet:   {spec.parquet.name}")
    print(f"  signal:    {spec.signal_col}")
    print(f"  return:    {spec.return_col} (h={spec.horizon_days}d)")
    if spec.filter_expr:
        print(f"  filter:    {spec.filter_expr}")
    print(f"  note:      {spec.note}")

    df = _load_event(spec)
    print(f"  n events:  {len(df):,}  "
          f"({df[spec.date_col].min().date()} -> {df[spec.date_col].max().date()})")

    # Pooled IC — biased across regimes but useful as a sanity check.
    pooled_ic, pooled_p = spearmanr(df[spec.signal_col], df[spec.return_col])

    yearly = _per_year_ic(df, spec.signal_col, spec.return_col, spec.date_col)
    if yearly.empty or len(yearly) < 2:
        raise SystemExit(
            f"Need >=2 yearly folds with n>=30 to compute walk-forward IC for {label}; "
            f"only got {len(yearly)}"
        )

    ics = yearly["ic"].dropna().to_numpy()
    n_folds = len(ics)
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1))

    # Naive t-stat assumes folds are iid.
    t_naive = float(mean_ic / (std_ic / np.sqrt(n_folds))) if std_ic > 0 else float("nan")
    p_naive = float(2 * (1 - t_dist.cdf(abs(t_naive), df=n_folds - 1))) \
        if np.isfinite(t_naive) else float("nan")

    # Newey-West HAC SE on the per-year IC series — yearly folds with h<=60d
    # mean lag=1 catches any cross-fold overlap; bump for longer horizons.
    nw_lag = max(1, int(np.ceil(spec.horizon_days / 252.0)))
    ic_series = pd.Series(ics)
    t_nw, p_nw = newey_west_tstat(ic_series, lag=nw_lag)

    ls = _ls_decile_spread(df, spec.signal_col, spec.return_col)

    print()
    print(f"  Pooled Spearman IC:        {pooled_ic:+.4f}  (p={pooled_p:.4g})  [biased — diagnostic only]")
    print(f"  Walk-forward folds (yrs):  {n_folds}")
    print(f"    mean IC:                 {mean_ic:+.4f}")
    print(f"    std IC:                  {std_ic:.4f}")
    print(f"    t-stat (naive):          {t_naive:+.3f}    p={p_naive:.4g}")
    print(f"    t-stat (Newey-West, l={nw_lag}): {t_nw:+.3f}    p={p_nw:.4g}")
    if ls["spread_bps"] is not None:
        print(f"  L/S decile spread:         {ls['spread_bps']:+.1f} bps  "
              f"({ls['mean_top']*1e4:+.1f} top / {ls['mean_bot']*1e4:+.1f} bot)")
    print("  Per-year IC:")
    for _, r in yearly.iterrows():
        print(f"    {int(r['year'])}  n={int(r['n']):>6d}  IC={r['ic']:+.4f}  "
              f"p={r['p_within_year']:.3g}")

    claim = EVENT_CLAIMS.get(label)
    verdict = None
    if claim:
        # Verdict taxonomy:
        #   INSIGNIFICANT — measured NW p >= 0.10 (the claim is unsupported)
        #   SIGN_FLIPPED  — measured significant but in the opposite direction
        #   REPLICATES    — within 0.05 of the claim, same sign, significant
        #   STRONGER      — same sign, |measured| > |claim| + 0.05, significant
        #   WEAKER        — same sign, |measured| < |claim| - 0.05, significant
        if not (np.isfinite(p_nw) and p_nw < 0.10):
            verdict = "INSIGNIFICANT"
        elif np.sign(mean_ic) != np.sign(claim["ic"]):
            verdict = "SIGN_FLIPPED"
        elif abs(mean_ic - claim["ic"]) <= 0.05:
            verdict = "REPLICATES"
        elif abs(mean_ic) > abs(claim["ic"]):
            verdict = "STRONGER"
        else:
            verdict = "WEAKER"
        print()
        print(f"  Claim ({claim['source']}):  IC = {claim['ic']:+.3f}  "
              f"(claim p={claim['p']}, claim n={claim['n']})")
        print(f"  Measured:                IC = {mean_ic:+.3f}  "
              f"(NW p={p_nw:.4g}, n_folds={n_folds})  -> {verdict}")

    return {
        "label": label,
        "mode": "event",
        "spec": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(spec).items()},
        "n_events": int(len(df)),
        "date_range": [str(df[spec.date_col].min().date()),
                       str(df[spec.date_col].max().date())],
        "pooled_ic": float(pooled_ic) if np.isfinite(pooled_ic) else None,
        "pooled_p": float(pooled_p) if np.isfinite(pooled_p) else None,
        "walk_forward": {
            "n_folds": n_folds,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_naive": t_naive,
            "p_naive": p_naive,
            "t_newey_west": t_nw,
            "p_newey_west": p_nw,
            "newey_west_lag": nw_lag,
            "per_year": yearly.to_dict("records"),
        },
        "ls_spread": ls,
        "claim": claim,
        "verdict": verdict,
    }


def _build_custom_event_spec(args: argparse.Namespace) -> EventDatasetSpec:
    if not (args.signal_col and args.return_col):
        raise SystemExit("--event-parquet requires --signal-col and --return-col")
    return EventDatasetSpec(
        parquet=args.event_parquet,
        signal_col=args.signal_col,
        return_col=args.return_col,
        date_col=args.date_col or "date",
        horizon_days=args.horizon_days,
        note=f"custom: {args.event_parquet.name}",
        filter_expr=args.filter_expr,
    )


def _run_event_labels(labels: list[str]) -> dict:
    summaries: dict = {}
    for label in labels:
        spec = PRESETS[label]
        if not spec.parquet.exists():
            print(f"\n[skip] {label}: parquet not found at {spec.parquet}")
            continue
        summaries[label] = validate_event(spec, label)
        out = EVENT_OUT_DIR / f"validate_signal_ic_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries[label], indent=2, default=str))
        print(f"  wrote {out}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Event-mode args
    parser.add_argument("--dataset", choices=sorted(PRESETS.keys()),
                        help="Pre-registered event-style dataset.")
    parser.add_argument("--all-presets", action="store_true",
                        help="Run every event-mode preset.")
    parser.add_argument("--allow-period-end-dates", action="store_true",
                        help="Skip the fiscal-period-end check (reproduces the "
                             "look-ahead-contaminated PEAD numbers; do not cite).")
    parser.add_argument("--event-parquet", type=Path,
                        help="Custom event-mode parquet path.")
    parser.add_argument("--signal-col", help="Signal column (event mode).")
    parser.add_argument("--return-col", help="Realized forward return column (event mode).")
    parser.add_argument("--date-col", help="Date column (event mode).")
    parser.add_argument("--horizon-days", type=int, default=40,
                        help="Forward-return horizon in trading days (event mode).")
    parser.add_argument("--filter", dest="filter_expr",
                        help="Optional pandas .query() filter (event mode).")
    parser.add_argument("--label", help="Output label (event mode).")

    # Cross-sectional mode args (legacy)
    parser.add_argument("--signal", type=Path,
                        help="Cross-sectional signal parquet/csv (date,ticker,score).")
    parser.add_argument("--name", type=str,
                        help="Output label for cross-sectional mode.")
    parser.add_argument("--horizon", type=int, default=21,
                        help="Forward-return horizon (cross-sectional mode).")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES,
                        help="Prices CSV (cross-sectional mode).")
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help="Configurations tried when picking this signal "
             "(deflated-Sharpe adjustment).",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    global ALLOW_PERIOD_END_DATES
    ALLOW_PERIOD_END_DATES = args.allow_period_end_dates

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Dispatch.
    if args.all_presets or args.dataset or args.event_parquet:
        if args.all_presets:
            labels = list(PRESETS.keys())
        elif args.dataset:
            labels = [args.dataset]
        else:
            label = args.label or args.event_parquet.stem
            PRESETS[label] = _build_custom_event_spec(args)
            labels = [label]
        from auto_researcher.validation.event_dates import PeriodEndDateError

        try:
            summaries = _run_event_labels(labels)
        except PeriodEndDateError as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        if len(summaries) > 1:
            roll = EVENT_OUT_DIR / "validate_signal_ic_summary.json"
            roll.write_text(json.dumps(summaries, indent=2, default=str))
            print(f"\n=== summary across {len(summaries)} signals -> {roll}")
            for label, s in summaries.items():
                wf = s["walk_forward"]
                claim_ic = (s.get("claim") or {}).get("ic", "—")
                print(f"  {label:25s}  claim={claim_ic}  measured={wf['mean_ic']:+.3f}  "
                      f"NW p={wf['p_newey_west']:.4g}  ({s['verdict']})")
        return

    # Cross-sectional fallback.
    if args.signal is None or args.name is None:
        parser.error(
            "Pick a mode: --dataset NAME / --all-presets / --event-parquet ... "
            "(event mode) OR --signal ... --name ... (cross-sectional mode)"
        )

    signal_df = load_signal(args.signal)
    prices = load_prices(args.prices)
    validate(
        signal_df=signal_df,
        prices=prices,
        name=args.name,
        horizon_days=args.horizon,
        n_trials=args.n_trials,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
