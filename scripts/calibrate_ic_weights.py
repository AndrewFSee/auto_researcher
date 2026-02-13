#!/usr/bin/env python3
"""
Empirical IC Calibration for Agent Weights.

Computes the historical Information Coefficient (Spearman rank correlation
between agent signal and forward returns) for each agent model using
existing backtest data.

Outputs:
    data/agent_ic.json  — Per-agent IC stats used by compute_composite_scores

Usage:
    python scripts/calibrate_ic_weights.py
    python scripts/calibrate_ic_weights.py --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "agent_ic.json"

# Forward return horizon to use for IC measurement (trading days)
HORIZON = "ret20d"  # ~1 month, consistent with ML model's 21-day target


# =============================================================================
# IC COMPUTATION HELPERS
# =============================================================================

def spearman_ic(predicted: pd.Series, realized: pd.Series) -> float:
    """Compute Spearman rank IC between predicted signal and realized returns."""
    common = predicted.index.intersection(realized.index)
    p = predicted.loc[common].dropna()
    r = realized.loc[common].dropna()
    common2 = p.index.intersection(r.index)
    p, r = p.loc[common2], r.loc[common2]
    if len(p) < 5:
        return np.nan
    corr, _ = stats.spearmanr(p, r)
    return corr


def cross_sectional_ic_series(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
    date_col: str,
) -> pd.Series:
    """
    Compute cross-sectional Spearman IC at each rebalance date.

    For each date, correlates the signal across tickers with forward returns.
    Returns a Series of ICs indexed by date.
    """
    ics = {}
    for date, group in df.groupby(date_col):
        sig = group[signal_col].dropna()
        ret = group[return_col].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 5:
            continue
        corr, _ = stats.spearmanr(sig.loc[common], ret.loc[common])
        if not np.isnan(corr):
            ics[date] = corr
    return pd.Series(ics).sort_index()


def pooled_ic(
    df: pd.DataFrame,
    signal_col: str,
    return_col: str,
) -> float:
    """Compute pooled (time-series) Spearman IC across all observations."""
    valid = df[[signal_col, return_col]].dropna()
    if len(valid) < 10:
        return np.nan
    corr, _ = stats.spearmanr(valid[signal_col], valid[return_col])
    return corr


def ic_stats(ic_series: pd.Series) -> dict:
    """Compute summary stats from a series of cross-sectional ICs."""
    if len(ic_series) == 0:
        return {"mean_ic": 0.0, "std_ic": 0.0, "ic_ir": 0.0, "hit_rate": 0.0, "n_periods": 0}
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_rate = float((ic_series > 0).mean())
    return {
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "ic_ir": round(ic_ir, 4),
        "hit_rate": round(hit_rate, 4),
        "n_periods": len(ic_series),
    }


# =============================================================================
# PER-AGENT IC CALIBRATION
# =============================================================================

def calibrate_earnings_ic(verbose: bool = False) -> dict:
    """
    Earnings/PEAD agent IC.

    Signal: SUE (standardized unexpected earnings)
    Return: ret20d (post-earnings drift, ~1 month)
    Source: pead_backtest_results.parquet
    """
    path = DATA_DIR / "pead_backtest_results.parquet"
    if not path.exists():
        logger.warning(f"Missing {path}, skipping earnings calibration")
        return {"mean_ic": 0.12, "source": "default"}

    df = pd.read_parquet(path)
    logger.info(f"Earnings/PEAD: {len(df)} observations, {df['ticker'].nunique()} tickers")

    # Use SUE as the signal — the core PEAD predictor
    ret_col = "ret20d" if "ret20d" in df.columns else "ret10d"
    ic_ser = cross_sectional_ic_series(df, "sue", ret_col, "quarter_date")
    result = ic_stats(ic_ser)
    result["signal"] = "sue"
    result["return_col"] = ret_col
    result["source"] = "pead_backtest_results.parquet"

    # Also compute pooled IC
    result["pooled_ic"] = round(pooled_ic(df, "sue", ret_col), 4)

    if verbose:
        logger.info(f"  Earnings IC: mean={result['mean_ic']:.4f}, IR={result['ic_ir']:.3f}, "
                     f"hit={result['hit_rate']:.1%}, n={result['n_periods']}")
    return result


def calibrate_sentiment_ic(verbose: bool = False) -> dict:
    """
    Sentiment agent IC.

    Signal: sentiment_mean (FinBERT news sentiment average)
    Return: ret5d or ret10d (shorter horizon for news momentum)
    Source: news_backtest_results.parquet or news_combined_results.parquet
    """
    # Prefer the combined results (more data)
    path = DATA_DIR / "news_combined_results.parquet"
    if not path.exists():
        path = DATA_DIR / "news_backtest_results.parquet"
    if not path.exists():
        logger.warning(f"Missing news backtest data, skipping sentiment calibration")
        return {"mean_ic": 0.08, "source": "default"}

    df = pd.read_parquet(path)
    logger.info(f"Sentiment: {len(df)} observations, {df['ticker'].nunique()} tickers")

    # Use sentiment_mean as signal, ret10d as return
    # News sentiment has a shorter alpha decay than PEAD
    ret_col = "ret10d" if "ret10d" in df.columns else "ret5d"
    signal_col = "sentiment_mean"

    ic_ser = cross_sectional_ic_series(df, signal_col, ret_col, "date")
    result = ic_stats(ic_ser)
    result["signal"] = signal_col
    result["return_col"] = ret_col
    result["source"] = path.name

    # Also try sentiment_momentum if available — captures trend changes
    if "sentiment_momentum" in df.columns:
        ic_mom = cross_sectional_ic_series(df, "sentiment_momentum", ret_col, "date")
        mom_stats = ic_stats(ic_mom)
        result["sentiment_momentum_ic"] = mom_stats["mean_ic"]

    result["pooled_ic"] = round(pooled_ic(df, signal_col, ret_col), 4)

    if verbose:
        logger.info(f"  Sentiment IC: mean={result['mean_ic']:.4f}, IR={result['ic_ir']:.3f}, "
                     f"hit={result['hit_rate']:.1%}, n={result['n_periods']}")
    return result


def calibrate_fundamental_ic(verbose: bool = False) -> dict:
    """
    Fundamental (quality+value) agent IC.

    Uses two sources:
    1. DCF backtest: dcf_gap signal vs ret20d
    2. Combined fundamentals: combined_weighted signal vs ret20d

    Takes the higher IC of the two (they measure different facets).
    """
    results = {}

    # DCF backtest
    dcf_path = DATA_DIR / "dcf_backtest_results.parquet"
    if dcf_path.exists():
        df_dcf = pd.read_parquet(dcf_path)
        # dcf_gap is the valuation signal (positive = undervalued)
        if "dcf_gap" in df_dcf.columns and "ret20d" in df_dcf.columns:
            valid = df_dcf[["dcf_gap", "ret20d"]].dropna()
            if len(valid) > 20:
                dcf_ic = float(stats.spearmanr(valid["dcf_gap"], valid["ret20d"])[0])
                results["dcf_pooled_ic"] = round(dcf_ic, 4)
                logger.info(f"  DCF pooled IC: {dcf_ic:.4f} ({len(valid)} obs)")

    # Combined fundamentals (revision + PEAD combined signal)
    combo_path = DATA_DIR / "combined_fundamentals_results.parquet"
    if combo_path.exists():
        df_combo = pd.read_parquet(combo_path)
        logger.info(f"Combined fundamentals: {len(df_combo)} observations, {df_combo['ticker'].nunique()} tickers")

        ret_col = "ret20d" if "ret20d" in df_combo.columns else "ret10d"
        signal_col = "combined_weighted" if "combined_weighted" in df_combo.columns else "combined_simple"

        if signal_col in df_combo.columns and ret_col in df_combo.columns:
            ic_ser = cross_sectional_ic_series(df_combo, signal_col, ret_col, "quarter")
            result = ic_stats(ic_ser)
            result["signal"] = signal_col
            result["return_col"] = ret_col
            result["source"] = "combined_fundamentals_results.parquet"
            result["pooled_ic"] = round(pooled_ic(df_combo, signal_col, ret_col), 4)
            results.update(result)

            if verbose:
                logger.info(f"  Fundamental IC: mean={result['mean_ic']:.4f}, IR={result['ic_ir']:.3f}, "
                             f"hit={result['hit_rate']:.1%}, n={result['n_periods']}")

    # Revision signal — estimate revisions predict returns
    rev_path = DATA_DIR / "revision_backtest_results.parquet"
    if rev_path.exists():
        df_rev = pd.read_parquet(rev_path)
        ret_col = "ret20d" if "ret20d" in df_rev.columns else "ret10d"
        signal_col = "revision_composite" if "revision_composite" in df_rev.columns else "revision_30d"

        if signal_col in df_rev.columns and ret_col in df_rev.columns:
            rev_pooled = pooled_ic(df_rev, signal_col, ret_col)
            results["revision_pooled_ic"] = round(rev_pooled, 4) if not np.isnan(rev_pooled) else 0.0
            logger.info(f"  Revision pooled IC: {results['revision_pooled_ic']:.4f} ({len(df_rev)} obs)")

    if not results or "mean_ic" not in results:
        return {"mean_ic": 0.10, "source": "default"}
    return results


def calibrate_insider_ic(verbose: bool = False) -> dict:
    """
    Insider agent IC.

    No dedicated insider backtest parquet exists with ticker-level signals.
    Use the default IC based on academic literature (Jeng, Metrick & Zeckhauser 2003;
    Lakonishok & Lee 2001) which finds IC ≈ 0.04-0.08 for insider signals.

    If ranking results exist with insider scores, compute a quick OOS estimate.
    """
    # Check if we have any ranking results with insider scores + enough time passed
    # for forward return computation
    result = {"mean_ic": 0.06, "source": "academic_literature"}

    # Try to get IC from existing ranking results
    ranking_dir = DATA_DIR / "ranking_results"
    if ranking_dir.exists():
        insider_scores = []
        for f in sorted(ranking_dir.glob("final_ranking_*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for item in data:
                    if "insider_score" in item and item.get("insider_score", 0) != 0:
                        insider_scores.append({
                            "ticker": item["ticker"],
                            "insider_score": item["insider_score"],
                            "date": f.stem.split("_")[-2],  # Extract date from filename
                        })
            except Exception:
                continue
        if insider_scores:
            result["n_nonzero_signals"] = len(insider_scores)
            logger.info(f"  Insider: {len(insider_scores)} non-zero signals found in ranking results")

    if verbose:
        logger.info(f"  Insider IC: {result['mean_ic']:.4f} (source={result['source']})")
    return result


def calibrate_thematic_ic(verbose: bool = False) -> dict:
    """
    Thematic/Early Adopter agent IC.

    Source: tech_backtest_results.csv (small dataset, 31 observations)
    Signal: tech_intensity
    Return: ret_21d
    """
    path = DATA_DIR / "tech_backtest_results.csv"
    if not path.exists():
        logger.warning(f"Missing {path}, using default thematic IC")
        return {"mean_ic": 0.05, "source": "default"}

    df = pd.read_csv(path)
    logger.info(f"Thematic: {len(df)} observations, {df['ticker'].nunique()} tickers")

    signal_col = "tech_intensity"
    ret_col = "ret_21d"

    if signal_col not in df.columns or ret_col not in df.columns:
        return {"mean_ic": 0.05, "source": "default"}

    valid = df[[signal_col, ret_col]].dropna()
    if len(valid) < 5:
        return {"mean_ic": 0.05, "source": "default", "n_obs": len(valid)}

    pooled = float(stats.spearmanr(valid[signal_col], valid[ret_col])[0])

    # Apply Bayesian shrinkage toward prior for small samples
    # Prior: IC ≈ 0.05 (generic factor IC). Shrink more with fewer observations.
    prior_ic = 0.05
    shrinkage_n = 100  # At n=100, weight evenly between empirical and prior
    empirical_weight = len(valid) / (len(valid) + shrinkage_n)
    shrunk_ic = empirical_weight * abs(pooled) + (1 - empirical_weight) * prior_ic

    result = {
        "mean_ic": round(shrunk_ic, 4),
        "raw_pooled_ic": round(pooled, 4),
        "shrunk_ic": round(shrunk_ic, 4),
        "n_obs": len(valid),
        "signal": signal_col,
        "return_col": ret_col,
        "source": "tech_backtest_results.csv",
        "note": f"Small sample (n={len(valid)}), Bayesian-shrunk toward prior={prior_ic}",
    }

    if verbose:
        logger.info(f"  Thematic IC: pooled={pooled:.4f}, n={len(valid)} (small sample)")
    return result


def calibrate_momentum_ic(verbose: bool = False) -> dict:
    """
    Sector Momentum agent IC.

    No existing backtest parquet. Compute from price data:
    - For each month, compute sector ETF relative strength vs SPY
    - Map stocks to sectors
    - Correlate sector momentum signal with stock forward returns

    Uses yfinance for historical sector ETF + stock prices.
    Falls back to academic literature estimates if data unavailable.
    """
    result = {
        "mean_ic": 0.07,
        "source": "academic_literature",
        "note": "Moskowitz & Grinblatt 1999: sector momentum IC ≈ 0.05-0.10",
    }

    try:
        import yfinance as yf
        from auto_researcher.models.sector_momentum import TICKER_SECTORS, SECTOR_ETFS, SECTOR_TO_ETF

        # Download sector ETF and SPY data
        # SECTOR_ETFS maps ETF ticker -> sector name (e.g. "XLK" -> "Technology")
        etf_tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
        logger.info(f"Momentum: downloading {len(etf_tickers)} ETF prices for IC calibration...")

        etf_data = yf.download(etf_tickers, period="3y", progress=False)
        if isinstance(etf_data.columns, pd.MultiIndex):
            prices = etf_data["Close"] if "Close" in etf_data.columns.get_level_values(0) else etf_data["Adj Close"]
        else:
            prices = etf_data

        if prices.empty or len(prices) < 252:
            logger.warning("Insufficient ETF price data for momentum calibration")
            return result

        # Compute monthly sector relative strength
        # For each month-end: 3-month sector return relative to SPY
        monthly = prices.resample("ME").last().dropna(how="all")
        spy_ret3m = monthly["SPY"].pct_change(3)

        # Get a set of stocks we have sector mappings for
        # Pick 30 tickers from TICKER_SECTORS for IC test
        test_tickers = list(TICKER_SECTORS.keys())[:40]
        logger.info(f"Momentum: downloading {len(test_tickers)} stock prices...")

        stock_data = yf.download(test_tickers, period="3y", progress=False)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_prices = stock_data["Close"] if "Close" in stock_data.columns.get_level_values(0) else stock_data["Adj Close"]
        else:
            stock_prices = stock_data

        stock_monthly = stock_prices.resample("ME").last().dropna(how="all")

        # Forward 1-month stock return
        stock_fwd_ret = stock_monthly.pct_change().shift(-1)

        # Build signal: for each stock at each month, look up its sector ETF's
        # relative strength vs SPY over trailing 3 months
        ics = []
        for date in stock_fwd_ret.index[3:-1]:  # Need 3m lookback and 1m forward
            signals = {}
            returns = {}
            for ticker in test_tickers:
                if ticker not in stock_fwd_ret.columns:
                    continue
                fwd = stock_fwd_ret.loc[date, ticker]
                if pd.isna(fwd):
                    continue

                sector = TICKER_SECTORS.get(ticker)
                if not sector or sector not in SECTOR_TO_ETF:
                    continue

                etf = SECTOR_TO_ETF[sector]
                if etf not in monthly.columns:
                    continue

                # Sector 3m return relative to SPY
                if date not in monthly.index:
                    continue
                date_idx = monthly.index.get_loc(date)
                if date_idx < 3:
                    continue

                etf_ret3m = (monthly[etf].iloc[date_idx] / monthly[etf].iloc[date_idx - 3]) - 1
                spy_ret = (monthly["SPY"].iloc[date_idx] / monthly["SPY"].iloc[date_idx - 3]) - 1
                rel_strength = etf_ret3m - spy_ret

                signals[ticker] = rel_strength
                returns[ticker] = fwd

            if len(signals) >= 8:
                sig = pd.Series(signals)
                ret = pd.Series(returns)
                corr, _ = stats.spearmanr(sig, ret)
                if not np.isnan(corr):
                    ics.append(corr)

        if len(ics) >= 6:
            ic_ser = pd.Series(ics)
            result = ic_stats(ic_ser)
            result["source"] = "empirical_sector_etf_replay"
            result["signal"] = "sector_3m_relative_strength"
            result["return_col"] = "stock_1m_forward"

        if verbose:
            logger.info(f"  Momentum IC: mean={result.get('mean_ic', 0):.4f}, "
                         f"n={result.get('n_periods', 0)} (source={result.get('source', 'n/a')})")

    except Exception as e:
        logger.warning(f"Momentum calibration failed: {e}, using academic default")

    return result


def calibrate_filing_tone_ic(verbose: bool = False) -> dict:
    """
    Filing Tone (10-K YoY tone change) agent IC.

    No dedicated filing tone backtest parquet exists.
    Use academic literature priors:
    - Loughran & McDonald (2011): IC ≈ 0.03-0.06 for tone-based signals
    - Li (2010): MD&A tone change predicts earnings, ~0.04 IC
    - Feldman et al. (2010): YoY tone change → ~1.5% abnormal return

    Since this is a seasonal signal (10-K filed annually), the effective
    IC is lower than raw because signal decays over months.
    We use 0.04 as a conservative prior.
    """
    result = {
        "mean_ic": 0.04,
        "source": "academic_literature",
        "note": "Loughran-McDonald 10-K tone change; seasonal annual signal with decay",
    }

    # Check if we have ranking results with filing_tone scores
    ranking_dir = DATA_DIR / "ranking_results"
    if ranking_dir.exists():
        tone_scores = []
        for f in sorted(ranking_dir.glob("final_ranking_*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for item in data:
                    if "filing_tone_score" in item and item.get("filing_tone_score", 0) != 0:
                        tone_scores.append({
                            "ticker": item["ticker"],
                            "filing_tone_score": item["filing_tone_score"],
                            "date": f.stem.split("_")[-2],
                        })
            except Exception:
                continue
        if tone_scores:
            result["n_nonzero_signals"] = len(tone_scores)
            logger.info(f"  FilingTone: {len(tone_scores)} non-zero signals found in ranking results")

    if verbose:
        logger.info(f"  FilingTone IC: {result['mean_ic']:.4f} (source={result['source']})")
    return result


def calibrate_earnings_call_qual_ic(verbose: bool = False) -> dict:
    """
    Earnings Call Qualitative Analysis agent IC.

    No dedicated earnings call qualitative backtest parquet exists.
    Use academic literature priors:
    - Brockman & Cicon (2013): Management tone in calls predicts returns, IC ≈ 0.04-0.06
    - Price et al. (2012): Q&A section tone more informative than prepared remarks
    - Matsumoto et al. (2011): Analyst-management tone gap predicts earnings quality
    - Davis et al. (2015): QoQ tone change is more informative than level

    This model combines 5 sub-signals (mgmt tone, tone gap, hedging,
    guidance specificity, QoQ change), so we use 0.05 as a moderate prior.
    """
    result = {
        "mean_ic": 0.05,
        "source": "academic_literature",
        "note": "Earnings call qualitative (FinBERT tone + hedging + guidance); multi-signal composite",
    }

    # Check if we have ranking results with call_qual scores
    ranking_dir = DATA_DIR / "ranking_results"
    if ranking_dir.exists():
        qual_scores = []
        for f in sorted(ranking_dir.glob("final_ranking_*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                for item in data:
                    if "earnings_call_qual_score" in item and item.get("earnings_call_qual_score", 0) != 0:
                        qual_scores.append({
                            "ticker": item["ticker"],
                            "earnings_call_qual_score": item["earnings_call_qual_score"],
                            "date": f.stem.split("_")[-2],
                        })
            except Exception:
                continue
        if qual_scores:
            result["n_nonzero_signals"] = len(qual_scores)
            logger.info(f"  EarningsCallQual: {len(qual_scores)} non-zero signals found in ranking results")

    if verbose:
        logger.info(f"  EarningsCallQual IC: {result['mean_ic']:.4f} (source={result['source']})")
    return result


def calibrate_ml_ic(verbose: bool = False) -> dict:
    """
    ML (XGBoost) model IC.

    This is already computed during training in recommend.py as historical_ic.
    We just record the default for reference.

    The actual IC is set dynamically each run by the ML training process.
    """
    result = {
        "mean_ic": 0.15,
        "source": "computed_at_runtime",
        "note": "ML IC is computed OOS during each training run; this is the fallback default",
    }
    if verbose:
        logger.info(f"  ML IC: {result['mean_ic']:.4f} (computed at runtime)")
    return result


# =============================================================================
# MAIN CALIBRATION RUNNER
# =============================================================================

def run_calibration(verbose: bool = False) -> dict:
    """Run IC calibration for all agents and return results dict."""
    logger.info("=" * 60)
    logger.info("EMPIRICAL IC CALIBRATION")
    logger.info("=" * 60)

    results = {}

    # 1. ML (runtime IC, just record default)
    logger.info("\n[1/9] ML Model...")
    results["ml"] = calibrate_ml_ic(verbose)

    # 2. Earnings/PEAD
    logger.info("\n[2/9] Earnings/PEAD...")
    results["earnings"] = calibrate_earnings_ic(verbose)

    # 3. Sentiment
    logger.info("\n[3/9] Sentiment...")
    results["sentiment"] = calibrate_sentiment_ic(verbose)

    # 4. Fundamental
    logger.info("\n[4/9] Fundamental (quality+value)...")
    results["fundamental"] = calibrate_fundamental_ic(verbose)

    # 5. Insider
    logger.info("\n[5/9] Insider...")
    results["insider"] = calibrate_insider_ic(verbose)

    # 6. Thematic
    logger.info("\n[6/9] Thematic/Early Adopter...")
    results["thematic"] = calibrate_thematic_ic(verbose)

    # 7. Momentum
    logger.info("\n[7/9] Sector Momentum...")
    results["momentum"] = calibrate_momentum_ic(verbose)

    # 8. Filing Tone
    logger.info("\n[8/9] Filing Tone (10-K)...")
    results["filing_tone"] = calibrate_filing_tone_ic(verbose)

    # 9. Earnings Call Qualitative
    logger.info("\n[9/9] Earnings Call Qualitative...")
    results["earnings_call_qual"] = calibrate_earnings_call_qual_ic(verbose)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("IC CALIBRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Agent':<15} {'IC':>8} {'IC-IR':>8} {'Hit%':>8} {'N':>6} {'Source'}")
    logger.info("-" * 60)

    # Build simplified weight dict for the pipeline
    ic_weights = {}
    for agent, stats_dict in results.items():
        ic = stats_dict.get("mean_ic", 0.0)
        ir = stats_dict.get("ic_ir", 0.0)
        hit = stats_dict.get("hit_rate", 0.0)
        n = stats_dict.get("n_periods", stats_dict.get("n_obs", 0))
        src = stats_dict.get("source", "unknown")
        logger.info(f"  {agent:<13} {ic:>8.4f} {ir:>8.3f} {hit:>7.1%} {n:>6} {src}")
        ic_weights[agent] = abs(ic)  # Use absolute IC for weighting

    results["_ic_weights"] = ic_weights
    results["_calibrated_at"] = pd.Timestamp.now().isoformat()
    results["_horizon"] = HORIZON

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibrate IC weights for agent models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed IC stats")
    parser.add_argument("--output", "-o", type=str, default=str(OUTPUT_FILE), help="Output JSON file")
    args = parser.parse_args()

    results = run_calibration(verbose=args.verbose)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")

    # Print the IC weights for easy copy-paste
    ic_weights = results["_ic_weights"]
    total = sum(ic_weights.values())
    logger.info("\nNormalized IC weights for pipeline:")
    for agent, ic in sorted(ic_weights.items(), key=lambda x: -x[1]):
        pct = ic / total * 100
        logger.info(f"  {agent}: IC={ic:.4f} -> weight={pct:.1f}%")


if __name__ == "__main__":
    main()
