"""
Performance Tracker — measure realised returns for previous pipeline runs.

Given a list of ranked stocks and the date the pipeline was run, fetches
historical prices and computes returns from run-date to present (or a
user-specified end date).  Also computes SPY benchmark return over the
same window so we can measure alpha.

Key metrics produced per stock:
  - total_return_pct        (close-to-close)
  - excess_return_pct       (stock return − SPY return)
  - signal_correct           True if buy→positive or sell→negative

Aggregate metrics:
  - hit_rate                 fraction of correct signals
  - avg_return_by_signal     mean return per signal bucket
  - portfolio_return         equal-weight return of all stocks
  - spy_return               benchmark over same window
  - alpha                    portfolio_return − spy_return
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StockPerformance:
    """Realised performance for a single stock."""
    ticker: str
    signal: str
    composite_score: float
    final_rank: int
    company_name: str
    sector: str
    # Prices
    price_at_run: float | None = None
    price_current: float | None = None
    # Returns
    total_return_pct: float | None = None
    excess_return_pct: float | None = None
    # Validation
    signal_correct: bool | None = None
    # Daily price series (dates → prices) for sparklines
    price_series: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PerformanceSummary:
    """Aggregate performance metrics for a full pipeline run."""
    run_date: str
    end_date: str
    trading_days: int
    calendar_days: int
    # Per-stock
    stocks: list[StockPerformance] = field(default_factory=list)
    # Benchmark
    spy_return_pct: float | None = None
    spy_price_at_run: float | None = None
    spy_price_current: float | None = None
    # Aggregates
    portfolio_return_pct: float | None = None
    alpha_pct: float | None = None
    hit_rate: float | None = None
    correct_count: int = 0
    total_actionable: int = 0
    avg_return_by_signal: dict[str, float] = field(default_factory=dict)
    median_return_by_signal: dict[str, float] = field(default_factory=dict)
    best_performer: StockPerformance | None = None
    worst_performer: StockPerformance | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_run_date_from_filename(filename: str) -> datetime | None:
    """Extract datetime from filenames like final_ranking_sp100_20260203_0856.json."""
    m = re.search(r"(\d{8})_(\d{4})", filename)
    if m:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M")
    m = re.search(r"(\d{8})", filename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    return None


def _next_trading_day(dt: datetime) -> str:
    """Return the date string (YYYY-MM-DD) for dt, advancing past weekends."""
    d = dt.date() if hasattr(dt, "date") else dt
    # If run was after market close (typically 4pm ET), use next day
    if hasattr(dt, "hour") and dt.hour >= 16:
        d += timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def compute_performance(
    final_data: list[dict[str, Any]],
    run_filename: str | None = None,
    run_date_override: str | None = None,
    end_date: str | None = None,
) -> PerformanceSummary:
    """
    Compute realised returns for every stock in a pipeline run.

    Parameters
    ----------
    final_data : list[dict]
        The list of stock dicts from a final_ranking JSON file.
    run_filename : str, optional
        Filename to extract run date from (e.g. "final_ranking_sp100_20260203_0856.json").
    run_date_override : str, optional
        Explicit run date in YYYY-MM-DD format.  Takes priority over filename.
    end_date : str, optional
        End date in YYYY-MM-DD.  Defaults to today.

    Returns
    -------
    PerformanceSummary
    """
    import yfinance as yf

    # ── Determine dates ──────────────────────────────────────────────
    if run_date_override:
        run_dt = datetime.strptime(run_date_override, "%Y-%m-%d")
    elif run_filename:
        run_dt = _parse_run_date_from_filename(run_filename)
        if run_dt is None:
            raise ValueError(f"Cannot parse date from filename: {run_filename}")
    else:
        raise ValueError("Either run_filename or run_date_override must be provided")

    start_str = _next_trading_day(run_dt)
    end_str = end_date or datetime.now().strftime("%Y-%m-%d")
    # yfinance end is exclusive, so add 1 day
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    end_fetch = (end_dt + timedelta(days=3)).strftime("%Y-%m-%d")
    calendar_days = (end_dt - run_dt).days

    # ── Collect tickers ──────────────────────────────────────────────
    tickers = [s.get("ticker", "") for s in final_data if s.get("ticker")]
    all_tickers = list(set(tickers + ["SPY"]))

    # ── Fetch prices ─────────────────────────────────────────────────
    logger.info(f"Fetching prices for {len(all_tickers)} tickers from {start_str} to {end_fetch}")

    try:
        price_data = yf.download(
            all_tickers,
            start=start_str,
            end=end_fetch,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return PerformanceSummary(
            run_date=start_str,
            end_date=end_str,
            trading_days=0,
            calendar_days=calendar_days,
        )

    # Handle single-ticker edge case (no MultiIndex)
    if len(all_tickers) == 1:
        close = price_data[["Close"]].copy()
        close.columns = [all_tickers[0]]
    else:
        if "Close" in price_data.columns.get_level_values(0):
            close = price_data["Close"]
        else:
            close = price_data.xs("Close", axis=1, level=0) if isinstance(
                price_data.columns, pd.MultiIndex
            ) else price_data

    if close.empty:
        return PerformanceSummary(
            run_date=start_str,
            end_date=end_str,
            trading_days=0,
            calendar_days=calendar_days,
        )

    trading_days = len(close)

    # ── SPY benchmark ────────────────────────────────────────────────
    spy_first = close["SPY"].dropna().iloc[0] if "SPY" in close.columns and not close["SPY"].dropna().empty else None
    spy_last = close["SPY"].dropna().iloc[-1] if "SPY" in close.columns and not close["SPY"].dropna().empty else None
    spy_return = ((spy_last / spy_first) - 1) * 100 if spy_first and spy_last else None

    # ── Per-stock returns ────────────────────────────────────────────
    stocks: list[StockPerformance] = []
    for s in final_data:
        ticker = s.get("ticker", "")
        if not ticker:
            continue

        perf = StockPerformance(
            ticker=ticker,
            signal=s.get("signal", "hold"),
            composite_score=s.get("composite_score", 0),
            final_rank=s.get("final_rank", 0),
            company_name=s.get("company_name", ticker),
            sector=s.get("sector", ""),
        )

        if ticker not in close.columns:
            perf.error = "No price data available"
            stocks.append(perf)
            continue

        series = close[ticker].dropna()
        if len(series) < 2:
            perf.error = "Insufficient price data"
            stocks.append(perf)
            continue

        first_price = series.iloc[0]
        last_price = series.iloc[-1]
        ret = ((last_price / first_price) - 1) * 100

        perf.price_at_run = round(float(first_price), 2)
        perf.price_current = round(float(last_price), 2)
        perf.total_return_pct = round(float(ret), 2)
        perf.excess_return_pct = round(float(ret) - float(spy_return), 2) if spy_return is not None else None

        # Signal correctness:
        #   buy / strong_buy  → correct if return > 0
        #   sell / strong_sell → correct if return < 0
        #   hold → not counted
        if perf.signal in ("buy", "strong_buy"):
            perf.signal_correct = ret > 0
        elif perf.signal in ("sell", "strong_sell"):
            perf.signal_correct = ret < 0
        else:
            perf.signal_correct = None  # hold — not actionable

        # Sparkline data (daily closes as dict)
        perf.price_series = {
            d.strftime("%Y-%m-%d"): round(float(p), 2)
            for d, p in series.items()
        }

        stocks.append(perf)

    # ── Aggregate metrics ────────────────────────────────────────────
    valid_returns = [sp.total_return_pct for sp in stocks if sp.total_return_pct is not None]
    portfolio_return = sum(valid_returns) / len(valid_returns) if valid_returns else None
    alpha = (portfolio_return - spy_return) if (portfolio_return is not None and spy_return is not None) else None

    actionable = [sp for sp in stocks if sp.signal_correct is not None]
    correct = [sp for sp in actionable if sp.signal_correct]
    hit_rate = len(correct) / len(actionable) if actionable else None

    # Average return by signal
    signal_returns: dict[str, list[float]] = {}
    for sp in stocks:
        if sp.total_return_pct is not None:
            signal_returns.setdefault(sp.signal, []).append(sp.total_return_pct)

    avg_by_signal = {sig: sum(rets) / len(rets) for sig, rets in signal_returns.items()}
    median_by_signal = {
        sig: sorted(rets)[len(rets) // 2] for sig, rets in signal_returns.items()
    }

    # Best / worst
    valid_stocks = [sp for sp in stocks if sp.total_return_pct is not None]
    best = max(valid_stocks, key=lambda sp: sp.total_return_pct) if valid_stocks else None
    worst = min(valid_stocks, key=lambda sp: sp.total_return_pct) if valid_stocks else None

    return PerformanceSummary(
        run_date=start_str,
        end_date=end_str,
        trading_days=trading_days,
        calendar_days=calendar_days,
        stocks=stocks,
        spy_return_pct=round(spy_return, 2) if spy_return is not None else None,
        spy_price_at_run=round(float(spy_first), 2) if spy_first is not None else None,
        spy_price_current=round(float(spy_last), 2) if spy_last is not None else None,
        portfolio_return_pct=round(portfolio_return, 2) if portfolio_return is not None else None,
        alpha_pct=round(alpha, 2) if alpha is not None else None,
        hit_rate=round(hit_rate, 3) if hit_rate is not None else None,
        correct_count=len(correct),
        total_actionable=len(actionable),
        avg_return_by_signal=avg_by_signal,
        median_return_by_signal=median_by_signal,
        best_performer=best,
        worst_performer=worst,
    )
