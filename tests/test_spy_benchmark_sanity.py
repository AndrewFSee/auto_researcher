"""
SPY Benchmark Sanity Tests.

This module tests that SPY returns are computed correctly to catch
any bugs in return calculations, benchmark handling, or metrics.

The key insight is that SPY from 2015-2023 should have positive returns.
If our metrics show SPY with near-zero or negative returns, something is wrong.
"""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.data.price_loader import download_price_history, get_adjusted_close
from auto_researcher.backtest.metrics import (
    compute_cumulative_returns,
    compute_annualized_return,
    compute_total_return,
)


class TestSPYBenchmarkReturns:
    """Test that SPY benchmark returns are computed correctly."""

    @pytest.fixture
    def spy_prices(self) -> pd.DataFrame:
        """Download SPY prices for 2015-2023."""
        raw = download_price_history(
            tickers=["SPY"],
            start="2015-01-01",
            end="2024-01-01",
        )
        prices = get_adjusted_close(raw)
        return prices

    def test_spy_cumulative_return_positive(self, spy_prices: pd.DataFrame) -> None:
        """
        SPY cumulative return 2015-2023 should be positive.
        
        SPY roughly doubled from ~$200 to ~$475 in this period.
        This test catches bugs where we incorrectly compute returns.
        """
        # Compute daily returns
        daily_returns = spy_prices["SPY"].pct_change().dropna()
        
        # Compute cumulative return
        cum_ret = compute_cumulative_returns(daily_returns)
        final_cum_ret = cum_ret.iloc[-1] - 1  # Convert from growth factor to return
        
        # SPY should have positive cumulative return
        assert final_cum_ret > 0, f"SPY cumulative return {final_cum_ret:.2%} should be positive"
        
        # More specifically, it should be substantial (>50% over 9 years is conservative)
        assert final_cum_ret > 0.50, f"SPY cumulative return {final_cum_ret:.2%} should be > 50%"

    def test_spy_annualized_return_positive(self, spy_prices: pd.DataFrame) -> None:
        """
        SPY annualized return 2015-2023 should be positive and reasonable.
        
        Historically SPY averages ~10% annualized.
        We test for > 2% to catch obviously broken calculations.
        """
        # Compute daily returns
        daily_returns = spy_prices["SPY"].pct_change().dropna()
        
        # Compute annualized return (daily data, 252 trading days/year)
        ann_ret = compute_annualized_return(daily_returns, periods_per_year=252)
        
        # SPY should have positive annualized return
        assert ann_ret > 0, f"SPY annualized return {ann_ret:.2%} should be positive"
        
        # Should be at least 2% annualized over this period
        assert ann_ret > 0.02, f"SPY annualized return {ann_ret:.2%} should be > 2%"
        
        # Should not be absurdly high (< 30% is reasonable)
        assert ann_ret < 0.30, f"SPY annualized return {ann_ret:.2%} should be < 30%"

    def test_spy_simple_price_return_positive(self, spy_prices: pd.DataFrame) -> None:
        """
        Simple sanity check: SPY ending price > starting price.
        
        This is the most basic check and should never fail if data is correct.
        """
        start_price = spy_prices["SPY"].iloc[0]
        end_price = spy_prices["SPY"].iloc[-1]
        
        simple_return = (end_price / start_price) - 1
        
        assert simple_return > 0, f"SPY simple return {simple_return:.2%} should be positive"
        assert simple_return > 0.50, f"SPY simple return {simple_return:.2%} should be > 50%"


class TestMonthlyReturnAggregation:
    """Test that monthly returns are computed correctly from daily prices."""

    @pytest.fixture
    def spy_prices(self) -> pd.DataFrame:
        """Download SPY prices for 2015-2023."""
        raw = download_price_history(
            tickers=["SPY"],
            start="2015-01-01",
            end="2024-01-01",
        )
        prices = get_adjusted_close(raw)
        return prices

    def test_monthly_returns_match_price_changes(self, spy_prices: pd.DataFrame) -> None:
        """
        Monthly returns should match actual price changes over the month.
        
        This catches the bug where we use daily pct_change on monthly dates
        instead of computing the full period return.
        """
        prices = spy_prices["SPY"]
        
        # Get month-end dates
        monthly_prices = prices.resample("ME").last()
        
        # Compute monthly returns correctly: (P_end / P_start) - 1
        correct_monthly_returns = monthly_prices.pct_change().dropna()
        
        # WRONG way: daily pct_change reindexed to month-ends
        daily_returns = prices.pct_change()
        wrong_monthly_returns = daily_returns.reindex(monthly_prices.index).dropna()
        
        # The correct and wrong methods should NOT match
        # (This is the bug we're catching)
        diff = (correct_monthly_returns - wrong_monthly_returns).abs()
        
        # If they're close, something is wrong with our test logic
        # They should be very different because wrong method only captures 1 day
        assert diff.mean() > 0.01, (
            "Expected significant difference between correct and wrong monthly returns. "
            "Correct method captures full month, wrong method captures only 1 day."
        )

    def test_compounded_monthly_matches_total(self, spy_prices: pd.DataFrame) -> None:
        """
        Compounded monthly returns should approximately match total return.
        
        This validates our monthly return calculation is consistent.
        """
        prices = spy_prices["SPY"]
        
        # Total return from prices
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        
        # Monthly returns (correct method)
        monthly_prices = prices.resample("ME").last()
        monthly_returns = monthly_prices.pct_change().dropna()
        
        # Compound monthly returns
        compounded = (1 + monthly_returns).prod() - 1
        
        # Should be close (not exact due to timing of month-ends vs actual dates)
        # Allow 5% relative tolerance
        rel_diff = abs(compounded - total_return) / abs(total_return)
        assert rel_diff < 0.05, (
            f"Compounded monthly return {compounded:.2%} should be close to "
            f"total return {total_return:.2%}, but diff is {rel_diff:.1%}"
        )


class TestBenchmarkInBacktest:
    """Test that benchmark returns in backtest match actual SPY performance."""

    def test_benchmark_period_returns(self) -> None:
        """
        Benchmark returns over holding periods should match actual SPY changes.
        
        This is a unit test that doesn't require data download.
        """
        # Create synthetic price data with business days
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        np.random.seed(42)
        spy_returns = np.random.normal(0.0005, 0.01, len(dates))
        spy_prices = pd.Series(
            100 * np.exp(np.cumsum(spy_returns)),
            index=dates,
            name="SPY"
        )
        
        # Get month-end prices (last available business day in each month)
        # Use .last() to get actual dates that exist in our series
        monthly_prices = spy_prices.resample("ME").last()
        monthly_dates = monthly_prices.index
        
        # The dates in monthly_prices.index are the actual business days,
        # not calendar month-ends, so we can safely use them
        actual_dates_in_series = [d for d in monthly_dates if d in spy_prices.index]
        
        # Compute period returns correctly: (P_end / P_start) - 1
        correct_period_returns = []
        for i in range(len(actual_dates_in_series) - 1):
            start_date = actual_dates_in_series[i]
            end_date = actual_dates_in_series[i + 1]
            
            start_price = spy_prices.loc[start_date]
            end_price = spy_prices.loc[end_date]
            period_ret = (end_price / start_price) - 1
            correct_period_returns.append((end_date, period_ret))
        
        # WRONG: daily pct_change on the dates
        daily_pct = spy_prices.pct_change()
        wrong_period_returns = daily_pct.reindex(actual_dates_in_series[1:]).tolist()
        
        # Compare
        correct_vals = [r for _, r in correct_period_returns]
        
        # Correct returns should compound to roughly total return
        total_from_correct = (pd.Series(correct_vals) + 1).prod() - 1
        first_date = actual_dates_in_series[0]
        last_date = actual_dates_in_series[-1]
        total_actual = (spy_prices.loc[last_date] / spy_prices.loc[first_date]) - 1
        
        assert abs(total_from_correct - total_actual) < 0.01, (
            f"Correct period returns should compound to total return. "
            f"Got {total_from_correct:.4f}, expected ~{total_actual:.4f}"
        )
        
        # Wrong returns should NOT compound to total return
        total_from_wrong = (pd.Series(wrong_period_returns) + 1).prod() - 1
        assert abs(total_from_wrong - total_actual) > 0.05, (
            f"Wrong period returns should NOT match total return. "
            f"Got {total_from_wrong:.4f}, actual is {total_actual:.4f}"
        )
