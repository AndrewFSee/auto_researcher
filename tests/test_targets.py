"""Tests for target engineering module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.features.targets import (
    compute_vol_normalized_targets,
    compute_forward_returns,
    validate_forward_returns,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    tickers = ["AAPL", "MSFT", "GOOG"]
    returns = np.random.randn(300, 3) * 0.02
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


class TestExpandingWinsorization:
    """Tests for expanding-window winsorization (no look-ahead bias)."""

    def test_expanding_mode_no_lookahead(self, sample_prices: pd.DataFrame) -> None:
        """Extreme outlier at end should NOT affect winsorization bounds for earlier rows."""
        prices = sample_prices.copy()

        # Compute targets without the outlier
        targets_clean = compute_vol_normalized_targets(
            prices, horizon_days=21, winsorize_mode="expanding"
        )

        # Now inject a massive outlier in the last price row for AAPL
        # This creates an extreme forward return for rows ~21 days before the end
        prices_dirty = prices.copy()
        prices_dirty.iloc[-1, 0] = prices_dirty.iloc[-1, 0] * 100  # 100x spike

        targets_dirty = compute_vol_normalized_targets(
            prices_dirty, horizon_days=21, winsorize_mode="expanding"
        )

        # With expanding-window winsorization, rows BEFORE the outlier window
        # should be identical (the outlier at the end can't affect past bounds)
        # Check first 200 rows (well before the outlier region)
        early_clean = targets_clean.iloc[:200]
        early_dirty = targets_dirty.iloc[:200]

        pd.testing.assert_frame_equal(early_clean, early_dirty)

    def test_full_mode_uses_all_data_for_bounds(self, sample_prices: pd.DataFrame) -> None:
        """Full-sample winsorization computes bounds from all data (legacy look-ahead behavior).

        Since validate_forward_returns NaN-outs extreme outliers before winsorization,
        we test with a moderate outlier that passes validation but shifts the distribution.
        """
        prices = sample_prices.copy()

        targets_clean = compute_vol_normalized_targets(
            prices, horizon_days=21, winsorize_mode="full"
        )

        # Inject a moderate price change (2x, not 100x) that won't be caught by
        # validation but will shift the vol-normalized distribution
        prices_shifted = prices.copy()
        prices_shifted.iloc[-5:, 0] = prices_shifted.iloc[-5:, 0] * 2

        targets_shifted = compute_vol_normalized_targets(
            prices_shifted, horizon_days=21, winsorize_mode="full"
        )

        # Full-sample mode uses mean/std from ALL data, so bounds differ.
        # The moderate shift changes the distribution of vol-normalized targets
        # for AAPL, producing different clip bounds overall.
        all_clean = targets_clean["AAPL"].dropna()
        all_shifted = targets_shifted["AAPL"].dropna()

        # The overall distributions should differ
        assert not all_clean.equals(all_shifted)

    def test_expanding_mode_is_default(self, sample_prices: pd.DataFrame) -> None:
        """Default winsorize_mode should be 'expanding'."""
        targets_default = compute_vol_normalized_targets(
            sample_prices, horizon_days=21
        )
        targets_expanding = compute_vol_normalized_targets(
            sample_prices, horizon_days=21, winsorize_mode="expanding"
        )
        pd.testing.assert_frame_equal(targets_default, targets_expanding)

    def test_invalid_winsorize_mode_raises(self, sample_prices: pd.DataFrame) -> None:
        """Invalid winsorize_mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown winsorize_mode"):
            compute_vol_normalized_targets(
                sample_prices, horizon_days=21, winsorize_mode="invalid"
            )


class TestValidateForwardReturns:
    """Tests for forward return validation."""

    def test_extreme_positive_returns_become_nan(self) -> None:
        """Returns > +500% should be set to NaN (likely stock split)."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"AAPL": [0.01, 0.02, 6.0, -0.01, 0.03],  # 6.0 = +600%
             "MSFT": [0.01, 0.02, 0.03, -0.01, 0.02]},
            index=dates,
        )
        result = validate_forward_returns(data, max_return=5.0)

        assert np.isnan(result.loc[dates[2], "AAPL"])
        assert result.loc[dates[2], "MSFT"] == 0.03  # normal value preserved

    def test_extreme_negative_returns_become_nan(self) -> None:
        """Returns < -90% should be set to NaN."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"AAPL": [0.01, -0.95, 0.02, -0.01, 0.03]},  # -95%
            index=dates,
        )
        result = validate_forward_returns(data, min_return=-0.90)

        assert np.isnan(result.loc[dates[1], "AAPL"])

    def test_normal_returns_preserved(self) -> None:
        """Normal returns should pass through unchanged."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"AAPL": [0.01, 0.02, -0.03, 0.05, -0.01]},
            index=dates,
        )
        result = validate_forward_returns(data)
        pd.testing.assert_frame_equal(result, data)

    def test_low_coverage_warning(self, caplog) -> None:
        """Tickers with very low coverage should trigger a warning."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {"AAPL": [np.nan] * 95 + [0.01] * 5},  # only 5% coverage
            index=dates,
        )
        with caplog.at_level("WARNING"):
            validate_forward_returns(data, min_coverage=0.1)
        assert "very low forward return coverage" in caplog.text

    def test_existing_nans_preserved(self) -> None:
        """Pre-existing NaN values should remain NaN."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {"AAPL": [0.01, np.nan, 0.02, np.nan, 0.03]},
            index=dates,
        )
        result = validate_forward_returns(data)
        assert np.isnan(result.iloc[1, 0])
        assert np.isnan(result.iloc[3, 0])


class TestForwardReturnsIntegration:
    """Integration tests for forward returns with validation."""

    def test_compute_forward_returns_validates(self, sample_prices: pd.DataFrame) -> None:
        """compute_forward_returns should apply validation."""
        # Inject a stock split (price jumps 10x)
        prices = sample_prices.copy()
        prices.iloc[150:, 0] = prices.iloc[150:, 0] * 10

        result = compute_forward_returns(prices, horizon_days=21)

        # The forward return around the split should be NaN
        # (the jump from ~100 to ~1000 creates a ~900% return)
        split_region = result["AAPL"].iloc[129:150]  # 21 days before the jump
        assert split_region.isna().any(), "Stock split region should contain NaN"

    def test_vol_normalized_validates_before_normalization(
        self, sample_prices: pd.DataFrame
    ) -> None:
        """Vol-normalized targets should validate returns before dividing by vol."""
        prices = sample_prices.copy()
        prices.iloc[150:, 0] = prices.iloc[150:, 0] * 10

        result = compute_vol_normalized_targets(prices, horizon_days=21)

        # The split region should be NaN, not an extreme vol-normalized value
        split_region = result["AAPL"].iloc[129:150]
        non_nan = split_region.dropna()
        if len(non_nan) > 0:
            assert non_nan.abs().max() < 100, (
                "Vol-normalized targets near stock split should not be extreme"
            )
