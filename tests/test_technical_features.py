"""Tests for technical features module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.features.technical import (
    compute_returns,
    compute_log_returns,
    compute_momentum,
    compute_momentum_simple,
    compute_volatility,
    compute_sharpe,
    compute_rsi,
    compute_moving_average_ratio,
    compute_all_technical_features,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    
    # Generate random walk prices
    returns = np.random.randn(200, 3) * 0.02  # 2% daily vol
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return pd.DataFrame(prices, index=dates, columns=["AAPL", "MSFT", "GOOG"])


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns from sample prices."""
    return compute_returns(sample_prices)


class TestComputeReturns:
    """Tests for compute_returns function."""

    def test_returns_shape(self, sample_prices: pd.DataFrame) -> None:
        """Returns should have same shape as prices."""
        returns = compute_returns(sample_prices)
        assert returns.shape == sample_prices.shape

    def test_returns_index_preserved(self, sample_prices: pd.DataFrame) -> None:
        """Returns should preserve DatetimeIndex."""
        returns = compute_returns(sample_prices)
        assert (returns.index == sample_prices.index).all()

    def test_returns_columns_preserved(self, sample_prices: pd.DataFrame) -> None:
        """Returns should preserve columns."""
        returns = compute_returns(sample_prices)
        assert (returns.columns == sample_prices.columns).all()

    def test_first_return_is_nan(self, sample_prices: pd.DataFrame) -> None:
        """First return should be NaN (no previous price)."""
        returns = compute_returns(sample_prices)
        assert returns.iloc[0].isna().all()

    def test_returns_series(self) -> None:
        """Should work with Series input."""
        prices = pd.Series([100, 102, 101], index=pd.date_range("2023-01-01", periods=3))
        returns = compute_returns(prices)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == 3


class TestComputeMomentum:
    """Tests for momentum computation."""

    def test_momentum_shape(self, sample_returns: pd.DataFrame) -> None:
        """Momentum should have same shape as returns."""
        mom = compute_momentum_simple(sample_returns, window=21)
        assert mom.shape == sample_returns.shape

    def test_momentum_warmup_nans(self, sample_returns: pd.DataFrame) -> None:
        """First (window-1) values should be NaN."""
        window = 21
        mom = compute_momentum_simple(sample_returns, window=window)
        
        # First (window-1) rows should have NaN
        assert mom.iloc[:window - 1].isna().all().all()
        
        # After warmup, should have values
        assert not mom.iloc[window:].isna().all().all()

    def test_momentum_different_windows(self, sample_returns: pd.DataFrame) -> None:
        """Different windows should produce different results."""
        mom_21 = compute_momentum_simple(sample_returns, window=21)
        mom_63 = compute_momentum_simple(sample_returns, window=63)
        
        # After both warmups, values should differ
        idx = 63
        assert not np.allclose(
            mom_21.iloc[idx:].values,
            mom_63.iloc[idx:].values,
            equal_nan=True,
        )


class TestComputeVolatility:
    """Tests for volatility computation."""

    def test_volatility_shape(self, sample_returns: pd.DataFrame) -> None:
        """Volatility should have same shape as returns."""
        vol = compute_volatility(sample_returns, window=21)
        assert vol.shape == sample_returns.shape

    def test_volatility_positive(self, sample_returns: pd.DataFrame) -> None:
        """Volatility should be non-negative."""
        vol = compute_volatility(sample_returns, window=21)
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all().all()

    def test_volatility_annualized(self, sample_returns: pd.DataFrame) -> None:
        """Annualized volatility should be larger than daily."""
        vol_ann = compute_volatility(sample_returns, window=21, annualize=True)
        vol_daily = compute_volatility(sample_returns, window=21, annualize=False)
        
        # Annualized should be sqrt(252) times daily
        ratio = (vol_ann / vol_daily).dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio.values, expected_ratio, rtol=0.01)

    def test_volatility_warmup_nans(self, sample_returns: pd.DataFrame) -> None:
        """First (window-1) values should be NaN."""
        window = 21
        vol = compute_volatility(sample_returns, window=window)
        
        assert vol.iloc[:window - 1].isna().all().all()


class TestComputeRSI:
    """Tests for RSI computation."""

    def test_rsi_range(self, sample_prices: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        rsi = compute_rsi(sample_prices, window=14)
        valid_rsi = rsi.dropna()
        
        assert (valid_rsi >= 0).all().all()
        assert (valid_rsi <= 100).all().all()

    def test_rsi_shape(self, sample_prices: pd.DataFrame) -> None:
        """RSI should have same shape as prices."""
        rsi = compute_rsi(sample_prices)
        assert rsi.shape == sample_prices.shape


class TestComputeAllTechnicalFeatures:
    """Tests for complete technical feature pipeline."""

    def test_returns_multiindex_columns(self, sample_prices: pd.DataFrame) -> None:
        """Should return DataFrame with MultiIndex columns."""
        features = compute_all_technical_features(sample_prices)
        
        assert isinstance(features.columns, pd.MultiIndex)
        assert features.columns.nlevels == 2
        assert features.columns.names == ["ticker", "feature"]

    def test_contains_expected_features(self, sample_prices: pd.DataFrame) -> None:
        """Should contain momentum and volatility features."""
        features = compute_all_technical_features(
            sample_prices,
            momentum_windows=(21, 63),
            volatility_windows=(21,),
        )
        
        # Get feature names
        feature_names = features.columns.get_level_values("feature").unique()
        
        assert "mom_21" in feature_names
        assert "mom_63" in feature_names
        assert "vol_21" in feature_names
        assert "rsi_14" in feature_names

    def test_all_tickers_present(self, sample_prices: pd.DataFrame) -> None:
        """All input tickers should be in output."""
        features = compute_all_technical_features(sample_prices)
        
        tickers = features.columns.get_level_values("ticker").unique()
        for t in sample_prices.columns:
            assert t in tickers

    def test_index_preserved(self, sample_prices: pd.DataFrame) -> None:
        """DatetimeIndex should be preserved."""
        features = compute_all_technical_features(sample_prices)
        
        assert (features.index == sample_prices.index).all()
