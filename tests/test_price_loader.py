"""Tests for price_loader module."""

import pytest
import pandas as pd
import numpy as np

from auto_researcher.data.price_loader import (
    download_price_history,
    get_adjusted_close,
    get_volume,
    validate_price_data,
)


class TestDownloadPriceHistory:
    """Tests for download_price_history function."""

    @pytest.fixture
    def sample_tickers(self) -> list[str]:
        """Small set of tickers for testing."""
        return ["AAPL", "MSFT"]

    @pytest.fixture
    def date_range(self) -> tuple[str, str]:
        """Short date range for fast tests."""
        return ("2023-01-01", "2023-03-31")

    def test_returns_dataframe_with_datetime_index(
        self, sample_tickers: list[str], date_range: tuple[str, str]
    ) -> None:
        """Downloaded data should have DatetimeIndex."""
        start, end = date_range
        df = download_price_history(sample_tickers, start, end)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_returns_multiindex_columns(
        self, sample_tickers: list[str], date_range: tuple[str, str]
    ) -> None:
        """Downloaded data should have MultiIndex columns."""
        start, end = date_range
        df = download_price_history(sample_tickers, start, end)
        
        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.nlevels == 2

    def test_contains_expected_fields(
        self, sample_tickers: list[str], date_range: tuple[str, str]
    ) -> None:
        """Downloaded data should contain standard price fields."""
        start, end = date_range
        df = download_price_history(sample_tickers, start, end)
        
        fields = df.columns.get_level_values(0).unique().tolist()
        assert "Close" in fields or "Adj Close" in fields

    def test_has_data_for_all_tickers(
        self, sample_tickers: list[str], date_range: tuple[str, str]
    ) -> None:
        """Downloaded data should include all requested tickers."""
        start, end = date_range
        df = download_price_history(sample_tickers, start, end)
        
        tickers_in_data = df.columns.get_level_values(1).unique().tolist()
        for ticker in sample_tickers:
            assert ticker in tickers_in_data

    def test_data_is_non_null(
        self, sample_tickers: list[str], date_range: tuple[str, str]
    ) -> None:
        """At least some data should be non-null."""
        start, end = date_range
        df = download_price_history(sample_tickers, start, end)
        
        # Check that we have some valid data
        assert not df.isna().all().all()

    def test_single_ticker_still_multiindex(self, date_range: tuple[str, str]) -> None:
        """Single ticker should still return MultiIndex columns."""
        start, end = date_range
        df = download_price_history(["AAPL"], start, end)
        
        assert isinstance(df.columns, pd.MultiIndex)


class TestGetAdjustedClose:
    """Tests for get_adjusted_close function."""

    @pytest.fixture
    def mock_price_data(self) -> pd.DataFrame:
        """Create mock price data with MultiIndex columns."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        tickers = ["AAPL", "MSFT"]
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], tickers])
        data = np.random.randn(5, 4)
        
        return pd.DataFrame(data, index=dates, columns=columns)

    def test_extracts_adj_close(self, mock_price_data: pd.DataFrame) -> None:
        """Should extract Adj Close prices."""
        adj_close = get_adjusted_close(mock_price_data)
        
        assert "AAPL" in adj_close.columns
        assert "MSFT" in adj_close.columns
        assert len(adj_close) == 5


class TestFallbackCacheDateValidation:
    """Tests for cache fallback date range validation."""

    def _create_cache_file(self, cache_dir, start, end, tickers):
        """Helper to create a fake cache parquet file."""
        dates = pd.date_range(start, end, freq="D")
        columns = pd.MultiIndex.from_product([["Adj Close"], tickers])
        data = np.random.randn(len(dates), len(tickers)) * 10 + 100
        df = pd.DataFrame(data, index=dates, columns=columns)
        path = cache_dir / f"prices_{start}_{end}.parquet"
        df.to_parquet(path)
        return path

    def test_rejects_non_overlapping_cache(self, tmp_path, monkeypatch) -> None:
        """Cache from a completely different date range should be rejected."""
        # Create cache for 2020 data
        self._create_cache_file(tmp_path, "2020-01-01", "2020-12-31", ["AAPL", "MSFT"])

        # Mock yfinance to simulate download failure
        import yfinance
        monkeypatch.setattr(yfinance, "download", lambda **kwargs: pd.DataFrame())

        # Request 2023 data -- should fail, not silently return 2020 data
        with pytest.raises(ValueError, match="No data returned"):
            download_price_history(
                ["AAPL", "MSFT"],
                "2023-01-01",
                "2023-12-31",
                cache_dir=tmp_path,
                max_retries=1,
                retry_delay=0.1,
                use_default_cache=False,
            )

    def test_accepts_overlapping_cache(self, tmp_path, monkeypatch) -> None:
        """Cache with >50% date overlap should be accepted as fallback."""
        # Create cache covering most of the requested range
        self._create_cache_file(tmp_path, "2023-01-01", "2023-11-30", ["AAPL", "MSFT"])

        # Mock yfinance to simulate download failure
        import yfinance
        monkeypatch.setattr(yfinance, "download", lambda **kwargs: pd.DataFrame())

        # Request slightly wider range -- cache covers >50% so should be accepted
        result = download_price_history(
            ["AAPL", "MSFT"],
            "2023-01-01",
            "2023-12-31",
            cache_dir=tmp_path,
            max_retries=1,
            retry_delay=0.1,
            use_default_cache=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_picks_best_overlap(self, tmp_path, monkeypatch) -> None:
        """When multiple caches exist, should pick the one with most date overlap."""
        # Create two cache files
        self._create_cache_file(tmp_path, "2023-01-01", "2023-06-30", ["AAPL", "MSFT"])
        self._create_cache_file(tmp_path, "2023-01-01", "2023-11-30", ["AAPL", "MSFT"])

        # Mock yfinance to simulate download failure
        import yfinance
        monkeypatch.setattr(yfinance, "download", lambda **kwargs: pd.DataFrame())

        result = download_price_history(
            ["AAPL", "MSFT"],
            "2023-01-01",
            "2023-12-31",
            cache_dir=tmp_path,
            max_retries=1,
            retry_delay=0.1,
            use_default_cache=False,
        )
        # Should get the larger cache (Jan-Nov vs Jan-Jun)
        assert len(result) > 180  # More than 6 months of data


class TestValidatePriceData:
    """Tests for validate_price_data function."""

    @pytest.fixture
    def mock_price_data(self) -> pd.DataFrame:
        """Create mock price data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        columns = pd.MultiIndex.from_product([["Adj Close"], ["AAPL", "MSFT"]])
        data = np.array([[100, 200], [101, 201], [102, np.nan], [103, 203], [104, 204]])
        
        return pd.DataFrame(data, index=dates, columns=columns)

    def test_validates_existing_tickers(self, mock_price_data: pd.DataFrame) -> None:
        """Should return True for tickers with data."""
        result = validate_price_data(mock_price_data, ["AAPL", "MSFT"])
        
        assert result["AAPL"] == True
        assert result["MSFT"] == True

    def test_returns_false_for_missing_tickers(self, mock_price_data: pd.DataFrame) -> None:
        """Should return False for missing tickers."""
        result = validate_price_data(mock_price_data, ["AAPL", "GOOG"])
        
        assert result["AAPL"] == True
        assert result["GOOG"] == False
