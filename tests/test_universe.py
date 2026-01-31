"""Tests for universe module."""

import pytest

from auto_researcher.config import ResearchConfig, DEFAULT_UNIVERSE
from auto_researcher.data.universe import (
    get_universe,
    get_sp500_universe,
    filter_universe,
)


class TestGetUniverse:
    """Tests for get_universe function."""

    def test_default_universe_is_non_empty(self) -> None:
        """DEFAULT_UNIVERSE should contain at least one ticker."""
        assert len(DEFAULT_UNIVERSE) > 0

    def test_get_universe_returns_list_of_strings(self) -> None:
        """get_universe should return a list of strings."""
        tickers = get_universe()
        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)

    def test_get_universe_returns_copy(self) -> None:
        """get_universe should return a copy, not the original."""
        tickers1 = get_universe()
        tickers2 = get_universe()
        tickers1.append("TEST")
        assert "TEST" not in tickers2

    def test_get_universe_with_config(self) -> None:
        """get_universe should respect custom config."""
        config = ResearchConfig(universe=("GOOG", "TSLA"))
        tickers = get_universe(config)
        assert tickers == ["GOOG", "TSLA"]


class TestGetSP500Universe:
    """Tests for get_sp500_universe function."""

    def test_sp500_is_non_empty(self) -> None:
        """S&P 500 universe should contain tickers."""
        tickers = get_sp500_universe()
        assert len(tickers) > 0

    def test_sp500_contains_expected_stocks(self) -> None:
        """S&P 500 should contain well-known large caps."""
        tickers = get_sp500_universe()
        expected = {"AAPL", "MSFT", "AMZN", "GOOGL", "JPM"}
        assert expected.issubset(set(tickers))


class TestFilterUniverse:
    """Tests for filter_universe function."""

    def test_filter_with_exclude(self) -> None:
        """Should exclude specified tickers."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        result = filter_universe(tickers, exclude=["MSFT"])
        assert result == ["AAPL", "GOOG"]

    def test_filter_with_include_only(self) -> None:
        """Should only include specified tickers."""
        tickers = ["AAPL", "MSFT", "GOOG"]
        result = filter_universe(tickers, include_only=["AAPL", "GOOG"])
        assert result == ["AAPL", "GOOG"]

    def test_filter_with_both(self) -> None:
        """Should apply include_only first, then exclude."""
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        result = filter_universe(
            tickers,
            include_only=["AAPL", "MSFT", "GOOG"],
            exclude=["MSFT"],
        )
        assert result == ["AAPL", "GOOG"]

    def test_filter_preserves_order(self) -> None:
        """Filter should preserve original order."""
        tickers = ["C", "B", "A"]
        result = filter_universe(tickers, include_only=["A", "C"])
        assert result == ["C", "A"]
