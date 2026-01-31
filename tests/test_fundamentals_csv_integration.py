"""
Integration tests for fundamental data CSV loading.

These tests verify that:
1. load_raw_fundamentals correctly loads data from a CSV file
2. The loaded data integrates properly with build_fundamental_factors
3. The schema matches what download_fundamentals.py produces
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_researcher.features.fundamentals import (
    RAW_FUNDAMENTAL_COLUMNS,
    FundamentalFactorConfig,
    build_fundamental_factors,
    load_raw_fundamentals,
)


# =============================================================================
# Helper Functions
# =============================================================================


def create_sample_csv(csv_path: Path) -> None:
    """
    Create a sample fundamentals CSV matching the download script output format.

    Args:
        csv_path: Path where to save the CSV file.
    """
    # Create sample data for 3 tickers x 4 quarters
    data = {
        "date": [
            "2023-03-31", "2023-03-31", "2023-03-31",
            "2023-06-30", "2023-06-30", "2023-06-30",
            "2023-09-30", "2023-09-30", "2023-09-30",
            "2023-12-31", "2023-12-31", "2023-12-31",
        ],
        "ticker": [
            "AAPL", "MSFT", "GOOGL",
            "AAPL", "MSFT", "GOOGL",
            "AAPL", "MSFT", "GOOGL",
            "AAPL", "MSFT", "GOOGL",
        ],
        "market_cap": [
            2.5e12, 2.2e12, 1.5e12,
            2.6e12, 2.3e12, 1.6e12,
            2.7e12, 2.4e12, 1.7e12,
            2.8e12, 2.5e12, 1.8e12,
        ],
        "pe_ratio": [
            25.0, 30.0, 22.0,
            26.0, 31.0, 23.0,
            27.0, 32.0, 24.0,
            28.0, 33.0, 25.0,
        ],
        "pb_ratio": [
            40.0, 12.0, 6.0,
            41.0, 12.5, 6.2,
            42.0, 13.0, 6.4,
            43.0, 13.5, 6.6,
        ],
        "ps_ratio": [
            7.0, 10.0, 5.0,
            7.2, 10.2, 5.2,
            7.4, 10.4, 5.4,
            7.6, 10.6, 5.6,
        ],
        "dividend_yield": [
            0.005, 0.008, 0.0,
            0.005, 0.008, 0.0,
            0.006, 0.009, 0.0,
            0.006, 0.009, 0.0,
        ],
        "roe": [
            1.5, 0.45, 0.25,
            1.55, 0.46, 0.26,
            1.6, 0.47, 0.27,
            1.65, 0.48, 0.28,
        ],
        "roa": [
            0.3, 0.15, 0.12,
            0.31, 0.155, 0.125,
            0.32, 0.16, 0.13,
            0.33, 0.165, 0.135,
        ],
        "gross_margin": [
            0.43, 0.68, 0.55,
            0.435, 0.685, 0.555,
            0.44, 0.69, 0.56,
            0.445, 0.695, 0.565,
        ],
        "operating_margin": [
            0.30, 0.42, 0.28,
            0.305, 0.425, 0.285,
            0.31, 0.43, 0.29,
            0.315, 0.435, 0.295,
        ],
        "net_margin": [
            0.25, 0.35, 0.22,
            0.255, 0.355, 0.225,
            0.26, 0.36, 0.23,
            0.265, 0.365, 0.235,
        ],
        "revenue_ttm": [
            3.8e11, 2.0e11, 2.8e11,
            3.9e11, 2.1e11, 2.9e11,
            4.0e11, 2.2e11, 3.0e11,
            4.1e11, 2.3e11, 3.1e11,
        ],
        "revenue_growth_1y": [
            0.08, 0.12, 0.15,
            0.09, 0.13, 0.16,
            0.10, 0.14, 0.17,
            0.11, 0.15, 0.18,
        ],
        "eps_ttm": [
            6.0, 10.0, 5.5,
            6.2, 10.3, 5.7,
            6.4, 10.6, 5.9,
            6.6, 10.9, 6.1,
        ],
        "eps_growth_1y": [
            0.10, 0.15, 0.20,
            0.11, 0.16, 0.21,
            0.12, 0.17, 0.22,
            0.13, 0.18, 0.23,
        ],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


# =============================================================================
# Test Classes
# =============================================================================


class TestLoadFromCSV:
    """Tests for loading fundamentals from CSV files."""

    def test_loads_csv_with_correct_structure(self) -> None:
        """CSV data should be loaded with correct MultiIndex structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            result = load_raw_fundamentals(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            assert isinstance(result, pd.DataFrame)
            assert isinstance(result.index, pd.MultiIndex)
            assert result.index.names == ["date", "ticker"]

    def test_loads_all_expected_columns(self) -> None:
        """All expected fundamental columns should be present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            result = load_raw_fundamentals(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            for col in RAW_FUNDAMENTAL_COLUMNS:
                assert col in result.columns, f"Missing column: {col}"

    def test_filters_by_ticker(self) -> None:
        """Should only return data for requested tickers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            result = load_raw_fundamentals(
                tickers=["AAPL"],  # Only request AAPL
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            tickers_in_result = result.index.get_level_values("ticker").unique()
            assert list(tickers_in_result) == ["AAPL"]

    def test_filters_by_date_range(self) -> None:
        """Should only return data within the requested date range."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            result = load_raw_fundamentals(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start="2023-04-01",  # Exclude Q1
                end="2023-09-30",  # Exclude Q4
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            dates = result.index.get_level_values("date").unique()
            assert len(dates) == 2  # Q2 and Q3 only
            assert pd.Timestamp("2023-03-31") not in dates
            assert pd.Timestamp("2023-12-31") not in dates

    def test_loads_correct_values(self) -> None:
        """Data values should match the source CSV."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            result = load_raw_fundamentals(
                tickers=["AAPL"],
                start="2023-01-01",
                end="2023-03-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            # Check specific values for AAPL Q1 2023
            aapl_q1 = result.loc[(pd.Timestamp("2023-03-31"), "AAPL")]
            assert aapl_q1["market_cap"] == 2.5e12
            assert aapl_q1["pe_ratio"] == 25.0
            assert aapl_q1["roe"] == 1.5


class TestCSVIntegrationWithFactors:
    """Tests for end-to-end integration of CSV data with factor computation."""

    def test_build_factors_from_csv_data(self) -> None:
        """Should be able to build factors from CSV-loaded data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            raw = load_raw_fundamentals(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            config = FundamentalFactorConfig(
                use_value=True,
                use_quality=True,
                use_growth=True,
                use_profitability=True,
                use_size=True,
                zscore_by_cross_section=False,  # Don't zscore for easier testing
            )

            factors = build_fundamental_factors(raw, config)

            assert isinstance(factors, pd.DataFrame)
            assert len(factors) > 0

    def test_value_factors_computed_correctly(self) -> None:
        """Value factors should be correctly computed from CSV data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            raw = load_raw_fundamentals(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            config = FundamentalFactorConfig(
                use_value=True,
                use_quality=False,
                use_growth=False,
                use_profitability=False,
                use_size=False,
                zscore_by_cross_section=False,
            )

            factors = build_fundamental_factors(raw, config)

            # Check earnings yield = 1 / PE
            assert "value_ey" in factors.columns
            # AAPL Q1: PE = 25, so EY = 0.04
            aapl_q1_ey = factors.loc[(pd.Timestamp("2023-03-31"), "AAPL"), "value_ey"]
            assert abs(aapl_q1_ey - 0.04) < 0.001

    def test_all_factor_families_included(self) -> None:
        """All enabled factor families should have columns in output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "fundamentals.csv"
            create_sample_csv(csv_path)

            raw = load_raw_fundamentals(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            config = FundamentalFactorConfig(
                use_value=True,
                use_quality=True,
                use_growth=True,
                use_profitability=True,
                use_size=True,
            )

            factors = build_fundamental_factors(raw, config)

            # Check for presence of at least one column from each family
            assert any(col.startswith("value_") for col in factors.columns)
            assert any(col.startswith("quality_") for col in factors.columns)
            assert any(col.startswith("growth_") for col in factors.columns)
            assert any(col.startswith("prof_") for col in factors.columns)
            assert any(col.startswith("size_") for col in factors.columns)


class TestMissingCSV:
    """Tests for handling missing or invalid CSV files."""

    def test_missing_csv_falls_back_to_yfinance(self) -> None:
        """When CSV doesn't exist, should fall back to yfinance or placeholder."""
        result = load_raw_fundamentals(
            tickers=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            fundamentals_csv_path="/nonexistent/path.csv",
            use_yfinance=False,  # Disable yfinance, use placeholder
        )

        # Should return placeholder with NaN values
        assert isinstance(result, pd.DataFrame)
        assert result.isna().all().all()

    def test_csv_with_missing_columns_fills_nan(self) -> None:
        """CSV missing some columns should have those filled with NaN."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "partial.csv"
            df = pd.DataFrame({
                "date": ["2023-03-31"],
                "ticker": ["AAPL"],
                "market_cap": [2.5e12],
                "pe_ratio": [25.0],
                # Missing other columns
            })
            df.to_csv(csv_path, index=False)

            result = load_raw_fundamentals(
                tickers=["AAPL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            # Should have all expected columns
            assert "roe" in result.columns
            assert "roa" in result.columns
            # Missing columns should be NaN
            assert result["roe"].isna().all()
            assert result["roa"].isna().all()
            # Provided columns should have values
            assert result["market_cap"].iloc[0] == 2.5e12


class TestSchemaCompatibility:
    """Tests for schema compatibility with download_fundamentals.py output."""

    def test_column_names_match_schema(self) -> None:
        """Column names should match the expected schema."""
        expected_columns = [
            "market_cap",
            "pe_ratio",
            "pb_ratio",
            "ps_ratio",
            "dividend_yield",
            "roe",
            "roa",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "revenue_ttm",
            "revenue_growth_1y",
            "eps_ttm",
            "eps_growth_1y",
        ]

        assert RAW_FUNDAMENTAL_COLUMNS == expected_columns

    def test_date_parsing_with_iso_format(self) -> None:
        """ISO date format (YYYY-MM-DD) should be parsed correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "dates.csv"
            df = pd.DataFrame({
                "date": ["2023-01-15", "2023-06-30", "2023-12-31"],
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "market_cap": [2.5e12, 2.6e12, 2.7e12],
                "pe_ratio": [25.0, 26.0, 27.0],
                "pb_ratio": [40.0, 41.0, 42.0],
                "ps_ratio": [7.0, 7.2, 7.4],
                "dividend_yield": [0.005, 0.005, 0.006],
                "roe": [1.5, 1.55, 1.6],
                "roa": [0.3, 0.31, 0.32],
                "gross_margin": [0.43, 0.435, 0.44],
                "operating_margin": [0.30, 0.305, 0.31],
                "net_margin": [0.25, 0.255, 0.26],
                "revenue_ttm": [3.8e11, 3.9e11, 4.0e11],
                "revenue_growth_1y": [0.08, 0.09, 0.10],
                "eps_ttm": [6.0, 6.2, 6.4],
                "eps_growth_1y": [0.10, 0.11, 0.12],
            })
            df.to_csv(csv_path, index=False)

            result = load_raw_fundamentals(
                tickers=["AAPL"],
                start="2023-01-01",
                end="2023-12-31",
                fundamentals_csv_path=str(csv_path),
                use_yfinance=False,
            )

            dates = result.index.get_level_values("date")
            assert pd.Timestamp("2023-01-15") in dates
            assert pd.Timestamp("2023-06-30") in dates
            assert pd.Timestamp("2023-12-31") in dates
