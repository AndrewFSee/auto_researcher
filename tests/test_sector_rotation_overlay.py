"""
Tests for SectorRotationOverlay: breadth computation, divergence detection,
tilt calculation, and edge cases.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest


# ── Minimal StockRanking stub matching pipeline dataclass shape ──────────
@dataclass
class FakeStockRanking:
    """Minimal replica of StockRanking for testing."""
    ticker: str
    sector: str = ""
    earnings_score: float = 0.0
    fundamental_score: float = 0.0
    filing_tone_score: float = 0.0
    earnings_call_qual_score: float = 0.0
    sentiment_score: float = 0.0
    insider_score: float = 0.0
    thematic_score: float = 0.0
    momentum_score: float = 0.0
    ml_score: float = 0.0
    ml_percentile: float = 50.0
    composite_score: float = 0.0
    agent_rationales: dict = field(default_factory=dict)


def make_sector_stocks(sector: str, n: int, avg_signal: float) -> list:
    """Create n stocks in a sector with signals centered around avg_signal."""
    stocks = []
    for i in range(n):
        # Add some dispersion
        noise = (i - n / 2) * 0.05
        sig = avg_signal + noise
        stocks.append(FakeStockRanking(
            ticker=f"{sector[:3].upper()}{i}",
            sector=sector,
            earnings_score=sig * 1.1,
            fundamental_score=sig * 0.9,
            filing_tone_score=sig * 0.8,
            earnings_call_qual_score=sig * 0.7,
        ))
    return stocks


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def overlay():
    """Create SectorRotationOverlay instance."""
    from auto_researcher.models.sector_rotation_overlay import SectorRotationOverlay
    return SectorRotationOverlay()


@pytest.fixture
def diverse_rankings():
    """
    Create rankings with clear divergence:
    - Technology: high breadth (many improving stocks)
    - Energy: low breadth (most stocks deteriorating)
    - Financials: mixed / neutral
    """
    stocks = []
    stocks.extend(make_sector_stocks("Technology", 8, 0.40))
    stocks.extend(make_sector_stocks("Energy", 6, -0.30))
    stocks.extend(make_sector_stocks("Financials", 5, 0.05))
    return stocks


@pytest.fixture
def uniform_rankings():
    """All sectors have similar breadth — should produce neutral tilts."""
    stocks = []
    for sector in ["Technology", "Energy", "Financials", "Healthcare"]:
        stocks.extend(make_sector_stocks(sector, 5, 0.10))
    return stocks


# ── Test: Breadth Computation ────────────────────────────────────────────


class TestBreadthComputation:
    """Test the _compute_sector_breadths method."""

    def test_high_breadth_sector(self, overlay, diverse_rankings):
        """Technology with positive scores should have high breadth."""
        breadths = overlay._compute_sector_breadths(diverse_rankings)
        assert "Technology" in breadths
        tech = breadths["Technology"]
        assert tech.is_reliable  # 8 stocks > min_sector_stocks (3)
        assert tech.breadth > 0.6  # Most tech stocks improving
        assert tech.n_stocks == 8

    def test_low_breadth_sector(self, overlay, diverse_rankings):
        """Energy with negative scores should have low breadth."""
        breadths = overlay._compute_sector_breadths(diverse_rankings)
        assert "Energy" in breadths
        energy = breadths["Energy"]
        assert energy.is_reliable
        assert energy.breadth < 0.4  # Most energy stocks deteriorating

    def test_sector_with_few_stocks_unreliable(self, overlay):
        """Sectors with < min_sector_stocks should be marked unreliable."""
        stocks = [
            FakeStockRanking(ticker="AAA", sector="Utilities", earnings_score=0.5),
            FakeStockRanking(ticker="BBB", sector="Utilities", earnings_score=0.5),
        ]
        breadths = overlay._compute_sector_breadths(stocks)
        assert "Utilities" in breadths
        assert not breadths["Utilities"].is_reliable

    def test_empty_rankings(self, overlay):
        """Empty rankings should produce empty breadths."""
        breadths = overlay._compute_sector_breadths([])
        assert len(breadths) == 0

    def test_dispersion_computed(self, overlay, diverse_rankings):
        """Signal dispersion should be positive for diverse signals."""
        breadths = overlay._compute_sector_breadths(diverse_rankings)
        tech = breadths["Technology"]
        assert tech.signal_dispersion >= 0.0

    def test_agent_breadths_tracked(self, overlay, diverse_rankings):
        """Per-agent breadths should be tracked in diagnostics."""
        breadths = overlay._compute_sector_breadths(diverse_rankings)
        tech = breadths["Technology"]
        assert "earnings_score" in tech.agent_breadths
        assert "fundamental_score" in tech.agent_breadths


# ── Test: Divergence & Tilt ──────────────────────────────────────────────


class TestDivergenceTilt:
    """Test divergence detection and tilt multiplier computation."""

    def test_positive_divergence_positive_tilt(self, overlay, diverse_rankings):
        """
        When breadth is high but price momentum is low,
        divergence should be positive -> tilt > 1.0.
        """
        breadths = overlay._compute_sector_breadths(diverse_rankings)

        # Mock price momentum: Technology lagging, Energy strong
        price_momentum = {
            "Technology": -0.03,   # Lagging prices
            "Energy": 0.05,       # Strong prices
            "Financials": 0.00,
        }

        tilts = overlay._compute_divergence_tilts(breadths, price_momentum)
        assert "Technology" in tilts
        # Tech: high breadth + low price = positive divergence = bullish tilt
        assert tilts["Technology"].tilt > 1.0
        assert tilts["Technology"].signal in ("positive_divergence", "aligned")

    def test_negative_divergence_negative_tilt(self, overlay, diverse_rankings):
        """
        When breadth is low but price momentum is high,
        divergence should be negative -> tilt < 1.0.
        """
        breadths = overlay._compute_sector_breadths(diverse_rankings)

        price_momentum = {
            "Technology": -0.03,
            "Energy": 0.05,       # Strong prices despite weak breadth
            "Financials": 0.00,
        }

        tilts = overlay._compute_divergence_tilts(breadths, price_momentum)
        assert "Energy" in tilts
        # Energy: low breadth + high price = negative divergence = bearish tilt
        assert tilts["Energy"].tilt < 1.0

    def test_tilt_within_bounds(self, overlay, diverse_rankings):
        """All tilts should be clamped to [min_tilt, max_tilt]."""
        breadths = overlay._compute_sector_breadths(diverse_rankings)
        price_momentum = {
            "Technology": -0.10,  # Extreme lag
            "Energy": 0.10,      # Extreme strength
            "Financials": 0.00,
        }

        tilts = overlay._compute_divergence_tilts(breadths, price_momentum)
        for sector, tilt_signal in tilts.items():
            assert 0.80 <= tilt_signal.tilt <= 1.20, (
                f"{sector} tilt {tilt_signal.tilt} out of bounds"
            )

    def test_uniform_breadth_near_neutral(self, overlay, uniform_rankings):
        """When all sectors have similar breadth, tilts should be near 1.0."""
        breadths = overlay._compute_sector_breadths(uniform_rankings)
        # All sectors same price momentum
        price_momentum = {s: 0.0 for s in breadths}
        tilts = overlay._compute_divergence_tilts(breadths, price_momentum)

        for sector, tilt_signal in tilts.items():
            if tilt_signal.signal != "insufficient_data":
                assert abs(tilt_signal.tilt - 1.0) < 0.10, (
                    f"{sector} should be near-neutral but got {tilt_signal.tilt}"
                )

    def test_insufficient_data_neutral(self, overlay):
        """Sectors with insufficient data should get tilt=1.0."""
        # Only 2 stocks in one sector — unreliable
        stocks = [
            FakeStockRanking(ticker="A", sector="Materials", earnings_score=0.5),
            FakeStockRanking(ticker="B", sector="Materials", earnings_score=0.3),
        ]
        breadths = overlay._compute_sector_breadths(stocks)
        tilts = overlay._compute_divergence_tilts(breadths, {"Materials": 0.05})
        assert tilts["Materials"].tilt == 1.0
        assert tilts["Materials"].signal == "insufficient_data"


# ── Test: Full compute_tilts Flow ────────────────────────────────────────


class TestComputeTilts:
    """Test the end-to-end compute_tilts method."""

    def test_returns_dict_of_floats(self, overlay, diverse_rankings):
        """compute_tilts should return {sector: float} dict."""
        with patch.object(
            overlay, "_fetch_sector_price_momentum",
            return_value={
                "Technology": -0.02,
                "Energy": 0.04,
                "Financials": 0.00,
            },
        ):
            tilts = overlay.compute_tilts(diverse_rankings)
            assert isinstance(tilts, dict)
            for sector, tilt in tilts.items():
                assert isinstance(tilt, float)

    def test_empty_rankings_returns_empty(self, overlay):
        """Empty rankings should produce empty tilts."""
        with patch.object(
            overlay, "_fetch_sector_price_momentum",
            return_value={},
        ):
            tilts = overlay.compute_tilts([])
            assert len(tilts) == 0

    def test_snapshot_contains_diagnostics(self, overlay, diverse_rankings):
        """get_snapshot should return full OverlaySnapshot with diagnostics."""
        with patch.object(
            overlay, "_fetch_sector_price_momentum",
            return_value={
                "Technology": -0.02,
                "Energy": 0.04,
                "Financials": 0.00,
            },
        ):
            snapshot = overlay.get_snapshot(diverse_rankings)
            assert snapshot.sectors_with_data > 0
            assert "Technology" in snapshot.sector_tilts


# ── Test: Sector Lookup ──────────────────────────────────────────────────


class TestSectorLookup:
    """Test _get_sector helper."""

    def test_uses_stock_sector_attribute(self, overlay):
        """Should prefer sector from stock object."""
        stock = FakeStockRanking(ticker="AAPL", sector="Technology")
        assert overlay._get_sector(stock) == "Technology"

    def test_fallback_to_ticker_sectors(self, overlay):
        """Should fallback to TICKER_SECTORS if stock.sector is empty."""
        stock = FakeStockRanking(ticker="AAPL", sector="")
        sector = overlay._get_sector(stock)
        # AAPL is in TICKER_SECTORS
        assert sector == "Technology"

    def test_unknown_ticker_returns_none(self, overlay):
        """Unknown ticker with no sector should return None."""
        stock = FakeStockRanking(ticker="ZZZZZZZ", sector="")
        assert overlay._get_sector(stock) is None


# ── Test: Configuration ──────────────────────────────────────────────────


class TestConfiguration:
    """Test custom configuration."""

    def test_custom_tilt_bounds(self):
        from auto_researcher.models.sector_rotation_overlay import SectorRotationOverlay
        overlay = SectorRotationOverlay(config={
            "max_tilt": 1.10,
            "min_tilt": 0.90,
        })
        assert overlay.config["max_tilt"] == 1.10
        assert overlay.config["min_tilt"] == 0.90

    def test_custom_min_sector_stocks(self):
        from auto_researcher.models.sector_rotation_overlay import SectorRotationOverlay
        overlay = SectorRotationOverlay(config={"min_sector_stocks": 5})
        stocks = make_sector_stocks("Technology", 4, 0.3)
        breadths = overlay._compute_sector_breadths(stocks)
        # 4 stocks < 5 min -> unreliable
        assert not breadths["Technology"].is_reliable


# ── Test: Module imports ─────────────────────────────────────────────────


class TestModuleImport:
    """Test that the module is importable from the package."""

    def test_import_from_models(self):
        from auto_researcher.models import SectorRotationOverlay
        assert SectorRotationOverlay is not None

    def test_import_dataclasses(self):
        from auto_researcher.models import SectorBreadth, SectorTilt, OverlaySnapshot
        assert SectorBreadth is not None
        assert SectorTilt is not None
        assert OverlaySnapshot is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
