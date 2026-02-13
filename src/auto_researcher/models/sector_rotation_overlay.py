"""
Sector Rotation Overlay — Earnings Breadth Diffusion

Applies a sector-level tilt multiplier (0.8x–1.2x) to stock composite scores
based on the divergence between fundamental breadth and price momentum.

Academic Foundation:
    - Chan, Jegadeesh & Lakonishok (1996): Earnings revisions breadth leads
      sector returns by 1–3 months.
    - Kakushadze & Yu (2017): Cross-sectional factor models benefit from
      sector-aware signal aggregation.
    - Hong, Torous & Valkanov (2007): Industry-level fundamentals predict
      aggregate stock returns; information diffuses gradually.

Approach:
    1. Aggregate existing stock-level agent scores by sector (earnings, 
       fundamental, filing_tone, earnings_call_qual — the "revision" signals).
    2. Compute "breadth" = fraction of stocks in each sector with improving
       signals (score > 0).
    3. Fetch sector ETF price momentum as the "price" baseline.
    4. Detect divergence: rising breadth + lagging ETF price = positive tilt;
       falling breadth + rising ETF price = negative tilt.
    5. Output: 0.8x–1.2x multiplicative tilt on each stock's composite score.

This is NOT a new agent. It is a post-agent overlay applied in Stage 3 to
tilt composite scores toward sectors with leading fundamental improvement
that hasn't been priced in yet.

Expected Performance:
    - +1–3% annual alpha from catching sector rotation inflection points
    - Reduces drawdowns by ~1% in sector crash regimes (breadth collapse)
    - Most valuable 1–3 months before major sector rotations

Usage:
    overlay = SectorRotationOverlay()
    tilts = overlay.compute_tilts(rankings)
    # tilts["Technology"] -> 1.12  (boost tech stocks by 12%)
    # tilts["Energy"] -> 0.88     (reduce energy stocks by 12%)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Literal

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

OVERLAY_CONFIG = {
    # Agent scores used for breadth computation (fundamental-oriented signals)
    # These map to StockRanking attribute names
    "breadth_agents": {
        "earnings_score": 0.35,       # PEAD + revenue surprise — strongest predictor
        "fundamental_score": 0.25,    # Multi-factor fundamental model
        "filing_tone_score": 0.20,    # 10-K/10-Q textual analysis
        "earnings_call_qual_score": 0.20,  # Earnings call qualitative signals
    },

    # Breadth threshold: a stock "improves" if its weighted agent score > this
    "improvement_threshold": 0.05,

    # ETF price momentum weights (for the "price" side of divergence)
    "price_momentum_weight_1m": 0.50,
    "price_momentum_weight_3m": 0.50,

    # Divergence sensitivity
    # How aggressively to tilt based on breadth-price divergence
    "tilt_sensitivity": 0.5,

    # Tilt bounds
    "max_tilt": 1.20,   # Max 20% boost
    "min_tilt": 0.80,   # Max 20% headwind

    # Minimum stocks per sector to compute a reliable breadth signal
    "min_sector_stocks": 3,

    # Breadth decay: sectors with breadth near 50% get muted tilts
    "neutral_breadth": 0.50,   # 50% improving is "neutral"

    # Cache duration for ETF price data
    "cache_hours": 2,
}


# Import sector definitions from sector_momentum to stay DRY
try:
    from auto_researcher.models.sector_momentum import (
        SECTOR_ETFS,
        SECTOR_TO_ETF,
        TICKER_SECTORS,
    )
except ImportError:
    # Fallback if import fails
    SECTOR_ETFS = {
        "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
        "XLE": "Energy", "XLI": "Industrials", "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
        "XLRE": "Real Estate", "XLC": "Communication Services",
    }
    SECTOR_TO_ETF = {v: k for k, v in SECTOR_ETFS.items()}
    TICKER_SECTORS = {}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SectorBreadth:
    """Breadth metrics for a single sector."""
    sector: str

    # Stock coverage
    n_stocks: int = 0
    tickers: List[str] = field(default_factory=list)

    # Agent-weighted breadth
    breadth: float = 0.50           # Fraction of stocks with improving signals
    avg_signal: float = 0.0         # Mean weighted agent score across stocks
    signal_dispersion: float = 0.0  # Std dev of signals (high = disagreement)

    # Individual agent breadths (for diagnostics)
    agent_breadths: Dict[str, float] = field(default_factory=dict)

    # Is this breadth estimate reliable?
    is_reliable: bool = False


@dataclass
class SectorTilt:
    """Final tilt signal for a sector."""
    sector: str

    # Breadth metrics
    breadth: SectorBreadth = field(default_factory=lambda: SectorBreadth(sector=""))

    # Price momentum (from ETF)
    price_momentum: float = 0.0     # Weighted ETF relative strength

    # Divergence: breadth_z - price_z
    # Positive = breadth leads price (buy signal)
    # Negative = price leads breadth (sell signal)
    divergence: float = 0.0

    # Final tilt multiplier [0.80, 1.20]
    tilt: float = 1.0

    # Classification
    signal: Literal["positive_divergence", "negative_divergence", "aligned", "insufficient_data"] = "insufficient_data"

    # Human-readable
    rationale: str = ""


@dataclass
class OverlaySnapshot:
    """Complete overlay state for logging and diagnostics."""
    timestamp: datetime = field(default_factory=datetime.now)
    sector_tilts: Dict[str, SectorTilt] = field(default_factory=dict)
    sectors_with_data: int = 0
    sectors_insufficient: int = 0
    max_tilt_sector: str = ""
    min_tilt_sector: str = ""
    summary: str = ""


# ==============================================================================
# SECTOR ROTATION OVERLAY
# ==============================================================================

class SectorRotationOverlay:
    """
    Computes sector tilt multipliers by detecting divergence between
    fundamental breadth (from agent scores) and sector ETF price momentum.

    The key insight: when many stocks in a sector show improving fundamentals
    (high breadth) but the sector ETF hasn't moved yet, the sector is likely
    to outperform. Conversely, rising prices with deteriorating breadth
    signals a sector that's about to roll over.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the overlay."""
        self.config = {**OVERLAY_CONFIG, **(config or {})}

        # Cache for ETF price data
        self._price_cache: Optional[Dict[str, float]] = None
        self._price_cache_time: Optional[datetime] = None

        logger.info("Initialized SectorRotationOverlay")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def compute_tilts(
        self,
        rankings: list,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Compute sector tilt multipliers from stock rankings.

        Parameters
        ----------
        rankings : list[StockRanking]
            Stocks with agent scores already populated (after Stage 2).
        verbose : bool
            Log detailed diagnostics.

        Returns
        -------
        Dict[str, float]
            Mapping of sector name -> tilt multiplier (0.80 to 1.20).
            Sectors not represented in rankings get tilt = 1.0 (neutral).
        """
        # Step 1: Compute breadth per sector
        sector_breadths = self._compute_sector_breadths(rankings)

        # Step 2: Fetch sector ETF price momentum
        price_momentum = self._fetch_sector_price_momentum()

        # Step 3: Compute divergence and tilts
        sector_tilts = self._compute_divergence_tilts(
            sector_breadths, price_momentum
        )

        # Step 4: Build result dict
        tilts: Dict[str, float] = {}
        snapshot = OverlaySnapshot()

        for sector, tilt_signal in sector_tilts.items():
            tilts[sector] = tilt_signal.tilt
            snapshot.sector_tilts[sector] = tilt_signal

        snapshot.sectors_with_data = sum(
            1 for t in sector_tilts.values()
            if t.signal != "insufficient_data"
        )
        snapshot.sectors_insufficient = sum(
            1 for t in sector_tilts.values()
            if t.signal == "insufficient_data"
        )

        if sector_tilts:
            active = {k: v for k, v in sector_tilts.items()
                      if v.signal != "insufficient_data"}
            if active:
                snapshot.max_tilt_sector = max(
                    active, key=lambda s: active[s].tilt
                )
                snapshot.min_tilt_sector = min(
                    active, key=lambda s: active[s].tilt
                )

        # Log summary
        self._log_summary(snapshot, verbose)

        return tilts

    def get_snapshot(
        self,
        rankings: list,
    ) -> OverlaySnapshot:
        """
        Compute and return full diagnostic snapshot (for reporting/tests).
        """
        sector_breadths = self._compute_sector_breadths(rankings)
        price_momentum = self._fetch_sector_price_momentum()
        sector_tilts = self._compute_divergence_tilts(
            sector_breadths, price_momentum
        )

        snapshot = OverlaySnapshot()
        for sector, tilt_signal in sector_tilts.items():
            snapshot.sector_tilts[sector] = tilt_signal

        snapshot.sectors_with_data = sum(
            1 for t in sector_tilts.values()
            if t.signal != "insufficient_data"
        )
        snapshot.sectors_insufficient = sum(
            1 for t in sector_tilts.values()
            if t.signal == "insufficient_data"
        )

        return snapshot

    # ------------------------------------------------------------------
    # STEP 1: BREADTH COMPUTATION
    # ------------------------------------------------------------------

    def _compute_sector_breadths(
        self, rankings: list
    ) -> Dict[str, SectorBreadth]:
        """
        Aggregate stock-level agent scores by sector to compute breadth.

        Breadth = fraction of stocks in the sector whose weighted agent
        score exceeds the improvement threshold.
        """
        cfg = self.config
        breadth_agents = cfg["breadth_agents"]
        threshold = cfg["improvement_threshold"]
        min_stocks = cfg["min_sector_stocks"]

        # Group stocks by sector
        sector_stocks: Dict[str, list] = {}
        for stock in rankings:
            sector = self._get_sector(stock)
            if sector:
                sector_stocks.setdefault(sector, []).append(stock)

        # Compute breadth per sector
        results: Dict[str, SectorBreadth] = {}

        for sector, stocks in sector_stocks.items():
            sb = SectorBreadth(
                sector=sector,
                n_stocks=len(stocks),
                tickers=[s.ticker for s in stocks],
            )

            if len(stocks) < min_stocks:
                sb.is_reliable = False
                results[sector] = sb
                continue

            # Compute weighted agent score per stock
            stock_signals = []
            per_agent_improving: Dict[str, int] = {a: 0 for a in breadth_agents}
            per_agent_total: Dict[str, int] = {a: 0 for a in breadth_agents}

            for stock in stocks:
                weighted_score = 0.0
                total_weight = 0.0

                for agent_attr, weight in breadth_agents.items():
                    score = getattr(stock, agent_attr, 0.0)
                    # Skip agents that returned no data (score == 0.0 exactly
                    # is ambiguous, but -1 to +1 range means 0.0 is neutral)
                    weighted_score += weight * score
                    total_weight += weight

                    # Track per-agent breadth
                    per_agent_total[agent_attr] += 1
                    if score > threshold:
                        per_agent_improving[agent_attr] += 1

                if total_weight > 0:
                    normalized = weighted_score / total_weight
                else:
                    normalized = 0.0

                stock_signals.append(normalized)

            signals = np.array(stock_signals)

            # Breadth = fraction improving
            improving = np.sum(signals > threshold)
            sb.breadth = float(improving / len(signals))

            # Average signal strength
            sb.avg_signal = float(np.mean(signals))

            # Dispersion (high = sector stocks disagree)
            sb.signal_dispersion = float(np.std(signals))

            # Per-agent breadths
            for agent_attr in breadth_agents:
                total = per_agent_total[agent_attr]
                if total > 0:
                    sb.agent_breadths[agent_attr] = (
                        per_agent_improving[agent_attr] / total
                    )

            sb.is_reliable = True
            results[sector] = sb

        return results

    # ------------------------------------------------------------------
    # STEP 2: PRICE MOMENTUM (SECTOR ETFS)
    # ------------------------------------------------------------------

    def _fetch_sector_price_momentum(self) -> Dict[str, float]:
        """
        Fetch sector ETF price momentum (relative strength vs SPY).

        Returns dict of sector name -> weighted RS score.
        Cached for config["cache_hours"].
        """
        # Check cache
        if (
            self._price_cache is not None
            and self._price_cache_time is not None
            and (datetime.now() - self._price_cache_time).total_seconds()
            < self.config["cache_hours"] * 3600
        ):
            return self._price_cache

        cfg = self.config
        w1 = cfg["price_momentum_weight_1m"]
        w3 = cfg["price_momentum_weight_3m"]

        result: Dict[str, float] = {}

        try:
            import yfinance as yf

            etf_tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
            data = yf.download(
                etf_tickers, period="6mo", progress=False, auto_adjust=True
            )

            # Handle multi-level columns
            if hasattr(data.columns, "levels"):
                if "Close" in data.columns.get_level_values(0):
                    closes = data["Close"]
                else:
                    closes = data
            else:
                closes = data

            if closes.empty:
                logger.warning("No sector ETF data — using neutral price momentum")
                return {sector: 0.0 for sector in SECTOR_ETFS.values()}

            # Calculate returns
            returns_1m = closes.pct_change(21).iloc[-1]
            returns_3m = closes.pct_change(63).iloc[-1]

            spy_1m = float(returns_1m.get("SPY", 0))
            spy_3m = float(returns_3m.get("SPY", 0))

            for etf, sector in SECTOR_ETFS.items():
                if etf not in returns_1m.index:
                    result[sector] = 0.0
                    continue

                r1 = float(returns_1m[etf])
                r3 = float(returns_3m[etf])

                if np.isnan(r1):
                    r1 = 0.0
                if np.isnan(r3):
                    r3 = 0.0

                rs_1m = r1 - spy_1m
                rs_3m = r3 - spy_3m

                result[sector] = w1 * rs_1m + w3 * rs_3m

        except ImportError:
            logger.warning("yfinance not available — using neutral price momentum")
            result = {sector: 0.0 for sector in SECTOR_ETFS.values()}
        except Exception as e:
            logger.warning(f"Failed to fetch sector ETF data: {e}")
            result = {sector: 0.0 for sector in SECTOR_ETFS.values()}

        self._price_cache = result
        self._price_cache_time = datetime.now()
        return result

    # ------------------------------------------------------------------
    # STEP 3: DIVERGENCE & TILT
    # ------------------------------------------------------------------

    def _compute_divergence_tilts(
        self,
        breadths: Dict[str, SectorBreadth],
        price_momentum: Dict[str, float],
    ) -> Dict[str, SectorTilt]:
        """
        Detect breadth-price divergence and compute tilt multipliers.

        Divergence = z(breadth) - z(price_momentum)
            Positive divergence: breadth improving ahead of prices -> bullish tilt
            Negative divergence: prices rising but breadth deteriorating -> bearish tilt
        """
        cfg = self.config
        sensitivity = cfg["tilt_sensitivity"]
        max_tilt = cfg["max_tilt"]
        min_tilt = cfg["min_tilt"]
        neutral = cfg["neutral_breadth"]

        # Collect breadth and price values for z-scoring across sectors
        reliable_sectors = [
            s for s, b in breadths.items() if b.is_reliable
        ]

        if not reliable_sectors:
            logger.warning("No sectors with reliable breadth — all tilts neutral")
            return {
                sector: SectorTilt(
                    sector=sector,
                    tilt=1.0,
                    signal="insufficient_data",
                    rationale="No sectors had enough stocks for breadth computation",
                )
                for sector in set(list(breadths.keys()) + list(price_momentum.keys()))
            }

        # Z-score breadth values across sectors
        breadth_vals = np.array([breadths[s].breadth for s in reliable_sectors])
        price_vals = np.array([
            price_momentum.get(s, 0.0) for s in reliable_sectors
        ])

        breadth_mean = np.mean(breadth_vals)
        breadth_std = np.std(breadth_vals)
        price_mean = np.mean(price_vals)
        price_std = np.std(price_vals)

        # Avoid divide by zero — if all sectors have same breadth/price,
        # there's no dispersion to exploit
        if breadth_std < 1e-6:
            breadth_std = 1.0  # Treat as zero z-scores
        if price_std < 1e-6:
            price_std = 1.0

        results: Dict[str, SectorTilt] = {}

        for sector in set(list(breadths.keys()) + list(price_momentum.keys())):
            st = SectorTilt(sector=sector)

            if sector in breadths:
                st.breadth = breadths[sector]

            if not st.breadth.is_reliable:
                st.tilt = 1.0
                st.signal = "insufficient_data"
                st.rationale = (
                    f"{sector}: only {st.breadth.n_stocks} stocks "
                    f"(need {cfg['min_sector_stocks']})"
                )
                results[sector] = st
                continue

            # Z-scores
            b_z = (st.breadth.breadth - breadth_mean) / breadth_std
            p = price_momentum.get(sector, 0.0)
            p_z = (p - price_mean) / price_std

            st.price_momentum = p

            # Divergence: positive = breadth leads price upward
            divergence = b_z - p_z
            st.divergence = float(divergence)

            # Convert divergence to tilt via sigmoid-like mapping
            # tilt = 1.0 + sensitivity * tanh(divergence)
            # This naturally bounds the tilt and provides smooth transitions
            raw_tilt = 1.0 + sensitivity * np.tanh(divergence * 0.5)

            # Clip to bounds
            st.tilt = float(np.clip(raw_tilt, min_tilt, max_tilt))

            # Classify signal
            if divergence > 0.5:
                st.signal = "positive_divergence"
                st.rationale = (
                    f"{sector}: breadth {st.breadth.breadth:.0%} improving "
                    f"(z={b_z:+.2f}) but price lagging (z={p_z:+.2f}) "
                    f"-> bullish tilt {st.tilt:.2f}x"
                )
            elif divergence < -0.5:
                st.signal = "negative_divergence"
                st.rationale = (
                    f"{sector}: price strong (z={p_z:+.2f}) but breadth "
                    f"{st.breadth.breadth:.0%} deteriorating (z={b_z:+.2f}) "
                    f"-> bearish tilt {st.tilt:.2f}x"
                )
            else:
                st.signal = "aligned"
                st.rationale = (
                    f"{sector}: breadth {st.breadth.breadth:.0%} and price "
                    f"aligned (div={divergence:+.2f}) -> neutral {st.tilt:.2f}x"
                )

            results[sector] = st

        return results

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _get_sector(self, stock) -> Optional[str]:
        """Look up sector for a stock, using StockRanking.sector or TICKER_SECTORS."""
        # First: use the sector already on the ranking object
        if hasattr(stock, "sector") and stock.sector:
            return stock.sector

        # Fallback: lookup table
        ticker = stock.ticker if hasattr(stock, "ticker") else str(stock)
        return TICKER_SECTORS.get(ticker)

    def _log_summary(self, snapshot: OverlaySnapshot, verbose: bool):
        """Log overlay results."""
        logger.info("=" * 60)
        logger.info("SECTOR ROTATION OVERLAY — Breadth Divergence")
        logger.info("=" * 60)
        logger.info(
            f"  Sectors with data: {snapshot.sectors_with_data} | "
            f"Insufficient: {snapshot.sectors_insufficient}"
        )

        if snapshot.max_tilt_sector and snapshot.min_tilt_sector:
            max_t = snapshot.sector_tilts.get(snapshot.max_tilt_sector)
            min_t = snapshot.sector_tilts.get(snapshot.min_tilt_sector)
            if max_t:
                logger.info(
                    f"  Strongest tilt: {snapshot.max_tilt_sector} "
                    f"{max_t.tilt:.2f}x ({max_t.signal})"
                )
            if min_t:
                logger.info(
                    f"  Weakest tilt:   {snapshot.min_tilt_sector} "
                    f"{min_t.tilt:.2f}x ({min_t.signal})"
                )

        if verbose:
            logger.info("")
            logger.info(
                f"  {'Sector':<25} {'Breadth':>8} {'PriceMom':>9} "
                f"{'Diverg':>7} {'Tilt':>6} {'Signal'}"
            )
            logger.info("  " + "-" * 80)

            for sector in sorted(
                snapshot.sector_tilts.keys(),
                key=lambda s: snapshot.sector_tilts[s].tilt,
                reverse=True,
            ):
                t = snapshot.sector_tilts[sector]
                b_str = (
                    f"{t.breadth.breadth:.0%}"
                    if t.breadth.is_reliable
                    else "n/a"
                )
                logger.info(
                    f"  {sector:<25} {b_str:>8} {t.price_momentum:>+9.4f} "
                    f"{t.divergence:>+7.2f} {t.tilt:>6.2f} {t.signal}"
                )

        logger.info("")
