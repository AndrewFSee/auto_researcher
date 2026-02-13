"""
Factor Rotation Model — Regime-Aware Dynamic Factor Weighting

Detects factor regime shifts using LEADING indicators and adjusts agent
IC-weights in real time. Unlike traditional regime detection (lagging vol
+ trend), this module looks for signals that PRECEDE regime transitions.

Academic Foundation:
    - Asness, Moskowitz & Pedersen (2013): "Value and Momentum Everywhere"
      — factors rotate predictably with macro conditions.
    - Arnott, Beck & Kalesnik (2016): "Timing 'Smart Beta' Strategies"
      — factor momentum (recent IC trends) is a robust timing signal.
    - Bender, Sun, Thomas & Zdorovtsov (2018): "The Promises and Pitfalls
      of Factor Timing" — leading indicators beat lagging regime classifiers.
    - Gupta & Kelly (2019): "Factor Momentum Everywhere" — 1-12 month
      factor return momentum predicts future factor performance.

Leading Indicator Framework:
    1. VIX Term Structure (VX1/VX2 ratio)
       - Contango (ratio < 1.0): complacency → momentum/growth favoured
       - Backwardation (ratio > 1.0): fear → quality/value favoured
       - Slope CHANGE is leading: contango flattening = early warning

    2. Credit Spread Momentum (HYG/IEF ratio momentum)
       - Tightening spreads: risk appetite rising → momentum/thematic
       - Widening spreads: risk aversion → fundamental/quality
       - Rate of change leads equity regime by 2-4 weeks

    3. Factor Momentum (rolling 1m/3m factor IC trends)
       - Factors that have been increasing in IC tend to continue
       - Factors whose IC is declining should be down-weighted
       - Academic consensus: 1-3 month IC momentum is #1 predictor

    4. Cross-Sectional Dispersion
       - High dispersion → stock-picking matters → fundamental agents shine
       - Low dispersion → macro dominates → momentum/sentiment dominate
       - Measured via cross-sectional stdev of returns

    5. Breadth Thrust (from sector rotation overlay)
       - Rapid breadth improvement across sectors = risk-on inflection
       - Breadth collapse = risk-off inflection

Usage:
    rotation = FactorRotationModel()
    regime = rotation.detect_regime()
    # regime.state = "risk_on_early" / "risk_on_late" / "risk_off_early" / ...
    # regime.factor_weights = {"fundamental": 0.12, "momentum": 0.05, ...}
    adjusted_ics = rotation.adjust_ic_weights(base_ics)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Regime States
# =============================================================================

class FactorRegime(str, Enum):
    """
    Factor regime states based on leading indicators.

    Unlike binary MOMENTUM/QUALITY, we model 5 states to capture
    the TRANSITION phases (where most alpha lives).
    """
    RISK_ON_EARLY = "risk_on_early"     # Inflection: shifting from defensive to risk-on
    RISK_ON_LATE = "risk_on_late"       # Established risk-on, momentum crowded
    NEUTRAL = "neutral"                  # No clear factor tilt
    RISK_OFF_EARLY = "risk_off_early"   # Inflection: shifting from risk-on to defensive
    RISK_OFF_LATE = "risk_off_late"     # Established risk-off, quality crowded


# =============================================================================
# Configuration
# =============================================================================

# Factor weight profiles per regime (multipliers on base IC weights)
# These represent HOW MUCH to tilt each factor relative to its base IC weight.
# 1.0 = no change, >1 = overweight, <1 = underweight
REGIME_FACTOR_PROFILES: dict[FactorRegime, dict[str, float]] = {
    FactorRegime.RISK_ON_EARLY: {
        # Inflection point: fundamentals improving but price hasn't moved
        # → fundamental + earnings most valuable (catching the turn)
        "ml": 1.0,
        "sentiment": 1.3,        # Sentiment shifts first
        "fundamental": 1.4,      # Quality at a discount — best entry
        "earnings": 1.3,         # Earnings revisions lead price
        "insider": 1.2,          # Insiders buy early
        "thematic": 1.0,
        "momentum": 0.6,         # Momentum still stale — lagging
        "filing_tone": 1.3,      # Tone improvements are leading
        "earnings_call_qual": 1.2,
    },
    FactorRegime.RISK_ON_LATE: {
        # Established uptrend, momentum is king but getting crowded
        # → momentum + sentiment, but fade fundamentals (priced in)
        "ml": 1.0,
        "sentiment": 1.2,
        "fundamental": 0.8,      # Already priced in
        "earnings": 0.9,
        "insider": 0.7,          # Insiders start selling into strength
        "thematic": 1.3,         # Thematic/growth stories dominate late cycle
        "momentum": 1.5,         # Momentum is strongest late risk-on
        "filing_tone": 0.8,
        "earnings_call_qual": 0.9,
    },
    FactorRegime.NEUTRAL: {
        # No clear direction — use base weights
        "ml": 1.0,
        "sentiment": 1.0,
        "fundamental": 1.0,
        "earnings": 1.0,
        "insider": 1.0,
        "thematic": 1.0,
        "momentum": 1.0,
        "filing_tone": 1.0,
        "earnings_call_qual": 1.0,
    },
    FactorRegime.RISK_OFF_EARLY: {
        # Inflection: cracks forming, need to rotate to defense
        # → quality + fundamentals to avoid falling knives
        "ml": 1.0,
        "sentiment": 0.7,        # Sentiment lags in downturns (still bullish)
        "fundamental": 1.5,      # Quality screens protect capital
        "earnings": 1.4,         # Earnings resilience matters most
        "insider": 1.3,          # Insider selling is a strong warning
        "thematic": 0.6,         # Growth narratives get punished first
        "momentum": 0.5,         # Momentum reversals are violent
        "filing_tone": 1.4,      # Tone deterioration = early warning
        "earnings_call_qual": 1.3,
    },
    FactorRegime.RISK_OFF_LATE: {
        # Deep risk-off — max quality, value matters for recovery
        "ml": 1.0,
        "sentiment": 0.6,        # Sentiment extremely negative (contrarian value)
        "fundamental": 1.5,      # Survivors have strong fundamentals
        "earnings": 1.3,
        "insider": 1.4,          # Insider buying in risk-off = strong conviction
        "thematic": 0.5,         # Growth gets crushed
        "momentum": 0.4,         # Avoid momentum (crash losers)
        "filing_tone": 1.2,
        "earnings_call_qual": 1.2,
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LeadingIndicators:
    """Raw leading indicator readings."""

    # VIX term structure
    vix_spot: float | None = None
    vix_1m: float | None = None        # VX1 (front-month futures)
    vix_2m: float | None = None        # VX2 (second-month futures)
    vix_term_ratio: float | None = None  # VX1/VX2; <1 = contango, >1 = backwardation
    vix_term_ratio_5d_chg: float | None = None  # Directional change

    # Credit spreads
    hyg_ief_ratio: float | None = None       # HYG/IEF as credit spread proxy
    hyg_ief_ratio_1m_chg: float | None = None  # 1-month rate of change
    hyg_ief_ratio_3m_chg: float | None = None  # 3-month rate of change

    # Cross-sectional dispersion
    cs_dispersion_21d: float | None = None   # 21-day cross-sectional return stdev
    cs_dispersion_63d: float | None = None   # 63-day (3m)
    dispersion_trend: float | None = None    # Rising or falling?

    # Breadth (from current run's sector overlay data)
    overall_breadth: float | None = None     # Fraction of stocks improving
    breadth_change: float | None = None      # Change vs 1-month ago

    # Factor momentum (rolling IC trends, loaded from calibration history)
    factor_ic_1m: dict[str, float] = field(default_factory=dict)
    factor_ic_3m: dict[str, float] = field(default_factory=dict)
    factor_ic_momentum: dict[str, float] = field(default_factory=dict)  # 1m - 3m

    # Data availability
    indicators_available: int = 0
    indicators_total: int = 5
    data_quality: str = "limited"  # "full", "partial", "limited"


@dataclass
class FactorRegimeState:
    """Complete factor regime detection output."""

    # Detected regime
    regime: FactorRegime = FactorRegime.NEUTRAL
    regime_confidence: float = 0.0          # 0-1

    # Transition detection (the key value-add)
    transition_probability: float = 0.0     # P(regime is shifting)
    transition_direction: str = "none"      # "toward_risk_on", "toward_risk_off", "none"
    days_in_current_regime: int | None = None

    # Leading indicator scores (-1 to +1, positive = risk-on)
    vix_signal: float = 0.0
    credit_signal: float = 0.0
    dispersion_signal: float = 0.0
    breadth_signal: float = 0.0
    factor_momentum_signal: float = 0.0

    # Composite regime score (-1 = deep risk-off, +1 = deep risk-on)
    regime_score: float = 0.0

    # Adjusted factor weights (the actionable output)
    factor_weights: dict[str, float] = field(default_factory=dict)

    # Raw data for transparency
    indicators: LeadingIndicators = field(default_factory=LeadingIndicators)

    # Diagnostics
    rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Factor Rotation Model
# =============================================================================

class FactorRotationModel:
    """
    Detects factor regime shifts using leading indicators and adjusts
    agent IC-weights dynamically.

    Key insight: regime TRANSITIONS are where the alpha is. By the time
    a regime is "established", the factor rotation is already priced in.
    Leading indicators (VIX term structure, credit spreads, breadth momentum)
    signal transitions 2-6 weeks before they show up in trailing vol/trend.
    """

    def __init__(
        self,
        tilt_strength: float = 0.5,
        use_factor_momentum: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            tilt_strength: How aggressively to tilt (0 = no tilt, 1 = full profile).
                Default 0.5 = half-strength tilts (conservative).
            use_factor_momentum: Whether to incorporate rolling IC trends.
            verbose: Log diagnostic details.
        """
        self.tilt_strength = max(0.0, min(1.0, tilt_strength))
        self.use_factor_momentum = use_factor_momentum
        self.verbose = verbose

        # Price data cache
        self._price_cache: dict[str, pd.Series] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl = timedelta(hours=2)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def detect_regime(
        self,
        rankings: list | None = None,
    ) -> FactorRegimeState:
        """
        Detect the current factor regime using leading indicators.

        Args:
            rankings: Optional list of StockRanking dicts (used for breadth).

        Returns:
            FactorRegimeState with detected regime and adjusted weights.
        """
        t0 = time.time()

        # 1. Gather leading indicators
        indicators = self._gather_indicators(rankings)

        # 2. Score each indicator dimension (-1 to +1)
        vix_signal = self._score_vix_term_structure(indicators)
        credit_signal = self._score_credit_spreads(indicators)
        dispersion_signal = self._score_dispersion(indicators)
        breadth_signal = self._score_breadth(indicators)
        fm_signal = self._score_factor_momentum(indicators)

        # 3. Composite regime score (weighted average of available signals)
        signals = {
            "vix": (vix_signal, 0.25),
            "credit": (credit_signal, 0.25),
            "dispersion": (dispersion_signal, 0.15),
            "breadth": (breadth_signal, 0.15),
            "factor_momentum": (fm_signal, 0.20),
        }

        total_weight = 0.0
        weighted_sum = 0.0
        available_count = 0
        for name, (signal, weight) in signals.items():
            if signal is not None and abs(signal) > 0.001:
                weighted_sum += signal * weight
                total_weight += weight
                available_count += 1

        regime_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # 4. Map to regime state
        regime, confidence, transition_prob, transition_dir = self._classify_regime(
            regime_score, indicators
        )

        # 5. Compute adjusted factor weights
        factor_weights = self._compute_factor_weights(regime, confidence)

        # 6. Apply factor momentum overlay if enabled
        if self.use_factor_momentum and indicators.factor_ic_momentum:
            factor_weights = self._apply_factor_momentum(
                factor_weights, indicators.factor_ic_momentum
            )

        # Build result
        state = FactorRegimeState(
            regime=regime,
            regime_confidence=confidence,
            transition_probability=transition_prob,
            transition_direction=transition_dir,
            vix_signal=vix_signal or 0.0,
            credit_signal=credit_signal or 0.0,
            dispersion_signal=dispersion_signal or 0.0,
            breadth_signal=breadth_signal or 0.0,
            factor_momentum_signal=fm_signal or 0.0,
            regime_score=round(regime_score, 4),
            factor_weights=factor_weights,
            indicators=indicators,
            rationale=self._build_rationale(
                regime, confidence, regime_score,
                vix_signal, credit_signal, dispersion_signal,
                breadth_signal, fm_signal, transition_prob, transition_dir,
            ),
        )

        elapsed = int((time.time() - t0) * 1000)
        logger.info(f"Factor regime: {regime.value} (conf={confidence:.0%}, "
                     f"score={regime_score:+.3f}, transition={transition_prob:.0%} "
                     f"{transition_dir}) [{elapsed}ms]")

        return state

    def adjust_ic_weights(
        self,
        base_ics: dict[str, float],
        regime_state: FactorRegimeState | None = None,
        rankings: list | None = None,
    ) -> dict[str, float]:
        """
        Adjust IC weights based on current factor regime.

        This is the main entry point for compute_composite_scores to call.

        Args:
            base_ics: Original IC values {"sentiment": 0.08, "fundamental": 0.10, ...}
            regime_state: Pre-computed regime. If None, will detect.
            rankings: Passed to detect_regime if regime_state is None.

        Returns:
            Adjusted ICs (same keys, different values).
        """
        if regime_state is None:
            regime_state = self.detect_regime(rankings)

        adjusted = {}
        for factor, base_ic in base_ics.items():
            if factor == "ml":
                # Don't adjust ML — it's already regime-aware via features
                adjusted[factor] = base_ic
            elif factor in regime_state.factor_weights:
                multiplier = regime_state.factor_weights[factor]
                adjusted[factor] = base_ic * multiplier
            else:
                adjusted[factor] = base_ic

        # Log adjustments
        if self.verbose:
            logger.info("Factor rotation IC adjustments:")
            for f in base_ics:
                old = base_ics[f]
                new = adjusted[f]
                if abs(new - old) > 0.001:
                    pct = ((new / old) - 1) * 100 if old > 0 else 0
                    logger.info(f"  {f}: {old:.3f} → {new:.3f} ({pct:+.0f}%)")

        return adjusted

    # ------------------------------------------------------------------
    # INDICATOR GATHERING
    # ------------------------------------------------------------------

    def _gather_indicators(
        self,
        rankings: list | None = None,
    ) -> LeadingIndicators:
        """Fetch all leading indicators from market data."""
        indicators = LeadingIndicators()
        available = 0

        # 1. VIX term structure
        try:
            vix_data = self._fetch_vix_data()
            if vix_data:
                indicators.vix_spot = vix_data.get("spot")
                indicators.vix_1m = vix_data.get("vx1")
                indicators.vix_2m = vix_data.get("vx2")
                if indicators.vix_1m and indicators.vix_2m and indicators.vix_2m > 0:
                    indicators.vix_term_ratio = indicators.vix_1m / indicators.vix_2m
                indicators.vix_term_ratio_5d_chg = vix_data.get("term_ratio_5d_chg")
                available += 1
        except Exception as e:
            logger.debug(f"VIX data unavailable: {e}")

        # 2. Credit spreads (HYG/IEF ratio)
        try:
            credit_data = self._fetch_credit_data()
            if credit_data:
                indicators.hyg_ief_ratio = credit_data.get("ratio")
                indicators.hyg_ief_ratio_1m_chg = credit_data.get("ratio_1m_chg")
                indicators.hyg_ief_ratio_3m_chg = credit_data.get("ratio_3m_chg")
                available += 1
        except Exception as e:
            logger.debug(f"Credit data unavailable: {e}")

        # 3. Cross-sectional dispersion
        try:
            disp_data = self._fetch_dispersion_data()
            if disp_data:
                indicators.cs_dispersion_21d = disp_data.get("disp_21d")
                indicators.cs_dispersion_63d = disp_data.get("disp_63d")
                indicators.dispersion_trend = disp_data.get("trend")
                available += 1
        except Exception as e:
            logger.debug(f"Dispersion data unavailable: {e}")

        # 4. Breadth from rankings
        try:
            if rankings:
                breadth = self._compute_breadth_from_rankings(rankings)
                if breadth is not None:
                    indicators.overall_breadth = breadth.get("breadth")
                    indicators.breadth_change = breadth.get("breadth_change")
                    available += 1
        except Exception as e:
            logger.debug(f"Breadth computation failed: {e}")

        # 5. Factor momentum from IC history
        try:
            fm_data = self._load_factor_momentum()
            if fm_data:
                indicators.factor_ic_1m = fm_data.get("ic_1m", {})
                indicators.factor_ic_3m = fm_data.get("ic_3m", {})
                indicators.factor_ic_momentum = fm_data.get("momentum", {})
                available += 1
        except Exception as e:
            logger.debug(f"Factor momentum unavailable: {e}")

        indicators.indicators_available = available
        if available >= 4:
            indicators.data_quality = "full"
        elif available >= 2:
            indicators.data_quality = "partial"
        else:
            indicators.data_quality = "limited"

        if self.verbose:
            logger.info(f"Leading indicators: {available}/{indicators.indicators_total} "
                        f"available ({indicators.data_quality})")

        return indicators

    # ------------------------------------------------------------------
    # DATA FETCHING
    # ------------------------------------------------------------------

    def _get_cached_prices(self, ticker: str, period: str = "6mo") -> pd.Series | None:
        """Fetch price series with caching."""
        now = datetime.now()
        if (self._cache_time and (now - self._cache_time) < self._cache_ttl
                and ticker in self._price_cache):
            return self._price_cache[ticker]

        try:
            import yfinance as yf
            data = yf.Ticker(ticker).history(period=period)
            if data is not None and len(data) > 20:
                prices = data["Close"]
                self._price_cache[ticker] = prices
                self._cache_time = now
                return prices
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")

        return None

    def _fetch_vix_data(self) -> dict | None:
        """
        Fetch VIX term structure data.

        Uses ^VIX (spot), ^VIX3M (3-month VIX) as a proxy for VX1/VX2 ratio.
        The VIX/VIX3M ratio captures the same contango/backwardation signal
        that institutional desks monitor via VX1/VX2 futures.
        """
        vix_prices = self._get_cached_prices("^VIX")
        vix3m_prices = self._get_cached_prices("^VIX3M")

        if vix_prices is None or len(vix_prices) < 10:
            return None

        result: dict[str, Any] = {"spot": float(vix_prices.iloc[-1])}

        if vix3m_prices is not None and len(vix3m_prices) >= 10:
            # Align dates
            common = vix_prices.index.intersection(vix3m_prices.index)
            if len(common) >= 10:
                vix_aligned = vix_prices.loc[common]
                vix3m_aligned = vix3m_prices.loc[common]
                ratio = vix_aligned / vix3m_aligned

                result["vx1"] = float(vix_aligned.iloc[-1])
                result["vx2"] = float(vix3m_aligned.iloc[-1])

                # 5-day change in term structure ratio
                if len(ratio) >= 6:
                    result["term_ratio_5d_chg"] = float(
                        ratio.iloc[-1] - ratio.iloc[-6]
                    )
        return result

    def _fetch_credit_data(self) -> dict | None:
        """
        Fetch credit spread proxy via HYG/IEF ratio.

        HYG (high yield bonds) / IEF (7-10yr treasuries) ratio rises when
        credit spreads tighten (risk-on) and falls when spreads widen (risk-off).
        """
        hyg = self._get_cached_prices("HYG")
        ief = self._get_cached_prices("IEF")

        if hyg is None or ief is None or len(hyg) < 63:
            return None

        common = hyg.index.intersection(ief.index)
        if len(common) < 63:
            return None

        hyg_aligned = hyg.loc[common]
        ief_aligned = ief.loc[common]
        ratio = hyg_aligned / ief_aligned

        current = float(ratio.iloc[-1])
        ratio_1m = float(ratio.iloc[-22]) if len(ratio) >= 22 else None
        ratio_3m = float(ratio.iloc[-63]) if len(ratio) >= 63 else None

        result: dict[str, Any] = {"ratio": current}
        if ratio_1m:
            result["ratio_1m_chg"] = (current - ratio_1m) / ratio_1m
        if ratio_3m:
            result["ratio_3m_chg"] = (current - ratio_3m) / ratio_3m

        return result

    def _fetch_dispersion_data(self) -> dict | None:
        """
        Compute cross-sectional dispersion using sector ETF returns.

        High dispersion = stock-picking environment (fundamental factors matter).
        Low dispersion = macro-driven market (momentum/sentiment dominate).
        """
        etf_tickers = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU"]

        returns_list = []
        for etf in etf_tickers:
            prices = self._get_cached_prices(etf, period="6mo")
            if prices is not None and len(prices) > 63:
                ret = prices.pct_change().dropna()
                returns_list.append(ret)

        if len(returns_list) < 5:
            return None

        # Align all returns to common dates
        df = pd.concat(returns_list, axis=1, join="inner")
        if len(df) < 63:
            return None

        # Cross-sectional standard deviation at each date
        cs_std = df.std(axis=1)

        disp_21d = float(cs_std.iloc[-21:].mean()) if len(cs_std) >= 21 else None
        disp_63d = float(cs_std.iloc[-63:].mean()) if len(cs_std) >= 63 else None

        result: dict[str, Any] = {}
        if disp_21d is not None:
            result["disp_21d"] = disp_21d
        if disp_63d is not None:
            result["disp_63d"] = disp_63d
        if disp_21d is not None and disp_63d is not None and disp_63d > 0:
            # Trend: positive = dispersion rising, negative = dispersion falling
            result["trend"] = (disp_21d - disp_63d) / disp_63d

        return result if result else None

    def _compute_breadth_from_rankings(
        self, rankings: list
    ) -> dict | None:
        """Compute overall breadth from the current ranking run."""
        improving = 0
        total = 0
        for stock in rankings:
            if isinstance(stock, dict):
                cs = stock.get("composite_score", 0)
            else:
                cs = getattr(stock, "composite_score", 0)
            total += 1
            if cs > 0.05:
                improving += 1

        if total < 10:
            return None

        breadth = improving / total
        # breadth_change requires historical data — use None for now
        return {"breadth": breadth, "breadth_change": None}

    def _load_factor_momentum(self) -> dict | None:
        """
        Load factor momentum from IC calibration history.

        Reads data/agent_ic.json looking for time-series IC data.
        If only aggregate ICs are available, treats them as 3m IC
        and skips momentum computation.
        """
        import json
        from pathlib import Path

        ic_file = Path("data/agent_ic.json")
        if not ic_file.exists():
            return None

        try:
            with open(ic_file) as f:
                calibrated = json.load(f)
        except Exception:
            return None

        factors = [
            "sentiment", "fundamental", "earnings", "insider",
            "thematic", "momentum", "filing_tone", "earnings_call_qual",
        ]

        ic_3m: dict[str, float] = {}
        for factor in factors:
            if factor in calibrated and "mean_ic" in calibrated[factor]:
                ic_3m[factor] = abs(calibrated[factor]["mean_ic"])

        if not ic_3m:
            return None

        # If we have rolling window ICs (e.g., ic_1m), use them
        ic_1m: dict[str, float] = {}
        momentum: dict[str, float] = {}
        for factor in factors:
            if factor in calibrated:
                ic_recent = calibrated[factor].get("ic_1m") or calibrated[factor].get("recent_ic")
                if ic_recent is not None:
                    ic_1m[factor] = abs(ic_recent)
                    if factor in ic_3m and ic_3m[factor] > 0:
                        # Momentum = whether the factor's IC is rising or falling
                        momentum[factor] = (ic_1m[factor] - ic_3m[factor]) / ic_3m[factor]

        return {
            "ic_1m": ic_1m,
            "ic_3m": ic_3m,
            "momentum": momentum,
        }

    # ------------------------------------------------------------------
    # SIGNAL SCORING (each returns -1 to +1, positive = risk-on)
    # ------------------------------------------------------------------

    def _score_vix_term_structure(self, ind: LeadingIndicators) -> float | None:
        """
        Score VIX term structure for regime direction.

        Key insight: the CHANGE in term structure predicts regime transitions
        better than the level. Flattening contango = early risk-off warning.
        """
        if ind.vix_term_ratio is None:
            return None

        # Base level signal: contango = risk-on, backwardation = risk-off
        # Normal contango: ratio ~0.85-0.95. Backwardation: ratio > 1.0
        level_signal = np.clip(1.0 - ind.vix_term_ratio, -1.0, 1.0) * 1.5
        level_signal = np.clip(level_signal, -1.0, 1.0)

        # Rate of change is the LEADING component
        if ind.vix_term_ratio_5d_chg is not None:
            # Negative change = contango deepening = risk-on momentum
            # Positive change = contango flattening = risk-off warning
            change_signal = np.clip(-ind.vix_term_ratio_5d_chg * 10, -1.0, 1.0)
            # Weight change signal more heavily (it's the leading indicator)
            return float(np.clip(0.4 * level_signal + 0.6 * change_signal, -1.0, 1.0))

        return float(np.clip(level_signal, -1.0, 1.0))

    def _score_credit_spreads(self, ind: LeadingIndicators) -> float | None:
        """
        Score credit spread momentum.

        Tightening spreads (rising HYG/IEF) = risk appetite.
        Rate of change leads equity factors by 2-4 weeks (Bender et al. 2018).
        """
        if ind.hyg_ief_ratio_1m_chg is None:
            return None

        # 1m change is the primary signal
        # Typical range: -3% to +3% per month
        signal_1m = np.clip(ind.hyg_ief_ratio_1m_chg * 20, -1.0, 1.0)

        # Add 3m trend for confirmation
        if ind.hyg_ief_ratio_3m_chg is not None:
            signal_3m = np.clip(ind.hyg_ief_ratio_3m_chg * 10, -1.0, 1.0)
            return float(np.clip(0.6 * signal_1m + 0.4 * signal_3m, -1.0, 1.0))

        return float(signal_1m)

    def _score_dispersion(self, ind: LeadingIndicators) -> float | None:
        """
        Score cross-sectional dispersion.

        High dispersion → fundamentals matter → positive for fundamental factors
        Low dispersion → macro dominates → positive for momentum/sentiment

        We invert the sign: high dispersion = slightly risk-off (more defensive
        factor mix), low dispersion = slightly risk-on (momentum works).
        """
        if ind.dispersion_trend is None:
            return None

        # Rising dispersion = more stock-picking, slightly risk-off
        # This is a mild signal — cap at ±0.5 to avoid over-influence
        signal = np.clip(-ind.dispersion_trend * 3, -0.5, 0.5)
        return float(signal)

    def _score_breadth(self, ind: LeadingIndicators) -> float | None:
        """
        Score market breadth.

        High breadth (>60%) with positive change = risk-on
        Low breadth (<40%) with negative change = risk-off
        """
        if ind.overall_breadth is None:
            return None

        # Normalize: 50% breadth = 0, 70% = +1, 30% = -1
        breadth_signal = np.clip((ind.overall_breadth - 0.5) * 5, -1.0, 1.0)

        if ind.breadth_change is not None:
            # Change component is leading
            change_signal = np.clip(ind.breadth_change * 10, -1.0, 1.0)
            return float(np.clip(0.5 * breadth_signal + 0.5 * change_signal, -1.0, 1.0))

        return float(breadth_signal)

    def _score_factor_momentum(self, ind: LeadingIndicators) -> float | None:
        """
        Score factor momentum (which factors are gaining/losing IC).

        Rising IC in risk-on factors (momentum, sentiment) = risk-on
        Rising IC in risk-off factors (fundamental, earnings) = risk-off
        """
        if not ind.factor_ic_momentum:
            return None

        risk_on_factors = {"momentum", "sentiment", "thematic"}
        risk_off_factors = {"fundamental", "earnings", "filing_tone", "earnings_call_qual", "insider"}

        on_score = 0.0
        on_count = 0
        off_score = 0.0
        off_count = 0

        for factor, mom in ind.factor_ic_momentum.items():
            if factor in risk_on_factors:
                on_score += mom
                on_count += 1
            elif factor in risk_off_factors:
                off_score += mom
                off_count += 1

        if on_count == 0 and off_count == 0:
            return None

        # If risk-on factors are gaining IC faster → positive signal
        avg_on = on_score / max(on_count, 1)
        avg_off = off_score / max(off_count, 1)
        net = avg_on - avg_off

        return float(np.clip(net * 2, -1.0, 1.0))

    # ------------------------------------------------------------------
    # REGIME CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        regime_score: float,
        indicators: LeadingIndicators,
    ) -> tuple[FactorRegime, float, float, str]:
        """
        Classify the factor regime from composite score.

        Returns:
            (regime, confidence, transition_probability, transition_direction)
        """
        score = regime_score

        # Primary regime classification
        if score > 0.4:
            regime = FactorRegime.RISK_ON_LATE
        elif score > 0.15:
            regime = FactorRegime.RISK_ON_EARLY
        elif score < -0.4:
            regime = FactorRegime.RISK_OFF_LATE
        elif score < -0.15:
            regime = FactorRegime.RISK_OFF_EARLY
        else:
            regime = FactorRegime.NEUTRAL

        # Confidence based on signal agreement + data availability
        data_factor = indicators.indicators_available / indicators.indicators_total
        strength_factor = min(abs(score) * 2.5, 1.0)
        confidence = round(data_factor * 0.4 + strength_factor * 0.6, 3)

        # Transition detection: look for leading indicators diverging from
        # the current regime. E.g., regime is risk-on but VIX term structure
        # is flattening and credit is widening → transition toward risk-off.
        transition_prob = 0.0
        transition_dir = "none"

        if indicators.vix_term_ratio_5d_chg is not None:
            # VIX term structure change opposing current regime
            if score > 0.1 and indicators.vix_term_ratio_5d_chg > 0.02:
                # Risk-on regime but VIX contango flattening → risk-off early warning
                transition_prob += 0.3
                transition_dir = "toward_risk_off"
            elif score < -0.1 and indicators.vix_term_ratio_5d_chg < -0.02:
                # Risk-off regime but VIX backwardation resolving → risk-on early warning
                transition_prob += 0.3
                transition_dir = "toward_risk_on"

        if indicators.hyg_ief_ratio_1m_chg is not None:
            # Credit spread direction opposing regime
            if score > 0.1 and indicators.hyg_ief_ratio_1m_chg < -0.01:
                transition_prob += 0.3
                if transition_dir == "none":
                    transition_dir = "toward_risk_off"
            elif score < -0.1 and indicators.hyg_ief_ratio_1m_chg > 0.01:
                transition_prob += 0.3
                if transition_dir == "none":
                    transition_dir = "toward_risk_on"

        transition_prob = round(min(transition_prob, 0.9), 2)

        return regime, confidence, transition_prob, transition_dir

    def _compute_factor_weights(
        self,
        regime: FactorRegime,
        confidence: float,
    ) -> dict[str, float]:
        """
        Compute factor weight multipliers from regime and confidence.

        Blends the regime profile with neutral (1.0) based on:
        - Confidence (low confidence → closer to neutral)
        - tilt_strength parameter (user control)
        """
        profile = REGIME_FACTOR_PROFILES[regime]
        effective_strength = self.tilt_strength * confidence

        weights = {}
        for factor, target_mult in profile.items():
            # Blend between 1.0 (neutral) and target multiplier
            weights[factor] = 1.0 + (target_mult - 1.0) * effective_strength

        return weights

    def _apply_factor_momentum(
        self,
        weights: dict[str, float],
        factor_momentum: dict[str, float],
    ) -> dict[str, float]:
        """
        Overlay factor momentum: tilt toward factors whose IC is rising.

        This is the single strongest timing signal in the literature
        (Gupta & Kelly 2019). Applied as a secondary overlay on top of
        the regime-based weights.
        """
        adjusted = dict(weights)
        fm_strength = 0.3  # Factor momentum overlay at 30% strength

        for factor, mom in factor_momentum.items():
            if factor in adjusted:
                # Positive momentum = IC rising = boost weight
                # Cap the momentum adjustment at ±20%
                fm_tilt = np.clip(mom * fm_strength, -0.2, 0.2)
                adjusted[factor] *= (1.0 + fm_tilt)

        return adjusted

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def _build_rationale(
        self,
        regime: FactorRegime,
        confidence: float,
        score: float,
        vix: float | None,
        credit: float | None,
        dispersion: float | None,
        breadth: float | None,
        fm: float | None,
        transition_prob: float,
        transition_dir: str,
    ) -> str:
        """Build human-readable rationale for the regime classification."""
        parts = [f"Factor regime: {regime.value} (score={score:+.3f}, conf={confidence:.0%})"]

        signals = []
        if vix is not None:
            direction = "risk-on" if vix > 0 else "risk-off"
            signals.append(f"VIX term structure → {direction} ({vix:+.2f})")
        if credit is not None:
            direction = "tightening" if credit > 0 else "widening"
            signals.append(f"Credit spreads {direction} ({credit:+.2f})")
        if dispersion is not None:
            direction = "falling" if dispersion > 0 else "rising"
            signals.append(f"Dispersion {direction} ({dispersion:+.2f})")
        if breadth is not None:
            direction = "expanding" if breadth > 0 else "contracting"
            signals.append(f"Breadth {direction} ({breadth:+.2f})")
        if fm is not None:
            direction = "risk-on gaining" if fm > 0 else "risk-off gaining"
            signals.append(f"Factor momentum: {direction} ({fm:+.2f})")

        if signals:
            parts.append("Leading indicators: " + "; ".join(signals))

        if transition_prob > 0.1:
            parts.append(
                f"⚠️ Transition warning: {transition_prob:.0%} probability of "
                f"shift {transition_dir.replace('_', ' ')}"
            )

        return " | ".join(parts)
