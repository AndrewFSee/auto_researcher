"""
Fundamentals Alpha Model - DISABLED (Forward Bias Detected)

================================================================================
⚠️  WARNING: SIGNALS DISABLED DUE TO FORWARD BIAS (2026-01-27 Audit)
================================================================================

This module has been DISABLED after a comprehensive forward-bias audit revealed
that ALL signals from DefeatBeta data are unreliable for backtesting:

1. ESTIMATE REVISIONS: COMPLETELY INVALID
   - DefeatBeta stock_earning_estimates contains only a SINGLE SNAPSHOT
   - 28,602 total rows, but only 1 row per ticker-period (not historical)
   - 9,003 rows have future dates (2026-2027) - this is CURRENT consensus data
   - Cannot backtest what estimates were at any historical point in time
   - Originally reported IC=+0.082*** was entirely fake from look-ahead bias
   
2. PEAD (Post-Earnings Drift): SEVERELY BIASED
   - DefeatBeta stock_historical_eps uses 'quarter_date' as fiscal period END
   - This is NOT the earnings announcement date
   - Returns were calculated 30-45 days BEFORE the EPS surprise was known
   - Originally reported IC=+0.173*** was contaminated by forward information
   
3. FCF YIELD / DCF VALUATION: NO ALPHA AFTER CORRECTION
   - DefeatBeta stock_statement uses 'report_date' as fiscal period END
   - After applying proper filing lags (60+ days for 10-K), IC dropped:
     * Before correction: IC=+0.10**
     * After correction: IC=+0.02 (not significant)
   - December fiscal year-end was driving fake signal (IC=+0.41***)

================================================================================
VALID DATA SOURCES IDENTIFIED (for future implementation)
================================================================================

1. EARNINGS DATES: yfinance .earnings_dates attribute
   - Provides ACTUAL announcement dates with timestamps (e.g., "2024-10-31 16:00:00")
   - Has 5+ years of historical data with EPS actual, estimate, and surprise %
   - Can be used to fix PEAD signal

2. SEC FILING DATES: SEC EDGAR API
   - data.sec.gov/submissions/CIK{cik}.json provides filingDate vs reportDate
   - Large accelerated filers: 10-K filed ~30-40 days after period end
   - Can be used to fix DCF/valuation signals

================================================================================
USAGE
================================================================================

The model class still exists for backwards compatibility, but all signals
will return DISABLED status with no actionable recommendations.

To properly implement these signals, use:
- yfinance for earnings announcement dates
- SEC EDGAR for filing dates
- Build historical time-series of estimates (not available in DefeatBeta)

Original module documentation preserved below for reference.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal, List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# ALPHA SIGNAL CONSTANTS - DISABLED (Forward Bias Detected)
# ==============================================================================

# ⚠️ WARNING: These statistics are INVALID due to forward bias in DefeatBeta data
# Kept for reference only - DO NOT USE FOR TRADING DECISIONS

ALPHA_CONFIG = {
    # ⚠️ INVALID - Forward bias detected 2026-01-27
    'backtest_date': '2026-01-24',
    'backtest_n_events': 11947,
    'oos_period': '2025',
    'STATUS': 'DISABLED_FORWARD_BIAS',
    
    # ❌ INVALID: DefeatBeta has only single snapshot, not historical time series
    'revision': {
        'STATUS': 'COMPLETELY_INVALID',
        'REASON': 'DefeatBeta stock_earning_estimates is single current snapshot, not historical',
        'optimal_horizon_days': (5, 20),
        'ic_ret5d': None,  # Was 0.062 - FAKE
        'ic_ret10d': None, # Was 0.052 - FAKE
        'ic_ret20d': None, # Was 0.079 - FAKE
        'ic_ret60d': None,
        'oos_ic_ret20d': None, # Was 0.082 - FAKE
        'oos_ls_spread': None,
        'description': 'DISABLED: Cannot backtest without historical estimate snapshots',
    },
    
    # ❌ INVALID: quarter_date is fiscal period end, not announcement date
    'pead': {
        'STATUS': 'SEVERELY_BIASED',
        'REASON': 'quarter_date is fiscal period END, not earnings announcement date',
        'optimal_horizon_days': (40, 60),
        'ic_ret5d': None,  # Was 0.008 - BIASED
        'ic_ret20d': None, # Was 0.002 - BIASED
        'ic_ret20d': 0.002,  # Weak
        'ic_ret40d': None, # Was 0.148 - SEVERELY BIASED
        'ic_ret60d': None, # Was 0.161 - SEVERELY BIASED
        'oos_ic_ret60d': None,
        'description': 'DISABLED: Need actual earnings announcement dates (use yfinance)',
    },
    
    # ❌ INVALID: Combined signal inherits bias from both components
    'combined': {
        'STATUS': 'INVALID',
        'REASON': 'Both component signals are biased',
        'double_positive_spread': None, # Was 0.0120 - FAKE
        'ic_ret60d_weighted': None,
        'ls_spread_ret60d': None,       # Was 0.0803 - FAKE
        'description': 'DISABLED: Combined signal invalid',
    },
    
    # Signal thresholds
    'thresholds': {
        'revision_strong_up': 0.05,    # >5% revision = strong bullish
        'revision_moderate_up': 0.02,   # 2-5% = moderate bullish
        'revision_moderate_down': -0.02,
        'revision_strong_down': -0.05,  # <-5% = strong bearish
        'sue_big_beat': 0.10,           # >10% surprise = big beat
        'sue_small_beat': 0.02,
        'sue_small_miss': -0.02,
        'sue_big_miss': -0.10,
    },
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class RevisionSignal:
    """Analyst estimate revision signal."""
    ticker: str
    revision_30d: Optional[float] = None  # % change in estimates last 30 days
    revision_60d: Optional[float] = None  # % change in estimates last 60 days
    revision_signal: Literal["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"] = "neutral"
    current_eps_estimate: Optional[float] = None
    num_analysts: Optional[int] = None
    data_date: Optional[datetime] = None
    
    @property
    def signal_strength(self) -> float:
        """Signal strength from -1 (bearish) to +1 (bullish)."""
        rev = self.revision_60d or self.revision_30d
        if rev is None:
            return 0.0
        return np.clip(rev * 10, -1, 1)  # Scale to [-1, 1]


@dataclass
class PEADSignal:
    """Post-Earnings Announcement Drift signal."""
    ticker: str
    sue: Optional[float] = None  # Standardized Unexpected Earnings
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    beat: bool = False
    surprise_bucket: Literal["big_miss", "small_miss", "inline", "small_beat", "big_beat"] = "inline"
    earnings_date: Optional[datetime] = None
    days_since_earnings: Optional[int] = None
    
    @property
    def signal_strength(self) -> float:
        """Signal strength from -1 (miss) to +1 (beat)."""
        if self.sue is None:
            return 0.0
        return np.clip(self.sue * 2, -1, 1)  # Scale to [-1, 1]
    
    @property
    def drift_potential(self) -> float:
        """Expected drift potential based on days since earnings."""
        if self.days_since_earnings is None:
            return 0.5
        
        # Drift is strongest 0-60 days after earnings
        if self.days_since_earnings <= 10:
            return 1.0  # Full drift potential
        elif self.days_since_earnings <= 40:
            return 0.8
        elif self.days_since_earnings <= 60:
            return 0.6
        elif self.days_since_earnings <= 90:
            return 0.3
        else:
            return 0.1  # Minimal drift remaining


@dataclass
class FundamentalsAlphaSignal:
    """Combined fundamentals alpha signal for a stock."""
    ticker: str
    
    # Component signals
    revision_signal: Optional[RevisionSignal] = None
    pead_signal: Optional[PEADSignal] = None
    
    # Combined scores
    short_term_score: Optional[float] = None  # For 5-20 day trades (revision-driven)
    long_term_score: Optional[float] = None   # For 40-60 day trades (PEAD-driven)
    combined_score: Optional[float] = None    # Blended
    
    # Classification
    signal_type: Literal[
        "double_positive",   # Revision up + beat (strongest)
        "revision_up_miss",  # Analysts bullish but company missed
        "revision_down_beat",# Analysts bearish but company beat
        "double_negative",   # Revision down + miss (weakest)
        "revision_only",     # Only revision signal available
        "pead_only",         # Only PEAD signal available
        "neutral",           # No clear signal
    ] = "neutral"
    
    # Recommendations
    recommended_horizon: Literal["short", "long", "either", "none"] = "none"
    expected_quintile: Optional[int] = None  # 1-5, predicted return bucket
    
    # Metadata
    data_freshness: Literal["fresh", "recent", "stale"] = "stale"
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # Summary
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'ticker': self.ticker,
            'short_term_score': self.short_term_score,
            'long_term_score': self.long_term_score,
            'combined_score': self.combined_score,
            'signal_type': self.signal_type,
            'recommended_horizon': self.recommended_horizon,
            'expected_quintile': self.expected_quintile,
            'data_freshness': self.data_freshness,
            'summary': self.summary,
            'strengths': self.strengths,
            'risks': self.risks,
            'revision_30d': self.revision_signal.revision_30d if self.revision_signal else None,
            'revision_60d': self.revision_signal.revision_60d if self.revision_signal else None,
            'sue': self.pead_signal.sue if self.pead_signal else None,
            'beat': self.pead_signal.beat if self.pead_signal else None,
            'days_since_earnings': self.pead_signal.days_since_earnings if self.pead_signal else None,
        }


# ==============================================================================
# FUNDAMENTALS ALPHA MODEL
# ==============================================================================

class FundamentalsAlphaModel:
    """
    Production model for fundamentals-based alpha signals.
    
    Combines analyst estimate revisions with post-earnings drift
    to generate actionable trading signals.
    """
    
    def __init__(self, cache_data: bool = True):
        """
        Initialize the model.
        
        Args:
            cache_data: Whether to cache DefeatBeta data (recommended).
        """
        self.cache_data = cache_data
        self._revision_cache: Optional[pd.DataFrame] = None
        self._pead_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None
        
        logger.info("Initialized FundamentalsAlphaModel")
    
    def _load_data(self) -> bool:
        """Load data from DefeatBeta if not cached."""
        now = datetime.now()
        
        # Check cache validity (24 hour TTL)
        if self._revision_cache is not None and self._cache_time is not None:
            cache_age_hours = (now - self._cache_time).total_seconds() / 3600
            if cache_age_hours < 24:
                return True
        
        try:
            from ..data.defeatbeta import DefeatBetaDataLoader
            
            # Load revision data
            logger.info("Loading estimate revision data from DefeatBeta...")
            rev_df = DefeatBetaDataLoader.load('stock_earning_estimates')
            if rev_df is not None:
                rev_df['ticker'] = rev_df['symbol']
                rev_df['date'] = pd.to_datetime(rev_df['report_date'])
                self._revision_cache = rev_df
            
            # Load PEAD data
            logger.info("Loading EPS history from DefeatBeta...")
            pead_df = DefeatBetaDataLoader.load('stock_historical_eps')
            if pead_df is not None:
                pead_df['ticker'] = pead_df['symbol']
                pead_df['earnings_date'] = pd.to_datetime(pead_df['quarter_date'], errors='coerce')
                pead_df['eps_actual'] = pd.to_numeric(pead_df['eps_actual'], errors='coerce')
                pead_df['eps_estimate'] = pd.to_numeric(pead_df['eps_estimate'], errors='coerce')
                self._pead_cache = pead_df
            
            self._cache_time = now
            logger.info(f"Loaded {len(self._revision_cache) if self._revision_cache is not None else 0} revision records, "
                       f"{len(self._pead_cache) if self._pead_cache is not None else 0} PEAD records")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load DefeatBeta data: {e}")
            return False
    
    def get_revision_signal(self, ticker: str) -> Optional[RevisionSignal]:
        """Get the revision signal for a ticker."""
        if not self._load_data() or self._revision_cache is None:
            return None
        
        ticker = ticker.upper()
        ticker_df = self._revision_cache[self._revision_cache['ticker'] == ticker]
        
        if ticker_df.empty:
            return None
        
        # Get most recent estimate
        ticker_df = ticker_df.sort_values('date', ascending=False)
        latest = ticker_df.iloc[0]
        
        # Calculate revisions
        current_eps = pd.to_numeric(latest.get('estimate_avg_eps'), errors='coerce')
        eps_30d = pd.to_numeric(latest.get('thirty_days_ago_estimate_avg_eps'), errors='coerce')
        eps_60d = pd.to_numeric(latest.get('sixty_days_ago_estimate_avg_eps'), errors='coerce')
        
        revision_30d = None
        revision_60d = None
        
        if pd.notna(current_eps) and pd.notna(eps_30d) and eps_30d != 0:
            revision_30d = (current_eps - eps_30d) / abs(eps_30d)
        
        if pd.notna(current_eps) and pd.notna(eps_60d) and eps_60d != 0:
            revision_60d = (current_eps - eps_60d) / abs(eps_60d)
        
        # Classify revision signal
        rev = revision_60d or revision_30d
        thresholds = ALPHA_CONFIG['thresholds']
        
        if rev is None:
            signal_class = "neutral"
        elif rev >= thresholds['revision_strong_up']:
            signal_class = "strong_bullish"
        elif rev >= thresholds['revision_moderate_up']:
            signal_class = "bullish"
        elif rev <= thresholds['revision_strong_down']:
            signal_class = "strong_bearish"
        elif rev <= thresholds['revision_moderate_down']:
            signal_class = "bearish"
        else:
            signal_class = "neutral"
        
        return RevisionSignal(
            ticker=ticker,
            revision_30d=revision_30d,
            revision_60d=revision_60d,
            revision_signal=signal_class,
            current_eps_estimate=current_eps if pd.notna(current_eps) else None,
            num_analysts=int(latest.get('number_of_analysts', 0)) or None,
            data_date=latest['date'] if pd.notna(latest['date']) else None,
        )
    
    def get_pead_signal(self, ticker: str) -> Optional[PEADSignal]:
        """Get the PEAD signal for a ticker."""
        if not self._load_data() or self._pead_cache is None:
            return None
        
        ticker = ticker.upper()
        ticker_df = self._pead_cache[self._pead_cache['ticker'] == ticker]
        
        if ticker_df.empty:
            return None
        
        # Get most recent earnings
        ticker_df = ticker_df.sort_values('earnings_date', ascending=False)
        latest = ticker_df.iloc[0]
        
        eps_actual = latest.get('eps_actual')
        eps_estimate = latest.get('eps_estimate')
        earnings_date = latest.get('earnings_date')
        
        if pd.isna(eps_actual) or pd.isna(eps_estimate) or eps_estimate == 0:
            return None
        
        sue = (eps_actual - eps_estimate) / abs(eps_estimate)
        sue = np.clip(sue, -2, 2)  # Winsorize
        
        beat = eps_actual > eps_estimate
        
        # Classify surprise bucket
        thresholds = ALPHA_CONFIG['thresholds']
        if sue <= thresholds['sue_big_miss']:
            bucket = "big_miss"
        elif sue <= thresholds['sue_small_miss']:
            bucket = "small_miss"
        elif sue >= thresholds['sue_big_beat']:
            bucket = "big_beat"
        elif sue >= thresholds['sue_small_beat']:
            bucket = "small_beat"
        else:
            bucket = "inline"
        
        # Calculate days since earnings
        days_since = None
        if pd.notna(earnings_date):
            days_since = (datetime.now() - pd.to_datetime(earnings_date)).days
        
        return PEADSignal(
            ticker=ticker,
            sue=sue,
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            beat=beat,
            surprise_bucket=bucket,
            earnings_date=earnings_date if pd.notna(earnings_date) else None,
            days_since_earnings=days_since,
        )
    
    def get_signal(self, ticker: str) -> FundamentalsAlphaSignal:
        """
        Get the combined alpha signal for a ticker.
        
        ⚠️ WARNING: This method is DISABLED due to forward bias in DefeatBeta data.
        Returns a signal with:
        - signal_type = "disabled"
        - recommended_horizon = "none"
        - summary explaining why signals are disabled
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            FundamentalsAlphaSignal with DISABLED status.
        """
        ticker = ticker.upper()
        
        # Return DISABLED signal - do not use DefeatBeta data
        signal = FundamentalsAlphaSignal(ticker=ticker)
        signal.signal_type = "neutral"  # Safe default
        signal.recommended_horizon = "none"
        signal.combined_score = 0.0
        signal.short_term_score = 0.0
        signal.long_term_score = 0.0
        signal.expected_quintile = 3  # Neutral
        signal.data_freshness = "stale"
        
        signal.summary = (
            "⚠️ SIGNALS DISABLED: Forward bias detected in DefeatBeta data (2026-01-27 audit). "
            "Estimate Revisions: single snapshot, not historical. "
            "PEAD: quarter_date is fiscal end, not announcement date. "
            "Do NOT use for trading decisions. See module docstring for details."
        )
        signal.strengths = []
        signal.risks = [
            "DefeatBeta estimate data is invalid for backtesting",
            "PEAD signal contaminated with look-ahead bias",
            "All reported ICs were fake due to forward bias"
        ]
        
        logger.warning(f"FundamentalsAlphaModel is DISABLED for {ticker} due to forward bias")
        
        return signal
    
    def get_signal_ORIGINAL_DISABLED(self, ticker: str) -> FundamentalsAlphaSignal:
        """
        ⚠️ ORIGINAL METHOD - DISABLED DUE TO FORWARD BIAS
        
        This method is preserved for reference but should NOT be used.
        The DefeatBeta data it relies on has critical timing issues.
        """
        ticker = ticker.upper()
        
        # Get component signals (BIASED DATA - DO NOT USE)
        revision = self.get_revision_signal(ticker)
        pead = self.get_pead_signal(ticker)
        
        # Initialize result
        signal = FundamentalsAlphaSignal(ticker=ticker)
        signal.revision_signal = revision
        signal.pead_signal = pead
        
        # Calculate scores
        rev_score = revision.signal_strength if revision else 0.0
        pead_score = pead.signal_strength if pead else 0.0
        drift_potential = pead.drift_potential if pead else 0.5
        
        # Short-term score (revision-dominated)
        signal.short_term_score = rev_score
        
        # Long-term score (PEAD-dominated, adjusted for drift potential)
        signal.long_term_score = pead_score * drift_potential
        
        # Combined score (weighted average)
        if revision and pead:
            signal.combined_score = 0.4 * rev_score + 0.6 * pead_score
        elif revision:
            signal.combined_score = rev_score
        elif pead:
            signal.combined_score = pead_score
        else:
            signal.combined_score = 0.0
        
        # Classify signal type
        if revision and pead:
            revision_up = (revision.revision_60d or revision.revision_30d or 0) > 0.02
            beat = pead.beat
            
            if revision_up and beat:
                signal.signal_type = "double_positive"
            elif revision_up and not beat:
                signal.signal_type = "revision_up_miss"
            elif not revision_up and beat:
                signal.signal_type = "revision_down_beat"
            else:
                signal.signal_type = "double_negative"
        elif revision:
            signal.signal_type = "revision_only"
        elif pead:
            signal.signal_type = "pead_only"
        else:
            signal.signal_type = "neutral"
        
        # Determine recommended horizon
        if signal.signal_type == "double_positive":
            signal.recommended_horizon = "either"  # Both signals align
        elif signal.signal_type == "revision_up_miss":
            signal.recommended_horizon = "short"   # Trust revisions short-term
        elif signal.signal_type == "revision_down_beat":
            signal.recommended_horizon = "long"    # Trust PEAD long-term
        elif signal.signal_type == "double_negative":
            signal.recommended_horizon = "none"    # Avoid
        elif revision and abs(rev_score) > 0.3:
            signal.recommended_horizon = "short"
        elif pead and abs(pead_score) > 0.3 and drift_potential > 0.5:
            signal.recommended_horizon = "long"
        else:
            signal.recommended_horizon = "none"
        
        # Estimate quintile (1-5, higher = better expected returns)
        if signal.combined_score >= 0.6:
            signal.expected_quintile = 5
        elif signal.combined_score >= 0.3:
            signal.expected_quintile = 4
        elif signal.combined_score >= -0.3:
            signal.expected_quintile = 3
        elif signal.combined_score >= -0.6:
            signal.expected_quintile = 2
        else:
            signal.expected_quintile = 1
        
        # Data freshness
        if pead and pead.days_since_earnings:
            if pead.days_since_earnings <= 10:
                signal.data_freshness = "fresh"
            elif pead.days_since_earnings <= 30:
                signal.data_freshness = "recent"
            else:
                signal.data_freshness = "stale"
        else:
            signal.data_freshness = "stale"
        
        # Generate summary and insights
        signal.summary, signal.strengths, signal.risks = self._generate_insights(signal)
        
        return signal
    
    def _generate_insights(self, signal: FundamentalsAlphaSignal) -> tuple[str, List[str], List[str]]:
        """Generate human-readable insights for the signal."""
        strengths = []
        risks = []
        
        rev = signal.revision_signal
        pead = signal.pead_signal
        
        # Revision insights
        if rev:
            rev_pct = (rev.revision_60d or rev.revision_30d or 0) * 100
            if rev.revision_signal in ["strong_bullish", "bullish"]:
                strengths.append(f"Analysts raising estimates ({rev_pct:+.1f}% last 60d)")
            elif rev.revision_signal in ["strong_bearish", "bearish"]:
                risks.append(f"Analysts cutting estimates ({rev_pct:+.1f}% last 60d)")
        
        # PEAD insights
        if pead:
            if pead.beat:
                strengths.append(f"Beat earnings ({pead.sue*100:+.1f}% surprise)")
                if pead.days_since_earnings and pead.days_since_earnings <= 30:
                    strengths.append(f"Fresh earnings drift ({pead.days_since_earnings}d ago)")
            else:
                risks.append(f"Missed earnings ({pead.sue*100:+.1f}% surprise)")
            
            if pead.days_since_earnings and pead.days_since_earnings > 60:
                risks.append(f"Stale earnings data ({pead.days_since_earnings}d old)")
        
        # Signal type specific summary
        summaries = {
            "double_positive": "🟢 BULLISH: Analysts raising estimates AND company beat → strong signal for any horizon",
            "revision_up_miss": "🟡 MIXED: Analysts bullish but company missed → trust revisions for short-term trades",
            "revision_down_beat": "🟡 MIXED: Analysts bearish but company beat → trust PEAD for long-term drift",
            "double_negative": "🔴 BEARISH: Analysts cutting estimates AND company missed → avoid or short",
            "revision_only": f"Revision signal only: {'bullish' if (rev and rev.signal_strength > 0) else 'bearish'} for short-term",
            "pead_only": f"PEAD signal only: {'bullish' if (pead and pead.beat) else 'bearish'} for long-term drift",
            "neutral": "Neutral: No strong signal in either direction",
        }
        
        summary = summaries.get(signal.signal_type, "Signal analysis unavailable")
        
        return summary, strengths, risks
    
    def get_batch_signals(self, tickers: List[str]) -> Dict[str, FundamentalsAlphaSignal]:
        """Get signals for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_signal(ticker)
        return results
    
    def to_dataframe(self, signals: Dict[str, FundamentalsAlphaSignal]) -> pd.DataFrame:
        """Convert signals to a DataFrame for easy analysis."""
        data = [s.to_dict() for s in signals.values()]
        return pd.DataFrame(data)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_fundamentals_alpha(ticker: str) -> FundamentalsAlphaSignal:
    """
    Quick function to get fundamentals alpha for a single ticker.
    
    Usage:
        signal = get_fundamentals_alpha("AAPL")
        print(signal.summary)
    """
    model = FundamentalsAlphaModel()
    return model.get_signal(ticker)


def get_fundamentals_alpha_batch(tickers: List[str]) -> Dict[str, FundamentalsAlphaSignal]:
    """Get fundamentals alpha for multiple tickers."""
    model = FundamentalsAlphaModel()
    return model.get_batch_signals(tickers)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get fundamentals alpha signals")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to analyze")
    args = parser.parse_args()
    
    model = FundamentalsAlphaModel()
    
    for ticker in args.tickers:
        signal = model.get_signal(ticker)
        print(f"\n{'='*60}")
        print(f"  {ticker} - Fundamentals Alpha Signal")
        print('='*60)
        print(f"  Signal Type: {signal.signal_type}")
        print(f"  Short-Term Score: {signal.short_term_score:+.2f}" if signal.short_term_score else "  Short-Term Score: N/A")
        print(f"  Long-Term Score: {signal.long_term_score:+.2f}" if signal.long_term_score else "  Long-Term Score: N/A")
        print(f"  Combined Score: {signal.combined_score:+.2f}" if signal.combined_score else "  Combined Score: N/A")
        print(f"  Recommended Horizon: {signal.recommended_horizon}")
        print(f"  Expected Quintile: Q{signal.expected_quintile}")
        print(f"  Data Freshness: {signal.data_freshness}")
        print(f"\n  {signal.summary}")
        if signal.strengths:
            print(f"\n  ✅ Strengths:")
            for s in signal.strengths:
                print(f"     - {s}")
        if signal.risks:
            print(f"\n  ⚠️  Risks:")
            for r in signal.risks:
                print(f"     - {r}")
