"""
Enhanced PEAD Alpha Model - Production Version
================================================

Based on forward-bias-free backtesting (2026-01-30), this model uses:
1. yfinance earnings_dates for ACTUAL announcement dates
2. Enhanced SUE signal focusing on BIG surprises (>20%)
3. Consecutive beat/miss pattern as additional factor
4. Revenue surprise as complementary signal (Jegadeesh & Livnat 2006)

================================================================================
BACKTEST RESULTS (2020-2024, N=1,948 events)
================================================================================

Base Signal (all earnings):
  - ret40d IC: +0.049** (p=0.029)
  - ret60d IC: +0.047** (p=0.038)
  - L/S Spread: +1.84% at 40d, +2.05% at 60d

Enhanced Signal (big surprises only, N=334):
  - ret40d IC: +0.152*** (p=0.006)  ← 3x improvement!
  - ret60d IC: +0.138** (p=0.012)
  - L/S Spread: +3.02% at 40d, +3.55% at 60d

Combined Signal (big surprise + consecutive beat pattern):
  - L/S Spread: +5.08% at 60d
  - Fewer trades but stronger signal

Revenue Surprise (Jegadeesh & Livnat 2006):
  - Revenue surprise is incrementally informative beyond EPS surprise
  - Dual-beat (both EPS and revenue) drift ~40% larger than EPS-only
  - Revenue miss + EPS beat = weaker drift (quality concern)

================================================================================
USAGE
================================================================================

    from auto_researcher.models.pead_enhanced import EnhancedPEADModel
    
    model = EnhancedPEADModel()
    signal = model.get_signal("AAPL")
    
    if signal.is_actionable:
        print(f"Signal: {signal.direction}")
        print(f"Strength: {signal.strength}")
        print(f"EPS surprise: {signal.sue:.1%}")
        print(f"Revenue surprise: {signal.revenue_sue:.1%}")
        print(f"Recommended hold: {signal.recommended_days}d")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal, List, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

PEAD_CONFIG = {
    'backtest_date': '2026-01-30',
    'backtest_period': '2020-2024',
    'total_events': 1948,
    
    # Base signal performance
    'base': {
        'ic_ret40d': 0.049,
        'ic_ret60d': 0.047,
        'ls_spread_40d': 0.0184,
        'ls_spread_60d': 0.0205,
    },
    
    # Enhanced signal (big surprises only)
    'enhanced': {
        'n_events': 334,
        'ic_ret40d': 0.152,  # 3x improvement!
        'ic_ret60d': 0.138,
        'ls_spread_40d': 0.0302,
        'ls_spread_60d': 0.0355,
    },
    
    # Thresholds
    'thresholds': {
        'big_surprise': 0.20,    # |SUE| > 20% = big surprise
        'small_surprise': 0.05,  # |SUE| < 5% = inline
        'optimal_hold_days': 40, # Peak IC around 40-60 days
        # Revenue surprise thresholds (Jegadeesh & Livnat 2006)
        'big_revenue_surprise': 0.05,    # >5% YoY revenue growth surprise
        'small_revenue_surprise': 0.02,  # >2% for modest surprise
    },
    
    # Revenue surprise weight in composite
    # Academic research: revenue is ~40% as informative as EPS for PEAD
    'revenue_weight': 0.30,  # 30% revenue, 70% EPS in composite surprise
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EarningsEvent:
    """Individual earnings announcement event."""
    ticker: str
    announcement_date: datetime
    eps_actual: float
    eps_estimate: float
    sue: float  # Standardized Unexpected Earnings
    beat: bool
    surprise_bucket: Literal["big_miss", "small_miss", "inline", "small_beat", "big_beat"]
    is_big_surprise: bool
    
    # Revenue surprise fields
    revenue_actual: Optional[float] = None
    revenue_year_ago: Optional[float] = None
    revenue_surprise: Optional[float] = None     # (actual - year_ago) / year_ago
    revenue_beat: Optional[bool] = None           # True if rev growth > expected
    
    # Optional: previous quarter info for consecutive analysis
    prev_beat: Optional[bool] = None
    is_consecutive_beat: bool = False
    is_consecutive_miss: bool = False


@dataclass 
class PEADSignal:
    """Enhanced PEAD trading signal."""
    ticker: str
    
    # Latest earnings info
    latest_earnings: Optional[EarningsEvent] = None
    
    # Signal components
    sue: Optional[float] = None
    is_big_surprise: bool = False
    is_consecutive: bool = False
    
    # Revenue surprise components
    revenue_sue: Optional[float] = None       # Revenue surprise (YoY growth)
    has_revenue_surprise: bool = False
    dual_beat: bool = False                   # Both EPS and revenue beat
    dual_miss: bool = False                   # Both EPS and revenue miss
    composite_sue: Optional[float] = None     # Weighted blend of EPS + revenue
    
    # Actionability
    is_actionable: bool = False
    direction: Literal["long", "short", "neutral"] = "neutral"
    strength: Literal["strong", "moderate", "weak"] = "weak"
    
    # Recommendations
    recommended_days: int = 40
    expected_return: Optional[float] = None
    
    # Metadata
    days_since_earnings: Optional[int] = None
    data_freshness: Literal["fresh", "recent", "stale", "expired"] = "expired"
    
    # Summary
    summary: str = ""
    reasons: List[str] = field(default_factory=list)


# ==============================================================================
# ENHANCED PEAD MODEL
# ==============================================================================

class EnhancedPEADModel:
    """
    Enhanced PEAD model using yfinance for bias-free earnings data.
    
    Key improvements over base model:
    1. Uses actual announcement dates (not fiscal period ends)
    2. Focuses on big surprises (|SUE| > 20%) for 3x IC improvement
    3. Incorporates consecutive beat/miss pattern
    """
    
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initialize the model.
        
        Args:
            cache_ttl_hours: How long to cache earnings data (hours).
        """
        self.cache_ttl_hours = cache_ttl_hours
        self._cache: Dict[str, tuple[datetime, List[EarningsEvent]]] = {}
        
        logger.info("Initialized EnhancedPEADModel")
    
    def _fetch_earnings(self, ticker: str) -> List[EarningsEvent]:
        """Fetch earnings history from yfinance, including revenue data."""
        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates
            
            if earnings is None or len(earnings) == 0:
                return []
            
            # Filter to historical only
            earnings = earnings[earnings['Reported EPS'].notna()].copy()
            
            if len(earnings) == 0:
                return []
            
            earnings = earnings.reset_index()
            earnings.columns = ['announcement_date', 'eps_estimate', 'eps_actual', 'surprise_pct']
            earnings['announcement_date'] = pd.to_datetime(earnings['announcement_date']).dt.tz_localize(None)
            earnings = earnings.sort_values('announcement_date', ascending=False)
            
            # Fetch quarterly revenue data for revenue surprise
            revenue_by_quarter = {}
            try:
                income_stmt = stock.quarterly_income_stmt
                if income_stmt is not None and 'Total Revenue' in income_stmt.index:
                    rev_row = income_stmt.loc['Total Revenue']
                    for col in rev_row.index:
                        q_end = pd.Timestamp(col)
                        if not pd.isna(rev_row[col]):
                            revenue_by_quarter[q_end] = float(rev_row[col])
            except Exception as e:
                logger.debug(f"Could not fetch revenue data for {ticker}: {e}")
            
            events = []
            prev_beat = None
            
            for i, row in earnings.iterrows():
                eps_actual = row['eps_actual']
                eps_estimate = row['eps_estimate']
                
                if pd.isna(eps_actual) or pd.isna(eps_estimate) or eps_estimate == 0:
                    continue
                
                sue = (eps_actual - eps_estimate) / abs(eps_estimate)
                beat = eps_actual > eps_estimate
                
                # Classify surprise
                if sue <= -0.20:
                    bucket = "big_miss"
                elif sue <= -0.05:
                    bucket = "small_miss"
                elif sue >= 0.20:
                    bucket = "big_beat"
                elif sue >= 0.05:
                    bucket = "small_beat"
                else:
                    bucket = "inline"
                
                is_big = bucket in ["big_beat", "big_miss"]
                
                # Match announcement date to fiscal quarter revenue
                # Earnings are announced ~1 month after quarter end
                # Find the closest quarter end that is before the announcement
                rev_actual = None
                rev_year_ago = None
                rev_surprise = None
                rev_beat = None
                ann_date = pd.Timestamp(row['announcement_date'])
                if revenue_by_quarter:
                    # Find the quarter end that this announcement corresponds to
                    # Typically announced 20-45 days after quarter end
                    candidates = [
                        q for q in revenue_by_quarter.keys()
                        if 15 <= (ann_date - q).days <= 100
                    ]
                    if candidates:
                        # Take the closest quarter end
                        fiscal_q = min(candidates, key=lambda q: abs((ann_date - q).days))
                        rev_actual = revenue_by_quarter[fiscal_q]
                        
                        # Find year-ago quarter
                        year_ago_q = fiscal_q - pd.DateOffset(years=1)
                        # Match within 15 days tolerance
                        year_ago_match = [
                            q for q in revenue_by_quarter.keys()
                            if abs((q - year_ago_q).days) <= 15
                        ]
                        if year_ago_match:
                            rev_year_ago = revenue_by_quarter[year_ago_match[0]]
                            if rev_year_ago and rev_year_ago != 0:
                                rev_surprise = (rev_actual - rev_year_ago) / abs(rev_year_ago)
                                # Revenue "beat" = positive YoY growth
                                # (In practice, would compare vs consensus, but YoY growth
                                # acceleration is a good proxy when estimates unavailable)
                                rev_beat = rev_surprise > 0
                
                event = EarningsEvent(
                    ticker=ticker,
                    announcement_date=row['announcement_date'],
                    eps_actual=eps_actual,
                    eps_estimate=eps_estimate,
                    sue=sue,
                    beat=beat,
                    surprise_bucket=bucket,
                    is_big_surprise=is_big,
                    revenue_actual=rev_actual,
                    revenue_year_ago=rev_year_ago,
                    revenue_surprise=rev_surprise,
                    revenue_beat=rev_beat,
                    prev_beat=prev_beat,
                    is_consecutive_beat=(beat and prev_beat == True),
                    is_consecutive_miss=(not beat and prev_beat == False),
                )
                
                events.append(event)
                prev_beat = beat
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching earnings for {ticker}: {e}")
            return []
    
    def _get_cached_earnings(self, ticker: str) -> List[EarningsEvent]:
        """Get earnings from cache or fetch fresh."""
        ticker = ticker.upper()
        now = datetime.now()
        
        if ticker in self._cache:
            cache_time, events = self._cache[ticker]
            age_hours = (now - cache_time).total_seconds() / 3600
            
            if age_hours < self.cache_ttl_hours:
                return events
        
        events = self._fetch_earnings(ticker)
        self._cache[ticker] = (now, events)
        return events
    
    def get_signal(self, ticker: str) -> PEADSignal:
        """
        Get the enhanced PEAD signal for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            PEADSignal with trading recommendations.
        """
        ticker = ticker.upper()
        signal = PEADSignal(ticker=ticker)
        
        # Fetch earnings
        events = self._get_cached_earnings(ticker)
        
        if not events:
            signal.summary = f"No earnings data available for {ticker}"
            return signal
        
        # Get most recent earnings
        latest = events[0]
        signal.latest_earnings = latest
        signal.sue = latest.sue
        signal.is_big_surprise = latest.is_big_surprise
        signal.is_consecutive = latest.is_consecutive_beat or latest.is_consecutive_miss
        
        # Revenue surprise
        if latest.revenue_surprise is not None:
            signal.revenue_sue = latest.revenue_surprise
            signal.has_revenue_surprise = True
            signal.dual_beat = (latest.beat and latest.revenue_beat == True)
            signal.dual_miss = (not latest.beat and latest.revenue_beat == False)
            
            # Composite SUE: weighted blend of EPS and revenue surprise
            rev_weight = PEAD_CONFIG['revenue_weight']
            eps_weight = 1.0 - rev_weight
            signal.composite_sue = eps_weight * latest.sue + rev_weight * latest.revenue_surprise
        else:
            signal.revenue_sue = None
            signal.has_revenue_surprise = False
            signal.composite_sue = latest.sue  # Fall back to EPS-only
        
        # Calculate days since earnings
        days_since = (datetime.now() - latest.announcement_date).days
        signal.days_since_earnings = days_since
        
        # Data freshness
        if days_since <= 10:
            signal.data_freshness = "fresh"
        elif days_since <= 30:
            signal.data_freshness = "recent"
        elif days_since <= 60:
            signal.data_freshness = "stale"
        else:
            signal.data_freshness = "expired"
        
        # Determine signal direction and strength
        reasons = []
        thresholds = PEAD_CONFIG['thresholds']
        
        if latest.is_big_surprise:
            # Strong signal - use enhanced model
            if latest.beat:
                signal.direction = "long"
                signal.strength = "strong" if latest.is_consecutive_beat else "moderate"
                reasons.append(f"Big beat: {latest.sue*100:+.1f}% EPS surprise")
                
                if latest.is_consecutive_beat:
                    reasons.append("Consecutive beat pattern (momentum)")
                
                # Revenue surprise modifier
                if signal.has_revenue_surprise:
                    if signal.dual_beat:
                        # Dual beat: upgrade strength, larger expected return
                        if signal.strength == "moderate":
                            signal.strength = "strong"
                        reasons.append(f"Revenue also beat: {latest.revenue_surprise*100:+.1f}% YoY (dual beat)")
                        signal.expected_return = PEAD_CONFIG['enhanced']['ls_spread_40d'] * 1.4
                    elif latest.revenue_beat == False:
                        # EPS beat but revenue miss: quality concern, downgrade
                        if signal.strength == "strong":
                            signal.strength = "moderate"
                        reasons.append(f"Revenue missed: {latest.revenue_surprise*100:+.1f}% YoY (quality concern)")
                        signal.expected_return = PEAD_CONFIG['enhanced']['ls_spread_40d'] * 0.7
                    else:
                        signal.expected_return = PEAD_CONFIG['enhanced']['ls_spread_40d']
                else:
                    signal.expected_return = PEAD_CONFIG['enhanced']['ls_spread_40d']
            else:
                signal.direction = "short"
                signal.strength = "strong" if latest.is_consecutive_miss else "moderate"
                reasons.append(f"Big miss: {latest.sue*100:+.1f}% EPS surprise")
                
                if latest.is_consecutive_miss:
                    reasons.append("Consecutive miss pattern (negative momentum)")
                
                # Revenue surprise modifier
                if signal.has_revenue_surprise:
                    if signal.dual_miss:
                        if signal.strength == "moderate":
                            signal.strength = "strong"
                        reasons.append(f"Revenue also missed: {latest.revenue_surprise*100:+.1f}% YoY (dual miss)")
                        signal.expected_return = -PEAD_CONFIG['enhanced']['ls_spread_40d'] * 1.4
                    elif latest.revenue_beat == True:
                        # EPS miss but revenue beat: less severe
                        if signal.strength == "strong":
                            signal.strength = "moderate"
                        reasons.append(f"Revenue grew: {latest.revenue_surprise*100:+.1f}% YoY (offsetting)")
                        signal.expected_return = -PEAD_CONFIG['enhanced']['ls_spread_40d'] * 0.7
                    else:
                        signal.expected_return = -PEAD_CONFIG['enhanced']['ls_spread_40d']
                else:
                    signal.expected_return = -PEAD_CONFIG['enhanced']['ls_spread_40d']
        else:
            # Weak signal - base model only
            if abs(latest.sue) >= 0.05:
                signal.direction = "long" if latest.beat else "short"
                signal.strength = "weak"
                reasons.append(f"Small {'beat' if latest.beat else 'miss'}: {latest.sue*100:+.1f}% EPS")
                
                # Revenue can upgrade small-surprise signals
                if signal.has_revenue_surprise and abs(latest.revenue_surprise) >= thresholds['big_revenue_surprise']:
                    if (latest.beat and latest.revenue_beat) or (not latest.beat and not latest.revenue_beat):
                        signal.strength = "moderate"
                        reasons.append(f"Revenue confirms: {latest.revenue_surprise*100:+.1f}% YoY (upgraded)")
                    elif latest.revenue_surprise is not None:
                        reasons.append(f"Revenue: {latest.revenue_surprise*100:+.1f}% YoY")
                elif signal.has_revenue_surprise and latest.revenue_surprise is not None:
                    reasons.append(f"Revenue: {latest.revenue_surprise*100:+.1f}% YoY")
                
                sign = 1 if latest.beat else -1
                signal.expected_return = PEAD_CONFIG['base']['ls_spread_40d'] * sign
            else:
                signal.direction = "neutral"
                signal.strength = "weak"
                reasons.append(f"Inline result: {latest.sue*100:+.1f}% EPS (no signal)")
                if signal.has_revenue_surprise and latest.revenue_surprise is not None:
                    reasons.append(f"Revenue: {latest.revenue_surprise*100:+.1f}% YoY")
        
        # Actionability check
        if signal.data_freshness in ["fresh", "recent"] and signal.strength in ["strong", "moderate"]:
            signal.is_actionable = True
            signal.recommended_days = 40  # Peak IC
        elif signal.data_freshness == "stale" and signal.strength in ["strong", "moderate"]:
            signal.is_actionable = True
            signal.recommended_days = 20  # Less time for drift
            reasons.append("Note: Earnings are 30-60d old, some drift may have occurred")
        elif signal.data_freshness in ["fresh", "recent"] and signal.strength == "weak" and abs(latest.sue) >= 0.05:
            # Weak but still tradeable
            signal.is_actionable = True
            signal.recommended_days = 40
            reasons.append("Note: Small surprise - weaker signal, consider smaller position")
        else:
            signal.is_actionable = False
            if signal.data_freshness == "expired":
                reasons.append("Earnings too old (>60d) - drift likely exhausted")
        
        signal.reasons = reasons
        
        # Generate summary
        rev_info = ""
        if signal.has_revenue_surprise and signal.revenue_sue is not None:
            rev_info = f" | Rev: {signal.revenue_sue*100:+.1f}% YoY"
            if signal.dual_beat:
                rev_info += " (dual beat)"
            elif signal.dual_miss:
                rev_info += " (dual miss)"
        
        if signal.is_actionable:
            signal.summary = (
                f"{'🟢' if signal.direction == 'long' else '🔴'} "
                f"{signal.strength.upper()} {signal.direction.upper()}: "
                f"{latest.surprise_bucket.replace('_', ' ').title()} "
                f"({latest.sue*100:+.1f}% EPS surprise{rev_info}) "
                f"| Hold ~{signal.recommended_days}d "
                f"| Expected: {signal.expected_return*100:+.1f}%"
            )
        else:
            signal.summary = f"⚪ NO SIGNAL: {'; '.join(reasons)}"
        
        return signal
    
    def get_signals_batch(self, tickers: List[str]) -> Dict[str, PEADSignal]:
        """Get signals for multiple tickers."""
        return {t: self.get_signal(t) for t in tickers}
    
    def get_actionable_signals(self, tickers: List[str]) -> List[PEADSignal]:
        """Get only actionable signals from a list of tickers."""
        signals = self.get_signals_batch(tickers)
        return [s for s in signals.values() if s.is_actionable]


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_pead_signal(ticker: str) -> PEADSignal:
    """Quick function to get PEAD signal for a single ticker."""
    model = EnhancedPEADModel()
    return model.get_signal(ticker)


def scan_for_pead_opportunities(tickers: List[str]) -> List[PEADSignal]:
    """Scan a list of tickers for actionable PEAD opportunities."""
    model = EnhancedPEADModel()
    return model.get_actionable_signals(tickers)


if __name__ == "__main__":
    # Demo
    print("="*70)
    print("ENHANCED PEAD MODEL - DEMO")
    print("="*70)
    
    model = EnhancedPEADModel()
    
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    
    for ticker in test_tickers:
        signal = model.get_signal(ticker)
        print(f"\n{ticker}:")
        print(f"  {signal.summary}")
        if signal.reasons:
            for reason in signal.reasons:
                print(f"    - {reason}")
        print(f"  EPS surprise: {signal.sue*100:+.1f}%" if signal.sue else "  EPS surprise: N/A")
        if signal.has_revenue_surprise:
            print(f"  Revenue surprise: {signal.revenue_sue*100:+.1f}% YoY")
            if signal.dual_beat:
                print(f"  *** DUAL BEAT ***")
            elif signal.dual_miss:
                print(f"  *** DUAL MISS ***")
        print(f"  Days since earnings: {signal.days_since_earnings}")
        print(f"  Data freshness: {signal.data_freshness}")
