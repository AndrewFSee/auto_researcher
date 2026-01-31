"""
Insider Cluster Alpha Model - Production Version
=================================================

This model generates trading signals based on insider cluster buying/selling patterns.

================================================================================
ACADEMIC FOUNDATION
================================================================================

Key Research:
1. Lakonishok & Lee (2001) "Are Insider Trades Informative?"
   - Aggregate insider trading predicts future returns
   - Purchases more informative than sales

2. Seyhun (1986) "Insiders' Profits, Costs of Trading, and Market Efficiency"
   - Insiders earn abnormal returns on purchases
   - CEO/CFO trades most informative

3. Cohen, Malloy, Pomorski (2012) "Decoding Inside Information"
   - "Opportunistic" trades (not routine) are predictive
   - Cluster buying by multiple insiders is strongest signal

Expected Performance (from literature):
- Cluster buying: +3-5% annual excess return
- CEO/CFO buying: Higher than other insiders
- Selling: Less informative (many non-information reasons)

================================================================================
SIGNAL CONSTRUCTION
================================================================================

The model identifies cluster events where 2+ insiders buy/sell within 30 days:

Signal Strength Components:
1. Number of insiders (more = stronger): 40% weight
2. Role weight (CEO/CFO = 1.0, Director = 0.6, VP = 0.5): 30% weight
3. Transaction value (normalized by log): 30% weight

Signal Direction:
- LONG: Multiple insiders acquiring shares
- SHORT: Multiple insiders disposing shares (weaker signal)

================================================================================
IMPORTANT NOTES
================================================================================

- Open market purchases (type P) are most informative but rare
- Options exercises (type M) are more common, still somewhat informative
- 10b5-1 pre-planned trades are filtered out
- Signal is computed at real-time filing date (no look-ahead)

================================================================================
USAGE
================================================================================

    from auto_researcher.models.insider_cluster import InsiderClusterModel
    
    model = InsiderClusterModel()
    signal = model.get_signal("AAPL")
    
    if signal.is_actionable:
        print(f"Signal: {signal.direction}")
        print(f"Strength: {signal.strength}")
        print(f"Cluster: {signal.num_insiders} insiders in {signal.days_span} days")
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal, List
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INSIDER_CONFIG = {
    # Cluster detection parameters
    'cluster_window_days': 30,
    'min_cluster_insiders': 2,
    'lookback_days': 90,  # How far back to look for transactions
    
    # Signal strength weights
    'weight_num_insiders': 0.40,
    'weight_role': 0.30,
    'weight_value': 0.30,
    
    # Transaction value thresholds
    'min_transaction_value': 1000,  # Skip tiny transactions
    
    # Expected performance (from literature)
    'expected_return_cluster_buy': 0.03,  # 3% over 60 days
    'expected_return_ceo_buy': 0.05,  # 5% if CEO/CFO buying
    
    # Signal decay
    'signal_decay_days': 60,  # Signal decays after 60 days
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class InsiderSignal:
    """Signal output from the insider cluster model."""
    ticker: str
    
    # Signal properties
    direction: Optional[Literal["long", "short"]] = None
    strength: Literal["strong", "moderate", "weak", "none"] = "none"
    is_actionable: bool = False
    
    # Cluster details
    has_cluster: bool = False
    cluster_direction: Optional[str] = None  # "buy" or "sell"
    num_insiders: int = 0
    total_value: float = 0.0
    has_ceo_cfo: bool = False
    avg_role_weight: float = 0.0
    
    # Timing
    cluster_start_date: Optional[datetime] = None
    cluster_end_date: Optional[datetime] = None
    days_since_cluster: Optional[int] = None
    
    # Transaction breakdown
    open_market_count: int = 0  # Type P transactions
    exercise_count: int = 0      # Type M transactions
    
    # Expected outcomes
    expected_return: Optional[float] = None
    recommended_days: int = 60
    
    # Summary
    summary: str = ""
    confidence: float = 0.0  # 0-1


# ==============================================================================
# INSIDER CLUSTER MODEL
# ==============================================================================

class InsiderClusterModel:
    """
    Generates trading signals from insider cluster buying/selling patterns.
    
    Uses the InsiderTradingAgent's SEC client and cluster detection logic,
    but provides a focused signal interface for model integration.
    """
    
    def __init__(
        self,
        lookback_days: int = 90,
        cluster_window_days: int = 30,
        min_cluster_insiders: int = 2,
    ):
        """
        Initialize the model.
        
        Args:
            lookback_days: How far back to look for transactions.
            cluster_window_days: Window for cluster detection.
            min_cluster_insiders: Minimum insiders for a cluster.
        """
        self.lookback_days = lookback_days
        self.cluster_window_days = cluster_window_days
        self.min_cluster_insiders = min_cluster_insiders
        
        # Lazy-load the insider trading agent (which has the SEC client and parsing)
        self._agent = None
        
        logger.info(f"Initialized InsiderClusterModel (lookback={lookback_days}d, "
                   f"cluster_window={cluster_window_days}d, min_insiders={min_cluster_insiders})")
    
    @property
    def agent(self):
        """Lazy-load InsiderTradingAgent."""
        if self._agent is None:
            try:
                from ..agents.insider_trading_agent import InsiderTradingAgent
                self._agent = InsiderTradingAgent(
                    lookback_days=self.lookback_days,
                    cluster_window_days=self.cluster_window_days,
                    min_cluster_insiders=self.min_cluster_insiders,
                )
            except ImportError:
                logger.error("Could not import InsiderTradingAgent")
                raise
        return self._agent
    
    def _compute_signal_strength(self, cluster) -> float:
        """Compute signal strength from 0 to 1."""
        # Component 1: Number of insiders
        # 2 insiders = 0.0, 3 = 0.33, 4 = 0.67, 5+ = 1.0
        insider_score = min(1.0, (cluster.num_insiders - 2) / 3)
        
        # Component 2: Role weight
        role_score = cluster.avg_role_weight
        
        # Component 3: Value (log-scaled)
        # $10K = 0.1, $100K = 0.3, $1M = 0.5, $10M = 0.7, $100M = 0.9
        if cluster.total_value > 0:
            value_score = min(1.0, np.log10(cluster.total_value) / 8)
        else:
            value_score = 0.0
        
        # Weighted combination
        strength = (
            INSIDER_CONFIG['weight_num_insiders'] * insider_score +
            INSIDER_CONFIG['weight_role'] * role_score +
            INSIDER_CONFIG['weight_value'] * value_score
        )
        
        # Selling is less informative
        if cluster.direction == "sell":
            strength *= 0.5
        
        return min(1.0, strength)
    
    def _has_ceo_cfo(self, cluster) -> bool:
        """Check if cluster has CEO/CFO transactions."""
        for t in cluster.transactions:
            title = t.insider_title.lower()
            if any(x in title for x in ["ceo", "cfo", "chief executive", "chief financial"]):
                return True
        return False
    
    def get_signal(self, ticker: str) -> InsiderSignal:
        """
        Get insider cluster signal for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            InsiderSignal with current signal status.
        """
        ticker = ticker.upper()
        signal = InsiderSignal(ticker=ticker)
        
        try:
            # Use the existing agent's analyze method
            analysis = self.agent.analyze_ticker(ticker)
            
            if analysis.total_transactions == 0:
                signal.summary = "No recent insider transactions found"
                return signal
            
            # Count transaction types
            signal.open_market_count = analysis.open_market_buys + analysis.open_market_sells
            signal.exercise_count = sum(
                1 for t in analysis.all_transactions 
                if t.transaction_type == 'M'
            )
            
            # Check for clusters
            clusters = analysis.cluster_events
            
            if not clusters:
                signal.summary = f"No cluster events ({analysis.total_transactions} individual transactions)"
                return signal
            
            # Use most recent cluster
            cluster = clusters[0]
            signal.has_cluster = True
            signal.cluster_direction = cluster.direction
            signal.num_insiders = cluster.num_insiders
            signal.total_value = cluster.total_value
            signal.has_ceo_cfo = self._has_ceo_cfo(cluster)
            signal.avg_role_weight = cluster.avg_role_weight
            signal.cluster_start_date = cluster.start_date
            signal.cluster_end_date = cluster.end_date
            
            # Days since cluster
            days_since = (datetime.now() - cluster.end_date).days
            signal.days_since_cluster = days_since
            
            # Compute strength
            strength_score = self._compute_signal_strength(cluster)
            
            # Map to strength category
            if strength_score >= 0.6:
                signal.strength = "strong"
            elif strength_score >= 0.35:
                signal.strength = "moderate"
            elif strength_score >= 0.15:
                signal.strength = "weak"
            else:
                signal.strength = "none"
            
            # Direction
            if cluster.direction == "buy":
                signal.direction = "long"
            else:
                signal.direction = "short"
            
            # Check if actionable (not too old)
            if days_since <= INSIDER_CONFIG['signal_decay_days']:
                signal.is_actionable = True
            
            # Expected return
            if cluster.direction == "buy":
                base_return = INSIDER_CONFIG['expected_return_cluster_buy']
                if signal.has_ceo_cfo:
                    base_return = INSIDER_CONFIG['expected_return_ceo_buy']
                # Decay based on time elapsed
                decay = max(0, 1 - days_since / INSIDER_CONFIG['signal_decay_days'])
                signal.expected_return = base_return * decay
            else:
                # Selling signal is weaker and inverted
                signal.expected_return = -0.01 * max(0, 1 - days_since / INSIDER_CONFIG['signal_decay_days'])
            
            # Confidence (use agent's confidence as base)
            signal.confidence = analysis.confidence * (0.8 if signal.is_actionable else 0.3)
            
            # Summary
            dir_emoji = "🟢" if cluster.direction == "buy" else "🔴"
            strength_text = signal.strength.upper()
            ceo_note = " (incl. CEO/CFO)" if signal.has_ceo_cfo else ""
            value_str = f"${cluster.total_value/1e6:.1f}M" if cluster.total_value >= 1e6 else f"${cluster.total_value/1e3:.0f}K"
            
            if signal.is_actionable:
                signal.summary = (
                    f"{dir_emoji} {strength_text} {cluster.direction.upper()}: "
                    f"{cluster.num_insiders} insiders{ceo_note} | "
                    f"{value_str} | {days_since}d ago"
                )
            else:
                signal.summary = f"⚪ EXPIRED: Cluster was {days_since}d ago (>{INSIDER_CONFIG['signal_decay_days']}d)"
            
        except Exception as e:
            logger.error(f"Error getting insider signal for {ticker}: {e}")
            signal.summary = f"Error: {str(e)}"
        
        return signal
    
    def get_signal_batch(self, tickers: List[str]) -> dict[str, InsiderSignal]:
        """Get signals for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_signal(ticker)
        return results


# ==============================================================================
# CONVENIENCE FUNCTIONS  
# ==============================================================================

def get_insider_signal(ticker: str) -> InsiderSignal:
    """Quick helper to get insider signal for a single ticker."""
    model = InsiderClusterModel()
    return model.get_signal(ticker)


def get_insider_signals(tickers: List[str]) -> dict[str, InsiderSignal]:
    """Quick helper to get insider signals for multiple tickers."""
    model = InsiderClusterModel()
    return model.get_signal_batch(tickers)
