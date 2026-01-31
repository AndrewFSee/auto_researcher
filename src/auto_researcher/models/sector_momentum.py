"""
Sector Momentum Model
=====================

A quantitative model that generates trading signals based on sector-relative 
momentum and rotation patterns.

================================================================================
ACADEMIC FOUNDATION
================================================================================

Key Research:
1. Moskowitz & Grinblatt (1999) "Do Industries Explain Momentum?"
   - Industry momentum explains a large portion of stock momentum
   - Buying stocks in winning industries beats individual momentum

2. Hong, Torous, Valkanov (2007) "Do Industries Lead Stock Markets?"
   - Industry returns lead market returns
   - Predicts market direction 1-2 months ahead

3. Menzly & Ozbas (2010) "Market Segmentation and Cross-predictability"
   - Related industries predict each other
   - Supply chain relationships create predictable patterns

4. Sector Rotation Theory (classic)
   - Different sectors lead at different economic cycle phases
   - Early cycle: Financials, Consumer Discretionary
   - Mid cycle: Technology, Industrials
   - Late cycle: Energy, Materials
   - Recession: Utilities, Healthcare, Consumer Staples

Expected Performance:
- Sector momentum: +2-4% annual alpha (overweight winning sectors)
- Industry momentum: +3-5% annual alpha (more granular)
- Combined with stock selection: Additive effects

================================================================================
SIGNAL CONSTRUCTION
================================================================================

MOMENTUM CALCULATION:
1. 1-month relative return vs SPY: 30% weight
2. 3-month relative return vs SPY: 50% weight
3. 6-month relative return vs SPY: 20% weight

SIGNAL LOGIC:
- Sector Rank 1-3 (top 3): OVERWEIGHT (+1 to +3 signal)
- Sector Rank 4-8 (middle): NEUTRAL (0 signal)
- Sector Rank 9-11 (bottom 3): UNDERWEIGHT (-1 to -3 signal)

STOCK SIGNAL:
- Stock in overweight sector: +0.15 to +0.30 boost
- Stock in underweight sector: -0.10 to -0.25 headwind

================================================================================
IMPORTANT NOTES
================================================================================

- Uses sector ETFs (XLK, XLF, etc.) for momentum calculation
- No look-ahead bias: uses prior day's close
- Signal rebalanced monthly for best results
- Works best for medium-term (1-3 month) holding periods

================================================================================
USAGE
================================================================================

    from auto_researcher.models.sector_momentum import SectorMomentumModel
    
    model = SectorMomentumModel()
    
    # Get all sector signals
    sectors = model.get_sector_signals()
    for sector, signal in sectors.items():
        print(f"{sector}: {signal.rotation_signal} (rank {signal.rank})")
    
    # Get signal for a specific stock
    signal = model.get_stock_signal("NVDA")
    print(f"NVDA sector momentum: {signal.summary}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Literal, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTOR DEFINITIONS
# ==============================================================================

# Sector ETFs for rotation analysis
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

ETF_TO_SECTOR = SECTOR_ETFS
SECTOR_TO_ETF = {v: k for k, v in SECTOR_ETFS.items()}

# Major tickers to sector mapping (for quick lookup)
TICKER_SECTORS = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "ADBE": "Technology", "CSCO": "Technology",
    "AVGO": "Technology", "TXN": "Technology", "QCOM": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "PYPL": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "UPS": "Industrials",
    "HON": "Industrials", "UNP": "Industrials", "RTX": "Industrials",
    "DE": "Industrials", "LMT": "Industrials", "GE": "Industrials",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples", "MDLZ": "Consumer Staples",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "SRE": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "ECL": "Materials", "NEM": "Materials", "FCX": "Materials",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate", "O": "Real Estate",
    # Communication Services
    "META": "Communication Services", "GOOGL": "Communication Services",
    "GOOG": "Communication Services", "DIS": "Communication Services",
    "NFLX": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services",
}


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SECTOR_MOMENTUM_CONFIG = {
    # Momentum calculation weights
    'weight_1m': 0.30,
    'weight_3m': 0.50,
    'weight_6m': 0.20,
    
    # Signal thresholds
    'overweight_threshold': 0.03,   # +3% relative strength
    'underweight_threshold': -0.03, # -3% relative strength
    
    # Rank-based signals
    'top_sectors': 3,      # Ranks 1-3 get overweight
    'bottom_sectors': 3,   # Ranks 9-11 get underweight
    
    # Stock impact from sector
    'sector_boost_max': 0.30,      # Max boost for top sector
    'sector_headwind_max': -0.25,  # Max headwind for bottom sector
    
    # Expected performance (from literature)
    'expected_alpha_overweight': 0.03,  # 3% annual for overweight sectors
    'expected_alpha_underweight': -0.02, # -2% for underweight
    
    # Cache duration
    'cache_hours': 1,
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SectorSignal:
    """Signal for a single sector."""
    sector: str
    etf: str
    
    # Returns
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    
    # Relative strength (vs SPY)
    rs_1m: float = 0.0
    rs_3m: float = 0.0
    rs_6m: float = 0.0
    
    # Composite momentum score
    momentum_score: float = 0.0
    
    # Rank (1 = strongest)
    rank: int = 0
    total_sectors: int = 11
    
    # Signal
    rotation_signal: Literal["overweight", "neutral", "underweight"] = "neutral"
    strength: Literal["strong", "moderate", "weak"] = "weak"
    
    # Expected outcomes
    expected_alpha: Optional[float] = None
    
    # Summary
    summary: str = ""


@dataclass
class StockSectorSignal:
    """Sector momentum signal for a specific stock."""
    ticker: str
    sector: str
    
    # Sector signal
    sector_signal: Optional[SectorSignal] = None
    
    # Stock-level impact
    sector_boost: float = 0.0  # -0.25 to +0.30
    direction: Optional[Literal["tailwind", "headwind"]] = None
    strength: Literal["strong", "moderate", "weak", "none"] = "none"
    is_actionable: bool = False
    
    # Expected impact
    expected_alpha: Optional[float] = None
    
    # Summary
    summary: str = ""
    rationale: str = ""


@dataclass
class SectorRotationSnapshot:
    """Complete sector rotation snapshot."""
    timestamp: datetime
    spy_return_1m: float
    spy_return_3m: float
    spy_return_6m: float
    
    # All sector signals
    sectors: Dict[str, SectorSignal] = field(default_factory=dict)
    
    # Rankings
    overweight_sectors: List[str] = field(default_factory=list)
    underweight_sectors: List[str] = field(default_factory=list)
    
    # Market regime
    regime: str = "neutral"  # "risk_on", "risk_off", "neutral", "rotating"


# ==============================================================================
# SECTOR MOMENTUM MODEL
# ==============================================================================

class SectorMomentumModel:
    """
    Generates sector rotation and momentum signals.
    
    Uses sector ETF relative strength to identify winning and losing sectors,
    then provides stock-level signals based on sector membership.
    """
    
    def __init__(self):
        """Initialize the model."""
        self._cache: Optional[SectorRotationSnapshot] = None
        self._cache_time: Optional[datetime] = None
        
        logger.info("Initialized SectorMomentumModel")
    
    def _fetch_sector_data(self) -> Optional[SectorRotationSnapshot]:
        """Fetch sector ETF data and compute momentum."""
        try:
            import yfinance as yf
            
            # Download sector ETF data
            tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
            data = yf.download(tickers, period="9mo", progress=False, auto_adjust=True)
            
            # Handle multi-level columns
            if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels'):
                if 'Close' in data.columns.get_level_values(0):
                    closes = data['Close']
                else:
                    closes = data
            else:
                closes = data
            
            if closes.empty:
                logger.error("No sector data fetched")
                return None
            
            # Calculate returns for different periods
            # Use business days: 21 = 1 month, 63 = 3 months, 126 = 6 months
            returns_1m = closes.pct_change(21).iloc[-1]
            returns_3m = closes.pct_change(63).iloc[-1]
            returns_6m = closes.pct_change(126).iloc[-1] if len(closes) > 126 else closes.pct_change(len(closes)-1).iloc[-1]
            
            # SPY returns for relative strength
            spy_1m = float(returns_1m.get("SPY", 0)) if "SPY" in returns_1m.index else 0
            spy_3m = float(returns_3m.get("SPY", 0)) if "SPY" in returns_3m.index else 0
            spy_6m = float(returns_6m.get("SPY", 0)) if "SPY" in returns_6m.index else 0
            
            cfg = SECTOR_MOMENTUM_CONFIG
            
            # Build sector signals
            sectors: Dict[str, SectorSignal] = {}
            
            for etf, sector in SECTOR_ETFS.items():
                if etf not in returns_1m.index:
                    continue
                
                ret_1m = float(returns_1m[etf]) if not np.isnan(returns_1m[etf]) else 0
                ret_3m = float(returns_3m[etf]) if not np.isnan(returns_3m[etf]) else 0
                ret_6m = float(returns_6m[etf]) if not np.isnan(returns_6m[etf]) else 0
                
                # Relative strength
                rs_1m = ret_1m - spy_1m
                rs_3m = ret_3m - spy_3m
                rs_6m = ret_6m - spy_6m
                
                # Weighted momentum score
                momentum = (
                    cfg['weight_1m'] * rs_1m +
                    cfg['weight_3m'] * rs_3m +
                    cfg['weight_6m'] * rs_6m
                )
                
                sectors[sector] = SectorSignal(
                    sector=sector,
                    etf=etf,
                    return_1m=ret_1m,
                    return_3m=ret_3m,
                    return_6m=ret_6m,
                    rs_1m=rs_1m,
                    rs_3m=rs_3m,
                    rs_6m=rs_6m,
                    momentum_score=momentum,
                )
            
            # Rank sectors by momentum
            sorted_sectors = sorted(sectors.values(), key=lambda x: x.momentum_score, reverse=True)
            
            for i, sig in enumerate(sorted_sectors):
                sig.rank = i + 1
                sig.total_sectors = len(sorted_sectors)
                
                # Determine signal based on rank
                if sig.rank <= cfg['top_sectors']:
                    sig.rotation_signal = "overweight"
                    sig.strength = "strong" if sig.rank == 1 else ("moderate" if sig.rank == 2 else "weak")
                    sig.expected_alpha = cfg['expected_alpha_overweight']
                elif sig.rank > len(sorted_sectors) - cfg['bottom_sectors']:
                    sig.rotation_signal = "underweight"
                    sig.strength = "strong" if sig.rank == len(sorted_sectors) else "moderate"
                    sig.expected_alpha = cfg['expected_alpha_underweight']
                else:
                    sig.rotation_signal = "neutral"
                    sig.strength = "weak"
                    sig.expected_alpha = 0.0
                
                # Summary
                emoji = "🟢" if sig.rotation_signal == "overweight" else ("🔴" if sig.rotation_signal == "underweight" else "🟡")
                sig.summary = (
                    f"{emoji} {sig.sector}: Rank {sig.rank}/{sig.total_sectors} | "
                    f"RS 1M: {sig.rs_1m*100:+.1f}% | RS 3M: {sig.rs_3m*100:+.1f}% | "
                    f"{sig.rotation_signal.upper()}"
                )
            
            # Update sectors dict with ranked signals
            sectors = {s.sector: s for s in sorted_sectors}
            
            # Identify overweight/underweight lists
            overweight = [s.sector for s in sorted_sectors if s.rotation_signal == "overweight"]
            underweight = [s.sector for s in sorted_sectors if s.rotation_signal == "underweight"]
            
            # Determine market regime
            avg_rs = np.mean([s.rs_3m for s in sorted_sectors])
            spread = sorted_sectors[0].momentum_score - sorted_sectors[-1].momentum_score
            
            if spread > 0.10:
                regime = "rotating"  # High dispersion = sector rotation matters
            elif avg_rs > 0.02:
                regime = "risk_on"
            elif avg_rs < -0.02:
                regime = "risk_off"
            else:
                regime = "neutral"
            
            return SectorRotationSnapshot(
                timestamp=datetime.now(),
                spy_return_1m=spy_1m,
                spy_return_3m=spy_3m,
                spy_return_6m=spy_6m,
                sectors=sectors,
                overweight_sectors=overweight,
                underweight_sectors=underweight,
                regime=regime,
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch sector data: {e}")
            return None
    
    def _get_cached_snapshot(self) -> Optional[SectorRotationSnapshot]:
        """Get cached snapshot or fetch new data."""
        cache_hours = SECTOR_MOMENTUM_CONFIG['cache_hours']
        
        if (self._cache and self._cache_time and
            datetime.now() - self._cache_time < timedelta(hours=cache_hours)):
            return self._cache
        
        snapshot = self._fetch_sector_data()
        if snapshot:
            self._cache = snapshot
            self._cache_time = datetime.now()
        
        return snapshot
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Get sector for a ticker."""
        ticker = ticker.upper()
        
        # Check cache first
        if ticker in TICKER_SECTORS:
            return TICKER_SECTORS[ticker]
        
        # Try yfinance
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            # Cache for future use
            TICKER_SECTORS[ticker] = sector
            return sector
        except:
            return "Unknown"
    
    def get_sector_signals(self) -> Dict[str, SectorSignal]:
        """
        Get all sector momentum signals.
        
        Returns:
            Dictionary mapping sector name to SectorSignal.
        """
        snapshot = self._get_cached_snapshot()
        if not snapshot:
            return {}
        return snapshot.sectors
    
    def get_rotation_snapshot(self) -> Optional[SectorRotationSnapshot]:
        """
        Get complete sector rotation snapshot.
        
        Returns:
            SectorRotationSnapshot with all data.
        """
        return self._get_cached_snapshot()
    
    def get_stock_signal(self, ticker: str) -> StockSectorSignal:
        """
        Get sector momentum signal for a specific stock.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            StockSectorSignal with sector impact for the stock.
        """
        ticker = ticker.upper()
        signal = StockSectorSignal(ticker=ticker, sector="Unknown")
        
        try:
            # Get sector
            sector = self._get_ticker_sector(ticker)
            signal.sector = sector
            
            if sector == "Unknown":
                signal.summary = f"⚪ {ticker}: Unknown sector"
                return signal
            
            # Get sector signals
            snapshot = self._get_cached_snapshot()
            if not snapshot or sector not in snapshot.sectors:
                signal.summary = f"⚪ {ticker}: No sector data for {sector}"
                return signal
            
            sector_sig = snapshot.sectors[sector]
            signal.sector_signal = sector_sig
            
            cfg = SECTOR_MOMENTUM_CONFIG
            
            # Calculate stock-level impact based on sector rank
            if sector_sig.rotation_signal == "overweight":
                # Boost based on rank (rank 1 = max boost, rank 3 = min boost)
                boost_fraction = 1.0 - (sector_sig.rank - 1) / cfg['top_sectors']
                signal.sector_boost = cfg['sector_boost_max'] * boost_fraction
                signal.direction = "tailwind"
                signal.is_actionable = True
                
                if sector_sig.rank == 1:
                    signal.strength = "strong"
                elif sector_sig.rank == 2:
                    signal.strength = "moderate"
                else:
                    signal.strength = "weak"
                    
            elif sector_sig.rotation_signal == "underweight":
                # Headwind based on rank (last = max headwind)
                num_sectors = sector_sig.total_sectors
                underweight_rank = sector_sig.rank - (num_sectors - cfg['bottom_sectors'])
                boost_fraction = underweight_rank / cfg['bottom_sectors']
                signal.sector_boost = cfg['sector_headwind_max'] * boost_fraction
                signal.direction = "headwind"
                signal.is_actionable = True
                
                if sector_sig.rank == num_sectors:
                    signal.strength = "strong"
                else:
                    signal.strength = "moderate"
            else:
                signal.sector_boost = 0.0
                signal.direction = None
                signal.strength = "none"
            
            # Expected alpha
            signal.expected_alpha = sector_sig.expected_alpha
            
            # Summary
            if signal.direction == "tailwind":
                emoji = "🟢"
                dir_text = "TAILWIND"
            elif signal.direction == "headwind":
                emoji = "🔴"
                dir_text = "HEADWIND"
            else:
                emoji = "🟡"
                dir_text = "NEUTRAL"
            
            signal.summary = (
                f"{emoji} {signal.strength.upper()} {dir_text}: "
                f"{sector} (Rank {sector_sig.rank}/{sector_sig.total_sectors}) | "
                f"Boost: {signal.sector_boost*100:+.1f}%"
            )
            
            signal.rationale = (
                f"{sector} is {'outperforming' if sector_sig.rs_3m > 0 else 'underperforming'} "
                f"(RS 3M: {sector_sig.rs_3m*100:+.1f}%)"
            )
            
        except Exception as e:
            logger.error(f"Error getting sector signal for {ticker}: {e}")
            signal.summary = f"Error: {str(e)}"
        
        return signal
    
    def get_stock_signals_batch(self, tickers: List[str]) -> Dict[str, StockSectorSignal]:
        """Get sector signals for multiple stocks."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_stock_signal(ticker)
        return results
    
    def print_rotation_table(self) -> None:
        """Print a formatted sector rotation table."""
        snapshot = self._get_cached_snapshot()
        if not snapshot:
            print("No sector data available")
            return
        
        print("\n" + "=" * 75)
        print("  SECTOR ROTATION TABLE")
        print(f"  As of: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Market Regime: {snapshot.regime.upper()}")
        print("=" * 75)
        
        print(f"\n{'Rank':<6}{'Sector':<25}{'RS 1M':<10}{'RS 3M':<10}{'Signal':<15}")
        print("-" * 75)
        
        sorted_sectors = sorted(snapshot.sectors.values(), key=lambda x: x.rank)
        
        for s in sorted_sectors:
            emoji = "🟢" if s.rotation_signal == "overweight" else ("🔴" if s.rotation_signal == "underweight" else "🟡")
            print(
                f"{s.rank:<6}{s.sector:<25}"
                f"{s.rs_1m*100:+6.1f}%   {s.rs_3m*100:+6.1f}%   "
                f"{emoji} {s.rotation_signal.upper()}"
            )
        
        print("\n" + "-" * 75)
        print(f"SPY Returns: 1M {snapshot.spy_return_1m*100:+.1f}% | 3M {snapshot.spy_return_3m*100:+.1f}%")
        print(f"Overweight: {', '.join(snapshot.overweight_sectors)}")
        print(f"Underweight: {', '.join(snapshot.underweight_sectors)}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_sector_signals() -> Dict[str, SectorSignal]:
    """Quick helper to get all sector signals."""
    model = SectorMomentumModel()
    return model.get_sector_signals()


def get_stock_sector_signal(ticker: str) -> StockSectorSignal:
    """Quick helper to get sector signal for a stock."""
    model = SectorMomentumModel()
    return model.get_stock_signal(ticker)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    model = SectorMomentumModel()
    
    # Print rotation table
    model.print_rotation_table()
    
    # If tickers provided, show stock-level signals
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
        print("\n" + "=" * 75)
        print("  STOCK SECTOR SIGNALS")
        print("=" * 75)
        
        for ticker in tickers:
            signal = model.get_stock_signal(ticker)
            print(f"\n{ticker}:")
            print(f"  {signal.summary}")
            if signal.rationale:
                print(f"  {signal.rationale}")
            if signal.expected_alpha:
                print(f"  Expected Alpha: {signal.expected_alpha*100:+.1f}% annually")
