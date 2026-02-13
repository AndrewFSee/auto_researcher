"""
Quality-Value Composite Model
=============================

A quantitative model combining profitability metrics (quality) with valuation 
metrics (value) to generate trading signals.

================================================================================
ACADEMIC FOUNDATION
================================================================================

Key Research:
1. Novy-Marx (2013) "The Other Side of Value"
   - Gross profitability (GP/Assets) is strongly predictive of returns
   - Quality and value are complementary: cheap + profitable = alpha

2. Fama-French (2015) "A Five-Factor Model"
   - Added profitability (RMW) and investment (CMA) factors
   - High profitability firms outperform, especially when cheap

3. Asness, Frazzini, Pedersen (2019) "Quality Minus Junk"
   - Combined quality metrics: profitability, growth, safety, payout
   - QMJ factor earns ~4% annual alpha

4. Piotroski (2000) "Value Investing with F-Score"
   - 9-point fundamental score for value stocks
   - High F-Score value stocks outperform by 7.5% annually

Expected Performance:
- Quality alone: +2-3% annual excess
- Value alone: +2-3% annual excess (long-term)
- Quality + Value composite: +4-6% annual excess

================================================================================
SIGNAL CONSTRUCTION
================================================================================

QUALITY SCORE (0-100) - Higher is better:
1. ROE > 15%: +20 points
2. ROA > 8%: +15 points
3. Gross Margin > sector median: +15 points
4. Operating Margin > 15%: +15 points
5. Low Debt/Equity < 0.5: +15 points
6. Current Ratio > 1.5: +10 points
7. Positive earnings growth: +10 points

VALUE SCORE (0-100) - Lower valuations score higher:
1. P/E < 15: +25 points (or P/E < sector median: +15)
2. P/B < 3: +20 points
3. EV/EBITDA < 10: +20 points
4. P/S < 2: +15 points
5. FCF Yield > 5%: +20 points (if available)

COMPOSITE SIGNAL:
- Quality Weight: 50%
- Value Weight: 50%
- Combined score 0-100

Signal Direction:
- LONG: Score > 65 (quality stock at reasonable price)
- NEUTRAL: 35-65
- SHORT: Score < 35 (low quality or overvalued)

================================================================================
USAGE
================================================================================

    from auto_researcher.models.quality_value import QualityValueModel
    
    model = QualityValueModel()
    signal = model.get_signal("AAPL")
    
    print(f"Quality: {signal.quality_score}")
    print(f"Value: {signal.value_score}")
    print(f"Composite: {signal.composite_score}")
    print(f"Signal: {signal.direction}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTOR BENCHMARKS
# ==============================================================================

# Median P/E by sector (approximate, based on historical data)
SECTOR_PE_MEDIANS = {
    'Technology': 28,
    'Healthcare': 22,
    'Financial Services': 12,
    'Consumer Cyclical': 18,
    'Consumer Defensive': 20,
    'Industrials': 18,
    'Basic Materials': 12,
    'Energy': 10,
    'Utilities': 16,
    'Real Estate': 35,
    'Communication Services': 18,
}

# Median Gross Margin by sector
SECTOR_GM_MEDIANS = {
    'Technology': 0.55,
    'Healthcare': 0.50,
    'Financial Services': 0.60,
    'Consumer Cyclical': 0.35,
    'Consumer Defensive': 0.35,
    'Industrials': 0.30,
    'Basic Materials': 0.25,
    'Energy': 0.30,
    'Utilities': 0.35,
    'Real Estate': 0.55,
    'Communication Services': 0.45,
}

DEFAULT_PE_MEDIAN = 18
DEFAULT_GM_MEDIAN = 0.40


# ==============================================================================
# CONFIGURATION
# ==============================================================================

QUALITY_VALUE_CONFIG = {
    # Quality thresholds
    'roe_threshold': 0.15,          # 15% ROE is good
    'roa_threshold': 0.08,          # 8% ROA is good
    'operating_margin_threshold': 0.15,  # 15% operating margin
    'debt_equity_max': 0.50,        # Low leverage threshold
    'current_ratio_min': 1.5,       # Healthy liquidity
    
    # Value thresholds
    'pe_cheap': 15,                 # Cheap P/E
    'pb_cheap': 3,                  # Cheap P/B
    'ev_ebitda_cheap': 10,          # Cheap EV/EBITDA
    'ps_cheap': 2,                  # Cheap P/S
    'fcf_yield_good': 0.05,         # 5% FCF yield is good
    
    # Weights
    'quality_weight': 0.50,
    'value_weight': 0.50,
    
    # Signal thresholds
    'long_threshold': 65,
    'short_threshold': 35,
}


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class QualityValueSignal:
    """Signal output from the quality-value model."""
    ticker: str
    
    # Company name (from yfinance)
    company_name: str = ""
    
    # Quality components (0-100)
    quality_score: float = 0.0
    quality_grade: str = "N/A"  # A, B, C, D, F
    
    # Value components (0-100)
    value_score: float = 0.0
    value_grade: str = "N/A"
    
    # Composite
    composite_score: float = 0.0
    direction: Optional[Literal["long", "short"]] = None
    strength: Literal["strong", "moderate", "weak", "none"] = "none"
    is_actionable: bool = False
    
    # Individual quality metrics used
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # Individual value metrics used
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    
    # Context
    sector: Optional[str] = None
    sector_pe_median: Optional[float] = None
    
    # Quality factor breakdown
    quality_factors: Dict[str, float] = field(default_factory=dict)
    value_factors: Dict[str, float] = field(default_factory=dict)
    
    # Summary
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    
    # Expected outcomes (from literature)
    expected_annual_alpha: Optional[float] = None


# ==============================================================================
# QUALITY-VALUE MODEL
# ==============================================================================

class QualityValueModel:
    """
    Computes quality-value composite signals for stocks.
    
    Based on academic research showing that cheap, high-quality stocks
    generate significant alpha.
    """
    
    def __init__(
        self,
        quality_weight: float = 0.50,
        value_weight: float = 0.50,
    ):
        """
        Initialize the model.
        
        Args:
            quality_weight: Weight for quality score (0-1).
            value_weight: Weight for value score (0-1).
        """
        self.quality_weight = quality_weight
        self.value_weight = value_weight
        
        # Normalize weights
        total = quality_weight + value_weight
        self.quality_weight = quality_weight / total
        self.value_weight = value_weight / total
        
        logger.info(f"Initialized QualityValueModel (quality={self.quality_weight:.0%}, value={self.value_weight:.0%})")
    
    def _fetch_metrics(self, ticker: str) -> Dict[str, Any]:
        """Fetch financial metrics from yfinance."""
        try:
            import yfinance as yf
            import pandas as pd
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            def safe_get(key, default=None):
                val = info.get(key, default)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return default
                return val
            
            # Try to compute FCF yield
            fcf = safe_get('freeCashflow')
            market_cap = safe_get('marketCap')
            fcf_yield = None
            if fcf and market_cap and market_cap > 0:
                fcf_yield = fcf / market_cap
            
            return {
                'ticker': ticker,
                'company_name': safe_get('shortName', safe_get('longName', ticker)),
                'sector': safe_get('sector'),
                'industry': safe_get('industry'),
                # Quality metrics
                'roe': safe_get('returnOnEquity'),
                'roa': safe_get('returnOnAssets'),
                'gross_margin': safe_get('grossMargins'),
                'operating_margin': safe_get('operatingMargins'),
                'profit_margin': safe_get('profitMargins'),
                'debt_to_equity': safe_get('debtToEquity'),
                'current_ratio': safe_get('currentRatio'),
                'earnings_growth': safe_get('earningsGrowth'),
                'revenue_growth': safe_get('revenueGrowth'),
                # Value metrics
                'pe_ratio': safe_get('trailingPE'),
                'forward_pe': safe_get('forwardPE'),
                'pb_ratio': safe_get('priceToBook'),
                'ps_ratio': safe_get('priceToSalesTrailing12Months'),
                'ev_ebitda': safe_get('enterpriseToEbitda'),
                'fcf_yield': fcf_yield,
                'dividend_yield': safe_get('dividendYield'),
                'market_cap': market_cap,
            }
            
        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {e}")
            return {'ticker': ticker}
    
    def _compute_quality_score(self, metrics: Dict[str, Any]) -> tuple[float, Dict[str, float], List[str], List[str]]:
        """
        Compute quality score from metrics.
        
        Returns:
            (score, factors_dict, strengths, concerns)
        """
        score = 0.0
        factors = {}
        strengths = []
        concerns = []
        
        cfg = QUALITY_VALUE_CONFIG
        sector = metrics.get('sector', '')
        sector_gm = SECTOR_GM_MEDIANS.get(sector, DEFAULT_GM_MEDIAN)
        
        # 1. ROE (20 points)
        roe = metrics.get('roe')
        if roe is not None:
            if roe >= cfg['roe_threshold']:
                pts = 20
                strengths.append(f"Strong ROE ({roe*100:.1f}%)")
            elif roe >= 0.10:
                pts = 12
            elif roe >= 0.05:
                pts = 5
            else:
                pts = 0
                if roe < 0:
                    concerns.append(f"Negative ROE ({roe*100:.1f}%)")
            factors['roe'] = pts
            score += pts
        
        # 2. ROA (15 points)
        roa = metrics.get('roa')
        if roa is not None:
            if roa >= cfg['roa_threshold']:
                pts = 15
                strengths.append(f"Strong ROA ({roa*100:.1f}%)")
            elif roa >= 0.05:
                pts = 10
            elif roa >= 0.02:
                pts = 5
            else:
                pts = 0
            factors['roa'] = pts
            score += pts
        
        # 3. Gross Margin vs sector (15 points)
        gm = metrics.get('gross_margin')
        if gm is not None:
            if gm >= sector_gm * 1.2:  # 20% above sector
                pts = 15
                strengths.append(f"Above-sector gross margin ({gm*100:.0f}%)")
            elif gm >= sector_gm:
                pts = 10
            elif gm >= sector_gm * 0.8:
                pts = 5
            else:
                pts = 0
                concerns.append(f"Below-sector gross margin ({gm*100:.0f}%)")
            factors['gross_margin'] = pts
            score += pts
        
        # 4. Operating Margin (15 points)
        om = metrics.get('operating_margin')
        if om is not None:
            if om >= cfg['operating_margin_threshold']:
                pts = 15
            elif om >= 0.10:
                pts = 10
            elif om >= 0.05:
                pts = 5
            else:
                pts = 0
                if om < 0:
                    concerns.append("Negative operating margin")
            factors['operating_margin'] = pts
            score += pts
        
        # 5. Low Debt/Equity (15 points)
        de = metrics.get('debt_to_equity')
        if de is not None:
            # Convert from percentage if needed
            if de > 10:  # Likely in percentage form
                de = de / 100
            if de <= cfg['debt_equity_max']:
                pts = 15
                strengths.append(f"Low leverage (D/E={de:.1f})")
            elif de <= 1.0:
                pts = 10
            elif de <= 2.0:
                pts = 5
            else:
                pts = 0
                concerns.append(f"High leverage (D/E={de:.1f})")
            factors['debt_equity'] = pts
            score += pts
        
        # 6. Current Ratio (10 points)
        cr = metrics.get('current_ratio')
        if cr is not None:
            if cr >= cfg['current_ratio_min']:
                pts = 10
            elif cr >= 1.0:
                pts = 6
            else:
                pts = 0
                concerns.append(f"Low liquidity (current ratio={cr:.1f})")
            factors['current_ratio'] = pts
            score += pts
        
        # 7. Positive Earnings Growth (10 points)
        eg = metrics.get('earnings_growth')
        if eg is not None:
            if eg >= 0.20:  # 20%+ growth
                pts = 10
                strengths.append(f"Strong earnings growth ({eg*100:.0f}%)")
            elif eg >= 0.10:
                pts = 7
            elif eg >= 0:
                pts = 4
            else:
                pts = 0
                concerns.append(f"Declining earnings ({eg*100:.0f}%)")
            factors['earnings_growth'] = pts
            score += pts
        
        return score, factors, strengths, concerns
    
    def _compute_value_score(self, metrics: Dict[str, Any]) -> tuple[float, Dict[str, float], List[str], List[str]]:
        """
        Compute value score from metrics.
        
        Returns:
            (score, factors_dict, strengths, concerns)
        """
        score = 0.0
        factors = {}
        strengths = []
        concerns = []
        
        cfg = QUALITY_VALUE_CONFIG
        sector = metrics.get('sector', '')
        sector_pe = SECTOR_PE_MEDIANS.get(sector, DEFAULT_PE_MEDIAN)
        
        # 1. P/E Ratio (25 points)
        pe = metrics.get('pe_ratio')
        if pe is not None and pe > 0:
            if pe < cfg['pe_cheap']:
                pts = 25
                strengths.append(f"Cheap P/E ({pe:.1f})")
            elif pe < sector_pe:
                pts = 15
                strengths.append(f"Below-sector P/E ({pe:.1f} vs {sector_pe:.0f})")
            elif pe < sector_pe * 1.5:
                pts = 8
            else:
                pts = 0
                concerns.append(f"High P/E ({pe:.1f} vs sector {sector_pe:.0f})")
            factors['pe'] = pts
            score += pts
        elif pe is not None and pe < 0:
            # Negative P/E (losses)
            concerns.append("Negative earnings (no P/E)")
            factors['pe'] = 0
        
        # 2. P/B Ratio (20 points)
        pb = metrics.get('pb_ratio')
        if pb is not None and pb > 0:
            if pb < cfg['pb_cheap']:
                pts = 20
                strengths.append(f"Cheap P/B ({pb:.1f})")
            elif pb < 5:
                pts = 12
            elif pb < 10:
                pts = 5
            else:
                pts = 0
                concerns.append(f"High P/B ({pb:.1f})")
            factors['pb'] = pts
            score += pts
        
        # 3. EV/EBITDA (20 points)
        ev_ebitda = metrics.get('ev_ebitda')
        if ev_ebitda is not None and ev_ebitda > 0:
            if ev_ebitda < cfg['ev_ebitda_cheap']:
                pts = 20
                strengths.append(f"Cheap EV/EBITDA ({ev_ebitda:.1f})")
            elif ev_ebitda < 15:
                pts = 12
            elif ev_ebitda < 20:
                pts = 5
            else:
                pts = 0
            factors['ev_ebitda'] = pts
            score += pts
        
        # 4. P/S Ratio (15 points)
        ps = metrics.get('ps_ratio')
        if ps is not None and ps > 0:
            if ps < cfg['ps_cheap']:
                pts = 15
            elif ps < 5:
                pts = 8
            elif ps < 10:
                pts = 3
            else:
                pts = 0
                concerns.append(f"High P/S ({ps:.1f})")
            factors['ps'] = pts
            score += pts
        
        # 5. FCF Yield (20 points)
        fcf_yield = metrics.get('fcf_yield')
        if fcf_yield is not None:
            if fcf_yield >= cfg['fcf_yield_good']:
                pts = 20
                strengths.append(f"Strong FCF yield ({fcf_yield*100:.1f}%)")
            elif fcf_yield >= 0.03:
                pts = 12
            elif fcf_yield >= 0.01:
                pts = 5
            else:
                pts = 0
                if fcf_yield < 0:
                    concerns.append("Negative free cash flow")
            factors['fcf_yield'] = pts
            score += pts
        
        return score, factors, strengths, concerns
    
    def _score_to_grade(self, score: float) -> str:
        """Convert 0-100 score to letter grade."""
        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 35:
            return "D"
        else:
            return "F"
    
    def get_signal(self, ticker: str) -> QualityValueSignal:
        """
        Get quality-value signal for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            QualityValueSignal with composite score and direction.
        """
        ticker = ticker.upper()
        signal = QualityValueSignal(ticker=ticker)
        
        try:
            # Fetch metrics
            metrics = self._fetch_metrics(ticker)
            
            if not metrics or len(metrics) <= 1:
                signal.summary = "Could not fetch financial metrics"
                return signal
            
            # Store raw metrics
            signal.company_name = metrics.get('company_name', ticker)
            signal.sector = metrics.get('sector')
            signal.roe = metrics.get('roe')
            signal.roa = metrics.get('roa')
            signal.gross_margin = metrics.get('gross_margin')
            signal.operating_margin = metrics.get('operating_margin')
            signal.debt_to_equity = metrics.get('debt_to_equity')
            signal.current_ratio = metrics.get('current_ratio')
            signal.earnings_growth = metrics.get('earnings_growth')
            signal.pe_ratio = metrics.get('pe_ratio')
            signal.pb_ratio = metrics.get('pb_ratio')
            signal.ps_ratio = metrics.get('ps_ratio')
            signal.ev_ebitda = metrics.get('ev_ebitda')
            signal.fcf_yield = metrics.get('fcf_yield')
            signal.sector_pe_median = SECTOR_PE_MEDIANS.get(signal.sector, DEFAULT_PE_MEDIAN)
            
            # Compute quality score
            quality_raw, quality_factors, q_strengths, q_concerns = self._compute_quality_score(metrics)
            signal.quality_factors = quality_factors
            
            # Compute value score
            value_raw, value_factors, v_strengths, v_concerns = self._compute_value_score(metrics)
            signal.value_factors = value_factors
            
            # Normalize scores (handle missing data gracefully)
            quality_max = sum([20, 15, 15, 15, 15, 10, 10])  # 100 max
            value_max = sum([25, 20, 20, 15, 20])  # 100 max
            
            quality_possible = sum(quality_factors.values()) + sum(
                [20 if 'roe' not in quality_factors else 0,
                 15 if 'roa' not in quality_factors else 0,
                 15 if 'gross_margin' not in quality_factors else 0,
                 15 if 'operating_margin' not in quality_factors else 0,
                 15 if 'debt_equity' not in quality_factors else 0,
                 10 if 'current_ratio' not in quality_factors else 0,
                 10 if 'earnings_growth' not in quality_factors else 0]
            )
            
            # Compute actual scores (as % of max possible given available data)
            if quality_factors:
                quality_available_max = sum([
                    20 if 'roe' in quality_factors else 0,
                    15 if 'roa' in quality_factors else 0,
                    15 if 'gross_margin' in quality_factors else 0,
                    15 if 'operating_margin' in quality_factors else 0,
                    15 if 'debt_equity' in quality_factors else 0,
                    10 if 'current_ratio' in quality_factors else 0,
                    10 if 'earnings_growth' in quality_factors else 0,
                ])
                signal.quality_score = (quality_raw / quality_available_max * 100) if quality_available_max > 0 else 0
            
            if value_factors:
                value_available_max = sum([
                    25 if 'pe' in value_factors else 0,
                    20 if 'pb' in value_factors else 0,
                    20 if 'ev_ebitda' in value_factors else 0,
                    15 if 'ps' in value_factors else 0,
                    20 if 'fcf_yield' in value_factors else 0,
                ])
                signal.value_score = (value_raw / value_available_max * 100) if value_available_max > 0 else 0
            
            signal.quality_grade = self._score_to_grade(signal.quality_score)
            signal.value_grade = self._score_to_grade(signal.value_score)
            
            # Combine strengths and concerns
            signal.strengths = q_strengths + v_strengths
            signal.concerns = q_concerns + v_concerns
            
            # Compute composite score
            signal.composite_score = (
                self.quality_weight * signal.quality_score +
                self.value_weight * signal.value_score
            )
            
            # Determine direction
            cfg = QUALITY_VALUE_CONFIG
            if signal.composite_score >= cfg['long_threshold']:
                signal.direction = "long"
                signal.is_actionable = True
                if signal.composite_score >= 80:
                    signal.strength = "strong"
                else:
                    signal.strength = "moderate"
            elif signal.composite_score <= cfg['short_threshold']:
                signal.direction = "short"
                signal.is_actionable = True
                if signal.composite_score <= 20:
                    signal.strength = "strong"
                else:
                    signal.strength = "weak"  # Short signals are less reliable
            else:
                signal.direction = None
                signal.strength = "none"
            
            # Expected alpha (from literature)
            if signal.direction == "long":
                if signal.quality_score >= 70 and signal.value_score >= 70:
                    signal.expected_annual_alpha = 0.05  # 5% for cheap + quality
                elif signal.quality_score >= 70:
                    signal.expected_annual_alpha = 0.025  # 2.5% for quality alone
                elif signal.value_score >= 70:
                    signal.expected_annual_alpha = 0.025  # 2.5% for value alone
                else:
                    signal.expected_annual_alpha = 0.02
            elif signal.direction == "short":
                signal.expected_annual_alpha = -0.03  # -3% for junk/overvalued
            
            # Summary
            q_emoji = "✅" if signal.quality_score >= 65 else ("⚠️" if signal.quality_score >= 35 else "❌")
            v_emoji = "✅" if signal.value_score >= 65 else ("⚠️" if signal.value_score >= 35 else "❌")
            
            if signal.is_actionable:
                dir_text = "LONG" if signal.direction == "long" else "SHORT"
                signal.summary = (
                    f"{'🟢' if signal.direction == 'long' else '🔴'} {signal.strength.upper()} {dir_text}: "
                    f"Quality {signal.quality_grade} ({signal.quality_score:.0f}) {q_emoji} | "
                    f"Value {signal.value_grade} ({signal.value_score:.0f}) {v_emoji} | "
                    f"Composite: {signal.composite_score:.0f}"
                )
            else:
                signal.summary = (
                    f"⚪ NEUTRAL: "
                    f"Quality {signal.quality_grade} ({signal.quality_score:.0f}) {q_emoji} | "
                    f"Value {signal.value_grade} ({signal.value_score:.0f}) {v_emoji} | "
                    f"Composite: {signal.composite_score:.0f}"
                )
            
        except Exception as e:
            logger.error(f"Error getting quality-value signal for {ticker}: {e}")
            signal.summary = f"Error: {str(e)}"
        
        return signal
    
    def get_signal_batch(self, tickers: List[str]) -> Dict[str, QualityValueSignal]:
        """Get signals for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_signal(ticker)
        return results


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_quality_value_signal(ticker: str) -> QualityValueSignal:
    """Quick helper to get quality-value signal for a single ticker."""
    model = QualityValueModel()
    return model.get_signal(ticker)


def get_quality_value_signals(tickers: List[str]) -> Dict[str, QualityValueSignal]:
    """Quick helper to get quality-value signals for multiple tickers."""
    model = QualityValueModel()
    return model.get_signal_batch(tickers)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "MSFT", "JPM", "XOM", "GM"]
    
    model = QualityValueModel()
    
    print("=" * 70)
    print("  QUALITY-VALUE ANALYSIS")
    print("=" * 70)
    
    for ticker in tickers:
        print(f"\n{ticker}:")
        signal = model.get_signal(ticker)
        
        print(f"  {signal.summary}")
        
        if signal.strengths:
            print(f"  Strengths: {', '.join(signal.strengths[:3])}")
        if signal.concerns:
            print(f"  Concerns: {', '.join(signal.concerns[:3])}")
        
        if signal.expected_annual_alpha:
            print(f"  Expected Alpha: {signal.expected_annual_alpha*100:+.1f}% annually")
