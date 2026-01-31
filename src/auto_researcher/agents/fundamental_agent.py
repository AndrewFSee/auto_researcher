"""
Fundamental Agent.

This agent analyzes company fundamentals using financial data and LLM analysis.
It fetches key metrics and provides qualitative insights on business quality.

================================================================================
QUALITY-VALUE MODEL - STRENGTHS & WEAKNESSES
================================================================================

ACADEMIC BASIS:
    Novy-Marx (2013): Gross profitability predicts returns independent of value
    Fama-French (2015): Profitability (RMW) is a priced factor
    Asness QMJ (2019): Quality-minus-Junk earns 4% annual alpha
    
STRENGTHS:
    ✅ Combines two proven factors (quality + value)
    ✅ Quality and value are complementary - cheap + profitable = best
    ✅ Uses sector-relative benchmarks (avoids comparing NVDA to XOM)
    ✅ Multiple metrics reduce noise (7 quality + 5 value factors)
    ✅ Works well over medium-term holding periods (6-12 months)
    ✅ Easy to interpret letter grades (A-F)
    
WEAKNESSES:
    ⚠️ VALUE TRAP RISK - cheap stocks can stay cheap or get cheaper
    ⚠️ Quality metrics are backward-looking (trailing 12 months)
    ⚠️ Doesn't capture momentum or sentiment
    ⚠️ Banks/financials may score poorly due to leverage
    ⚠️ Growth stocks will score poorly on value (by design)
    ⚠️ yfinance data may have delays or gaps
    
BEST USE CASES:
    • Screening for high-quality companies at reasonable prices
    • Identifying potential value traps (low value, low quality)
    • Confirming fundamental thesis on stock picks
    
CAUTIONS:
    • High quality + low value = may be overpriced for good reason
    • Low quality + high value = potential value trap
    • Sector matters - compare within sectors
    • Combine with momentum/sentiment for timing

Usage:
    from auto_researcher.agents.fundamental_agent import FundamentalAgent
    
    agent = FundamentalAgent(model="gpt-4o-mini")
    results = agent.analyze_tickers(["GOOGL", "NVDA", "AAPL"])
    
    for ticker, analysis in results.items():
        print(f"{ticker}: {analysis.quality_score:.2f} - {analysis.summary}")
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Literal

import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    litellm = None

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# Try to import the fundamentals alpha model
try:
    from ..models.fundamentals_alpha import FundamentalsAlphaModel, FundamentalsAlphaSignal
    HAS_ALPHA_MODEL = True
except ImportError:
    HAS_ALPHA_MODEL = False
    FundamentalsAlphaModel = None
    FundamentalsAlphaSignal = None

# Try to import the quality-value model
try:
    from ..models.quality_value import QualityValueModel, QualityValueSignal
    HAS_QV_MODEL = True
except ImportError:
    HAS_QV_MODEL = False
    QualityValueModel = None
    QualityValueSignal = None


# ==============================================================================
# BASE AGENT CLASS
# ==============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Agents are modular components that can be composed to build
    complex research pipelines.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name for identification."""
        pass

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary task."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class FinancialMetrics:
    """Key financial metrics for a company."""
    ticker: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    business_summary: Optional[str] = None
    
    # DefeatBeta historical data (new)
    revenue_yoy_growth: Optional[float] = None  # From actual financials
    net_income_yoy_growth: Optional[float] = None
    revenue_trend: Optional[str] = None  # 'growing' or 'declining'
    net_income_trend: Optional[str] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    
    # EPS estimates and revisions (alpha signal)
    eps_estimate: Optional[float] = None
    eps_estimate_growth: Optional[float] = None
    num_analysts: Optional[int] = None
    revision_30d: Optional[float] = None  # Estimate revision last 30 days
    revision_90d: Optional[float] = None  # Estimate revision last 90 days
    revision_signal: Optional[str] = None  # 'positive', 'neutral', 'negative'
    
    # EPS surprise history
    avg_eps_surprise: Optional[float] = None  # Average surprise over last 4 quarters
    beat_rate: Optional[float] = None  # % of quarters beating estimates


@dataclass
class FundamentalResult:
    """Fundamental analysis result for a ticker."""
    ticker: str
    quality_score: float  # 0 (poor) to 1 (excellent)
    quality_label: Literal["poor", "fair", "good", "excellent"]
    value_assessment: Literal["undervalued", "fairly_valued", "overvalued"]
    growth_outlook: Literal["declining", "stable", "growing", "high_growth"]
    risk_level: Literal["low", "moderate", "high"]
    summary: str
    strengths: list[str]
    concerns: list[str]
    metrics: FinancialMetrics
    analysis_date: datetime
    raw_response: Optional[str] = None
    
    # Quantitative Alpha Signals (from backtest-validated model)
    # Based on 11,947 earnings events, OOS validated on 2025 data
    alpha_signal_type: Optional[str] = None  # double_positive, double_negative, etc.
    alpha_short_term_score: Optional[float] = None  # For 5-20 day horizon (revision-driven)
    alpha_long_term_score: Optional[float] = None   # For 40-60 day horizon (PEAD-driven)
    alpha_combined_score: Optional[float] = None    # Blended signal
    alpha_recommended_horizon: Optional[str] = None  # short, long, either, none
    alpha_expected_quintile: Optional[int] = None    # 1-5, predicted return bucket
    alpha_data_freshness: Optional[str] = None       # fresh (<10d), recent (<30d), stale
    alpha_summary: Optional[str] = None              # Human-readable signal summary
    alpha_strengths: Optional[list[str]] = None      # Model-detected strengths
    alpha_risks: Optional[list[str]] = None          # Model-detected risks
    
    # Quality-Value Composite Signal (backtested model)
    # Based on Novy-Marx, Fama-French, Asness QMJ research
    qv_quality_score: Optional[float] = None         # 0-100 quality score
    qv_quality_grade: Optional[str] = None           # A, B, C, D, F
    qv_value_score: Optional[float] = None           # 0-100 value score
    qv_value_grade: Optional[str] = None             # A, B, C, D, F
    qv_composite_score: Optional[float] = None       # 0-100 combined score
    qv_direction: Optional[str] = None               # "long", "short", or None
    qv_strength: Optional[str] = None                # "strong", "moderate", "weak", "none"
    qv_is_actionable: bool = False
    qv_expected_alpha: Optional[float] = None        # Expected annual alpha
    qv_strengths: Optional[list[str]] = None         # Quality-value strengths
    qv_concerns: Optional[list[str]] = None          # Quality-value concerns
    qv_summary: Optional[str] = None                 # Human-readable signal summary


@dataclass
class FundamentalAgentConfig:
    """Configuration for the fundamental agent."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    timeout: int = 60
    include_business_summary: bool = True
    use_defeatbeta: bool = True  # Use DefeatBeta for historical data
    use_alpha_model: bool = True  # Use quantitative alpha signals (backtested)
    use_qv_model: bool = True  # Use quality-value composite model


# ==============================================================================
# DEFEATBETA DATA INTEGRATION
# ==============================================================================

def _fetch_defeatbeta_data(ticker: str) -> dict:
    """
    Fetch fundamental data from DefeatBeta (free HuggingFace dataset).
    
    Returns dict with trends, estimates, and EPS history.
    """
    try:
        from ..data.defeatbeta import (
            get_financial_trends,
            get_earnings_estimates,
            get_eps_history,
            calculate_estimate_revisions,
        )
        
        data = {}
        
        # Get financial trends
        trends = get_financial_trends(ticker)
        if trends:
            data['trends'] = trends
        
        # Get earnings estimates and revisions
        estimates = get_earnings_estimates(ticker)
        if estimates:
            data['estimates'] = estimates
            data['revisions'] = calculate_estimate_revisions(estimates)
        
        # Get EPS history for beat rate
        eps_history = get_eps_history(ticker)
        if eps_history:
            surprises = []
            for e in eps_history[:8]:  # Last 8 quarters
                surprise = e.get('surprise_pct')
                if surprise is not None:
                    try:
                        surprises.append(float(surprise))
                    except:
                        pass
            if surprises:
                data['avg_eps_surprise'] = sum(surprises) / len(surprises)
                data['beat_rate'] = sum(1 for s in surprises if s > 0) / len(surprises)
        
        return data
        
    except ImportError:
        logger.debug("DefeatBeta module not available")
        return {}
    except Exception as e:
        logger.debug(f"DefeatBeta fetch failed for {ticker}: {e}")
        return {}


# ==============================================================================
# FUNDAMENTAL AGENT
# ==============================================================================

class FundamentalAgent(BaseAgent):
    """
    Agent for analyzing company fundamentals using financial data and LLM.
    
    Fetches key financial metrics from yfinance and uses an LLM to
    provide qualitative analysis and investment insights.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        config: Optional[FundamentalAgentConfig] = None,
    ) -> None:
        """
        Initialize the Fundamental Agent.
        
        Args:
            model: LLM model to use for analysis.
            config: Optional configuration object.
        """
        if not HAS_LITELLM:
            raise ImportError("litellm is required. Install with: pip install litellm")
        
        self.config = config or FundamentalAgentConfig(model=model)
        self.model = self.config.model
        
        # Initialize alpha model if available and enabled
        self.alpha_model = None
        if self.config.use_alpha_model and HAS_ALPHA_MODEL:
            try:
                self.alpha_model = FundamentalsAlphaModel()
                logger.info("Initialized FundamentalsAlphaModel for quantitative signals")
            except Exception as e:
                logger.warning(f"Failed to initialize alpha model: {e}")
        
        # Initialize quality-value model if available and enabled
        self.qv_model = None
        if self.config.use_qv_model and HAS_QV_MODEL:
            try:
                self.qv_model = QualityValueModel()
                logger.info("Initialized QualityValueModel for quality-value signals")
            except Exception as e:
                logger.warning(f"Failed to initialize quality-value model: {e}")
        
        logger.info(f"Initialized FundamentalAgent with model: {self.model}")
    
    @property
    def name(self) -> str:
        return "FundamentalAgent"
    
    def run(
        self,
        tickers: list[str],
        start_date: str = None,
        end_date: str = None,
    ) -> dict[str, FundamentalResult]:
        """Execute fundamental analysis for tickers."""
        return self.analyze_tickers(tickers)
    
    # ==========================================================================
    # DATA FETCHING
    # ==========================================================================
    
    def fetch_metrics(self, ticker: str) -> FinancialMetrics:
        """
        Fetch financial metrics for a ticker from yfinance and DefeatBeta.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            FinancialMetrics object with available data.
        """
        if not HAS_YFINANCE:
            logger.warning("yfinance not available")
            return FinancialMetrics(ticker=ticker)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Helper to safely get values
            def get(key, default=None):
                val = info.get(key, default)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return default
                return val
            
            metrics = FinancialMetrics(
                ticker=ticker,
                market_cap=get("marketCap"),
                pe_ratio=get("trailingPE"),
                forward_pe=get("forwardPE"),
                pb_ratio=get("priceToBook"),
                ps_ratio=get("priceToSalesTrailing12Months"),
                ev_ebitda=get("enterpriseToEbitda"),
                profit_margin=get("profitMargins"),
                operating_margin=get("operatingMargins"),
                roe=get("returnOnEquity"),
                roa=get("returnOnAssets"),
                debt_to_equity=get("debtToEquity"),
                current_ratio=get("currentRatio"),
                revenue_growth=get("revenueGrowth"),
                earnings_growth=get("earningsGrowth"),
                dividend_yield=get("dividendYield"),
                beta=get("beta"),
                sector=get("sector"),
                industry=get("industry"),
                business_summary=get("longBusinessSummary") if self.config.include_business_summary else None,
            )
            
            # Enrich with DefeatBeta historical data
            if self.config.use_defeatbeta:
                db_data = _fetch_defeatbeta_data(ticker)
                
                # Add trends from actual financials
                if 'trends' in db_data:
                    trends = db_data['trends']
                    metrics.revenue_yoy_growth = float(trends.get('revenue_yoy_growth')) if trends.get('revenue_yoy_growth') else None
                    metrics.net_income_yoy_growth = float(trends.get('net_income_yoy_growth')) if trends.get('net_income_yoy_growth') else None
                    metrics.revenue_trend = trends.get('revenue_trend')
                    metrics.net_income_trend = trends.get('net_income_trend')
                    metrics.gross_margin = float(trends.get('gross_margin')) if trends.get('gross_margin') else None
                    metrics.net_margin = float(trends.get('net_margin')) if trends.get('net_margin') else None
                
                # Add estimates and revisions
                if 'estimates' in db_data:
                    est = db_data['estimates']
                    metrics.eps_estimate = float(est.get('estimate_avg_eps')) if est.get('estimate_avg_eps') else None
                    metrics.eps_estimate_growth = float(est.get('estimate_eps_growth')) if est.get('estimate_eps_growth') else None
                    metrics.num_analysts = int(est.get('num_analysts')) if est.get('num_analysts') else None
                
                if 'revisions' in db_data:
                    rev = db_data['revisions']
                    metrics.revision_30d = float(rev.get('revision_30d')) if rev.get('revision_30d') else None
                    metrics.revision_90d = float(rev.get('revision_90d')) if rev.get('revision_90d') else None
                    metrics.revision_signal = rev.get('revision_signal')
                
                # Add EPS beat rate
                if 'avg_eps_surprise' in db_data:
                    metrics.avg_eps_surprise = db_data['avg_eps_surprise']
                if 'beat_rate' in db_data:
                    metrics.beat_rate = db_data['beat_rate']
            
            logger.debug(f"Fetched metrics for {ticker}")
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to fetch metrics for {ticker}: {e}")
            return FinancialMetrics(ticker=ticker)
    
    # ==========================================================================
    # LLM ANALYSIS
    # ==========================================================================
    
    def _format_metric(self, value, fmt: str = ".2f", suffix: str = "") -> str:
        """Format a metric for display."""
        if value is None:
            return "N/A"
        if fmt == ".2f":
            return f"{value:.2f}{suffix}"
        elif fmt == ".1%":
            return f"{value*100:.1f}%"
        elif fmt == ".0f":
            return f"{value:,.0f}{suffix}"
        else:
            return str(value)
    
    def _build_prompt(self, metrics: FinancialMetrics) -> str:
        """Build the analysis prompt for the LLM."""
        
        # Format metrics for the prompt
        metrics_text = f"""
COMPANY: {metrics.ticker}
Sector: {metrics.sector or 'Unknown'}
Industry: {metrics.industry or 'Unknown'}

VALUATION METRICS:
- Market Cap: ${self._format_metric(metrics.market_cap, '.0f')}
- P/E Ratio (TTM): {self._format_metric(metrics.pe_ratio)}
- Forward P/E: {self._format_metric(metrics.forward_pe)}
- P/B Ratio: {self._format_metric(metrics.pb_ratio)}
- P/S Ratio: {self._format_metric(metrics.ps_ratio)}
- EV/EBITDA: {self._format_metric(metrics.ev_ebitda)}

PROFITABILITY:
- Profit Margin: {self._format_metric(metrics.profit_margin, '.1%')}
- Operating Margin: {self._format_metric(metrics.operating_margin, '.1%')}
- Gross Margin: {self._format_metric(metrics.gross_margin, '.1%')}
- Net Margin: {self._format_metric(metrics.net_margin, '.1%')}
- ROE: {self._format_metric(metrics.roe, '.1%')}
- ROA: {self._format_metric(metrics.roa, '.1%')}

GROWTH (from actual financials):
- Revenue Growth (YoY): {self._format_metric(metrics.revenue_yoy_growth, '.1%')}
- Revenue Trend: {metrics.revenue_trend or 'N/A'}
- Net Income Growth (YoY): {self._format_metric(metrics.net_income_yoy_growth, '.1%')}
- Net Income Trend: {metrics.net_income_trend or 'N/A'}

FINANCIAL HEALTH:
- Debt/Equity: {self._format_metric(metrics.debt_to_equity)}
- Current Ratio: {self._format_metric(metrics.current_ratio)}
- Beta: {self._format_metric(metrics.beta)}
- Dividend Yield: {self._format_metric(metrics.dividend_yield, '.1%')}
"""

        # Add analyst estimates section if available
        if metrics.eps_estimate or metrics.revision_signal:
            metrics_text += f"""
ANALYST ESTIMATES (alpha signal):
- EPS Estimate: {self._format_metric(metrics.eps_estimate)}
- EPS Growth Expected: {self._format_metric(metrics.eps_estimate_growth, '.1%')}
- Number of Analysts: {metrics.num_analysts or 'N/A'}
- 30-Day Revision: {self._format_metric(metrics.revision_30d, '.1%')}
- 90-Day Revision: {self._format_metric(metrics.revision_90d, '.1%')}
- Revision Signal: {metrics.revision_signal or 'N/A'} (positive revisions are bullish)
"""

        # Add EPS beat history if available
        if metrics.beat_rate is not None:
            metrics_text += f"""
EARNINGS TRACK RECORD:
- Average EPS Surprise: {self._format_metric(metrics.avg_eps_surprise, '.1%')}
- Beat Rate: {self._format_metric(metrics.beat_rate, '.1%')} (% of quarters beating estimates)
"""

        if metrics.business_summary:
            # Truncate if too long
            summary = metrics.business_summary[:500] + "..." if len(metrics.business_summary) > 500 else metrics.business_summary
            metrics_text += f"\nBUSINESS SUMMARY:\n{summary}\n"
        
        prompt = f"""Analyze the fundamental quality of this company for investment purposes.

{metrics_text}

Provide your analysis in the following EXACT format:

QUALITY_SCORE: [number from 0.0 to 1.0, where 0 is poor and 1 is excellent quality]
QUALITY_LABEL: [one of: poor, fair, good, excellent]
VALUE_ASSESSMENT: [one of: undervalued, fairly_valued, overvalued]
GROWTH_OUTLOOK: [one of: declining, stable, growing, high_growth]
RISK_LEVEL: [one of: low, moderate, high]
SUMMARY: [one sentence overall assessment]
STRENGTHS: [comma-separated list of 2-3 key strengths]
CONCERNS: [comma-separated list of 2-3 key concerns or risks]

Base your assessment on:
1. Quality: Profitability, margins, returns on capital
2. Value: Valuation multiples vs growth, sector norms
3. Growth: Revenue and earnings trajectory, analyst revision trends
4. Risk: Leverage, beta, business stability
5. Track Record: EPS beat rate and surprise history

NOTE: Positive analyst estimate revisions and high beat rates are bullish signals.
Be objective and specific to the data provided."""

        return prompt
    
    def _parse_response(self, ticker: str, response: str, metrics: FinancialMetrics) -> FundamentalResult:
        """Parse the LLM response into a FundamentalResult."""
        
        # Defaults
        quality_score = 0.5
        quality_label = "fair"
        value_assessment = "fairly_valued"
        growth_outlook = "stable"
        risk_level = "moderate"
        summary = "Analysis not available."
        strengths = []
        concerns = []
        
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("QUALITY_SCORE:"):
                    try:
                        quality_score = float(line.split(":", 1)[1].strip())
                        quality_score = max(0.0, min(1.0, quality_score))
                    except:
                        pass
                elif line.startswith("QUALITY_LABEL:"):
                    label = line.split(":", 1)[1].strip().lower()
                    if label in ["poor", "fair", "good", "excellent"]:
                        quality_label = label
                elif line.startswith("VALUE_ASSESSMENT:"):
                    val = line.split(":", 1)[1].strip().lower()
                    if val in ["undervalued", "fairly_valued", "overvalued"]:
                        value_assessment = val
                elif line.startswith("GROWTH_OUTLOOK:"):
                    val = line.split(":", 1)[1].strip().lower()
                    if val in ["declining", "stable", "growing", "high_growth"]:
                        growth_outlook = val
                elif line.startswith("RISK_LEVEL:"):
                    val = line.split(":", 1)[1].strip().lower()
                    if val in ["low", "moderate", "high"]:
                        risk_level = val
                elif line.startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.startswith("STRENGTHS:"):
                    strengths_str = line.split(":", 1)[1].strip()
                    strengths = [s.strip() for s in strengths_str.split(",") if s.strip()]
                elif line.startswith("CONCERNS:"):
                    concerns_str = line.split(":", 1)[1].strip()
                    concerns = [c.strip() for c in concerns_str.split(",") if c.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
        
        return FundamentalResult(
            ticker=ticker,
            quality_score=quality_score,
            quality_label=quality_label,
            value_assessment=value_assessment,
            growth_outlook=growth_outlook,
            risk_level=risk_level,
            summary=summary,
            strengths=strengths,
            concerns=concerns,
            metrics=metrics,
            analysis_date=datetime.now(),
            raw_response=response,
        )
    
    def analyze_ticker(self, ticker: str) -> FundamentalResult:
        """
        Analyze fundamentals for a single ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            FundamentalResult with analysis.
        """
        # Fetch metrics
        metrics = self.fetch_metrics(ticker)
        
        # Check if we have enough data
        if metrics.market_cap is None and metrics.pe_ratio is None:
            return FundamentalResult(
                ticker=ticker,
                quality_score=0.0,
                quality_label="poor",
                value_assessment="fairly_valued",
                growth_outlook="stable",
                risk_level="high",
                summary="Insufficient financial data available.",
                strengths=[],
                concerns=["No financial data available"],
                metrics=metrics,
                analysis_date=datetime.now(),
            )
        
        # Build prompt
        prompt = self._build_prompt(metrics)
        
        # Call LLM
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional equity research analyst. Provide objective, data-driven fundamental analysis."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
            
            response_text = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM call failed for {ticker}: {e}")
            return FundamentalResult(
                ticker=ticker,
                quality_score=0.0,
                quality_label="fair",
                value_assessment="fairly_valued",
                growth_outlook="stable",
                risk_level="moderate",
                summary=f"Analysis failed: {str(e)[:50]}",
                strengths=[],
                concerns=[],
                metrics=metrics,
                analysis_date=datetime.now(),
            )
        
        # Parse response
        result = self._parse_response(ticker, response_text, metrics)
        
        # Enrich with quantitative alpha signals
        result = self._add_alpha_signals(result)
        
        # Enrich with quality-value composite signals
        result = self._add_qv_signals(result)
        
        return result
    
    def _add_alpha_signals(self, result: FundamentalResult) -> FundamentalResult:
        """
        Add quantitative alpha signals from the backtest-validated model.
        
        These signals are based on:
        - Analyst estimate revisions (IC=+0.082*** OOS for short-term)
        - Post-Earnings Announcement Drift (IC=+0.161*** for long-term)
        
        The model understands that different signals work at different horizons.
        """
        if self.alpha_model is None:
            return result
        
        try:
            alpha_signal = self.alpha_model.get_signal(result.ticker)
            
            # Add alpha signal fields to result
            result.alpha_signal_type = alpha_signal.signal_type
            result.alpha_short_term_score = alpha_signal.short_term_score
            result.alpha_long_term_score = alpha_signal.long_term_score
            result.alpha_combined_score = alpha_signal.combined_score
            result.alpha_recommended_horizon = alpha_signal.recommended_horizon
            result.alpha_expected_quintile = alpha_signal.expected_quintile
            result.alpha_data_freshness = alpha_signal.data_freshness
            result.alpha_summary = alpha_signal.summary
            result.alpha_strengths = alpha_signal.strengths
            result.alpha_risks = alpha_signal.risks
            
            # Merge alpha strengths/risks with LLM analysis
            if alpha_signal.strengths:
                for s in alpha_signal.strengths:
                    if s not in result.strengths:
                        result.strengths.append(f"📊 {s}")
            
            if alpha_signal.risks:
                for r in alpha_signal.risks:
                    if r not in result.concerns:
                        result.concerns.append(f"📊 {r}")
            
            logger.debug(f"Added alpha signals for {result.ticker}: {alpha_signal.signal_type}")
            
        except Exception as e:
            logger.warning(f"Failed to get alpha signal for {result.ticker}: {e}")
        
        return result
    
    def _add_qv_signals(self, result: FundamentalResult) -> FundamentalResult:
        """
        Add quality-value composite signals from the backtested model.
        
        These signals are based on:
        - Novy-Marx (2013): Gross profitability predicts returns
        - Fama-French (2015): Profitability + Investment factors
        - Asness QMJ (2019): Quality minus Junk factor
        
        Expected alpha: +4-6% annually for high quality + cheap stocks.
        """
        if self.qv_model is None:
            return result
        
        try:
            qv_signal = self.qv_model.get_signal(result.ticker)
            
            # Add quality-value signal fields to result
            result.qv_quality_score = qv_signal.quality_score
            result.qv_quality_grade = qv_signal.quality_grade
            result.qv_value_score = qv_signal.value_score
            result.qv_value_grade = qv_signal.value_grade
            result.qv_composite_score = qv_signal.composite_score
            result.qv_direction = qv_signal.direction
            result.qv_strength = qv_signal.strength
            result.qv_is_actionable = qv_signal.is_actionable
            result.qv_expected_alpha = qv_signal.expected_annual_alpha
            result.qv_strengths = qv_signal.strengths
            result.qv_concerns = qv_signal.concerns
            result.qv_summary = qv_signal.summary
            
            # Merge QV strengths/concerns with LLM analysis
            if qv_signal.strengths:
                for s in qv_signal.strengths[:3]:  # Top 3 QV strengths
                    if s not in result.strengths:
                        result.strengths.append(f"📈 {s}")
            
            if qv_signal.concerns:
                for c in qv_signal.concerns[:3]:  # Top 3 QV concerns
                    if c not in result.concerns:
                        result.concerns.append(f"📉 {c}")
            
            logger.debug(f"Added QV signals for {result.ticker}: {qv_signal.summary}")
            
        except Exception as e:
            logger.warning(f"Failed to get QV signal for {result.ticker}: {e}")
        
        return result
    
    def analyze_tickers(self, tickers: list[str]) -> dict[str, FundamentalResult]:
        """
        Analyze fundamentals for multiple tickers.
        
        Args:
            tickers: List of ticker symbols.
            
        Returns:
            Dictionary mapping ticker to FundamentalResult.
        """
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Analyzing {ticker} ({i}/{len(tickers)})...")
            try:
                result = self.analyze_ticker(ticker)
                results[ticker] = result
            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")
                results[ticker] = FundamentalResult(
                    ticker=ticker,
                    quality_score=0.0,
                    quality_label="fair",
                    value_assessment="fairly_valued",
                    growth_outlook="stable",
                    risk_level="moderate",
                    summary=f"Analysis failed: {str(e)[:50]}",
                    strengths=[],
                    concerns=[],
                    metrics=FinancialMetrics(ticker=ticker),
                    analysis_date=datetime.now(),
                )
        
        return results
    
    # ==========================================================================
    # OUTPUT FORMATTING
    # ==========================================================================
    
    def format_results(self, results: dict[str, FundamentalResult]) -> str:
        """Format results as a readable string."""
        
        lines = []
        lines.append("=" * 70)
        lines.append("  FUNDAMENTAL ANALYSIS RESULTS")
        lines.append("=" * 70)
        lines.append("")
        
        # Sort by quality score
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.quality_score,
            reverse=True,
        )
        
        lines.append(f"{'Ticker':<8} {'Quality':>7} {'Value':<14} {'Growth':<12} {'Risk':<8}")
        lines.append("-" * 70)
        
        for r in sorted_results:
            quality_str = f"{r.quality_score:.2f}"
            lines.append(f"{r.ticker:<8} {quality_str:>7} {r.value_assessment:<14} {r.growth_outlook:<12} {r.risk_level:<8}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # Detailed view
        lines.append("DETAILED ANALYSIS:")
        lines.append("")
        
        for r in sorted_results:
            # Quality indicator
            if r.quality_score >= 0.7:
                indicator = "🟢"
            elif r.quality_score >= 0.4:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            lines.append(f"{indicator} {r.ticker} - {r.quality_label.upper()} (score: {r.quality_score:.2f})")
            lines.append(f"   {r.summary}")
            
            if r.strengths:
                lines.append(f"   ✅ Strengths: {', '.join(r.strengths)}")
            if r.concerns:
                lines.append(f"   ⚠️  Concerns: {', '.join(r.concerns)}")
            
            # Key metrics
            m = r.metrics
            metrics_str = []
            if m.pe_ratio:
                metrics_str.append(f"P/E: {m.pe_ratio:.1f}")
            if m.roe:
                metrics_str.append(f"ROE: {m.roe*100:.1f}%")
            if m.profit_margin:
                metrics_str.append(f"Margin: {m.profit_margin*100:.1f}%")
            if m.revenue_growth:
                metrics_str.append(f"Rev Growth: {m.revenue_growth*100:.1f}%")
            
            if metrics_str:
                lines.append(f"   📊 {' | '.join(metrics_str)}")
            
            # Alpha signal section (quantitative model)
            if r.alpha_signal_type:
                lines.append("")
                lines.append(f"   📈 ALPHA SIGNAL (Backtest-Validated Model):")
                lines.append(f"      {r.alpha_summary}")
                
                if r.alpha_short_term_score is not None:
                    lines.append(f"      Short-Term Score (5-20d): {r.alpha_short_term_score:+.2f}")
                if r.alpha_long_term_score is not None:
                    lines.append(f"      Long-Term Score (40-60d): {r.alpha_long_term_score:+.2f}")
                
                if r.alpha_recommended_horizon != "none":
                    horizon_text = {
                        "short": "📅 RECOMMENDED: Short-term trade (5-20 days)",
                        "long": "📅 RECOMMENDED: Long-term hold (40-60 days)",
                        "either": "📅 RECOMMENDED: Both horizons viable",
                    }.get(r.alpha_recommended_horizon, "")
                    if horizon_text:
                        lines.append(f"      {horizon_text}")
                
                if r.alpha_expected_quintile:
                    quintile_emoji = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🟢"}.get(r.alpha_expected_quintile, "")
                    lines.append(f"      {quintile_emoji} Expected Quintile: Q{r.alpha_expected_quintile}/5")
                
                if r.alpha_data_freshness == "stale":
                    lines.append(f"      ⚠️  Data Freshness: {r.alpha_data_freshness} (signal may be less reliable)")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dataframe(self, results: dict[str, FundamentalResult]) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        
        data = []
        for ticker, r in results.items():
            m = r.metrics
            data.append({
                "ticker": ticker,
                "quality_score": r.quality_score,
                "quality_label": r.quality_label,
                "value_assessment": r.value_assessment,
                "growth_outlook": r.growth_outlook,
                "risk_level": r.risk_level,
                "summary": r.summary,
                "pe_ratio": m.pe_ratio,
                "roe": m.roe,
                "profit_margin": m.profit_margin,
                "revenue_growth": m.revenue_growth,
                "debt_to_equity": m.debt_to_equity,
                "sector": m.sector,
                # Alpha signal fields
                "alpha_signal_type": r.alpha_signal_type,
                "alpha_short_term_score": r.alpha_short_term_score,
                "alpha_long_term_score": r.alpha_long_term_score,
                "alpha_combined_score": r.alpha_combined_score,
                "alpha_recommended_horizon": r.alpha_recommended_horizon,
                "alpha_expected_quintile": r.alpha_expected_quintile,
                "alpha_data_freshness": r.alpha_data_freshness,
            })
        
        return pd.DataFrame(data)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_fundamentals(
    tickers: list[str],
    model: str = "gpt-4o-mini",
) -> dict[str, FundamentalResult]:
    """
    Convenience function to analyze fundamentals for tickers.
    """
    agent = FundamentalAgent(model=model)
    return agent.analyze_tickers(tickers)


# ==============================================================================
# CLI FOR TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze company fundamentals")
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols to analyze",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    
    args = parser.parse_args()
    
    # Run analysis
    agent = FundamentalAgent(model=args.model)
    results = agent.analyze_tickers(args.tickers)
    
    # Print formatted results
    print(agent.format_results(results))
