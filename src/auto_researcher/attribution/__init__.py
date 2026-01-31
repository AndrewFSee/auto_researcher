"""
Performance Attribution Module

Comprehensive performance attribution for institutional reporting:
- Returns-based attribution
- Holdings-based (Brinson) attribution
- Factor-based attribution
- Multi-period linking
- Currency effects
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np
import pandas as pd
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Attribution methodology."""
    BRINSON_FACHLER = "brinson_fachler"
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"
    FACTOR_BASED = "factor_based"
    RETURNS_BASED = "returns_based"


@dataclass
class PeriodAttribution:
    """Attribution results for a single period."""
    period_start: date
    period_end: date
    portfolio_return: float
    benchmark_return: float
    active_return: float
    
    # Brinson components
    allocation_effect: float = 0.0
    selection_effect: float = 0.0
    interaction_effect: float = 0.0
    
    # Factor components
    factor_effects: Dict[str, float] = field(default_factory=dict)
    
    # Sector breakdown
    sector_attribution: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class AttributionSummary:
    """Multi-period attribution summary."""
    start_date: date
    end_date: date
    
    # Cumulative returns
    portfolio_cumulative_return: float
    benchmark_cumulative_return: float
    active_cumulative_return: float
    
    # Annualized
    portfolio_annualized_return: float
    benchmark_annualized_return: float
    active_annualized_return: float
    
    # Attribution breakdown (geometrically linked)
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    
    # Risk-adjusted
    information_ratio: float
    tracking_error: float
    
    # Period details
    periods: List[PeriodAttribution] = field(default_factory=list)


class PerformanceAttributor:
    """
    Institutional performance attribution engine.
    
    Supports:
    - Brinson-Fachler and Brinson-Hood-Beebower methods
    - Multi-period geometric linking
    - Factor-based attribution
    - Sector/country/style decomposition
    
    Usage:
        attributor = PerformanceAttributor()
        
        summary = attributor.calculate_attribution(
            portfolio_holdings=portfolio_df,
            benchmark_holdings=benchmark_df,
            returns=returns_df,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
    """
    
    def __init__(
        self,
        method: AttributionMethod = AttributionMethod.BRINSON_FACHLER,
        frequency: str = "monthly",  # daily, weekly, monthly, quarterly
        geometric_linking: bool = True,
    ):
        self.method = method
        self.frequency = frequency
        self.geometric_linking = geometric_linking
        
        logger.info(f"PerformanceAttributor initialized: method={method.value}, freq={frequency}")
    
    def calculate_attribution(
        self,
        portfolio_weights: pd.DataFrame,  # Date x Ticker weights
        benchmark_weights: pd.DataFrame,
        returns: pd.DataFrame,  # Date x Ticker returns
        sectors: Dict[str, str],  # Ticker -> Sector mapping
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> AttributionSummary:
        """
        Calculate full attribution over a period.
        
        Args:
            portfolio_weights: DataFrame with portfolio weights by date
            benchmark_weights: DataFrame with benchmark weights by date
            returns: DataFrame with asset returns by date
            sectors: Mapping of ticker to sector
            start_date: Start of attribution period
            end_date: End of attribution period
            
        Returns:
            AttributionSummary with full breakdown
        """
        # Align dates
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        common_dates = common_dates.intersection(returns.index)
        
        if start_date:
            common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
        if end_date:
            common_dates = common_dates[common_dates <= pd.to_datetime(end_date)]
        
        if len(common_dates) < 2:
            raise ValueError("Insufficient data for attribution")
        
        # Resample to frequency
        periods = self._get_periods(common_dates)
        
        # Calculate attribution for each period
        period_results = []
        
        for period_start, period_end in periods:
            period_attr = self._single_period_attribution(
                portfolio_weights=portfolio_weights,
                benchmark_weights=benchmark_weights,
                returns=returns,
                sectors=sectors,
                period_start=period_start,
                period_end=period_end,
            )
            period_results.append(period_attr)
        
        # Link periods
        summary = self._link_periods(period_results)
        
        return summary
    
    def _get_periods(self, dates: pd.DatetimeIndex) -> List[Tuple[date, date]]:
        """Split dates into periods based on frequency."""
        periods = []
        
        if self.frequency == "daily":
            for i in range(len(dates) - 1):
                periods.append((dates[i].date(), dates[i + 1].date()))
        
        elif self.frequency == "weekly":
            # Group by week
            weeks = dates.to_series().groupby(pd.Grouper(freq='W'))
            for _, week_dates in weeks:
                if len(week_dates) >= 2:
                    periods.append((week_dates.iloc[0].date(), week_dates.iloc[-1].date()))
        
        elif self.frequency == "monthly":
            # Group by month
            months = dates.to_series().groupby(pd.Grouper(freq='M'))
            for _, month_dates in months:
                if len(month_dates) >= 2:
                    periods.append((month_dates.iloc[0].date(), month_dates.iloc[-1].date()))
        
        elif self.frequency == "quarterly":
            quarters = dates.to_series().groupby(pd.Grouper(freq='Q'))
            for _, q_dates in quarters:
                if len(q_dates) >= 2:
                    periods.append((q_dates.iloc[0].date(), q_dates.iloc[-1].date()))
        
        return periods
    
    def _single_period_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        returns: pd.DataFrame,
        sectors: Dict[str, str],
        period_start: date,
        period_end: date,
    ) -> PeriodAttribution:
        """Calculate attribution for a single period."""
        
        # Get weights at start of period
        start_ts = pd.Timestamp(period_start)
        end_ts = pd.Timestamp(period_end)
        
        port_weights = portfolio_weights.loc[start_ts] if start_ts in portfolio_weights.index else \
                       portfolio_weights.iloc[portfolio_weights.index.get_indexer([start_ts], method='ffill')[0]]
        bench_weights = benchmark_weights.loc[start_ts] if start_ts in benchmark_weights.index else \
                        benchmark_weights.iloc[benchmark_weights.index.get_indexer([start_ts], method='ffill')[0]]
        
        # Get period returns
        period_returns = returns.loc[start_ts:end_ts]
        if len(period_returns) < 1:
            return PeriodAttribution(
                period_start=period_start,
                period_end=period_end,
                portfolio_return=0,
                benchmark_return=0,
                active_return=0,
            )
        
        # Compound returns over period
        cumulative_returns = (1 + period_returns).prod() - 1
        
        # Portfolio and benchmark returns
        port_return = sum(port_weights.get(t, 0) * cumulative_returns.get(t, 0) 
                          for t in set(port_weights.index) | set(cumulative_returns.index))
        bench_return = sum(bench_weights.get(t, 0) * cumulative_returns.get(t, 0)
                           for t in set(bench_weights.index) | set(cumulative_returns.index))
        
        active_return = port_return - bench_return
        
        # Sector-level attribution
        sector_attr = self._sector_attribution(
            port_weights, bench_weights, cumulative_returns, sectors, bench_return
        )
        
        # Aggregate effects
        allocation = sum(s.get("allocation", 0) for s in sector_attr.values())
        selection = sum(s.get("selection", 0) for s in sector_attr.values())
        interaction = sum(s.get("interaction", 0) for s in sector_attr.values())
        
        return PeriodAttribution(
            period_start=period_start,
            period_end=period_end,
            portfolio_return=port_return,
            benchmark_return=bench_return,
            active_return=active_return,
            allocation_effect=allocation,
            selection_effect=selection,
            interaction_effect=interaction,
            sector_attribution=sector_attr,
        )
    
    def _sector_attribution(
        self,
        port_weights: pd.Series,
        bench_weights: pd.Series,
        returns: pd.Series,
        sectors: Dict[str, str],
        bench_total_return: float,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate sector-level Brinson attribution."""
        
        # Aggregate weights and returns by sector
        all_tickers = set(port_weights.index) | set(bench_weights.index) | set(returns.index)
        
        sector_data: Dict[str, Dict[str, float]] = {}
        
        for ticker in all_tickers:
            sector = sectors.get(ticker, "Unknown")
            
            if sector not in sector_data:
                sector_data[sector] = {
                    "port_weight": 0,
                    "bench_weight": 0,
                    "port_return_contrib": 0,
                    "bench_return_contrib": 0,
                }
            
            port_w = port_weights.get(ticker, 0)
            bench_w = bench_weights.get(ticker, 0)
            ret = returns.get(ticker, 0)
            
            sector_data[sector]["port_weight"] += port_w
            sector_data[sector]["bench_weight"] += bench_w
            sector_data[sector]["port_return_contrib"] += port_w * ret
            sector_data[sector]["bench_return_contrib"] += bench_w * ret
        
        # Calculate sector returns and attribution
        sector_attr = {}
        
        for sector, data in sector_data.items():
            port_w = data["port_weight"]
            bench_w = data["bench_weight"]
            
            # Sector returns
            port_sect_ret = data["port_return_contrib"] / port_w if port_w > 0 else 0
            bench_sect_ret = data["bench_return_contrib"] / bench_w if bench_w > 0 else 0
            
            # Brinson-Fachler
            if self.method == AttributionMethod.BRINSON_FACHLER:
                allocation = (port_w - bench_w) * (bench_sect_ret - bench_total_return)
                selection = bench_w * (port_sect_ret - bench_sect_ret)
                interaction = (port_w - bench_w) * (port_sect_ret - bench_sect_ret)
            else:  # BRINSON_HOOD_BEEBOWER
                allocation = (port_w - bench_w) * bench_sect_ret
                selection = bench_w * (port_sect_ret - bench_sect_ret)
                interaction = (port_w - bench_w) * (port_sect_ret - bench_sect_ret)
            
            sector_attr[sector] = {
                "portfolio_weight": port_w,
                "benchmark_weight": bench_w,
                "portfolio_return": port_sect_ret,
                "benchmark_return": bench_sect_ret,
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
                "total": allocation + selection + interaction,
            }
        
        return sector_attr
    
    def _link_periods(self, periods: List[PeriodAttribution]) -> AttributionSummary:
        """Geometrically link multi-period attribution."""
        
        if not periods:
            raise ValueError("No periods to link")
        
        start_date = periods[0].period_start
        end_date = periods[-1].period_end
        
        # Geometric linking for returns
        port_cumulative = np.prod([1 + p.portfolio_return for p in periods]) - 1
        bench_cumulative = np.prod([1 + p.benchmark_return for p in periods]) - 1
        active_cumulative = port_cumulative - bench_cumulative
        
        # Calculate number of years
        days = (end_date - start_date).days
        years = days / 365.25
        
        # Annualized returns
        if years > 0:
            port_annual = (1 + port_cumulative) ** (1 / years) - 1
            bench_annual = (1 + bench_cumulative) ** (1 / years) - 1
        else:
            port_annual = port_cumulative
            bench_annual = bench_cumulative
        
        # Geometric linking for attribution effects (Carino method simplified)
        if self.geometric_linking:
            total_allocation = self._geometric_link_effect(
                [p.allocation_effect for p in periods],
                [p.portfolio_return for p in periods]
            )
            total_selection = self._geometric_link_effect(
                [p.selection_effect for p in periods],
                [p.portfolio_return for p in periods]
            )
            total_interaction = self._geometric_link_effect(
                [p.interaction_effect for p in periods],
                [p.portfolio_return for p in periods]
            )
        else:
            # Simple arithmetic
            total_allocation = sum(p.allocation_effect for p in periods)
            total_selection = sum(p.selection_effect for p in periods)
            total_interaction = sum(p.interaction_effect for p in periods)
        
        # Tracking error and information ratio
        active_returns = [p.active_return for p in periods]
        tracking_error = np.std(active_returns) * np.sqrt(12)  # Annualized (assuming monthly)
        
        if tracking_error > 0:
            information_ratio = (active_cumulative / years) / tracking_error if years > 0 else 0
        else:
            information_ratio = 0
        
        return AttributionSummary(
            start_date=start_date,
            end_date=end_date,
            portfolio_cumulative_return=port_cumulative,
            benchmark_cumulative_return=bench_cumulative,
            active_cumulative_return=active_cumulative,
            portfolio_annualized_return=port_annual,
            benchmark_annualized_return=bench_annual,
            active_annualized_return=port_annual - bench_annual,
            total_allocation_effect=total_allocation,
            total_selection_effect=total_selection,
            total_interaction_effect=total_interaction,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            periods=periods,
        )
    
    def _geometric_link_effect(
        self,
        effects: List[float],
        returns: List[float],
    ) -> float:
        """Geometrically link attribution effects (simplified Carino)."""
        
        if not effects:
            return 0
        
        # Calculate linking coefficients
        cumulative_return = np.prod([1 + r for r in returns]) - 1
        
        if abs(cumulative_return) < 1e-10:
            return sum(effects)
        
        total_effect = 0
        cumulative = 1
        
        for effect, ret in zip(effects, returns):
            if abs(ret) < 1e-10:
                coefficient = 1
            else:
                # Carino linking coefficient
                log_cum = np.log(1 + cumulative_return)
                log_ret = np.log(1 + ret) if ret > -1 else 0
                coefficient = log_cum / log_ret if abs(log_ret) > 1e-10 else 1
            
            total_effect += effect * coefficient * cumulative
            cumulative *= (1 + ret)
        
        return total_effect
    
    def generate_report(self, summary: AttributionSummary) -> str:
        """Generate attribution report."""
        
        lines = [
            "=" * 70,
            "PERFORMANCE ATTRIBUTION REPORT",
            f"Period: {summary.start_date} to {summary.end_date}",
            "=" * 70,
            "",
            "RETURNS:",
            f"  Portfolio Cumulative:   {summary.portfolio_cumulative_return:+.2%}",
            f"  Benchmark Cumulative:   {summary.benchmark_cumulative_return:+.2%}",
            f"  Active Return:          {summary.active_cumulative_return:+.2%}",
            "",
            f"  Portfolio Annualized:   {summary.portfolio_annualized_return:+.2%}",
            f"  Benchmark Annualized:   {summary.benchmark_annualized_return:+.2%}",
            "",
            "ATTRIBUTION EFFECTS:",
            f"  Allocation Effect:      {summary.total_allocation_effect:+.2%}",
            f"  Selection Effect:       {summary.total_selection_effect:+.2%}",
            f"  Interaction Effect:     {summary.total_interaction_effect:+.2%}",
            f"  -----------------------------------",
            f"  Total Active Return:    {summary.active_cumulative_return:+.2%}",
            "",
            "RISK METRICS:",
            f"  Tracking Error:         {summary.tracking_error:.2%}",
            f"  Information Ratio:      {summary.information_ratio:.2f}",
            "",
        ]
        
        # Sector breakdown from last period
        if summary.periods:
            last_period = summary.periods[-1]
            if last_period.sector_attribution:
                lines.append("SECTOR ATTRIBUTION (Last Period):")
                sorted_sectors = sorted(
                    last_period.sector_attribution.items(),
                    key=lambda x: abs(x[1].get("total", 0)),
                    reverse=True
                )
                for sector, attr in sorted_sectors[:10]:
                    lines.append(
                        f"  {sector:25s}: Alloc={attr['allocation']:+.3%} "
                        f"Select={attr['selection']:+.3%} "
                        f"Total={attr['total']:+.3%}"
                    )
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)
