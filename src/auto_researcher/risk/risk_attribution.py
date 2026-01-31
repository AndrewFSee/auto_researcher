"""
Risk Attribution Module

Implements institutional risk attribution:
- Factor exposure decomposition
- Brinson-style attribution
- Risk budgeting
- Ex-ante vs ex-post risk comparison
- Marginal contribution to risk
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Factor exposure for a position or portfolio."""
    factor_name: str
    exposure: float  # Beta or loading
    t_stat: Optional[float] = None
    contribution_to_var: Optional[float] = None
    contribution_to_return: Optional[float] = None


@dataclass
class RiskDecomposition:
    """Decomposition of portfolio risk."""
    total_risk: float  # Portfolio volatility
    systematic_risk: float  # From factors
    idiosyncratic_risk: float  # Residual
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    position_contributions: Dict[str, float] = field(default_factory=dict)
    
    # VaR decomposition
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    component_var: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskBudget:
    """Risk budget allocation."""
    total_risk_budget: float  # Target portfolio vol
    used_risk: float  # Current portfolio vol
    available_risk: float  # Remaining budget
    position_budgets: Dict[str, float] = field(default_factory=dict)  # Allocated by position
    factor_budgets: Dict[str, float] = field(default_factory=dict)  # Allocated by factor


class RiskAttributor:
    """
    Institutional risk attribution and decomposition.
    
    Features:
    - Factor-based risk decomposition
    - Brinson performance attribution
    - Marginal contribution to risk (MCTR)
    - Risk budgeting
    - Ex-ante/ex-post comparison
    
    Usage:
        attributor = RiskAttributor()
        
        # Load factor returns
        attributor.set_factors(factor_returns_df)
        
        # Decompose portfolio risk
        decomp = attributor.decompose_risk(
            positions={"AAPL": 0.10, "MSFT": 0.10},
            returns=stock_returns_df
        )
        
        # Performance attribution
        attr = attributor.brinson_attribution(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns
        )
    """
    
    STANDARD_FACTORS = [
        "market",  # Market beta
        "size",  # SMB (small minus big)
        "value",  # HML (high minus low)
        "momentum",  # WML (winners minus losers)
        "quality",  # Profitability factor
        "volatility",  # Low vol factor
    ]
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        factor_names: Optional[List[str]] = None,
    ):
        self.risk_free_rate = risk_free_rate
        self.factor_names = factor_names or self.STANDARD_FACTORS
        self.factor_returns: Optional[pd.DataFrame] = None
        self.factor_covariance: Optional[pd.DataFrame] = None
        
        logger.info(f"RiskAttributor initialized with factors: {self.factor_names}")
    
    def set_factors(self, factor_returns: pd.DataFrame):
        """
        Set factor return series.
        
        Args:
            factor_returns: DataFrame with factor returns as columns
        """
        self.factor_returns = factor_returns
        self.factor_covariance = factor_returns.cov() * 252  # Annualized
        
        logger.info(f"Loaded {len(factor_returns.columns)} factors, "
                    f"{len(factor_returns)} observations")
    
    def calculate_factor_exposures(
        self,
        returns: pd.Series,
        min_observations: int = 60,
    ) -> List[FactorExposure]:
        """
        Estimate factor exposures using regression.
        
        Args:
            returns: Asset return series
            min_observations: Minimum obs for regression
            
        Returns:
            List of FactorExposure objects
        """
        if self.factor_returns is None:
            logger.warning("No factor returns loaded")
            return []
        
        # Align dates
        common_dates = returns.index.intersection(self.factor_returns.index)
        
        if len(common_dates) < min_observations:
            logger.warning(f"Insufficient observations: {len(common_dates)} < {min_observations}")
            return []
        
        y = returns.loc[common_dates].values
        X = self.factor_returns.loc[common_dates].values
        
        # Add constant for intercept (alpha)
        X_with_const = np.column_stack([np.ones(len(y)), X])
        
        try:
            # OLS regression
            betas, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            
            # Calculate t-stats (simplified)
            y_pred = X_with_const @ betas
            residual_std = np.std(y - y_pred)
            
            exposures = []
            
            # Skip intercept (index 0)
            for i, factor in enumerate(self.factor_returns.columns):
                beta = betas[i + 1]
                
                # Rough t-stat calculation
                x_col = X[:, i]
                se = residual_std / (np.std(x_col) * np.sqrt(len(y)))
                t_stat = beta / se if se > 0 else 0
                
                exposures.append(FactorExposure(
                    factor_name=factor,
                    exposure=beta,
                    t_stat=t_stat,
                ))
            
            return exposures
            
        except Exception as e:
            logger.error(f"Factor regression failed: {e}")
            return []
    
    def decompose_risk(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        factor_exposures: Optional[Dict[str, List[FactorExposure]]] = None,
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk into factor and idiosyncratic components.
        
        Args:
            positions: Dict of ticker -> weight
            returns: DataFrame with stock returns
            factor_exposures: Optional pre-calculated exposures
            
        Returns:
            RiskDecomposition object
        """
        # Filter to available stocks
        available = [t for t in positions.keys() if t in returns.columns]
        if not available:
            logger.warning("No matching positions in returns")
            return RiskDecomposition(
                total_risk=0, systematic_risk=0, idiosyncratic_risk=0
            )
        
        weights = np.array([positions[t] for t in available])
        weights = weights / weights.sum()  # Normalize
        
        ret_subset = returns[available].dropna()
        
        # Portfolio returns
        port_returns = (ret_subset * weights).sum(axis=1)
        total_risk = port_returns.std() * np.sqrt(252)
        
        # Calculate factor exposures if not provided
        if factor_exposures is None and self.factor_returns is not None:
            factor_exposures = {}
            for ticker in available:
                if ticker in returns.columns:
                    factor_exposures[ticker] = self.calculate_factor_exposures(
                        returns[ticker]
                    )
        
        # Calculate systematic and idiosyncratic risk
        systematic_risk = 0
        factor_contributions = {}
        position_contributions = {}
        
        if factor_exposures and self.factor_covariance is not None:
            # Aggregate factor exposures at portfolio level
            port_factor_exposures = {}
            for factor in self.factor_names:
                port_exposure = 0
                for ticker, weight in zip(available, weights):
                    ticker_exposures = factor_exposures.get(ticker, [])
                    factor_exp = next(
                        (e.exposure for e in ticker_exposures if e.factor_name == factor), 
                        0
                    )
                    port_exposure += weight * factor_exp
                port_factor_exposures[factor] = port_exposure
            
            # Calculate systematic variance
            for f1 in self.factor_names:
                for f2 in self.factor_names:
                    if f1 in self.factor_covariance.index and f2 in self.factor_covariance.columns:
                        cov = self.factor_covariance.loc[f1, f2]
                        systematic_risk += (
                            port_factor_exposures.get(f1, 0) * 
                            port_factor_exposures.get(f2, 0) * 
                            cov
                        )
            
            systematic_risk = np.sqrt(max(0, systematic_risk))
            
            # Factor contributions (marginal)
            for factor in self.factor_names:
                if factor in self.factor_returns.columns:
                    factor_ret = self.factor_returns[factor]
                    common_dates = port_returns.index.intersection(factor_ret.index)
                    if len(common_dates) > 20:
                        cov = np.cov(
                            port_returns.loc[common_dates], 
                            factor_ret.loc[common_dates]
                        )[0, 1]
                        factor_contributions[factor] = cov * 252 / total_risk if total_risk > 0 else 0
        
        # Idiosyncratic risk
        idiosyncratic_risk = np.sqrt(max(0, total_risk**2 - systematic_risk**2))
        
        # Position contributions to risk
        if len(ret_subset) > 20:
            cov_matrix = ret_subset.cov() * 252
            port_var = weights @ cov_matrix @ weights
            
            for i, ticker in enumerate(available):
                # Marginal contribution
                mctr = (cov_matrix.iloc[i] @ weights) / np.sqrt(port_var) if port_var > 0 else 0
                # Component contribution
                position_contributions[ticker] = weights[i] * mctr
        
        # VaR calculation
        var_95 = np.percentile(port_returns, 5)
        var_99 = np.percentile(port_returns, 1)
        
        return RiskDecomposition(
            total_risk=total_risk,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            factor_contributions=factor_contributions,
            position_contributions=position_contributions,
            var_95=var_95,
            var_99=var_99,
        )
    
    def marginal_contribution_to_risk(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate Marginal Contribution to Risk (MCTR) for each position.
        
        MCTR_i = (Cov(r_i, r_p) / σ_p)
        
        Returns:
            Dict of ticker -> MCTR
        """
        available = [t for t in positions.keys() if t in returns.columns]
        if not available:
            return {}
        
        weights = np.array([positions[t] for t in available])
        weights = weights / weights.sum()
        
        ret_subset = returns[available].dropna()
        if len(ret_subset) < 20:
            return {}
        
        cov_matrix = ret_subset.cov() * 252
        port_var = weights @ cov_matrix @ weights
        port_vol = np.sqrt(port_var)
        
        mctr = {}
        for i, ticker in enumerate(available):
            cov_with_port = cov_matrix.iloc[i] @ weights
            mctr[ticker] = cov_with_port / port_vol if port_vol > 0 else 0
        
        return mctr
    
    def component_var(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate Component VaR for each position.
        
        Component VaR shows how each position contributes to total VaR.
        
        Returns:
            Dict of ticker -> Component VaR
        """
        available = [t for t in positions.keys() if t in returns.columns]
        if not available:
            return {}
        
        weights = np.array([positions[t] for t in available])
        weights = weights / weights.sum()
        
        ret_subset = returns[available].dropna()
        if len(ret_subset) < 20:
            return {}
        
        # Portfolio returns
        port_returns = (ret_subset * weights).sum(axis=1)
        
        # Portfolio VaR
        var_pct = np.percentile(port_returns, (1 - confidence) * 100)
        
        # MCTR
        mctr = self.marginal_contribution_to_risk(positions, returns)
        
        # Component VaR = weight * MCTR * VaR_scalar
        z_score = abs(var_pct) / port_returns.std() if port_returns.std() > 0 else 1.65
        
        component_var = {}
        for ticker in available:
            w = positions[ticker] / sum(positions.values())
            component_var[ticker] = w * mctr.get(ticker, 0) * z_score
        
        return component_var
    
    def brinson_attribution(
        self,
        portfolio_weights: Dict[str, Dict[str, float]],  # sector -> ticker -> weight
        benchmark_weights: Dict[str, Dict[str, float]],
        portfolio_returns: Dict[str, Dict[str, float]],  # sector -> ticker -> return
        benchmark_returns: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Brinson-Fachler performance attribution.
        
        Decomposes active return into:
        - Allocation effect (sector weights)
        - Selection effect (stock picks)
        - Interaction effect
        
        Returns:
            Dict with sector-level attribution
        """
        sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        
        attribution = {}
        
        for sector in sectors:
            port_sector_weight = sum(portfolio_weights.get(sector, {}).values())
            bench_sector_weight = sum(benchmark_weights.get(sector, {}).values())
            
            # Sector returns (weighted average)
            port_sector_ret = 0
            port_sect_positions = portfolio_weights.get(sector, {})
            for ticker, weight in port_sect_positions.items():
                ret = portfolio_returns.get(sector, {}).get(ticker, 0)
                if port_sector_weight > 0:
                    port_sector_ret += (weight / port_sector_weight) * ret
            
            bench_sector_ret = 0
            bench_sect_positions = benchmark_weights.get(sector, {})
            for ticker, weight in bench_sect_positions.items():
                ret = benchmark_returns.get(sector, {}).get(ticker, 0)
                if bench_sector_weight > 0:
                    bench_sector_ret += (weight / bench_sector_weight) * ret
            
            # Benchmark total return (simplified)
            bench_total_ret = 0
            for sect, positions in benchmark_returns.items():
                sect_weight = sum(benchmark_weights.get(sect, {}).values())
                for ticker, ret in positions.items():
                    bench_total_ret += sect_weight * ret
            bench_total_ret = bench_total_ret / max(1, sum(
                sum(w.values()) for w in benchmark_weights.values()
            ))
            
            # Brinson-Fachler attribution
            allocation = (port_sector_weight - bench_sector_weight) * (bench_sector_ret - bench_total_ret)
            selection = bench_sector_weight * (port_sector_ret - bench_sector_ret)
            interaction = (port_sector_weight - bench_sector_weight) * (port_sector_ret - bench_sector_ret)
            
            attribution[sector] = {
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
                "total": allocation + selection + interaction,
                "portfolio_weight": port_sector_weight,
                "benchmark_weight": bench_sector_weight,
                "portfolio_return": port_sector_ret,
                "benchmark_return": bench_sector_ret,
            }
        
        return attribution
    
    def calculate_risk_budget(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        total_risk_budget: float = 0.15,  # 15% target vol
    ) -> RiskBudget:
        """
        Calculate risk budget utilization.
        
        Args:
            positions: Current portfolio weights
            returns: Historical returns
            total_risk_budget: Target portfolio volatility
            
        Returns:
            RiskBudget object
        """
        decomp = self.decompose_risk(positions, returns)
        
        used_risk = decomp.total_risk
        available_risk = max(0, total_risk_budget - used_risk)
        
        # Allocate budget by position contribution
        position_budgets = {}
        total_contribution = sum(abs(c) for c in decomp.position_contributions.values())
        
        if total_contribution > 0:
            for ticker, contribution in decomp.position_contributions.items():
                budget_share = abs(contribution) / total_contribution
                position_budgets[ticker] = budget_share * total_risk_budget
        
        # Factor budgets
        factor_budgets = {}
        total_factor_contrib = sum(abs(c) for c in decomp.factor_contributions.values())
        
        if total_factor_contrib > 0:
            for factor, contribution in decomp.factor_contributions.items():
                budget_share = abs(contribution) / total_factor_contrib
                factor_budgets[factor] = budget_share * total_risk_budget
        
        return RiskBudget(
            total_risk_budget=total_risk_budget,
            used_risk=used_risk,
            available_risk=available_risk,
            position_budgets=position_budgets,
            factor_budgets=factor_budgets,
        )
    
    def ex_ante_ex_post_comparison(
        self,
        positions: Dict[str, float],
        expected_returns: Dict[str, float],
        expected_volatilities: Dict[str, float],
        realized_returns: pd.DataFrame,
        period_days: int = 252,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare ex-ante (predicted) vs ex-post (realized) risk and return.
        
        Returns:
            Dict with comparison metrics
        """
        available = [t for t in positions.keys() if t in realized_returns.columns]
        if not available:
            return {}
        
        weights = np.array([positions[t] for t in available])
        weights = weights / weights.sum()
        
        # Ex-ante (expected)
        ex_ante_return = sum(
            weights[i] * expected_returns.get(t, 0) 
            for i, t in enumerate(available)
        )
        
        ex_ante_vol = np.sqrt(sum(
            (weights[i] * expected_volatilities.get(t, 0.20)) ** 2
            for i, t in enumerate(available)
        ))  # Simplified - ignores correlations
        
        # Ex-post (realized)
        ret_subset = realized_returns[available].tail(period_days).dropna()
        port_returns = (ret_subset * weights).sum(axis=1)
        
        ex_post_return = port_returns.mean() * 252
        ex_post_vol = port_returns.std() * np.sqrt(252)
        
        return {
            "portfolio": {
                "ex_ante_return": ex_ante_return,
                "ex_post_return": ex_post_return,
                "return_surprise": ex_post_return - ex_ante_return,
                "ex_ante_volatility": ex_ante_vol,
                "ex_post_volatility": ex_post_vol,
                "volatility_surprise": ex_post_vol - ex_ante_vol,
                "ex_ante_sharpe": (ex_ante_return - self.risk_free_rate) / ex_ante_vol if ex_ante_vol > 0 else 0,
                "ex_post_sharpe": (ex_post_return - self.risk_free_rate) / ex_post_vol if ex_post_vol > 0 else 0,
            },
            "positions": {
                t: {
                    "ex_ante_return": expected_returns.get(t, 0),
                    "ex_post_return": ret_subset[t].mean() * 252 if t in ret_subset else 0,
                    "ex_ante_vol": expected_volatilities.get(t, 0.20),
                    "ex_post_vol": ret_subset[t].std() * np.sqrt(252) if t in ret_subset else 0,
                }
                for t in available
            }
        }
    
    def generate_risk_report(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
    ) -> str:
        """Generate comprehensive risk report."""
        decomp = self.decompose_risk(positions, returns)
        mctr = self.marginal_contribution_to_risk(positions, returns)
        component_var = self.component_var(positions, returns)
        
        lines = [
            "=" * 70,
            "RISK ATTRIBUTION REPORT",
            "=" * 70,
            "",
            "PORTFOLIO RISK SUMMARY:",
            f"  Total Risk (Annualized):     {decomp.total_risk:.2%}",
            f"  Systematic Risk:             {decomp.systematic_risk:.2%}",
            f"  Idiosyncratic Risk:          {decomp.idiosyncratic_risk:.2%}",
            f"  VaR (95%, daily):            {decomp.var_95:.2%}" if decomp.var_95 else "",
            f"  VaR (99%, daily):            {decomp.var_99:.2%}" if decomp.var_99 else "",
            "",
        ]
        
        if decomp.factor_contributions:
            lines.append("FACTOR CONTRIBUTIONS:")
            sorted_factors = sorted(
                decomp.factor_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            for factor, contrib in sorted_factors:
                lines.append(f"  {factor:20s}: {contrib:+.4f}")
            lines.append("")
        
        if decomp.position_contributions:
            lines.append("POSITION RISK CONTRIBUTIONS:")
            sorted_positions = sorted(
                decomp.position_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]  # Top 10
            for ticker, contrib in sorted_positions:
                mctr_val = mctr.get(ticker, 0)
                cvar = component_var.get(ticker, 0)
                lines.append(f"  {ticker:8s}: Contrib={contrib:+.4f}  MCTR={mctr_val:.4f}  CVaR={cvar:.4f}")
            lines.append("")
        
        lines.extend([
            "=" * 70,
        ])
        
        return "\n".join(filter(None, lines))
