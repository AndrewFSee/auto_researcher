"""
Position Sizing Module

Implements institutional-grade position sizing methods:
- Kelly Criterion (full and fractional)
- Volatility Targeting
- Equal Risk Contribution
- Max Position Limits
- Liquidity-Adjusted Sizing
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    EQUAL_WEIGHT = "equal_weight"
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    VOLATILITY_TARGET = "volatility_target"
    EQUAL_RISK = "equal_risk"
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"


@dataclass
class PositionLimit:
    """Position limit constraints."""
    max_position_pct: float = 0.10  # Max 10% in single position
    min_position_pct: float = 0.01  # Min 1% position
    max_total_exposure: float = 1.0  # Max 100% gross exposure (no leverage)
    max_leverage: float = 1.0  # No leverage by default
    
    # Liquidity constraints
    max_adv_pct: float = 0.05  # Max 5% of avg daily volume
    min_market_cap: float = 1e9  # Min $1B market cap
    
    # Concentration limits
    max_sector_pct: float = 0.30  # Max 30% in single sector
    max_factor_exposure: float = 2.0  # Max 2 std factor exposure


@dataclass
class SizedPosition:
    """Result of position sizing calculation."""
    ticker: str
    raw_weight: float  # Weight before constraints
    adjusted_weight: float  # Weight after constraints
    shares: int  # Number of shares
    notional: float  # Dollar value
    
    # Sizing metadata
    method: PositionSizingMethod = PositionSizingMethod.EQUAL_WEIGHT
    constraints_applied: List[str] = field(default_factory=list)
    
    # Risk metrics
    expected_return: Optional[float] = None
    volatility: Optional[float] = None
    sharpe: Optional[float] = None
    var_contribution: Optional[float] = None


class PositionSizer:
    """
    Institutional-grade position sizing engine.
    
    Supports multiple sizing methods with constraint enforcement.
    
    Usage:
        sizer = PositionSizer(
            method=PositionSizingMethod.VOLATILITY_TARGET,
            target_volatility=0.15,
            limits=PositionLimit(max_position_pct=0.05)
        )
        
        positions = sizer.size_portfolio(
            tickers=["AAPL", "MSFT", "GOOGL"],
            signals=[0.8, 0.6, 0.4],
            prices={"AAPL": 150, "MSFT": 350, "GOOGL": 140},
            portfolio_value=1_000_000
        )
    """
    
    def __init__(
        self,
        method: PositionSizingMethod = PositionSizingMethod.EQUAL_WEIGHT,
        limits: Optional[PositionLimit] = None,
        target_volatility: float = 0.15,
        kelly_fraction: float = 0.25,
        lookback_days: int = 252,
        risk_free_rate: float = 0.05,
    ):
        self.method = method
        self.limits = limits or PositionLimit()
        self.target_volatility = target_volatility
        self.kelly_fraction = kelly_fraction
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        
        logger.info(f"PositionSizer initialized: method={method.value}")
    
    def size_portfolio(
        self,
        tickers: List[str],
        signals: List[float],
        prices: Dict[str, float],
        portfolio_value: float,
        returns: Optional[pd.DataFrame] = None,
        volumes: Optional[Dict[str, float]] = None,
        sectors: Optional[Dict[str, str]] = None,
    ) -> List[SizedPosition]:
        """
        Size positions for a portfolio.
        
        Args:
            tickers: List of ticker symbols
            signals: Model signals/scores for each ticker
            prices: Current prices per ticker
            portfolio_value: Total portfolio value
            returns: Historical returns DataFrame (tickers as columns)
            volumes: Average daily volume per ticker
            sectors: Sector mapping per ticker
            
        Returns:
            List of SizedPosition objects
        """
        if len(tickers) != len(signals):
            raise ValueError("tickers and signals must have same length")
        
        n = len(tickers)
        if n == 0:
            return []
        
        # Calculate raw weights based on method
        if self.method == PositionSizingMethod.EQUAL_WEIGHT:
            raw_weights = self._equal_weight(n)
        elif self.method == PositionSizingMethod.KELLY:
            raw_weights = self._kelly_sizing(signals, returns)
        elif self.method == PositionSizingMethod.FRACTIONAL_KELLY:
            raw_weights = self._fractional_kelly(signals, returns)
        elif self.method == PositionSizingMethod.VOLATILITY_TARGET:
            raw_weights = self._volatility_target(signals, returns)
        elif self.method == PositionSizingMethod.EQUAL_RISK:
            raw_weights = self._equal_risk_contribution(signals, returns)
        elif self.method == PositionSizingMethod.RISK_PARITY:
            raw_weights = self._risk_parity(returns)
        else:
            raw_weights = self._equal_weight(n)
        
        # Apply constraints
        adjusted_weights, constraints = self._apply_constraints(
            tickers, raw_weights, volumes, sectors
        )
        
        # Normalize to sum to target exposure
        weight_sum = sum(adjusted_weights)
        if weight_sum > 0:
            scale = min(self.limits.max_total_exposure, weight_sum) / weight_sum
            adjusted_weights = [w * scale for w in adjusted_weights]
        
        # Build position objects
        positions = []
        for i, ticker in enumerate(tickers):
            price = prices.get(ticker, 0)
            notional = adjusted_weights[i] * portfolio_value
            shares = int(notional / price) if price > 0 else 0
            
            # Calculate risk metrics if returns available
            vol = None
            exp_ret = None
            if returns is not None and ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                if len(ticker_returns) > 20:
                    vol = ticker_returns.std() * np.sqrt(252)
                    exp_ret = ticker_returns.mean() * 252
            
            pos = SizedPosition(
                ticker=ticker,
                raw_weight=raw_weights[i],
                adjusted_weight=adjusted_weights[i],
                shares=shares,
                notional=notional,
                method=self.method,
                constraints_applied=constraints.get(ticker, []),
                expected_return=exp_ret,
                volatility=vol,
                sharpe=(exp_ret - self.risk_free_rate) / vol if vol and vol > 0 else None,
            )
            positions.append(pos)
        
        return positions
    
    def _equal_weight(self, n: int) -> List[float]:
        """Simple equal weighting."""
        return [1.0 / n] * n
    
    def _kelly_sizing(
        self,
        signals: List[float],
        returns: Optional[pd.DataFrame],
    ) -> List[float]:
        """
        Full Kelly criterion sizing.
        
        Kelly = (p * b - q) / b
        where p = win probability, b = win/loss ratio, q = 1-p
        
        In practice, we estimate from historical returns and signals.
        """
        n = len(signals)
        
        if returns is None or returns.empty:
            logger.warning("No returns data, falling back to signal-weighted")
            return self._signal_weighted(signals)
        
        weights = []
        for i, (_, signal) in enumerate(zip(returns.columns, signals)):
            if signal <= 0:
                weights.append(0.0)
                continue
            
            col = returns.columns[i] if i < len(returns.columns) else None
            if col is None or col not in returns.columns:
                weights.append(signal / sum(signals) if sum(signals) > 0 else 1/n)
                continue
            
            r = returns[col].dropna()
            if len(r) < 30:
                weights.append(signal / sum(signals) if sum(signals) > 0 else 1/n)
                continue
            
            # Estimate Kelly from returns
            mean_ret = r.mean()
            var_ret = r.var()
            
            if var_ret > 0:
                kelly = mean_ret / var_ret
                # Scale by signal strength
                kelly *= signal
                weights.append(max(0, kelly))
            else:
                weights.append(signal / sum(signals) if sum(signals) > 0 else 1/n)
        
        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = self._equal_weight(n)
        
        return weights
    
    def _fractional_kelly(
        self,
        signals: List[float],
        returns: Optional[pd.DataFrame],
    ) -> List[float]:
        """Fractional Kelly for reduced variance."""
        full_kelly = self._kelly_sizing(signals, returns)
        return [w * self.kelly_fraction for w in full_kelly]
    
    def _volatility_target(
        self,
        signals: List[float],
        returns: Optional[pd.DataFrame],
    ) -> List[float]:
        """
        Volatility targeting: scale positions to achieve target portfolio vol.
        
        Each position sized inversely proportional to its volatility.
        """
        n = len(signals)
        
        if returns is None or returns.empty:
            return self._signal_weighted(signals)
        
        # Calculate individual volatilities
        vols = []
        for i, signal in enumerate(signals):
            if signal <= 0:
                vols.append(float('inf'))
                continue
            
            col = returns.columns[i] if i < len(returns.columns) else None
            if col is None or col not in returns.columns:
                vols.append(0.20)  # Assume 20% vol
                continue
            
            r = returns[col].dropna()
            if len(r) < 20:
                vols.append(0.20)
            else:
                vols.append(r.std() * np.sqrt(252))
        
        # Inverse volatility weighting
        inv_vols = [1/v if v > 0 else 0 for v in vols]
        
        # Scale by signal
        weighted = [inv_v * max(0, sig) for inv_v, sig in zip(inv_vols, signals)]
        
        # Normalize
        total = sum(weighted)
        if total > 0:
            weights = [w / total for w in weighted]
        else:
            weights = self._equal_weight(n)
        
        # Scale to target volatility
        # Estimate portfolio vol (simplified: assume no correlation)
        port_vol = np.sqrt(sum(w**2 * v**2 for w, v in zip(weights, vols) if v < float('inf')))
        
        if port_vol > 0:
            scale = self.target_volatility / port_vol
            weights = [w * min(scale, 2.0) for w in weights]  # Cap at 2x leverage
        
        return weights
    
    def _equal_risk_contribution(
        self,
        signals: List[float],
        returns: Optional[pd.DataFrame],
    ) -> List[float]:
        """
        Equal Risk Contribution: each position contributes equally to portfolio risk.
        
        Simplified version - full ERC requires optimization.
        """
        n = len(signals)
        
        if returns is None or returns.empty:
            return self._signal_weighted(signals)
        
        # Calculate volatilities
        vols = []
        for i in range(n):
            col = returns.columns[i] if i < len(returns.columns) else None
            if col and col in returns.columns:
                r = returns[col].dropna()
                vols.append(r.std() * np.sqrt(252) if len(r) > 20 else 0.20)
            else:
                vols.append(0.20)
        
        # Weight inversely by vol^2 (marginal risk contribution)
        inv_var = [1 / (v**2) if v > 0 else 0 for v in vols]
        
        # Scale by positive signals
        weighted = [iv * max(0, sig) for iv, sig in zip(inv_var, signals)]
        
        total = sum(weighted)
        if total > 0:
            weights = [w / total for w in weighted]
        else:
            weights = self._equal_weight(n)
        
        return weights
    
    def _risk_parity(self, returns: Optional[pd.DataFrame]) -> List[float]:
        """
        Risk Parity: weight so each asset contributes equally to portfolio variance.
        
        Uses simplified analytical solution for uncorrelated assets.
        """
        if returns is None or returns.empty:
            n = len(returns.columns) if returns is not None else 1
            return self._equal_weight(n)
        
        n = len(returns.columns)
        vols = []
        
        for col in returns.columns:
            r = returns[col].dropna()
            if len(r) > 20:
                vols.append(r.std() * np.sqrt(252))
            else:
                vols.append(0.20)
        
        # Risk parity weights (simplified)
        inv_vols = [1/v if v > 0 else 0 for v in vols]
        total = sum(inv_vols)
        
        if total > 0:
            return [iv / total for iv in inv_vols]
        return self._equal_weight(n)
    
    def _signal_weighted(self, signals: List[float]) -> List[float]:
        """Weight by signal strength."""
        positive_signals = [max(0, s) for s in signals]
        total = sum(positive_signals)
        if total > 0:
            return [s / total for s in positive_signals]
        return self._equal_weight(len(signals))
    
    def _apply_constraints(
        self,
        tickers: List[str],
        weights: List[float],
        volumes: Optional[Dict[str, float]],
        sectors: Optional[Dict[str, str]],
    ) -> tuple[List[float], Dict[str, List[str]]]:
        """Apply position and exposure constraints."""
        adjusted = list(weights)
        constraints_log: Dict[str, List[str]] = {t: [] for t in tickers}
        
        # Apply max position constraint
        for i, (ticker, w) in enumerate(zip(tickers, adjusted)):
            if w > self.limits.max_position_pct:
                adjusted[i] = self.limits.max_position_pct
                constraints_log[ticker].append(f"max_position: {w:.2%} -> {self.limits.max_position_pct:.2%}")
        
        # Apply min position constraint (set to 0 if below min)
        for i, (ticker, w) in enumerate(zip(tickers, adjusted)):
            if 0 < w < self.limits.min_position_pct:
                adjusted[i] = 0.0
                constraints_log[ticker].append(f"min_position: {w:.2%} -> 0%")
        
        # Apply liquidity constraint if volumes provided
        if volumes:
            for i, ticker in enumerate(tickers):
                vol = volumes.get(ticker, float('inf'))
                if vol > 0:
                    max_from_liquidity = self.limits.max_adv_pct
                    if adjusted[i] > max_from_liquidity:
                        adjusted[i] = max_from_liquidity
                        constraints_log[ticker].append(f"liquidity: capped at {max_from_liquidity:.2%} ADV")
        
        # Apply sector constraints if sectors provided
        if sectors:
            sector_weights: Dict[str, float] = {}
            for ticker, w in zip(tickers, adjusted):
                sector = sectors.get(ticker, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + w
            
            # Scale down sectors exceeding limit
            for sector, total_weight in sector_weights.items():
                if total_weight > self.limits.max_sector_pct:
                    scale = self.limits.max_sector_pct / total_weight
                    for i, ticker in enumerate(tickers):
                        if sectors.get(ticker) == sector:
                            old_w = adjusted[i]
                            adjusted[i] *= scale
                            constraints_log[ticker].append(
                                f"sector_limit ({sector}): {old_w:.2%} -> {adjusted[i]:.2%}"
                            )
        
        return adjusted, constraints_log
    
    def calculate_portfolio_risk(
        self,
        positions: List[SizedPosition],
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Returns:
            Dict with VaR, CVaR, volatility, max drawdown, etc.
        """
        if not positions:
            return {}
        
        weights = np.array([p.adjusted_weight for p in positions])
        tickers = [p.ticker for p in positions]
        
        # Filter returns to matching tickers
        available = [t for t in tickers if t in returns.columns]
        if not available:
            return {"error": "No matching return data"}
        
        ret_subset = returns[available].dropna()
        if len(ret_subset) < 20:
            return {"error": "Insufficient return history"}
        
        # Align weights
        weight_map = {p.ticker: p.adjusted_weight for p in positions}
        aligned_weights = np.array([weight_map.get(t, 0) for t in available])
        aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
        
        # Portfolio returns
        port_returns = (ret_subset * aligned_weights).sum(axis=1)
        
        # Risk metrics
        vol = port_returns.std() * np.sqrt(252)
        var_95 = np.percentile(port_returns, 5)
        var_99 = np.percentile(port_returns, 1)
        cvar_95 = port_returns[port_returns <= var_95].mean()
        
        # Max drawdown
        cum_returns = (1 + port_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_dd = drawdowns.min()
        
        return {
            "volatility_annual": vol,
            "var_95_daily": var_95,
            "var_99_daily": var_99,
            "cvar_95_daily": cvar_95,
            "max_drawdown": max_dd,
            "sharpe_ratio": (port_returns.mean() * 252 - self.risk_free_rate) / vol if vol > 0 else 0,
            "sortino_ratio": self._calculate_sortino(port_returns),
            "calmar_ratio": (port_returns.mean() * 252) / abs(max_dd) if max_dd < 0 else 0,
        }
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation."""
        excess = returns - self.risk_free_rate / 252
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
        annual_return = returns.mean() * 252
        return (annual_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
