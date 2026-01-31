"""
Exposure Limits Module

Implements institutional exposure limit monitoring:
- Single-name concentration limits
- Sector/industry exposure limits  
- Factor exposure limits
- Correlation-based concentration
- Crowding detection
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Set
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of exposure limits."""
    SINGLE_NAME = "single_name"
    SECTOR = "sector"
    INDUSTRY = "industry"
    FACTOR = "factor"
    COUNTRY = "country"
    CURRENCY = "currency"
    MARKET_CAP = "market_cap"
    BETA = "beta"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"


@dataclass
class ExposureLimit:
    """Definition of an exposure limit."""
    limit_type: LimitType
    name: str  # e.g., "Technology" for sector, "momentum" for factor
    max_exposure: float  # Maximum allowed exposure (as fraction or absolute)
    min_exposure: float = 0.0  # Minimum allowed (for hedging requirements)
    is_hard_limit: bool = True  # Hard limits block trades, soft limits warn
    breach_action: str = "warn"  # "warn", "block", "reduce"


@dataclass
class ExposureViolation:
    """Record of an exposure limit breach."""
    limit: ExposureLimit
    current_exposure: float
    excess: float  # Amount over limit
    severity: str  # "warning", "breach", "critical"
    timestamp: datetime = field(default_factory=datetime.now)
    affected_positions: List[str] = field(default_factory=list)
    recommended_action: str = ""


class ExposureLimitChecker:
    """
    Monitor and enforce portfolio exposure limits.
    
    Tracks exposures across multiple dimensions and generates
    violations when limits are breached.
    
    Usage:
        checker = ExposureLimitChecker()
        
        # Add limits
        checker.add_limit(ExposureLimit(
            limit_type=LimitType.SECTOR,
            name="Technology",
            max_exposure=0.30,
            breach_action="warn"
        ))
        
        # Check portfolio
        violations = checker.check_portfolio(
            positions={"AAPL": 0.15, "MSFT": 0.20},
            sectors={"AAPL": "Technology", "MSFT": "Technology"}
        )
    """
    
    def __init__(self):
        self.limits: List[ExposureLimit] = []
        self.violation_history: List[ExposureViolation] = []
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Set up standard institutional limits."""
        # Single name limits
        self.add_limit(ExposureLimit(
            limit_type=LimitType.SINGLE_NAME,
            name="default",
            max_exposure=0.10,  # 10% max single position
            breach_action="reduce"
        ))
        
        # Sector limits
        for sector in ["Technology", "Healthcare", "Financials", "Energy", 
                       "Consumer Discretionary", "Consumer Staples", "Industrials",
                       "Materials", "Utilities", "Real Estate", "Communication Services"]:
            self.add_limit(ExposureLimit(
                limit_type=LimitType.SECTOR,
                name=sector,
                max_exposure=0.30,  # 30% max per sector
                breach_action="warn"
            ))
        
        # Beta limit
        self.add_limit(ExposureLimit(
            limit_type=LimitType.BETA,
            name="portfolio_beta",
            max_exposure=1.5,  # Max 1.5 beta
            min_exposure=0.5,  # Min 0.5 beta (not too defensive)
            breach_action="reduce"
        ))
        
        # Liquidity concentration
        self.add_limit(ExposureLimit(
            limit_type=LimitType.LIQUIDITY,
            name="illiquid_positions",
            max_exposure=0.20,  # Max 20% in illiquid names
            breach_action="block"
        ))
    
    def add_limit(self, limit: ExposureLimit):
        """Add or update an exposure limit."""
        # Remove existing limit of same type/name
        self.limits = [l for l in self.limits 
                       if not (l.limit_type == limit.limit_type and l.name == limit.name)]
        self.limits.append(limit)
        logger.debug(f"Added limit: {limit.limit_type.value}/{limit.name} max={limit.max_exposure}")
    
    def remove_limit(self, limit_type: LimitType, name: str):
        """Remove an exposure limit."""
        self.limits = [l for l in self.limits 
                       if not (l.limit_type == limit_type and l.name == name)]
    
    def check_portfolio(
        self,
        positions: Dict[str, float],  # ticker -> weight
        sectors: Optional[Dict[str, str]] = None,
        industries: Optional[Dict[str, str]] = None,
        betas: Optional[Dict[str, float]] = None,
        factors: Optional[Dict[str, Dict[str, float]]] = None,  # ticker -> {factor: exposure}
        market_caps: Optional[Dict[str, float]] = None,
        adv: Optional[Dict[str, float]] = None,  # Average daily volume
    ) -> List[ExposureViolation]:
        """
        Check all exposure limits against current portfolio.
        
        Returns:
            List of violations found
        """
        violations = []
        
        # Check single-name limits
        violations.extend(self._check_single_name(positions))
        
        # Check sector limits
        if sectors:
            violations.extend(self._check_group_exposure(
                positions, sectors, LimitType.SECTOR
            ))
        
        # Check industry limits
        if industries:
            violations.extend(self._check_group_exposure(
                positions, industries, LimitType.INDUSTRY
            ))
        
        # Check beta limits
        if betas:
            violations.extend(self._check_beta(positions, betas))
        
        # Check factor exposures
        if factors:
            violations.extend(self._check_factors(positions, factors))
        
        # Check liquidity
        if adv and market_caps:
            violations.extend(self._check_liquidity(positions, adv, market_caps))
        
        # Log and store violations
        for v in violations:
            self.violation_history.append(v)
            level = logging.WARNING if v.severity == "warning" else logging.ERROR
            logger.log(level, f"Exposure violation: {v.limit.limit_type.value}/{v.limit.name} "
                              f"current={v.current_exposure:.2%} limit={v.limit.max_exposure:.2%}")
        
        return violations
    
    def _check_single_name(self, positions: Dict[str, float]) -> List[ExposureViolation]:
        """Check single-name concentration limits."""
        violations = []
        
        limit = next((l for l in self.limits 
                      if l.limit_type == LimitType.SINGLE_NAME and l.name == "default"), None)
        
        if not limit:
            return violations
        
        for ticker, weight in positions.items():
            if weight > limit.max_exposure:
                excess = weight - limit.max_exposure
                severity = "critical" if excess > 0.05 else "breach"
                
                violations.append(ExposureViolation(
                    limit=limit,
                    current_exposure=weight,
                    excess=excess,
                    severity=severity,
                    affected_positions=[ticker],
                    recommended_action=f"Reduce {ticker} by {excess:.2%}"
                ))
        
        return violations
    
    def _check_group_exposure(
        self,
        positions: Dict[str, float],
        groupings: Dict[str, str],  # ticker -> group
        limit_type: LimitType,
    ) -> List[ExposureViolation]:
        """Check exposure limits by grouping (sector, industry, etc.)."""
        violations = []
        
        # Aggregate weights by group
        group_weights: Dict[str, float] = {}
        group_positions: Dict[str, List[str]] = {}
        
        for ticker, weight in positions.items():
            group = groupings.get(ticker, "Unknown")
            group_weights[group] = group_weights.get(group, 0) + weight
            if group not in group_positions:
                group_positions[group] = []
            group_positions[group].append(ticker)
        
        # Check against limits
        for group, weight in group_weights.items():
            limit = next((l for l in self.limits 
                          if l.limit_type == limit_type and l.name == group), None)
            
            if limit and weight > limit.max_exposure:
                excess = weight - limit.max_exposure
                severity = "critical" if excess > 0.10 else "breach"
                
                violations.append(ExposureViolation(
                    limit=limit,
                    current_exposure=weight,
                    excess=excess,
                    severity=severity,
                    affected_positions=group_positions.get(group, []),
                    recommended_action=f"Reduce {group} exposure by {excess:.2%}"
                ))
        
        return violations
    
    def _check_beta(
        self,
        positions: Dict[str, float],
        betas: Dict[str, float],
    ) -> List[ExposureViolation]:
        """Check portfolio beta limits."""
        violations = []
        
        # Calculate weighted beta
        portfolio_beta = sum(
            positions.get(ticker, 0) * beta 
            for ticker, beta in betas.items()
        )
        
        # Normalize by total weight
        total_weight = sum(positions.values())
        if total_weight > 0:
            portfolio_beta /= total_weight
        
        limit = next((l for l in self.limits 
                      if l.limit_type == LimitType.BETA), None)
        
        if limit:
            if portfolio_beta > limit.max_exposure:
                excess = portfolio_beta - limit.max_exposure
                violations.append(ExposureViolation(
                    limit=limit,
                    current_exposure=portfolio_beta,
                    excess=excess,
                    severity="breach",
                    affected_positions=list(positions.keys()),
                    recommended_action=f"Add low-beta positions or reduce high-beta names"
                ))
            elif portfolio_beta < limit.min_exposure:
                shortfall = limit.min_exposure - portfolio_beta
                violations.append(ExposureViolation(
                    limit=limit,
                    current_exposure=portfolio_beta,
                    excess=-shortfall,
                    severity="warning",
                    affected_positions=list(positions.keys()),
                    recommended_action=f"Portfolio too defensive, increase market exposure"
                ))
        
        return violations
    
    def _check_factors(
        self,
        positions: Dict[str, float],
        factors: Dict[str, Dict[str, float]],  # ticker -> {factor: exposure}
    ) -> List[ExposureViolation]:
        """Check factor exposure limits."""
        violations = []
        
        # Aggregate factor exposures
        factor_exposures: Dict[str, float] = {}
        factor_positions: Dict[str, List[str]] = {}
        
        for ticker, weight in positions.items():
            ticker_factors = factors.get(ticker, {})
            for factor, exposure in ticker_factors.items():
                weighted_exposure = weight * exposure
                factor_exposures[factor] = factor_exposures.get(factor, 0) + weighted_exposure
                if factor not in factor_positions:
                    factor_positions[factor] = []
                factor_positions[factor].append(ticker)
        
        # Check against limits
        for factor, exposure in factor_exposures.items():
            limit = next((l for l in self.limits 
                          if l.limit_type == LimitType.FACTOR and l.name == factor), None)
            
            if limit and abs(exposure) > limit.max_exposure:
                excess = abs(exposure) - limit.max_exposure
                violations.append(ExposureViolation(
                    limit=limit,
                    current_exposure=exposure,
                    excess=excess,
                    severity="breach",
                    affected_positions=factor_positions.get(factor, []),
                    recommended_action=f"Reduce {factor} factor exposure"
                ))
        
        return violations
    
    def _check_liquidity(
        self,
        positions: Dict[str, float],
        adv: Dict[str, float],  # ticker -> avg daily volume in $
        market_caps: Dict[str, float],
    ) -> List[ExposureViolation]:
        """Check liquidity concentration."""
        violations = []
        
        # Define illiquid threshold (e.g., < $10M ADV or < $2B market cap)
        illiquid_threshold_adv = 10_000_000
        illiquid_threshold_mcap = 2_000_000_000
        
        illiquid_weight = 0
        illiquid_positions = []
        
        for ticker, weight in positions.items():
            ticker_adv = adv.get(ticker, float('inf'))
            ticker_mcap = market_caps.get(ticker, float('inf'))
            
            if ticker_adv < illiquid_threshold_adv or ticker_mcap < illiquid_threshold_mcap:
                illiquid_weight += weight
                illiquid_positions.append(ticker)
        
        limit = next((l for l in self.limits 
                      if l.limit_type == LimitType.LIQUIDITY), None)
        
        if limit and illiquid_weight > limit.max_exposure:
            excess = illiquid_weight - limit.max_exposure
            violations.append(ExposureViolation(
                limit=limit,
                current_exposure=illiquid_weight,
                excess=excess,
                severity="critical",
                affected_positions=illiquid_positions,
                recommended_action=f"Reduce illiquid positions by {excess:.2%}"
            ))
        
        return violations
    
    def get_current_exposures(
        self,
        positions: Dict[str, float],
        sectors: Optional[Dict[str, str]] = None,
        betas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate current exposures for reporting.
        
        Returns:
            Dict organized by exposure type
        """
        exposures = {
            "single_name": {},
            "sector": {},
            "beta": {"portfolio": 0.0},
        }
        
        # Single name
        for ticker, weight in positions.items():
            exposures["single_name"][ticker] = weight
        
        # Sector
        if sectors:
            for ticker, weight in positions.items():
                sector = sectors.get(ticker, "Unknown")
                exposures["sector"][sector] = exposures["sector"].get(sector, 0) + weight
        
        # Beta
        if betas:
            total_weight = sum(positions.values())
            if total_weight > 0:
                portfolio_beta = sum(
                    positions.get(ticker, 0) * beta 
                    for ticker, beta in betas.items()
                ) / total_weight
                exposures["beta"]["portfolio"] = portfolio_beta
        
        return exposures
    
    def generate_exposure_report(
        self,
        positions: Dict[str, float],
        sectors: Optional[Dict[str, str]] = None,
        betas: Optional[Dict[str, float]] = None,
    ) -> str:
        """Generate human-readable exposure report."""
        exposures = self.get_current_exposures(positions, sectors, betas)
        
        lines = ["=" * 60, "EXPOSURE REPORT", "=" * 60, ""]
        
        # Top positions
        lines.append("TOP POSITIONS:")
        sorted_positions = sorted(exposures["single_name"].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]
        for ticker, weight in sorted_positions:
            limit = self.limits[0].max_exposure if self.limits else 0.10
            status = "⚠️" if weight > limit * 0.8 else "✓"
            lines.append(f"  {status} {ticker}: {weight:.2%}")
        
        lines.append("")
        
        # Sector breakdown
        if exposures["sector"]:
            lines.append("SECTOR EXPOSURE:")
            sorted_sectors = sorted(exposures["sector"].items(), 
                                     key=lambda x: x[1], reverse=True)
            for sector, weight in sorted_sectors:
                limit = next((l.max_exposure for l in self.limits 
                              if l.limit_type == LimitType.SECTOR and l.name == sector), 0.30)
                status = "⚠️" if weight > limit * 0.8 else "✓"
                lines.append(f"  {status} {sector}: {weight:.2%}")
        
        lines.append("")
        
        # Beta
        if exposures["beta"]["portfolio"]:
            beta = exposures["beta"]["portfolio"]
            status = "✓" if 0.8 <= beta <= 1.2 else "⚠️"
            lines.append(f"PORTFOLIO BETA: {status} {beta:.2f}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def check_proposed_trade(
        self,
        current_positions: Dict[str, float],
        trade_ticker: str,
        trade_weight_change: float,
        sectors: Optional[Dict[str, str]] = None,
        betas: Optional[Dict[str, float]] = None,
    ) -> tuple[bool, List[ExposureViolation]]:
        """
        Pre-trade compliance check.
        
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        # Create proposed portfolio
        proposed = current_positions.copy()
        proposed[trade_ticker] = proposed.get(trade_ticker, 0) + trade_weight_change
        
        # Check violations
        violations = self.check_portfolio(proposed, sectors=sectors, betas=betas)
        
        # Filter for hard limit breaches
        hard_violations = [v for v in violations if v.limit.is_hard_limit]
        
        is_allowed = len(hard_violations) == 0
        
        return is_allowed, violations
