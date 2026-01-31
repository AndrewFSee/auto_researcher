"""
Drawdown Control Module

Implements institutional drawdown management:
- Real-time drawdown monitoring
- Circuit breakers (portfolio and position level)
- De-risking cascades
- Recovery tracking
- Volatility-based scaling
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Callable
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DrawdownAction(Enum):
    """Actions to take on drawdown events."""
    NONE = "none"
    ALERT = "alert"
    REDUCE_EXPOSURE = "reduce_exposure"
    HALT_NEW_POSITIONS = "halt_new_positions"
    CLOSE_LOSERS = "close_losers"
    FULL_DELEVERAGE = "full_deleverage"
    EMERGENCY_LIQUIDATE = "emergency_liquidate"


@dataclass
class CircuitBreaker:
    """Definition of a circuit breaker trigger."""
    name: str
    drawdown_threshold: float  # e.g., -0.05 for 5% drawdown
    action: DrawdownAction
    cooldown_hours: int = 24  # Hours before reset
    auto_reset: bool = True
    is_portfolio_level: bool = True  # vs position level
    target_exposure: Optional[float] = None  # Target exposure after trigger


@dataclass
class DrawdownState:
    """Current drawdown state of portfolio or position."""
    current_value: float
    peak_value: float
    drawdown_pct: float
    drawdown_start_date: Optional[datetime] = None
    days_in_drawdown: int = 0
    recovery_needed_pct: float = 0.0
    active_circuit_breakers: List[str] = field(default_factory=list)
    current_action: DrawdownAction = DrawdownAction.NONE


class DrawdownController:
    """
    Institutional-grade drawdown monitoring and control.
    
    Features:
    - Configurable circuit breakers at portfolio and position levels
    - Automatic de-risking cascades
    - Volatility-based position scaling
    - Recovery tracking and reporting
    
    Usage:
        controller = DrawdownController()
        
        # Add custom circuit breakers
        controller.add_circuit_breaker(CircuitBreaker(
            name="daily_loss_limit",
            drawdown_threshold=-0.02,  # 2% daily loss
            action=DrawdownAction.HALT_NEW_POSITIONS
        ))
        
        # Update with daily values
        state = controller.update(portfolio_value=980_000, date=today)
        
        # Check if trading is allowed
        if controller.is_trading_halted():
            print("Trading halted due to drawdown")
    """
    
    def __init__(
        self,
        initial_value: float = 1_000_000,
        max_drawdown: float = 0.20,  # 20% max drawdown
        volatility_scale: bool = True,
        volatility_lookback: int = 20,
        target_volatility: float = 0.15,
    ):
        self.initial_value = initial_value
        self.max_drawdown = max_drawdown
        self.volatility_scale = volatility_scale
        self.volatility_lookback = volatility_lookback
        self.target_volatility = target_volatility
        
        # State tracking
        self.current_value = initial_value
        self.peak_value = initial_value
        self.value_history: List[tuple[datetime, float]] = []
        self.drawdown_history: List[tuple[datetime, float]] = []
        
        # Circuit breakers
        self.circuit_breakers: List[CircuitBreaker] = []
        self.triggered_breakers: Dict[str, datetime] = {}
        
        # Position-level tracking
        self.position_peaks: Dict[str, float] = {}
        self.position_values: Dict[str, float] = {}
        
        # Callbacks
        self.on_drawdown_callbacks: List[Callable] = []
        self.on_recovery_callbacks: List[Callable] = []
        
        # Setup default circuit breakers
        self._setup_default_breakers()
        
        logger.info(f"DrawdownController initialized: max_dd={max_drawdown:.1%}")
    
    def _setup_default_breakers(self):
        """Set up standard institutional circuit breakers."""
        # Daily loss limits
        self.add_circuit_breaker(CircuitBreaker(
            name="daily_loss_2pct",
            drawdown_threshold=-0.02,
            action=DrawdownAction.ALERT,
            cooldown_hours=4,
        ))
        
        self.add_circuit_breaker(CircuitBreaker(
            name="daily_loss_3pct",
            drawdown_threshold=-0.03,
            action=DrawdownAction.HALT_NEW_POSITIONS,
            cooldown_hours=8,
        ))
        
        self.add_circuit_breaker(CircuitBreaker(
            name="daily_loss_5pct",
            drawdown_threshold=-0.05,
            action=DrawdownAction.REDUCE_EXPOSURE,
            target_exposure=0.50,  # Reduce to 50% exposure
            cooldown_hours=24,
        ))
        
        # Peak-to-trough drawdown limits
        self.add_circuit_breaker(CircuitBreaker(
            name="drawdown_10pct",
            drawdown_threshold=-0.10,
            action=DrawdownAction.REDUCE_EXPOSURE,
            target_exposure=0.75,
            cooldown_hours=48,
        ))
        
        self.add_circuit_breaker(CircuitBreaker(
            name="drawdown_15pct",
            drawdown_threshold=-0.15,
            action=DrawdownAction.CLOSE_LOSERS,
            cooldown_hours=72,
        ))
        
        self.add_circuit_breaker(CircuitBreaker(
            name="drawdown_20pct",
            drawdown_threshold=-0.20,
            action=DrawdownAction.FULL_DELEVERAGE,
            target_exposure=0.25,
            cooldown_hours=168,  # 1 week
            auto_reset=False,  # Manual reset required
        ))
    
    def add_circuit_breaker(self, breaker: CircuitBreaker):
        """Add or update a circuit breaker."""
        self.circuit_breakers = [b for b in self.circuit_breakers if b.name != breaker.name]
        self.circuit_breakers.append(breaker)
        # Sort by threshold (least severe first)
        self.circuit_breakers.sort(key=lambda b: b.drawdown_threshold, reverse=True)
        logger.debug(f"Added circuit breaker: {breaker.name} at {breaker.drawdown_threshold:.1%}")
    
    def update(
        self,
        portfolio_value: float,
        date: Optional[datetime] = None,
        position_values: Optional[Dict[str, float]] = None,
    ) -> DrawdownState:
        """
        Update with current portfolio value and check for breaker triggers.
        
        Args:
            portfolio_value: Current total portfolio value
            date: Date of update (defaults to now)
            position_values: Optional dict of position values for position-level tracking
            
        Returns:
            Current DrawdownState
        """
        date = date or datetime.now()
        self.current_value = portfolio_value
        
        # Update peak (high water mark)
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Record history
        self.value_history.append((date, portfolio_value))
        
        # Calculate drawdown
        drawdown_pct = (portfolio_value - self.peak_value) / self.peak_value
        self.drawdown_history.append((date, drawdown_pct))
        
        # Track position-level drawdowns
        if position_values:
            self._update_position_drawdowns(position_values)
        
        # Check circuit breakers
        triggered = self._check_circuit_breakers(drawdown_pct, date)
        
        # Calculate days in drawdown
        days_in_dd = self._calculate_days_in_drawdown(date)
        
        # Calculate recovery needed
        recovery_needed = (self.peak_value / portfolio_value) - 1 if portfolio_value > 0 else 0
        
        # Determine current action
        current_action = DrawdownAction.NONE
        if triggered:
            # Use most severe triggered action
            for breaker_name in triggered:
                breaker = next((b for b in self.circuit_breakers if b.name == breaker_name), None)
                if breaker and breaker.action.value > current_action.value:
                    current_action = breaker.action
        
        state = DrawdownState(
            current_value=portfolio_value,
            peak_value=self.peak_value,
            drawdown_pct=drawdown_pct,
            drawdown_start_date=self._find_drawdown_start(),
            days_in_drawdown=days_in_dd,
            recovery_needed_pct=recovery_needed,
            active_circuit_breakers=list(self.triggered_breakers.keys()),
            current_action=current_action,
        )
        
        # Fire callbacks
        if drawdown_pct < -0.01 and len(self.value_history) > 1:
            prev_dd = self.drawdown_history[-2][1] if len(self.drawdown_history) > 1 else 0
            if drawdown_pct < prev_dd:
                self._fire_drawdown_callbacks(state)
        
        return state
    
    def _check_circuit_breakers(
        self,
        drawdown_pct: float,
        date: datetime,
    ) -> List[str]:
        """Check and trigger circuit breakers."""
        newly_triggered = []
        
        for breaker in self.circuit_breakers:
            if not breaker.is_portfolio_level:
                continue
            
            # Check if already triggered and in cooldown
            if breaker.name in self.triggered_breakers:
                trigger_time = self.triggered_breakers[breaker.name]
                cooldown_end = trigger_time + timedelta(hours=breaker.cooldown_hours)
                
                if date < cooldown_end:
                    continue  # Still in cooldown
                elif breaker.auto_reset:
                    del self.triggered_breakers[breaker.name]
                    logger.info(f"Circuit breaker reset: {breaker.name}")
            
            # Check threshold
            if drawdown_pct <= breaker.drawdown_threshold:
                self.triggered_breakers[breaker.name] = date
                newly_triggered.append(breaker.name)
                
                logger.warning(
                    f"Circuit breaker TRIGGERED: {breaker.name} "
                    f"(DD={drawdown_pct:.2%} <= {breaker.drawdown_threshold:.2%}) "
                    f"Action: {breaker.action.value}"
                )
        
        return newly_triggered
    
    def _update_position_drawdowns(self, position_values: Dict[str, float]):
        """Track position-level drawdowns."""
        for ticker, value in position_values.items():
            # Update peak
            current_peak = self.position_peaks.get(ticker, 0)
            if value > current_peak:
                self.position_peaks[ticker] = value
            
            self.position_values[ticker] = value
    
    def _calculate_days_in_drawdown(self, current_date: datetime) -> int:
        """Calculate how many days we've been in current drawdown."""
        if not self.drawdown_history:
            return 0
        
        # Find last time we were at or near peak
        for date, dd in reversed(self.drawdown_history):
            if dd >= -0.001:  # Within 0.1% of peak
                return (current_date - date).days
        
        return len(self.drawdown_history)
    
    def _find_drawdown_start(self) -> Optional[datetime]:
        """Find when current drawdown started."""
        if not self.drawdown_history:
            return None
        
        for date, dd in reversed(self.drawdown_history):
            if dd >= -0.001:
                return date
        
        return self.drawdown_history[0][0] if self.drawdown_history else None
    
    def _fire_drawdown_callbacks(self, state: DrawdownState):
        """Fire registered drawdown callbacks."""
        for callback in self.on_drawdown_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Drawdown callback error: {e}")
    
    def is_trading_halted(self) -> bool:
        """Check if trading is halted due to circuit breakers."""
        halt_actions = {
            DrawdownAction.HALT_NEW_POSITIONS,
            DrawdownAction.FULL_DELEVERAGE,
            DrawdownAction.EMERGENCY_LIQUIDATE,
        }
        
        for breaker_name in self.triggered_breakers:
            breaker = next((b for b in self.circuit_breakers if b.name == breaker_name), None)
            if breaker and breaker.action in halt_actions:
                return True
        
        return False
    
    def get_target_exposure(self) -> float:
        """Get target exposure based on active circuit breakers."""
        min_exposure = 1.0
        
        for breaker_name in self.triggered_breakers:
            breaker = next((b for b in self.circuit_breakers if b.name == breaker_name), None)
            if breaker and breaker.target_exposure is not None:
                min_exposure = min(min_exposure, breaker.target_exposure)
        
        return min_exposure
    
    def get_volatility_scale(self, recent_returns: pd.Series) -> float:
        """
        Calculate volatility scaling factor.
        
        Reduces exposure when realized vol exceeds target.
        """
        if not self.volatility_scale or len(recent_returns) < 5:
            return 1.0
        
        realized_vol = recent_returns.tail(self.volatility_lookback).std() * np.sqrt(252)
        
        if realized_vol <= 0:
            return 1.0
        
        scale = self.target_volatility / realized_vol
        
        # Clamp to reasonable range
        scale = max(0.25, min(2.0, scale))
        
        if scale < 0.9:
            logger.info(f"Volatility scaling: {scale:.2f}x (realized vol={realized_vol:.1%})")
        
        return scale
    
    def calculate_position_risk_adjustment(
        self,
        ticker: str,
        position_returns: pd.Series,
    ) -> float:
        """
        Calculate position-level risk adjustment factor.
        
        Returns scaling factor based on:
        - Position drawdown
        - Position volatility
        - Correlation to portfolio
        """
        adjustment = 1.0
        
        # Drawdown adjustment
        if ticker in self.position_values and ticker in self.position_peaks:
            current = self.position_values[ticker]
            peak = self.position_peaks[ticker]
            if peak > 0:
                pos_dd = (current - peak) / peak
                if pos_dd < -0.20:  # Position down 20%+
                    adjustment *= 0.5
                    logger.info(f"Position {ticker} reduced due to {pos_dd:.1%} drawdown")
                elif pos_dd < -0.10:
                    adjustment *= 0.75
        
        # Volatility adjustment
        if len(position_returns) >= 20:
            pos_vol = position_returns.std() * np.sqrt(252)
            if pos_vol > 0.50:  # Very volatile
                adjustment *= 0.50
            elif pos_vol > 0.35:
                adjustment *= 0.75
        
        return adjustment
    
    def reset_circuit_breaker(self, name: str):
        """Manually reset a circuit breaker."""
        if name in self.triggered_breakers:
            del self.triggered_breakers[name]
            logger.info(f"Circuit breaker manually reset: {name}")
    
    def reset_all(self):
        """Reset all state (e.g., for new period)."""
        self.triggered_breakers.clear()
        self.peak_value = self.current_value
        logger.info("DrawdownController: All state reset")
    
    def get_drawdown_report(self) -> str:
        """Generate drawdown status report."""
        current_dd = (self.current_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0
        recovery_needed = (self.peak_value / self.current_value) - 1 if self.current_value > 0 else 0
        
        lines = [
            "=" * 60,
            "DRAWDOWN STATUS REPORT",
            "=" * 60,
            "",
            f"Current Value:     ${self.current_value:,.2f}",
            f"Peak Value:        ${self.peak_value:,.2f}",
            f"Current Drawdown:  {current_dd:.2%}",
            f"Recovery Needed:   {recovery_needed:.2%}",
            "",
            "CIRCUIT BREAKERS:",
        ]
        
        for breaker in self.circuit_breakers:
            status = "🔴 TRIGGERED" if breaker.name in self.triggered_breakers else "🟢 OK"
            lines.append(f"  {status} {breaker.name}: {breaker.drawdown_threshold:.1%} -> {breaker.action.value}")
        
        if self.triggered_breakers:
            lines.append("")
            lines.append("ACTIVE TRIGGERS:")
            for name, trigger_time in self.triggered_breakers.items():
                breaker = next((b for b in self.circuit_breakers if b.name == name), None)
                if breaker:
                    cooldown_end = trigger_time + timedelta(hours=breaker.cooldown_hours)
                    lines.append(f"  - {name}: triggered {trigger_time}, resets {cooldown_end}")
        
        lines.append("")
        lines.append(f"Target Exposure: {self.get_target_exposure():.0%}")
        lines.append(f"Trading Halted:  {'YES' if self.is_trading_halted() else 'NO'}")
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def on_drawdown(self, callback: Callable):
        """Register callback for drawdown events."""
        self.on_drawdown_callbacks.append(callback)
    
    def on_recovery(self, callback: Callable):
        """Register callback for recovery events."""
        self.on_recovery_callbacks.append(callback)
