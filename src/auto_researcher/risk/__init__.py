"""
Risk Management Module

Institutional-grade risk controls for portfolio construction and management.

Components:
- PositionSizer: Kelly criterion, volatility targeting, max position limits
- ExposureLimits: Sector, factor, and concentration limits
- DrawdownController: Max drawdown, circuit breakers, recovery mode
- RiskAttribution: Factor exposure decomposition and risk budgeting
"""

from .position_sizing import (
    PositionSizer,
    PositionSizingMethod,
    PositionLimit,
    SizedPosition,
)
from .exposure_limits import (
    ExposureLimitChecker,
    ExposureLimit,
    ExposureViolation,
    LimitType,
)
from .drawdown_control import (
    DrawdownController,
    DrawdownState,
    DrawdownAction,
    CircuitBreaker,
)
from .risk_attribution import (
    RiskAttributor,
    FactorExposure,
    RiskDecomposition,
    RiskBudget,
)

__all__ = [
    # Position Sizing
    "PositionSizer",
    "PositionSizingMethod",
    "PositionLimit",
    "SizedPosition",
    # Exposure Limits
    "ExposureLimitChecker",
    "ExposureLimit",
    "ExposureViolation",
    "LimitType",
    # Drawdown Control
    "DrawdownController",
    "DrawdownState",
    "DrawdownAction",
    "CircuitBreaker",
    # Risk Attribution
    "RiskAttributor",
    "FactorExposure",
    "RiskDecomposition",
    "RiskBudget",
]
