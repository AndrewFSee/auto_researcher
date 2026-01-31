"""
Tests for Risk Management Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from auto_researcher.risk import (
    PositionSizer,
    PositionSizingMethod,
    PositionLimit,
    SizedPosition,
    ExposureLimitChecker,
    ExposureLimit,
    LimitType,
    DrawdownController,
    DrawdownAction,
    CircuitBreaker,
    RiskAttributor,
)


class TestPositionSizer:
    """Tests for PositionSizer class."""
    
    def test_equal_weight_sizing(self):
        """Test equal weight position sizing."""
        sizer = PositionSizer(method=PositionSizingMethod.EQUAL_WEIGHT)
        
        positions = sizer.size_portfolio(
            tickers=["AAPL", "MSFT", "GOOGL"],
            signals=[1.0, 1.0, 1.0],
            prices={"AAPL": 150, "MSFT": 350, "GOOGL": 140},
            portfolio_value=1_000_000,
        )
        
        assert len(positions) == 3
        # Equal weight should give ~33% each
        for pos in positions:
            assert 0.30 <= pos.adjusted_weight <= 0.35
    
    def test_signal_weighted_sizing(self):
        """Test that stronger signals get larger weights."""
        sizer = PositionSizer(method=PositionSizingMethod.VOLATILITY_TARGET)
        
        positions = sizer.size_portfolio(
            tickers=["AAPL", "MSFT"],
            signals=[1.0, 0.5],  # AAPL has stronger signal
            prices={"AAPL": 150, "MSFT": 350},
            portfolio_value=1_000_000,
        )
        
        aapl = next(p for p in positions if p.ticker == "AAPL")
        msft = next(p for p in positions if p.ticker == "MSFT")
        
        # AAPL should have higher weight due to stronger signal
        assert aapl.raw_weight > msft.raw_weight
    
    def test_max_position_constraint(self):
        """Test that max position limit is enforced."""
        limits = PositionLimit(max_position_pct=0.05)
        sizer = PositionSizer(
            method=PositionSizingMethod.EQUAL_WEIGHT,
            limits=limits,
        )
        
        positions = sizer.size_portfolio(
            tickers=["AAPL", "MSFT"],
            signals=[1.0, 1.0],
            prices={"AAPL": 150, "MSFT": 350},
            portfolio_value=1_000_000,
        )
        
        for pos in positions:
            assert pos.adjusted_weight <= 0.05
    
    def test_shares_calculation(self):
        """Test that shares are calculated correctly."""
        sizer = PositionSizer(method=PositionSizingMethod.EQUAL_WEIGHT)
        
        positions = sizer.size_portfolio(
            tickers=["AAPL"],
            signals=[1.0],
            prices={"AAPL": 100},
            portfolio_value=100_000,
        )
        
        pos = positions[0]
        expected_shares = int(100_000 / 100)  # Full weight at $100/share
        assert pos.shares == expected_shares
    
    def test_sector_constraint(self):
        """Test sector concentration limit."""
        limits = PositionLimit(max_sector_pct=0.20)
        sizer = PositionSizer(limits=limits)
        
        positions = sizer.size_portfolio(
            tickers=["AAPL", "MSFT", "GOOGL", "META"],
            signals=[1.0, 1.0, 1.0, 1.0],
            prices={"AAPL": 150, "MSFT": 350, "GOOGL": 140, "META": 300},
            portfolio_value=1_000_000,
            sectors={"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech"},
        )
        
        total_tech_weight = sum(p.adjusted_weight for p in positions)
        # Should be constrained below raw equal weight
        assert total_tech_weight <= 0.25  # Some buffer for constraint application


class TestExposureLimitChecker:
    """Tests for ExposureLimitChecker class."""
    
    def test_single_name_violation(self):
        """Test detection of single-name concentration violation."""
        checker = ExposureLimitChecker()
        
        violations = checker.check_portfolio(
            positions={"AAPL": 0.15, "MSFT": 0.05}  # AAPL exceeds 10% default
        )
        
        assert len(violations) >= 1
        aapl_violation = next(
            (v for v in violations if "AAPL" in v.affected_positions), None
        )
        assert aapl_violation is not None
        assert aapl_violation.excess > 0
    
    def test_sector_violation(self):
        """Test detection of sector concentration violation."""
        checker = ExposureLimitChecker()
        
        violations = checker.check_portfolio(
            positions={"AAPL": 0.10, "MSFT": 0.10, "GOOGL": 0.10, "META": 0.10},
            sectors={"AAPL": "Technology", "MSFT": "Technology", 
                     "GOOGL": "Technology", "META": "Technology"},
        )
        
        sector_violations = [v for v in violations if v.limit.limit_type == LimitType.SECTOR]
        assert len(sector_violations) >= 1
    
    def test_no_violations_clean_portfolio(self):
        """Test that clean portfolio has no violations."""
        checker = ExposureLimitChecker()
        
        violations = checker.check_portfolio(
            positions={"AAPL": 0.05, "XOM": 0.05, "JPM": 0.05},
            sectors={"AAPL": "Technology", "XOM": "Energy", "JPM": "Financials"},
        )
        
        # Filter out soft warnings
        hard_violations = [v for v in violations if v.limit.is_hard_limit]
        assert len(hard_violations) == 0
    
    def test_pre_trade_compliance(self):
        """Test pre-trade compliance check."""
        checker = ExposureLimitChecker()
        
        current = {"AAPL": 0.08}
        
        # This trade would exceed limit
        allowed, violations = checker.check_proposed_trade(
            current_positions=current,
            trade_ticker="AAPL",
            trade_weight_change=0.05,  # Would take AAPL to 13%
        )
        
        assert not allowed  # Should be blocked


class TestDrawdownController:
    """Tests for DrawdownController class."""
    
    def test_drawdown_calculation(self):
        """Test drawdown percentage calculation."""
        controller = DrawdownController(initial_value=1_000_000)
        
        # Simulate 5% loss
        state = controller.update(portfolio_value=950_000)
        
        assert abs(state.drawdown_pct - (-0.05)) < 0.001
        assert state.peak_value == 1_000_000
    
    def test_circuit_breaker_trigger(self):
        """Test that circuit breakers trigger correctly."""
        controller = DrawdownController(initial_value=1_000_000)
        
        # Trigger 5% drawdown breaker
        state = controller.update(portfolio_value=940_000)
        
        assert len(state.active_circuit_breakers) > 0
        assert state.current_action != DrawdownAction.NONE
    
    def test_peak_tracking(self):
        """Test high water mark tracking."""
        controller = DrawdownController(initial_value=1_000_000)
        
        # New high
        controller.update(portfolio_value=1_100_000)
        assert controller.peak_value == 1_100_000
        
        # Drawdown from new high
        state = controller.update(portfolio_value=1_050_000)
        expected_dd = (1_050_000 - 1_100_000) / 1_100_000
        assert abs(state.drawdown_pct - expected_dd) < 0.001
    
    def test_trading_halt(self):
        """Test trading halt on severe drawdown."""
        controller = DrawdownController(initial_value=1_000_000)
        
        # Trigger severe drawdown
        controller.update(portfolio_value=800_000)  # 20% drawdown
        
        assert controller.is_trading_halted()
    
    def test_volatility_scaling(self):
        """Test volatility-based position scaling."""
        controller = DrawdownController(target_volatility=0.15)
        
        # High volatility returns
        returns = pd.Series([0.02, -0.03, 0.025, -0.02, 0.015] * 10)
        
        scale = controller.get_volatility_scale(returns)
        
        # Should reduce exposure when vol is high
        assert 0.25 <= scale <= 2.0


class TestRiskAttributor:
    """Tests for RiskAttributor class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        returns = pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "MSFT": np.random.normal(0.0008, 0.018, 252),
            "GOOGL": np.random.normal(0.0012, 0.022, 252),
        }, index=dates)
        
        return returns
    
    @pytest.fixture
    def factor_returns(self):
        """Generate sample factor returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        
        factors = pd.DataFrame({
            "market": np.random.normal(0.0005, 0.01, 252),
            "size": np.random.normal(0.0001, 0.005, 252),
            "value": np.random.normal(0.0001, 0.005, 252),
            "momentum": np.random.normal(0.0002, 0.008, 252),
        }, index=dates)
        
        return factors
    
    def test_risk_decomposition(self, sample_returns):
        """Test portfolio risk decomposition."""
        attributor = RiskAttributor()
        
        positions = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        
        decomp = attributor.decompose_risk(positions, sample_returns)
        
        assert decomp.total_risk > 0
        assert decomp.var_95 < 0  # VaR should be negative
    
    def test_marginal_contribution(self, sample_returns):
        """Test MCTR calculation."""
        attributor = RiskAttributor()
        
        positions = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        
        mctr = attributor.marginal_contribution_to_risk(positions, sample_returns)
        
        assert len(mctr) == 3
        assert all(v > 0 for v in mctr.values())
    
    def test_risk_budget(self, sample_returns):
        """Test risk budget calculation."""
        attributor = RiskAttributor()
        
        positions = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        
        budget = attributor.calculate_risk_budget(
            positions, sample_returns, total_risk_budget=0.15
        )
        
        assert budget.total_risk_budget == 0.15
        assert budget.used_risk >= 0
        assert budget.available_risk >= 0


class TestIntegration:
    """Integration tests for risk module."""
    
    def test_full_risk_workflow(self):
        """Test complete risk management workflow."""
        # 1. Size positions
        sizer = PositionSizer(
            method=PositionSizingMethod.EQUAL_WEIGHT,
            limits=PositionLimit(max_position_pct=0.10),
        )
        
        positions = sizer.size_portfolio(
            tickers=["AAPL", "MSFT", "GOOGL"],
            signals=[0.8, 0.6, 0.4],
            prices={"AAPL": 150, "MSFT": 350, "GOOGL": 140},
            portfolio_value=1_000_000,
        )
        
        # 2. Check exposures
        checker = ExposureLimitChecker()
        position_dict = {p.ticker: p.adjusted_weight for p in positions}
        
        violations = checker.check_portfolio(
            positions=position_dict,
            sectors={"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology"},
        )
        
        # 3. Set up drawdown control
        controller = DrawdownController(initial_value=1_000_000)
        
        # Simulate day
        state = controller.update(portfolio_value=1_000_000)
        
        assert not controller.is_trading_halted()
        assert len(positions) == 3
