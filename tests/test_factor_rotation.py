"""
Tests for FactorRotationModel — regime detection and factor weight adjustment.

Tests cover:
1. Basic instantiation and configuration
2. Regime classification from composite scores
3. Factor weight computation (blending profiles with confidence)
4. IC weight adjustment pipeline
5. Graceful degradation when market data is unavailable
6. Factor momentum overlay
7. Transition detection
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

from auto_researcher.models.factor_rotation import (
    FactorRotationModel,
    FactorRegime,
    FactorRegimeState,
    LeadingIndicators,
    REGIME_FACTOR_PROFILES,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model():
    """Basic factor rotation model."""
    return FactorRotationModel(tilt_strength=0.5, verbose=True)


@pytest.fixture
def base_ics():
    """Standard IC weights to adjust."""
    return {
        "ml": 0.15,
        "sentiment": 0.08,
        "fundamental": 0.10,
        "earnings": 0.12,
        "insider": 0.06,
        "thematic": 0.05,
        "momentum": 0.07,
        "filing_tone": 0.04,
        "earnings_call_qual": 0.05,
    }


@pytest.fixture
def risk_on_indicators():
    """Indicators consistent with risk-on environment."""
    ind = LeadingIndicators()
    ind.vix_spot = 14.0
    ind.vix_1m = 13.5
    ind.vix_2m = 15.0
    ind.vix_term_ratio = 0.90  # Deep contango
    ind.vix_term_ratio_5d_chg = -0.02  # Contango deepening
    ind.hyg_ief_ratio = 0.82
    ind.hyg_ief_ratio_1m_chg = 0.015  # Spreads tightening
    ind.hyg_ief_ratio_3m_chg = 0.025
    ind.overall_breadth = 0.65  # Strong breadth
    ind.indicators_available = 3
    return ind


@pytest.fixture
def risk_off_indicators():
    """Indicators consistent with risk-off environment."""
    ind = LeadingIndicators()
    ind.vix_spot = 28.0
    ind.vix_1m = 30.0
    ind.vix_2m = 27.0
    ind.vix_term_ratio = 1.11  # Backwardation
    ind.vix_term_ratio_5d_chg = 0.04  # Getting worse
    ind.hyg_ief_ratio = 0.75
    ind.hyg_ief_ratio_1m_chg = -0.02  # Spreads widening
    ind.hyg_ief_ratio_3m_chg = -0.04
    ind.overall_breadth = 0.30  # Weak breadth
    ind.indicators_available = 3
    return ind


# =============================================================================
# Tests: Instantiation
# =============================================================================

class TestInstantiation:
    def test_default_params(self):
        m = FactorRotationModel()
        assert m.tilt_strength == 0.5
        assert m.use_factor_momentum is True

    def test_custom_tilt_strength(self):
        m = FactorRotationModel(tilt_strength=0.8)
        assert m.tilt_strength == 0.8

    def test_tilt_strength_clamped(self):
        m = FactorRotationModel(tilt_strength=1.5)
        assert m.tilt_strength == 1.0
        m2 = FactorRotationModel(tilt_strength=-0.3)
        assert m2.tilt_strength == 0.0


# =============================================================================
# Tests: Regime Classification
# =============================================================================

class TestRegimeClassification:
    def test_strong_risk_on(self, model):
        regime, conf, trans_prob, trans_dir = model._classify_regime(
            0.6, LeadingIndicators(indicators_available=4)
        )
        assert regime == FactorRegime.RISK_ON_LATE

    def test_mild_risk_on(self, model):
        regime, *_ = model._classify_regime(
            0.25, LeadingIndicators(indicators_available=3)
        )
        assert regime == FactorRegime.RISK_ON_EARLY

    def test_neutral(self, model):
        regime, *_ = model._classify_regime(
            0.05, LeadingIndicators(indicators_available=3)
        )
        assert regime == FactorRegime.NEUTRAL

    def test_mild_risk_off(self, model):
        regime, *_ = model._classify_regime(
            -0.25, LeadingIndicators(indicators_available=3)
        )
        assert regime == FactorRegime.RISK_OFF_EARLY

    def test_strong_risk_off(self, model):
        regime, *_ = model._classify_regime(
            -0.6, LeadingIndicators(indicators_available=4)
        )
        assert regime == FactorRegime.RISK_OFF_LATE

    def test_transition_detection_risk_on_to_off(self, model):
        """Risk-on regime but VIX contango flattening → transition warning."""
        ind = LeadingIndicators(
            indicators_available=3,
            vix_term_ratio_5d_chg=0.05,  # Contango flattening
        )
        _, _, trans_prob, trans_dir = model._classify_regime(0.3, ind)
        assert trans_prob > 0
        assert trans_dir == "toward_risk_off"

    def test_transition_detection_risk_off_to_on(self, model):
        """Risk-off regime but VIX backwardation resolving → transition warning."""
        ind = LeadingIndicators(
            indicators_available=3,
            vix_term_ratio_5d_chg=-0.05,  # Backwardation resolving
        )
        _, _, trans_prob, trans_dir = model._classify_regime(-0.3, ind)
        assert trans_prob > 0
        assert trans_dir == "toward_risk_on"


# =============================================================================
# Tests: Factor Weight Profiles
# =============================================================================

class TestFactorWeightProfiles:
    def test_all_regimes_have_profiles(self):
        for regime in FactorRegime:
            assert regime in REGIME_FACTOR_PROFILES
            profile = REGIME_FACTOR_PROFILES[regime]
            assert "fundamental" in profile
            assert "momentum" in profile
            assert "ml" in profile

    def test_neutral_profile_is_all_ones(self):
        profile = REGIME_FACTOR_PROFILES[FactorRegime.NEUTRAL]
        for factor, weight in profile.items():
            assert weight == 1.0, f"{factor} should be 1.0 in neutral"

    def test_risk_on_early_boosts_fundamentals(self):
        profile = REGIME_FACTOR_PROFILES[FactorRegime.RISK_ON_EARLY]
        assert profile["fundamental"] > 1.0
        assert profile["momentum"] < 1.0

    def test_risk_on_late_boosts_momentum(self):
        profile = REGIME_FACTOR_PROFILES[FactorRegime.RISK_ON_LATE]
        assert profile["momentum"] > 1.0
        assert profile["fundamental"] < 1.0

    def test_risk_off_early_boosts_quality(self):
        profile = REGIME_FACTOR_PROFILES[FactorRegime.RISK_OFF_EARLY]
        assert profile["fundamental"] > 1.0
        assert profile["earnings"] > 1.0
        assert profile["momentum"] < 1.0

    def test_risk_off_late_avoids_momentum(self):
        profile = REGIME_FACTOR_PROFILES[FactorRegime.RISK_OFF_LATE]
        assert profile["momentum"] < 0.5
        assert profile["thematic"] < 1.0


# =============================================================================
# Tests: Factor Weight Computation
# =============================================================================

class TestFactorWeightComputation:
    def test_neutral_regime_weights_near_one(self, model):
        weights = model._compute_factor_weights(FactorRegime.NEUTRAL, 1.0)
        for factor, w in weights.items():
            assert abs(w - 1.0) < 0.01, f"{factor} should be ~1.0 in neutral"

    def test_zero_confidence_gives_neutral_weights(self, model):
        weights = model._compute_factor_weights(FactorRegime.RISK_ON_LATE, 0.0)
        for factor, w in weights.items():
            assert abs(w - 1.0) < 0.01, f"{factor} should be ~1.0 with 0 confidence"

    def test_tilt_strength_zero_gives_neutral(self):
        m = FactorRotationModel(tilt_strength=0.0)
        weights = m._compute_factor_weights(FactorRegime.RISK_ON_LATE, 1.0)
        for factor, w in weights.items():
            assert abs(w - 1.0) < 0.01

    def test_full_tilt_full_confidence_matches_profile(self):
        m = FactorRotationModel(tilt_strength=1.0)
        weights = m._compute_factor_weights(FactorRegime.RISK_ON_LATE, 1.0)
        profile = REGIME_FACTOR_PROFILES[FactorRegime.RISK_ON_LATE]
        for factor in profile:
            assert abs(weights[factor] - profile[factor]) < 0.01


# =============================================================================
# Tests: IC Adjustment
# =============================================================================

class TestICAdjustment:
    def test_ml_weight_unchanged(self, model, base_ics):
        """ML weight should never be adjusted (already regime-aware)."""
        state = FactorRegimeState(
            regime=FactorRegime.RISK_ON_LATE,
            regime_confidence=0.8,
            factor_weights={"ml": 1.0, "momentum": 1.5, "fundamental": 0.8},
        )
        adjusted = model.adjust_ic_weights(base_ics, state)
        assert adjusted["ml"] == base_ics["ml"]

    def test_risk_on_late_boosts_momentum_ic(self, model, base_ics):
        state = FactorRegimeState(
            regime=FactorRegime.RISK_ON_LATE,
            regime_confidence=0.9,
            factor_weights={
                "ml": 1.0, "sentiment": 1.2, "fundamental": 0.8,
                "earnings": 0.9, "insider": 0.7, "thematic": 1.3,
                "momentum": 1.5, "filing_tone": 0.8, "earnings_call_qual": 0.9,
            },
        )
        adjusted = model.adjust_ic_weights(base_ics, state)
        assert adjusted["momentum"] > base_ics["momentum"]
        assert adjusted["fundamental"] < base_ics["fundamental"]

    def test_missing_factor_in_weights_uses_base(self, model, base_ics):
        """Factors not in factor_weights should use base IC unchanged."""
        state = FactorRegimeState(
            regime=FactorRegime.NEUTRAL,
            factor_weights={"momentum": 1.2},  # Only one factor
        )
        adjusted = model.adjust_ic_weights(base_ics, state)
        assert adjusted["fundamental"] == base_ics["fundamental"]


# =============================================================================
# Tests: Signal Scoring
# =============================================================================

class TestSignalScoring:
    def test_vix_contango_is_risk_on(self, model):
        ind = LeadingIndicators(vix_term_ratio=0.88, vix_term_ratio_5d_chg=-0.01)
        signal = model._score_vix_term_structure(ind)
        assert signal is not None
        assert signal > 0  # Risk-on

    def test_vix_backwardation_is_risk_off(self, model):
        ind = LeadingIndicators(vix_term_ratio=1.10, vix_term_ratio_5d_chg=0.02)
        signal = model._score_vix_term_structure(ind)
        assert signal is not None
        assert signal < 0  # Risk-off

    def test_vix_none_returns_none(self, model):
        ind = LeadingIndicators()
        assert model._score_vix_term_structure(ind) is None

    def test_credit_tightening_is_risk_on(self, model):
        ind = LeadingIndicators(hyg_ief_ratio_1m_chg=0.02, hyg_ief_ratio_3m_chg=0.03)
        signal = model._score_credit_spreads(ind)
        assert signal is not None
        assert signal > 0

    def test_credit_widening_is_risk_off(self, model):
        ind = LeadingIndicators(hyg_ief_ratio_1m_chg=-0.02, hyg_ief_ratio_3m_chg=-0.03)
        signal = model._score_credit_spreads(ind)
        assert signal is not None
        assert signal < 0

    def test_breadth_high_is_risk_on(self, model):
        ind = LeadingIndicators(overall_breadth=0.70)
        signal = model._score_breadth(ind)
        assert signal is not None
        assert signal > 0

    def test_breadth_low_is_risk_off(self, model):
        ind = LeadingIndicators(overall_breadth=0.30)
        signal = model._score_breadth(ind)
        assert signal is not None
        assert signal < 0

    def test_signals_bounded(self, model, risk_on_indicators):
        """All signals should be in [-1, 1]."""
        for scorer in [
            model._score_vix_term_structure,
            model._score_credit_spreads,
            model._score_breadth,
        ]:
            signal = scorer(risk_on_indicators)
            if signal is not None:
                assert -1.0 <= signal <= 1.0


# =============================================================================
# Tests: Factor Momentum Overlay
# =============================================================================

class TestFactorMomentum:
    def test_rising_ic_boosts_weight(self, model):
        weights = {"momentum": 1.0, "fundamental": 1.0}
        fm = {"momentum": 0.5, "fundamental": -0.3}  # Momentum IC rising
        adjusted = model._apply_factor_momentum(weights, fm)
        assert adjusted["momentum"] > 1.0
        assert adjusted["fundamental"] < 1.0

    def test_empty_momentum_no_change(self, model):
        weights = {"momentum": 1.2, "fundamental": 0.9}
        adjusted = model._apply_factor_momentum(weights, {})
        assert adjusted == weights


# =============================================================================
# Tests: Graceful Degradation
# =============================================================================

class TestGracefulDegradation:
    @patch.object(FactorRotationModel, '_fetch_vix_data', return_value=None)
    @patch.object(FactorRotationModel, '_fetch_credit_data', return_value=None)
    @patch.object(FactorRotationModel, '_fetch_dispersion_data', return_value=None)
    @patch.object(FactorRotationModel, '_load_factor_momentum', return_value=None)
    def test_no_market_data_returns_neutral(self, *mocks):
        """With no market data, should return NEUTRAL with low confidence."""
        m = FactorRotationModel()
        state = m.detect_regime(rankings=None)
        assert state.regime == FactorRegime.NEUTRAL
        assert state.regime_confidence < 0.5

    @patch.object(FactorRotationModel, '_fetch_vix_data', return_value=None)
    @patch.object(FactorRotationModel, '_fetch_credit_data', return_value=None)
    @patch.object(FactorRotationModel, '_fetch_dispersion_data', return_value=None)
    @patch.object(FactorRotationModel, '_load_factor_momentum', return_value=None)
    def test_no_data_weights_near_neutral(self, *mocks):
        """With no data, IC adjustments should be minimal."""
        m = FactorRotationModel()
        base_ics = {"fundamental": 0.10, "momentum": 0.07, "ml": 0.15}
        adjusted = m.adjust_ic_weights(base_ics, rankings=None)
        # Should be very close to original (neutral)
        for k in base_ics:
            assert abs(adjusted[k] - base_ics[k]) < 0.02
