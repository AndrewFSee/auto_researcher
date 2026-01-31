"""
Data Validation Layer

Pandera schemas for institutional-grade data validation:
- Transcript data validation
- Price/returns data validation
- Signal/score validation
- Trade data validation
- Holdings validation
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import pandera as pa
from pandera import Column, Check, Index
from pandera.typing import DataFrame, Series
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TRANSCRIPT SCHEMAS
# =============================================================================

class TranscriptSchema(pa.DataFrameModel):
    """Schema for earnings transcript data."""
    
    ticker: Series[str] = pa.Field(
        nullable=False,
        str_matches=r"^[A-Z]{1,5}$",
        description="Stock ticker symbol"
    )
    
    date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        description="Transcript date"
    )
    
    quarter: Series[str] = pa.Field(
        nullable=False,
        str_matches=r"^Q[1-4]\s+\d{4}$",
        description="Fiscal quarter (e.g., 'Q1 2024')"
    )
    
    text: Series[str] = pa.Field(
        nullable=False,
        str_length={"min_value": 100},
        description="Full transcript text"
    )
    
    source: Series[str] = pa.Field(
        nullable=True,
        isin=["seeking_alpha", "refinitiv", "factset", "bloomberg", "internal"],
        description="Data source"
    )
    
    class Config:
        strict = True
        coerce = True


class TranscriptChunkSchema(pa.DataFrameModel):
    """Schema for chunked transcript segments."""
    
    ticker: Series[str] = pa.Field(nullable=False)
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    chunk_id: Series[int] = pa.Field(ge=0)
    chunk_text: Series[str] = pa.Field(str_length={"min_value": 10})
    speaker: Series[str] = pa.Field(nullable=True)
    section: Series[str] = pa.Field(
        nullable=True,
        isin=["prepared_remarks", "qa", "opening", "closing"]
    )
    
    class Config:
        coerce = True


# =============================================================================
# PRICE/RETURNS SCHEMAS
# =============================================================================

class PriceSchema(pa.DataFrameModel):
    """Schema for price data (OHLCV)."""
    
    date: Index[pd.Timestamp] = pa.Field(
        nullable=False,
        description="Trading date"
    )
    
    open: Series[float] = pa.Field(gt=0, nullable=True)
    high: Series[float] = pa.Field(gt=0, nullable=True)
    low: Series[float] = pa.Field(gt=0, nullable=True)
    close: Series[float] = pa.Field(gt=0, nullable=False)
    adj_close: Series[float] = pa.Field(gt=0, nullable=True)
    volume: Series[float] = pa.Field(ge=0, nullable=True)
    
    @pa.check("high")
    def high_gte_low(cls, high: Series[float], low: Series[float]) -> Series[bool]:
        """High must be >= Low."""
        return high >= low
    
    @pa.check("high")
    def high_gte_open_close(
        cls, 
        high: Series[float], 
        open: Series[float], 
        close: Series[float]
    ) -> Series[bool]:
        """High must be >= Open and Close."""
        return (high >= open) & (high >= close)
    
    class Config:
        coerce = True


class ReturnsSchema(pa.DataFrameModel):
    """Schema for returns data."""
    
    # Dynamic columns - returns for each ticker
    class Config:
        strict = False  # Allow dynamic columns
        coerce = True
    
    @pa.dataframe_check
    def returns_in_range(cls, df: pd.DataFrame) -> bool:
        """Returns should be within reasonable bounds (-1 to +10)."""
        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        for col in numeric_cols:
            if (df[col].min() < -1) or (df[col].max() > 10):
                logger.warning(f"Returns for {col} outside expected range")
                return False
        return True


class FactorReturnsSchema(pa.DataFrameModel):
    """Schema for factor returns data."""
    
    date: Index[pd.Timestamp] = pa.Field(nullable=False)
    
    market: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": -0.25, "max_value": 0.25},
        description="Market factor return"
    )
    
    size: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": -0.15, "max_value": 0.15},
        description="Size factor (SMB)"
    )
    
    value: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": -0.15, "max_value": 0.15},
        description="Value factor (HML)"
    )
    
    momentum: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": -0.20, "max_value": 0.20},
        description="Momentum factor (WML)"
    )
    
    class Config:
        coerce = True


# =============================================================================
# SIGNAL SCHEMAS
# =============================================================================

class SignalSchema(pa.DataFrameModel):
    """Schema for model signals/scores."""
    
    ticker: Series[str] = pa.Field(nullable=False)
    
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    
    signal: Series[float] = pa.Field(
        in_range={"min_value": -1.0, "max_value": 1.0},
        description="Normalized signal strength"
    )
    
    raw_score: Series[float] = pa.Field(
        nullable=True,
        description="Raw model output before normalization"
    )
    
    model_name: Series[str] = pa.Field(nullable=False)
    
    confidence: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": 0, "max_value": 1},
        description="Model confidence"
    )
    
    class Config:
        coerce = True


class EarlyAdopterSignalSchema(pa.DataFrameModel):
    """Schema for Early Adopter Model output."""
    
    ticker: Series[str] = pa.Field(nullable=False)
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    
    signal: Series[str] = pa.Field(
        isin=["strong_buy", "buy", "neutral", "avoid"],
        description="Categorical signal"
    )
    
    pioneer_score: Series[float] = pa.Field(
        in_range={"min_value": 0, "max_value": 1},
        nullable=True
    )
    
    total_techs_adopted: Series[int] = pa.Field(ge=0, nullable=True)
    techs_adopted_early: Series[int] = pa.Field(ge=0, nullable=True)
    months_ahead_avg: Series[float] = pa.Field(nullable=True)
    
    class Config:
        coerce = True


class ThematicScoreSchema(pa.DataFrameModel):
    """Schema for thematic analysis scores."""
    
    ticker: Series[str] = pa.Field(nullable=False)
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    
    # Core scores (0-1 range)
    forward_score: Series[float] = pa.Field(
        in_range={"min_value": 0, "max_value": 1}
    )
    management_quality: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": 0, "max_value": 1}
    )
    competitive_position: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": 0, "max_value": 1}
    )
    
    # Theme exposures
    ai_exposure: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": 0, "max_value": 1}
    )
    esg_exposure: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": 0, "max_value": 1}
    )
    
    class Config:
        coerce = True


# =============================================================================
# TRADE/POSITION SCHEMAS
# =============================================================================

class TradeSchema(pa.DataFrameModel):
    """Schema for trade records."""
    
    trade_id: Series[str] = pa.Field(nullable=False, unique=True)
    
    timestamp: Series[pd.Timestamp] = pa.Field(nullable=False)
    
    ticker: Series[str] = pa.Field(nullable=False)
    
    side: Series[str] = pa.Field(
        isin=["buy", "sell", "short", "cover"]
    )
    
    quantity: Series[float] = pa.Field(gt=0)
    
    price: Series[float] = pa.Field(gt=0)
    
    notional: Series[float] = pa.Field(gt=0)
    
    order_type: Series[str] = pa.Field(
        nullable=True,
        isin=["market", "limit", "stop", "stop_limit"]
    )
    
    signal_source: Series[str] = pa.Field(nullable=True)
    
    signal_strength: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": -1, "max_value": 1}
    )
    
    class Config:
        coerce = True


class PositionSchema(pa.DataFrameModel):
    """Schema for position records."""
    
    ticker: Series[str] = pa.Field(nullable=False)
    
    quantity: Series[float] = pa.Field(description="Shares held (negative for short)")
    
    cost_basis: Series[float] = pa.Field(gt=0)
    
    market_value: Series[float] = pa.Field()
    
    weight: Series[float] = pa.Field(
        in_range={"min_value": -1, "max_value": 1},
        description="Portfolio weight"
    )
    
    unrealized_pnl: Series[float] = pa.Field(nullable=True)
    
    unrealized_pnl_pct: Series[float] = pa.Field(nullable=True)
    
    sector: Series[str] = pa.Field(nullable=True)
    
    class Config:
        coerce = True


class HoldingsSchema(pa.DataFrameModel):
    """Schema for portfolio holdings snapshot."""
    
    date: Index[pd.Timestamp] = pa.Field(nullable=False)
    
    # Dynamic columns for ticker weights
    class Config:
        strict = False
        coerce = True
    
    @pa.dataframe_check
    def weights_sum_to_one(cls, df: pd.DataFrame) -> bool:
        """Weights should approximately sum to 1 (or less for cash)."""
        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        weight_sums = df[numeric_cols].sum(axis=1)
        return all(weight_sums <= 1.1)  # Allow small tolerance


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class DataValidator:
    """
    Utility class for validating data against schemas.
    
    Usage:
        validator = DataValidator()
        
        # Validate with error handling
        is_valid, errors = validator.validate(df, "transcript")
        
        # Validate and raise on error
        validated_df = validator.validate_or_raise(df, "price")
    """
    
    SCHEMAS = {
        "transcript": TranscriptSchema,
        "transcript_chunk": TranscriptChunkSchema,
        "price": PriceSchema,
        "returns": ReturnsSchema,
        "factor_returns": FactorReturnsSchema,
        "signal": SignalSchema,
        "early_adopter": EarlyAdopterSignalSchema,
        "thematic_score": ThematicScoreSchema,
        "trade": TradeSchema,
        "position": PositionSchema,
        "holdings": HoldingsSchema,
    }
    
    def __init__(self, strict: bool = False):
        self.strict = strict
    
    def validate(
        self,
        df: pd.DataFrame,
        schema_name: str,
    ) -> tuple[bool, List[str]]:
        """
        Validate DataFrame against schema.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        schema = self.SCHEMAS.get(schema_name)
        if schema is None:
            return False, [f"Unknown schema: {schema_name}"]
        
        try:
            schema.validate(df, lazy=True)
            return True, []
        except pa.errors.SchemaErrors as e:
            errors = [str(err) for err in e.failure_cases.itertuples()]
            logger.warning(f"Validation failed for {schema_name}: {len(errors)} errors")
            return False, errors
    
    def validate_or_raise(
        self,
        df: pd.DataFrame,
        schema_name: str,
    ) -> pd.DataFrame:
        """
        Validate and return typed DataFrame, or raise on error.
        """
        schema = self.SCHEMAS.get(schema_name)
        if schema is None:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        return schema.validate(df)
    
    def coerce_and_validate(
        self,
        df: pd.DataFrame,
        schema_name: str,
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Attempt to coerce types and validate.
        
        Returns:
            Tuple of (coerced_df, warnings)
        """
        schema = self.SCHEMAS.get(schema_name)
        if schema is None:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        warnings = []
        
        try:
            # Attempt validation with coercion
            validated = schema.validate(df, lazy=True)
            return validated, warnings
        except pa.errors.SchemaErrors as e:
            # Log warnings but return original with attempted coercion
            for case in e.failure_cases.itertuples():
                warnings.append(str(case))
            
            logger.warning(f"Partial validation for {schema_name}: {len(warnings)} issues")
            return df, warnings


def validate_dataframe(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Convenience function for quick validation.
    
    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    validator = DataValidator()
    return validator.validate_or_raise(df, schema_name)


def check_data_quality(df: pd.DataFrame, name: str = "data") -> Dict[str, Any]:
    """
    Generate data quality report.
    
    Returns:
        Dict with quality metrics
    """
    report = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    
    # Numeric column stats
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64']).columns
    if len(numeric_cols) > 0:
        report["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # Check for duplicates
    report["duplicate_rows"] = df.duplicated().sum()
    
    # Date range if datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        report["date_range"] = {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "gaps": len(pd.date_range(df.index.min(), df.index.max(), freq='B')) - len(df)
        }
    
    return report
