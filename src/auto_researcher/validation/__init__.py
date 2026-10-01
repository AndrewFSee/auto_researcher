"""
Validation package.

- ``cpcv`` and ``deflated_sharpe`` are always available (pure numpy/pandas).
- Pandera schemas live in ``schemas`` and are re-exported here only when
  the optional ``pandera`` dependency is installed. Consumers that need the
  schemas (``DataValidator``, ``TranscriptSchema``, etc.) get a clear
  ``ImportError`` on access if pandera isn't present.
"""

from __future__ import annotations

try:
    import pandera  # noqa: F401

    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False


if HAS_PANDERA:
    from auto_researcher.validation.schemas import (  # noqa: F401
        DataValidator,
        EarlyAdopterSignalSchema,
        FactorReturnsSchema,
        HoldingsSchema,
        PositionSchema,
        PriceSchema,
        ReturnsSchema,
        SignalSchema,
        ThematicScoreSchema,
        TradeSchema,
        TranscriptChunkSchema,
        TranscriptSchema,
        check_data_quality,
        validate_dataframe,
    )

    __all__ = [
        "DataValidator",
        "EarlyAdopterSignalSchema",
        "FactorReturnsSchema",
        "HAS_PANDERA",
        "HoldingsSchema",
        "PositionSchema",
        "PriceSchema",
        "ReturnsSchema",
        "SignalSchema",
        "ThematicScoreSchema",
        "TradeSchema",
        "TranscriptChunkSchema",
        "TranscriptSchema",
        "check_data_quality",
        "validate_dataframe",
    ]
else:
    __all__ = ["HAS_PANDERA"]
