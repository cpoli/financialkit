"""Abstract base classes and result containers for financialkit.time_series."""

from financialkit.time_series.core.base import (
    OHLC,
    ARMAResult,
    CointegrationResult,
    HurstResult,
    OrderSelectionResult,
    ReturnStatistics,
    TailIndexResult,
    TestResult,
    UnitRootResult,
    VarianceRatioResult,
    VolatilityFit,
    VolatilityModel,
    VolatilityPath,
)

__all__ = [
    "ReturnStatistics",
    "TailIndexResult",
    "TestResult",
    "VarianceRatioResult",
    "ARMAResult",
    "OrderSelectionResult",
    "UnitRootResult",
    "CointegrationResult",
    "HurstResult",
    "OHLC",
    "VolatilityPath",
    "VolatilityFit",
    "VolatilityModel",
]
