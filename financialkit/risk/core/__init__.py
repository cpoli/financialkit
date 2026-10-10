"""Abstract base classes and result containers for financialkit.risk."""

from financialkit.risk.core.base import (
    AllocationResult,
    BacktestResult,
    CoherenceResult,
    DrawdownResult,
    GEVResult,
    GPDResult,
    RiskMeasure,
    TrafficLightResult,
    VaRResult,
)

__all__ = [
    "VaRResult",
    "BacktestResult",
    "TrafficLightResult",
    "GEVResult",
    "GPDResult",
    "CoherenceResult",
    "AllocationResult",
    "DrawdownResult",
    "RiskMeasure",
]
