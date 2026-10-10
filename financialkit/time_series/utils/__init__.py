"""Supporting numerics and synthetic data for financialkit.time_series's systems/ modules."""

from financialkit.time_series.utils.numerics import OLSResult, lag_matrix, numerical_hessian, ols
from financialkit.time_series.utils.synthetic import simulate_cointegrated_pair, simulate_dickey_fuller, simulate_intraday_returns, simulate_ohlc

__all__ = [
    "OLSResult",
    "ols",
    "lag_matrix",
    "numerical_hessian",
    "simulate_ohlc",
    "simulate_intraday_returns",
    "simulate_cointegrated_pair",
    "simulate_dickey_fuller",
]
