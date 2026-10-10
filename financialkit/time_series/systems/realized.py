"""Volatility measured from prices: Parkinson's high-low range and realized variance.

Parkinson (1980) showed that the daily range of the log price carries far
more information about volatility than the close-to-close return: for
Brownian motion, :math:`E[(\\ln H/L)^2] = 4\\ln 2\\,\\sigma^2`, and the
range estimator is about five times as efficient. Andersen and Bollerslev
(1998) argued that GARCH forecasts only *look* poor when judged against
squared daily returns, a very noisy proxy for the day's variance; summing
squared intraday returns (realized variance) removes most of the noise and
reveals accurate forecasts. References: M. Parkinson, "The Extreme Value
Method for Estimating the Variance of the Rate of Return," Journal of
Business 53(1) (1980), 61-65; T. G. Andersen and T. Bollerslev,
"Answering the Skeptics: Yes, Standard Volatility Models Do Provide
Accurate Forecasts," International Economic Review 39(4) (1998), 885-905;
J. Mincer and V. Zarnowitz, in *Economic Forecasts and Expectations*
(NBER, 1969), 3-46.
"""

from __future__ import annotations

import math

import numpy as np

from financialkit.time_series.utils.numerics import OLSResult, ols

__all__ = ["close_to_close_variance", "parkinson_variance", "realized_variance", "mincer_zarnowitz"]

_PARKINSON = 1.0 / (4.0 * math.log(2.0))


def close_to_close_variance(close) -> np.ndarray:
    """Per-period variance proxies :math:`r_t^2 = (\\ln C_t / C_{t-1})^2` (zero mean assumed).

    Examples
    --------
    >>> import numpy as np
    >>> close_to_close_variance([100.0, 100.0 * np.exp(0.01)]).round(10).tolist()
    [0.0001]
    """
    c = np.asarray(close, dtype=np.float64)
    return np.diff(np.log(c)) ** 2


def parkinson_variance(high, low) -> np.ndarray:
    """Parkinson's per-period variance estimates :math:`(\\ln H_t/L_t)^2 / (4\\ln 2)`.

    Unbiased for driftless Brownian motion observed continuously; average
    over periods (or smooth) to estimate a variance, and multiply by
    periods per year to annualize. Discrete sampling of the high and low
    biases it slightly downward.

    Examples
    --------
    >>> import numpy as np
    >>> float(parkinson_variance([np.exp(0.02)], [1.0])[0].round(8))
    0.00014427
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    if np.any(h < l):
        raise ValueError("high must be at least low")
    return _PARKINSON * np.log(h / l) ** 2


def realized_variance(intraday_returns, axis: int = -1) -> np.ndarray:
    """Realized variance :math:`RV_t = \\sum_i r_{t,i}^2`, summing intraday returns along `axis`.

    As the sampling interval shrinks, :math:`RV_t` converges to the
    integrated variance :math:`\\int_{t-1}^{t}\\sigma_s^2\\,ds` of the day.

    Examples
    --------
    >>> realized_variance([[0.01, -0.01], [0.02, 0.0]]).round(8).tolist()
    [0.0002, 0.0004]
    """
    r = np.asarray(intraday_returns, dtype=np.float64)
    return np.sum(r**2, axis=axis)


def mincer_zarnowitz(proxy, forecast) -> OLSResult:
    """Mincer-Zarnowitz regression :math:`\\text{proxy}_t = a + b\\,\\text{forecast}_t + e_t`.

    An unbiased forecast has :math:`a = 0, b = 1`; the :math:`R^2` measures
    how much of the proxy's variation the forecast explains. Andersen and
    Bollerslev (1998) showed that with :math:`r_t^2` as the proxy even a
    perfect variance forecast has a small :math:`R^2` (at most
    :math:`1/\\kappa` of the explainable part, :math:`\\kappa` the
    kurtosis of :math:`z_t`), and that realized variance restores it.

    Returns
    -------
    OLSResult
        Coefficients ``[a, b]`` and :attr:`~financialkit.time_series.utils.numerics.OLSResult.r_squared`.
    """
    y = np.asarray(proxy, dtype=np.float64)
    f = np.asarray(forecast, dtype=np.float64)
    return ols(y, np.column_stack([np.ones_like(f), f]))
