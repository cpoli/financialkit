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
(NBER, 1969), 3-46. Corsi (2009) modelled realized variance as a cascade
of daily, weekly and monthly averages, an autoregression with only four
coefficients that reproduces the slowly decaying memory of volatility:
F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility,"
Journal of Financial Econometrics 7(2) (2009), 174-196; U. A. Müller et
al., Journal of Empirical Finance 4(2-3) (1997), 213-239.
"""

from __future__ import annotations

import math

import numpy as np

from financialkit.time_series.core.base import HARResult
from financialkit.time_series.utils.numerics import OLSResult, ols

__all__ = ["close_to_close_variance", "parkinson_variance", "realized_variance", "mincer_zarnowitz", "har_regressors", "fit_har"]

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
    It takes *returns* and is not annualized; for the annualized variance of
    a *price* path that a variance swap pays, use
    :func:`financialkit.exotic_options.annualized_realized_variance`.

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


def har_regressors(rv, horizons: tuple[int, ...] = (1, 5, 22)) -> np.ndarray:
    """Trailing means :math:`RV^{(h)}_t = \\frac1h\\sum_{j=0}^{h-1} RV_{t-j}` for each horizon `h`.

    Row ``i`` is date ``t = max(horizons) - 1 + i``.

    Returns
    -------
    ndarray, shape (n - max(horizons) + 1, len(horizons))

    Examples
    --------
    >>> har_regressors([1.0, 2.0, 3.0, 4.0], horizons=(1, 2)).tolist()
    [[2.0, 1.5], [3.0, 2.5], [4.0, 3.5]]
    """
    x = np.asarray(rv, dtype=np.float64)
    H = max(horizons)
    if x.size <= H:
        raise ValueError("series shorter than the longest horizon")
    c = np.r_[0.0, np.cumsum(x)]
    t = np.arange(H, x.size + 1)  # one past each date
    return np.column_stack([(c[t] - c[t - h]) / h for h in horizons])


def fit_har(rv, horizons: tuple[int, ...] = (1, 5, 22)) -> HARResult:
    """Corsi's (2009) HAR-RV model fitted by least squares.

    .. math:: RV_{t+1} = \\beta_0 + \\beta_d RV_t + \\beta_w RV^{(5)}_t + \\beta_m RV^{(22)}_t + e_{t+1}

    Traders acting on daily, weekly and monthly horizons (Müller et al.'s
    heterogeneous market hypothesis) each react to volatility at their own
    scale; the sum of three such components is an AR(22) with step-like
    weights that mimics long memory with four parameters.

    Parameters
    ----------
    rv : array_like, shape (n,)
        Realized variances (or their logs, or square roots).
    horizons : tuple of int
        Averaging windows.

    Returns
    -------
    HARResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> rv = np.empty(5000)
    >>> rv[:22] = 1.0
    >>> for t in range(22, 5000):
    ...     rv[t] = 0.1 + 0.4 * rv[t - 1] + 0.3 * rv[t - 5 : t].mean() + 0.2 * rv[t - 22 : t].mean() + 0.1 * rng.standard_normal()
    >>> fit_har(rv).coef.round(1).tolist()
    [0.1, 0.4, 0.3, 0.2]
    """
    x = np.asarray(rv, dtype=np.float64)
    hz = tuple(int(h) for h in horizons)
    X = har_regressors(x, hz)[:-1]
    res = ols(x[max(hz) :], np.column_stack([np.ones(X.shape[0]), X]))
    return HARResult(coef=res.coef, horizons=hz, ols=res, data=x)
