"""Value at risk: historical, parametric, Cornish-Fisher, Monte Carlo,
age-weighted (hybrid) and filtered historical simulation.

Every function takes profits and losses (losses negative) and reports
the VaR as a positive loss at confidence level ``alpha``. The scenario
estimators return a :class:`~financialkit.risk.core.base.VaRResult`
holding the scenarios and their weights, and all of them read the VaR
and ES off :func:`~financialkit.risk.utils.tail.weighted_tail`, so they
agree exactly in their common limits (``lam = 1`` in the hybrid method
is plain historical simulation).

References: J.P. Morgan/Reuters, *RiskMetrics -- Technical Document*,
4th ed. (1996); E. A. Cornish and R. A. Fisher, *Rev. Int. Stat. Inst.*
5 (1937) 307-320; M. Pritsker, *J. Financial Services Research* 12
(1997) 201-242; J. Boudoukh, M. Richardson and R. Whitelaw, *Risk* 11
(1998) 64-67; G. Barone-Adesi, K. Giannopoulos and L. Vosper, *J.
Futures Markets* 19 (1999) 583-602; Hull, *Options, Futures, and Other
Derivatives*, 10th ed., Ch. 22.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from scipy import stats

from financialkit._deprecation import renamed_parameter
from financialkit.integrators import RNGLike, correlated_increments
from financialkit.risk.core.base import VaRResult
from financialkit.risk.utils.tail import check_alpha, weighted_tail

__all__ = [
    "historical_var",
    "rolling_historical_var",
    "parametric_var",
    "delta_normal_var",
    "cornish_fisher_var",
    "monte_carlo_var",
    "hybrid_var",
    "filtered_historical_var",
]


def _as_output(x: ArrayLike) -> np.ndarray | float:
    x = np.asarray(x, dtype=np.float64)
    return float(x) if x.ndim == 0 else x


def _scenario_result(pnl: np.ndarray, alpha: float, method: str, weights: np.ndarray | None = None) -> VaRResult:
    w = np.full(pnl.size, 1.0 / pnl.size) if weights is None else weights / weights.sum()
    var, es = weighted_tail(pnl, alpha, w)
    return VaRResult(var=var, es=es, alpha=alpha, method=method, scenarios=pnl, weights=w)


def historical_var(pnl: ArrayLike, alpha: float = 0.99) -> VaRResult:
    """Historical-simulation VaR and ES: past outcomes, equally likely.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Historical profits and losses (or returns).
    alpha : float
        Confidence level.

    Returns
    -------
    VaRResult

    Examples
    --------
    With 100 outcomes the 95% VaR is the fifth-worst loss.

    >>> import numpy as np
    >>> historical_var(-np.arange(1.0, 101.0), 0.95).var
    96.0
    """
    alpha = check_alpha(alpha)
    return _scenario_result(np.asarray(pnl, dtype=np.float64).ravel(), alpha, "historical")


def rolling_historical_var(pnl: ArrayLike, window: int = 250, alpha: float = 0.99) -> np.ndarray:
    """Out-of-sample historical VaR forecasts from a rolling window.

    The forecast for day ``t`` uses days ``t - window`` to ``t - 1`` only,
    as in the Basel Committee's (1996) backtesting framework.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Daily profits and losses in time order.
    window : int
        Look-back length in days.
    alpha : float
        Confidence level.

    Returns
    -------
    ndarray, shape (n - window,)
        VaR forecasts for days ``window, ..., n - 1``.
    """
    p = 1.0 - check_alpha(alpha)
    x = np.asarray(pnl, dtype=np.float64).ravel()
    if not 1 <= window < x.size:
        raise ValueError(f"window must lie in [1, {x.size - 1}], got {window}")
    windows = np.sort(sliding_window_view(x[:-1], window), axis=1)
    # The same order statistic weighted_tail picks for equal weights.
    k = max(int(np.ceil(window * p * (1.0 - 1e-12))) - 1, 0)
    return -windows[:, k]


def parametric_var(
    mu: ArrayLike, sigma: ArrayLike, alpha: ArrayLike = 0.99, horizon: ArrayLike = 1.0, distribution: str = "normal", df: float | None = None
) -> np.ndarray | float:
    """Parametric VaR of a normal or Student-t profit and loss.

    :math:`\\mathrm{VaR} = -\\mu h + \\sigma \\sqrt{h}\\, q_\\alpha`, with
    :math:`q_\\alpha` the standard normal quantile, or the Student-t
    quantile rescaled to unit variance, :math:`t_\\nu^{-1}(\\alpha)\\sqrt{(\\nu - 2)/\\nu}`.

    Parameters
    ----------
    mu, sigma : array_like
        Mean and standard deviation of the one-period profit and loss.
    alpha : array_like
        Confidence level.
    horizon : array_like
        Number of periods; the mean scales with ``h`` and the volatility
        with ``sqrt(h)`` (independent increments).
    distribution : {"normal", "t"}
    df : float, optional
        Degrees of freedom (> 2) for ``distribution="t"``.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Ch. 22: a $10 million position with a 2% daily
    volatility has a one-day 99% VaR of about $465,300 and a ten-day VaR
    of about $1.47 million.

    >>> round(parametric_var(0.0, 200_000.0, 0.99))
    465270
    >>> round(parametric_var(0.0, 200_000.0, 0.99, horizon=10))
    1471312
    """
    mu, sigma, alpha, h = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, alpha, horizon))
    return _as_output(-mu * h + sigma * np.sqrt(h) * _std_quantile(alpha, distribution, df))


def _std_quantile(alpha: np.ndarray, distribution: str, df: float | None) -> np.ndarray:
    if np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("alpha must lie in (0, 1)")
    if distribution == "normal":
        return stats.norm.ppf(alpha)
    if distribution == "t":
        if df is None or df <= 2:
            raise ValueError("distribution='t' needs df > 2 so the variance is finite")
        return stats.t.ppf(alpha, df) * np.sqrt((df - 2.0) / df)
    raise ValueError(f"distribution must be 'normal' or 't', got {distribution!r}")


def delta_normal_var(exposures: ArrayLike, cov: ArrayLike, alpha: float = 0.99, horizon: float = 1.0) -> float:
    """RiskMetrics delta-normal portfolio VaR, :math:`q_\\alpha \\sqrt{h\\, w^\\top \\Sigma w}`.

    Parameters
    ----------
    exposures : array_like, shape (d,)
        Money exposure to each risk factor (position value, or delta
        times spot).
    cov : array_like, shape (d, d)
        Covariance of the one-period factor returns.
    alpha : float
        Confidence level.
    horizon : float
        Number of periods.

    Returns
    -------
    float

    Examples
    --------
    Hull, 10th ed., Ch. 22: $10 million in Microsoft (2% daily vol) and
    $5 million in AT&T (1%), correlation 0.3. The ten-day 99% VaR is
    about $1.62 million, against $1.83 million if the two VaRs are added.

    >>> import numpy as np
    >>> cov = np.array([[0.02**2, 0.3 * 0.02 * 0.01], [0.3 * 0.02 * 0.01, 0.01**2]])
    >>> round(delta_normal_var([10e6, 5e6], cov, 0.99, horizon=10))
    1620114
    """
    w = np.asarray(exposures, dtype=np.float64)
    sigma_p = float(np.sqrt(w @ np.asarray(cov, dtype=np.float64) @ w))
    return float(stats.norm.ppf(check_alpha(alpha)) * np.sqrt(horizon) * sigma_p)


def cornish_fisher_var(mu: ArrayLike, sigma: ArrayLike, skew: ArrayLike, excess_kurtosis: ArrayLike, alpha: ArrayLike = 0.99) -> np.ndarray | float:
    """VaR from the Cornish-Fisher expansion of the quantile in skewness and kurtosis.

    With :math:`z = \\Phi^{-1}(1 - \\alpha)`, skewness :math:`S` and excess
    kurtosis :math:`K`, the adjusted quantile is

    .. math::

       z_{cf} = z + \\frac{(z^2 - 1) S}{6} + \\frac{(z^3 - 3z) K}{24} - \\frac{(2z^3 - 5z) S^2}{36},

    and :math:`\\mathrm{VaR} = -(\\mu + \\sigma z_{cf})`.

    Parameters
    ----------
    mu, sigma, skew, excess_kurtosis : array_like
        First four moments of the profit and loss.
    alpha : array_like
        Confidence level.

    Returns
    -------
    float or ndarray

    Examples
    --------
    With no skew and no excess kurtosis it is the normal VaR.

    >>> round(cornish_fisher_var(0.0, 1.0, 0.0, 0.0, 0.99), 4)
    2.3263
    >>> round(cornish_fisher_var(0.0, 1.0, -0.5, 3.0, 0.99), 4)
    3.3013
    """
    mu, sigma, S, K, alpha = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, skew, excess_kurtosis, alpha))
    z = stats.norm.ppf(1.0 - alpha)
    z_cf = z + (z**2 - 1.0) * S / 6.0 + (z**3 - 3.0 * z) * K / 24.0 - (2.0 * z**3 - 5.0 * z) * S**2 / 36.0
    return _as_output(-(mu + sigma * z_cf))


@renamed_parameter("ci_level", "confidence", "financialkit.risk.monte_carlo_var")
def monte_carlo_var(
    pnl_fn: Callable[[np.ndarray], ArrayLike],
    cov: ArrayLike,
    alpha: float = 0.99,
    mean: ArrayLike | None = None,
    horizon: float = 1.0,
    n_paths: int = 10_000,
    confidence: float = 0.95,
    rng: RNGLike = None,
) -> VaRResult:
    """Full-revaluation Monte Carlo VaR (Pritsker 1997).

    Draws joint normal risk-factor moves
    :math:`\\Delta X \\sim N(\\mu h, \\Sigma h)` through
    :func:`financialkit.integrators.correlated_increments`, revalues the
    portfolio in every scenario with ``pnl_fn`` and reads the VaR and ES
    off the simulated distribution, with an exact order-statistic
    confidence interval for the VaR.

    Parameters
    ----------
    pnl_fn : callable
        ``pnl_fn(dX) -> pnl`` maps factor moves of shape ``(n_paths, d)``
        to the portfolio's profit and loss, shape ``(n_paths,)``.
    cov : array_like, shape (d, d)
        Covariance of the one-period factor moves.
    alpha : float
        Confidence level.
    mean : array_like, shape (d,), optional
        Mean one-period factor move; zero if omitted.
    horizon : float
        Number of periods.
    n_paths : int
        Number of scenarios.
    confidence : float
        Coverage of the confidence interval stored in ``result.ci``.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    VaRResult
    """
    alpha = check_alpha(alpha)
    c = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    sd = np.sqrt(np.diag(c))
    dX = correlated_increments(c / np.outer(sd, sd), T=horizon, n_steps=1, n_paths=n_paths, rng=rng)[:, 0, :] * sd
    if mean is not None:
        dX = dX + np.asarray(mean, dtype=np.float64) * horizon
    pnl = np.asarray(pnl_fn(dX), dtype=np.float64).ravel()
    if pnl.size != n_paths:
        raise ValueError(f"pnl_fn must return one value per scenario ({n_paths}), got {pnl.size}")
    res = _scenario_result(pnl, alpha, "monte_carlo")
    # The k-th order statistic lies below the p-quantile with probability
    # P(Binomial(n, p) >= k), which gives a distribution-free interval.
    p, x = 1.0 - alpha, np.sort(pnl)
    lo = int(stats.binom.ppf((1.0 - confidence) / 2.0, n_paths, p))
    hi = int(stats.binom.ppf((1.0 + confidence) / 2.0, n_paths, p))
    res.ci = (float(-x[min(hi, n_paths - 1)]), float(-x[max(lo - 1, 0)]))
    return res


def hybrid_var(pnl: ArrayLike, alpha: float = 0.99, lam: float = 0.98) -> VaRResult:
    """Boudoukh-Richardson-Whitelaw age-weighted historical simulation.

    The outcome ``i`` days old gets probability proportional to
    :math:`\\lambda^i`, so recent days count more and the VaR reacts to
    a change in volatility within weeks instead of a whole window;
    ``lam = 1`` is plain historical simulation.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Profits and losses in time order, most recent last.
    alpha : float
        Confidence level.
    lam : float
        Decay factor in ``(0, 1]``.

    Returns
    -------
    VaRResult
    """
    alpha = check_alpha(alpha)
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lam must lie in (0, 1], got {lam}")
    x = np.asarray(pnl, dtype=np.float64).ravel()
    age = np.arange(x.size - 1, -1, -1, dtype=np.float64)
    return _scenario_result(x, alpha, "hybrid", lam**age)


def filtered_historical_var(pnl: ArrayLike, sigma: ArrayLike, sigma_forecast: float, alpha: float = 0.99) -> VaRResult:
    """Filtered historical simulation (Barone-Adesi, Giannopoulos and Vosper 1999).

    Each past outcome is divided by the volatility of its day, giving
    roughly i.i.d. standardized residuals :math:`z_t = x_t / \\sigma_t`,
    and rescaled by tomorrow's forecast: the scenarios are
    :math:`\\hat\\sigma_{n+1} z_t`. The empirical shape of the tail is
    kept while its scale follows current volatility.

    Parameters
    ----------
    pnl : array_like, shape (n,)
        Historical profits and losses.
    sigma : array_like, shape (n,)
        Conditional volatility of each day, from any filter (EWMA, GARCH).
    sigma_forecast : float
        Volatility forecast for the day the VaR is for.
    alpha : float
        Confidence level.

    Returns
    -------
    VaRResult
    """
    alpha = check_alpha(alpha)
    x = np.asarray(pnl, dtype=np.float64).ravel()
    s = np.asarray(sigma, dtype=np.float64).ravel()
    if s.shape != x.shape or np.any(s <= 0):
        raise ValueError("sigma must be positive and match pnl")
    return _scenario_result(sigma_forecast * x / s, alpha, "filtered_historical")
