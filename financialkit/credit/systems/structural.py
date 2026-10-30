"""Merton's structural model, its calibration to equity, and Crosbie and Bohn's distance to default.

Merton (1974) saw that shareholders of a firm with assets :math:`V`
following geometric Brownian motion and zero-coupon debt of face
:math:`D` hold a call on the assets: at maturity they repay :math:`D` if
:math:`V_T > D` and walk away otherwise. Equity is the Black-Scholes call
:math:`E = V N(d_1) - D e^{-rT} N(d_2)`, debt is :math:`V - E`, and the
risk-neutral default probability is :math:`N(-d_2)`. Asset value and
volatility are not observed, but equity and its volatility are; the two
equations

.. math:: E = C_{BS}(V, D, T, r, \\sigma_V), \\qquad \\sigma_E E = N(d_1)\\,\\sigma_V V

pin them down (Jones, Mason and Rosenfeld 1984). Moody's KMV (Crosbie and
Bohn 2003) replaced the second equation with an iteration on a daily
equity series, set the default point to short-term debt plus half the
long-term debt, and reported the physical distance to default

.. math:: DD = \\frac{\\ln(V / D) + (\\mu - \\sigma_V^2/2) T}{\\sigma_V \\sqrt{T}}.

References: R. C. Merton, "On the Pricing of Corporate Debt: The Risk
Structure of Interest Rates," *J. Finance* 29 (1974) 449-470; P. J.
Crosbie and J. R. Bohn, *Modeling Default Risk* (Moody's KMV, 2003);
J. C. Hull, *Options, Futures, and Other Derivatives*, 10th ed. (2018),
Section 24.6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from financialkit._deprecation import renamed_parameter
from financialkit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL, TRADING_DAYS_PER_YEAR
from financialkit.credit.core.base import KMVResult, MertonResult
from financialkit.credit.utils.discounting import zero_rate_to
from financialkit.fixed_income import CurveLike
from financialkit.options import black_scholes_price
from financialkit.options.utils.common import as_output

__all__ = ["merton_model", "merton_calibrate", "distance_to_default", "kmv_default_point", "kmv_iterate"]


def _d1_d2(V, D, T, r, sigma):
    sd = sigma * np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * sigma**2) * T) / sd
    return d1, d1 - sd


def merton_model(V: float, D: float, T: float, r: CurveLike, sigma_V: float) -> MertonResult:
    """Value equity and debt in Merton's (1974) model.

    Parameters
    ----------
    V : float
        Current asset value.
    D : float
        Face value of the zero-coupon debt.
    T : float
        Debt maturity in years.
    r : float or YieldCurve
        Risk-free rate, or a ``fixed_income`` discount curve, whose zero
        rate to ``T`` is used: with deterministic rates the debt, paid
        only at ``T``, sees the curve only through :math:`P(0, T)`.
    sigma_V : float
        Asset volatility.

    Returns
    -------
    MertonResult

    Examples
    --------
    Hull, 10th ed., Example 24.3: assets 12.40, volatility 21.23%, debt
    of 10 due in a year, 5% rate.

    >>> m = merton_model(12.3954, 10.0, 1.0, 0.05, 0.2123)
    >>> round(m.equity, 2), round(m.default_probability, 3)
    (3.0, 0.127)
    """
    if V <= 0 or D <= 0 or T <= 0 or sigma_V <= 0:
        raise ValueError("V, D, T and sigma_V must be positive")
    r = zero_rate_to(r, T)
    E = float(black_scholes_price(V, D, T, r, sigma_V))
    d1, d2 = _d1_d2(V, D, T, r, sigma_V)
    debt = V - E
    pd = float(stats.norm.cdf(-d2))
    recovery = float(V * np.exp(r * T) * stats.norm.cdf(-d1) / (D * pd)) if pd > 0 else float("nan")
    return MertonResult(
        equity=E,
        debt=debt,
        asset_value=float(V),
        asset_volatility=float(sigma_V),
        equity_volatility=float(stats.norm.cdf(d1) * sigma_V * V / E) if E > 0 else float("inf"),
        default_probability=pd,
        distance_to_default=float(d2),
        credit_spread=float(-np.log(debt / D) / T - r),
        recovery_rate=recovery,
    )


def merton_calibrate(E: float, sigma_E: float, D: float, T: float, r: CurveLike) -> MertonResult:
    """Solve for the asset value and volatility implied by equity and its volatility.

    Solves :math:`E = C_{BS}(V)` and :math:`\\sigma_E E = N(d_1)\\sigma_V V`
    with :func:`scipy.optimize.least_squares`.

    Parameters
    ----------
    E : float
        Market value of equity.
    sigma_E : float
        Equity volatility.
    D, T : float
        Face value and maturity of the debt.
    r : float or YieldCurve
        Risk-free rate, or a ``fixed_income`` discount curve, whose zero
        rate to ``T`` is used: with deterministic rates the debt, paid
        only at ``T``, sees the curve only through :math:`P(0, T)`.

    Returns
    -------
    MertonResult

    Examples
    --------
    Hull, 10th ed., Example 24.3:

    >>> m = merton_calibrate(3.0, 0.80, 10.0, 1.0, 0.05)
    >>> round(m.asset_value, 2), round(m.asset_volatility, 4), round(m.debt, 2)
    (12.4, 0.2123, 9.4)
    """
    if E <= 0 or sigma_E <= 0:
        raise ValueError("E and sigma_E must be positive")
    r = zero_rate_to(r, T)

    def residuals(x):
        V, s = np.exp(x)
        d1, _ = _d1_d2(V, D, T, r, s)
        return [float(black_scholes_price(V, D, T, r, s)) / E - 1.0, stats.norm.cdf(d1) * s * V / (sigma_E * E) - 1.0]

    V0 = E + D * np.exp(-r * T)
    sol = optimize.least_squares(residuals, np.log([V0, sigma_E * E / V0]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    V, s = np.exp(sol.x)
    return merton_model(float(V), D, T, r, float(s))


def distance_to_default(V: ArrayLike, D: ArrayLike, mu: ArrayLike, sigma_V: ArrayLike, T: ArrayLike = 1.0) -> np.ndarray | float:
    """Crosbie and Bohn's distance to default under the physical drift.

    .. math:: DD = \\frac{\\ln(V/D) + (\\mu - \\sigma_V^2/2) T}{\\sigma_V\\sqrt{T}}

    With ``mu = r`` it is Merton's :math:`d_2`.

    Parameters
    ----------
    V, D, mu, sigma_V, T : array_like
        Asset value, default point, asset drift, asset volatility and horizon.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(distance_to_default(100.0, 60.0, 0.08, 0.25, 1.0), 4)
    2.2383
    """
    V, D, mu, s, T = (np.asarray(a, dtype=np.float64) for a in (V, D, mu, sigma_V, T))
    return as_output((np.log(V / D) + (mu - 0.5 * s**2) * T) / (s * np.sqrt(T)))


def kmv_default_point(short_term_debt: ArrayLike, long_term_debt: ArrayLike, weight: float = 0.5) -> np.ndarray | float:
    """Moody's KMV default point: short-term debt plus half of long-term debt.

    Parameters
    ----------
    short_term_debt, long_term_debt : array_like
    weight : float
        Weight on long-term debt.

    Returns
    -------
    float or ndarray
    """
    return as_output(np.asarray(short_term_debt, dtype=np.float64) + weight * np.asarray(long_term_debt, dtype=np.float64))


@renamed_parameter("rtol", "tol", "financialkit.credit.kmv_iterate")
def kmv_iterate(
    equity: ArrayLike,
    D: ArrayLike,
    r: CurveLike,
    T: float = 1.0,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    sigma0: float | None = None,
    tol: float = DEFAULT_RTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> KMVResult:
    """Crosbie and Bohn's (2003) iterative estimate of asset values and volatility.

    Start from a guess of :math:`\\sigma_V`, invert the Merton equity
    equation each day for :math:`V_t` (Newton's method, vectorised over
    days), re-estimate :math:`\\sigma_V` from the implied asset log
    returns, and repeat until it stops changing. The drift is the mean
    log return plus :math:`\\sigma_V^2/2`.

    Parameters
    ----------
    equity : array_like, shape (n,)
        Equity market values, one per period.
    D : array_like
        Default point(s), scalar or one per period.
    r : float or YieldCurve
        Risk-free rate, or a ``fixed_income`` discount curve, whose zero
        rate to ``T`` is used: with deterministic rates the debt, paid
        only at ``T``, sees the curve only through :math:`P(0, T)`.
    T : float
        Horizon of the distance to default (and the option maturity).
    dt : float
        Spacing of the observations in years.
    sigma0 : float, optional
        Starting asset volatility; defaults to equity volatility times ``E / (E + D)``.
    tol : float
        Relative change in :math:`\\sigma_V` that stops the iteration.
    max_iter : int

    Returns
    -------
    KMVResult
    """
    r = zero_rate_to(r, T)
    E = np.asarray(equity, dtype=np.float64)
    Dv = np.broadcast_to(np.asarray(D, dtype=np.float64), E.shape)
    if E.ndim != 1 or E.size < 3:
        raise ValueError("equity must be a 1-d series of at least three observations")
    sigma_E = float(np.std(np.diff(np.log(E)), ddof=1) / np.sqrt(dt))
    sigma = sigma0 if sigma0 is not None else sigma_E * float(np.mean(E / (E + Dv)))
    history = [sigma]
    V = E + Dv * np.exp(-r * T)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007
        s = sigma
        V = optimize.newton(
            lambda v: np.asarray(black_scholes_price(v, Dv, T, r, s)) - E,
            V,
            fprime=lambda v: stats.norm.cdf(_d1_d2(v, Dv, T, r, s)[0]),
        )
        log_ret = np.diff(np.log(V))
        new = float(np.std(log_ret, ddof=1) / np.sqrt(dt))
        history.append(new)
        converged = abs(new - sigma) <= tol * sigma
        sigma = new
        if converged:
            break
    mu = float(np.mean(np.diff(np.log(V))) / dt + 0.5 * sigma**2)
    dd = float(distance_to_default(V[-1], Dv[-1], mu, sigma, T))
    return KMVResult(
        asset_values=V,
        asset_volatility=sigma,
        drift=mu,
        distance_to_default=dd,
        expected_default_frequency=float(stats.norm.cdf(-dd)),
        n_iter=n_iter,
        volatility_history=np.asarray(history),
    )
