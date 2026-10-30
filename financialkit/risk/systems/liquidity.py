"""Liquidity-adjusted VaR: Bangia, Diebold, Schuermann and Stroughair's
exogenous cost of liquidation.

Market VaR values a position at the mid price, but closing it costs
half the bid-ask spread, and spreads widen when prices fall. Bangia et
al. (1999) added the cost of liquidating at a bad spread, the mean
relative spread plus ``a`` standard deviations, to a lognormal VaR:

.. math::

   \\mathrm{LVaR} = P\\big(1 - e^{\\mu h - z_\\alpha \\sigma \\sqrt h}\\big)
   + \\tfrac12 P (\\bar S + a\\, \\sigma_S).

Adding the two worst cases assumes the worst spread arrives with the
worst return, which errs on the side of caution.

References: A. Bangia, F. X. Diebold, T. Schuermann and J. D.
Stroughair, "Modeling Liquidity Risk, with Implications for Traditional
Market Risk Measurement and Management," Wharton Financial Institutions
Center working paper 99-06 (1999), and "Liquidity on the Outside,"
*Risk* 12(6) (1999) 68-73; K. Dowd, *Measuring Market Risk*, 2nd ed.
(2005), Sec. 14.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

__all__ = ["exogenous_liquidity_cost", "liquidity_adjusted_var"]


def _as_output(x: np.ndarray) -> np.ndarray | float:
    return float(x) if x.ndim == 0 else x


def exogenous_liquidity_cost(position: ArrayLike, spread_mean: ArrayLike, spread_sd: ArrayLike = 0.0, a: ArrayLike = 0.0) -> np.ndarray | float:
    """Cost of closing a position at a stressed bid-ask spread, :math:`\\tfrac12 P(\\bar S + a \\sigma_S)`.

    Parameters
    ----------
    position : array_like
        Market value of the position at the mid price.
    spread_mean, spread_sd : array_like
        Mean and standard deviation of the relative spread
        ``(ask - bid) / mid``.
    a : array_like
        Number of spread standard deviations to stress by.

    Returns
    -------
    float or ndarray

    Examples
    --------
    $1 million at a 20 basis point spread costs $1,000 to close.

    >>> round(exogenous_liquidity_cost(1e6, 0.002), 6)
    1000.0
    """
    P, S, sS, a = (np.asarray(v, dtype=np.float64) for v in (position, spread_mean, spread_sd, a))
    if np.any(S < 0) or np.any(sS < 0):
        raise ValueError("spread_mean and spread_sd must be non-negative")
    return _as_output(0.5 * np.abs(P) * (S + a * sS))


def liquidity_adjusted_var(
    position: ArrayLike,
    sigma: ArrayLike,
    spread_mean: ArrayLike,
    spread_sd: ArrayLike = 0.0,
    alpha: float = 0.99,
    mu: ArrayLike = 0.0,
    horizon: ArrayLike = 1.0,
    a: ArrayLike | None = None,
) -> np.ndarray | float:
    """Bangia-Diebold-Schuermann-Stroughair liquidity-adjusted VaR of a long position.

    Lognormal market VaR plus
    :func:`exogenous_liquidity_cost`. Bangia et al. chose ``a`` so that
    :math:`\\bar S + a\\sigma_S` is the empirical ``alpha`` quantile of the
    spread, between about 2 and 4.5 for their currencies; the default
    ``a = z_alpha`` is that quantile for a normal spread.

    Parameters
    ----------
    position : array_like
        Market value at the mid price.
    sigma : array_like
        Volatility of the one-period log return.
    spread_mean, spread_sd : array_like
        Mean and standard deviation of the relative spread over the
        horizon.
    alpha : float
        Confidence level.
    mu : array_like
        Mean one-period log return.
    horizon : array_like
        Number of periods.
    a : array_like, optional
        Spread stress multiple; ``scipy.stats.norm.ppf(alpha)`` if omitted.

    Returns
    -------
    float or ndarray
        A positive loss.

    Examples
    --------
    $1 million with 2% daily volatility and a spread of 50 bp on average
    and 20 bp standard deviation: the liquidity cost adds about $4,800 to
    a market VaR of about $45,500.

    >>> round(liquidity_adjusted_var(1e6, 0.02, 0.005, 0.002))
    50288
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    P, s, m, h = (np.asarray(v, dtype=np.float64) for v in (position, sigma, mu, horizon))
    z = stats.norm.ppf(alpha)
    market = P * (1.0 - np.exp(m * h - z * s * np.sqrt(h)))
    cost = exogenous_liquidity_cost(P, spread_mean, spread_sd, z if a is None else a)
    return _as_output(np.asarray(market + cost))
