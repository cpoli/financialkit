"""Variance swaps: realized variance, and its fair strike from a strip of vanilla options.

Demeterfi, Derman, Kamal and Zou (1999) showed that a variance swap is
replicated, whatever the volatility process (as long as the price is
continuous), by a static position in the log contract, which is itself a
strip of out-of-the-money puts and calls weighted by :math:`1/K^2`,
plus a dynamic delta hedge:

.. math:: K_{\\mathrm{var}} = \\frac{2}{T}\\left[\\ln\\frac{F}{K_0} - \\left(\\frac{F}{K_0} - 1\\right)\\right]
          + \\frac{2 e^{rT}}{T}\\left[\\int_0^{K_0} \\frac{P(K)}{K^2}\\,dK + \\int_{K_0}^\\infty \\frac{C(K)}{K^2}\\,dK\\right].

* :func:`variance_swap_strike` -- the fair variance from quoted option
  prices on a strike grid, discretised as in the CBOE's VIX calculation.
* :func:`annualized_realized_variance` -- the annualized sum of squared
  log returns of a price path, which the swap pays against the strike.

References: K. Demeterfi, E. Derman, M. Kamal and J. Zou, "A Guide to
Volatility and Variance Swaps," *J. Derivatives* 6(4) (1999) 9-32; Cboe,
*VIX White Paper* (2003, revised 2019); Hull, *Options, Futures, and
Other Derivatives*, 10th ed., Section 26.16.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit._deprecation import deprecated_alias
from financialkit.exotic_options.core.base import VarianceSwapResult

__all__ = ["variance_swap_strike", "annualized_realized_variance", "realized_variance"]


def variance_swap_strike(strikes: ArrayLike, call_prices: ArrayLike, put_prices: ArrayLike, S: float, T: float, r: float, q: float = 0.0) -> VarianceSwapResult:
    """Fair variance strike from out-of-the-money option prices (Demeterfi, Derman, Kamal and Zou 1999).

    With :math:`K_0` the largest strike at or below the forward
    :math:`F = S e^{(r-q)T}`, puts are used below :math:`K_0`, calls above
    and their average at :math:`K_0`, and

    .. math:: K_{\\mathrm{var}} = \\frac{2 e^{rT}}{T} \\sum_i \\frac{\\Delta K_i}{K_i^2} Q(K_i)
              - \\frac{1}{T}\\left(\\frac{F}{K_0} - 1\\right)^2,

    with :math:`\\Delta K_i` half the distance between the neighbouring
    strikes. The last term is the second-order expansion of the forward
    adjustment above. The strike range truncates the integrals, so the
    strip should reach well into both wings.

    Parameters
    ----------
    strikes : array_like, shape (n,)
        Increasing strikes.
    call_prices, put_prices : array_like, shape (n,)
        Call and put prices at those strikes.
    S, T, r : float
        Spot, maturity and rate.
    q : float
        Continuous dividend yield.

    Returns
    -------
    VarianceSwapResult

    Examples
    --------
    Under Black-Scholes the fair variance is :math:`\\sigma^2`:

    >>> import numpy as np
    >>> from financialkit.options import black_scholes_price
    >>> K = np.arange(20.0, 400.0, 1.0)
    >>> c, p = (black_scholes_price(100.0, K, 1.0, 0.05, 0.2, option_type=t) for t in ("call", "put"))
    >>> round(variance_swap_strike(K, c, p, 100.0, 1.0, 0.05).vol_strike, 4)
    0.2
    """
    K = np.asarray(strikes, dtype=np.float64)
    C, P = np.asarray(call_prices, dtype=np.float64), np.asarray(put_prices, dtype=np.float64)
    if K.ndim != 1 or K.size < 3 or np.any(np.diff(K) <= 0):
        raise ValueError("strikes must be an increasing 1-d grid of at least 3 points")
    F = S * np.exp((r - q) * T)
    below = np.nonzero(F >= K)[0]
    i0 = int(below[-1]) if below.size else 0
    K0 = float(K[i0])
    Q = np.where(K < K0, P, C)
    Q[i0] = 0.5 * (P[i0] + C[i0])
    dK = np.gradient(K)  # half the distance between neighbours; one-sided at the ends
    dK[0], dK[-1] = K[1] - K[0], K[-1] - K[-2]
    w = 2.0 * np.exp(r * T) * dK / (T * K**2)
    fair = float(w @ Q - (F / K0 - 1.0) ** 2 / T)
    return VarianceSwapResult(fair, K, w, Q, float(F), K0, float(T))


def annualized_realized_variance(paths: ArrayLike, T: float) -> np.ndarray:
    """Annualized realized variance :math:`\\frac{1}{T}\\sum_i \\ln^2(S_{t_{i+1}}/S_{t_i})` of each path.

    This is the floating leg of a variance swap: it takes *prices* and
    divides by the period length `T`. For the unannualized realized
    variance of Andersen and Bollerslev, a sum of squared *returns*, use
    :func:`financialkit.time_series.realized_variance`. This function was
    ``exotic_options.realized_variance`` in 0.1.0; that name remains as a
    deprecated alias.

    Parameters
    ----------
    paths : array_like, shape (n_paths, n_steps + 1) or (n_steps + 1,)
        Prices on the observation dates.
    T : float
        Length of the observation period in years.

    Returns
    -------
    ndarray, shape (n_paths,)
        A 0-d array for a single path.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(annualized_realized_variance(100.0 * np.exp([0.0, 0.1, 0.0]), 0.5)), 10)
    0.04
    """
    x = np.log(np.asarray(paths, dtype=np.float64))
    return np.sum(np.diff(x, axis=-1) ** 2, axis=-1) / T


realized_variance = deprecated_alias(annualized_realized_variance, "realized_variance", "financialkit.exotic_options.annualized_realized_variance")
