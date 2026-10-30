"""Analytic Black-Scholes-Merton Greeks.

The Greeks are the partial derivatives of the option price that the
Black-Scholes hedging argument rests on: holding :math:`\\Delta` shares
against a short option removes its exposure to small spot moves, gamma
measures how fast that hedge goes stale, and theta is the time decay that
pays for gamma (:math:`\\Theta + \\tfrac12\\sigma^2 S^2 \\Gamma + (r - q) S
\\Delta = r V`). All are checked against
:func:`~financialkit.options.utils.common.finite_difference_greeks` in the
tests.

References: F. Black and M. Scholes, *J. Political Economy* 81 (1973)
637-654; Hull, *Options, Futures, and Other Derivatives*, 10th ed.,
Ch. 19; Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.,
Ch. 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from financialkit._validation import nonnegative
from financialkit._validation import option_type as check_option_type
from financialkit.options.core.base import Greeks
from financialkit.options.utils.common import as_output, black_d1_d2

__all__ = ["black_scholes_greeks"]


def black_scholes_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call") -> Greeks:
    r"""All first-order and the main second-order Black-Scholes-Merton Greeks.

    With :math:`\phi = \pm 1` for a call/put:

    .. math::

        \Delta = \phi e^{-qT} N(\phi d_1), \quad
        \Gamma = \frac{e^{-qT} n(d_1)}{S \sigma \sqrt{T}}, \quad
        \mathcal{V} = S e^{-qT} n(d_1) \sqrt{T}, \\
        \Theta = -\frac{S e^{-qT} n(d_1) \sigma}{2\sqrt{T}}
                 - \phi r K e^{-rT} N(\phi d_2) + \phi q S e^{-qT} N(\phi d_1), \\
        \rho = \phi K T e^{-rT} N(\phi d_2), \quad
        \epsilon = -\phi S T e^{-qT} N(\phi d_1), \quad
        \text{vanna} = -e^{-qT} n(d_1) \frac{d_2}{\sigma}, \quad
        \text{volga} = \mathcal{V} \frac{d_1 d_2}{\sigma}.

    At ``T = 0`` or ``sigma = 0`` each Greek is its limit: the option is
    worth :math:`e^{-rT}\max(\pm(F - K), 0)`, so away from the money delta
    is :math:`\phi e^{-qT}` or 0 and gamma, vega, vanna and volga vanish.
    Exactly at the money the payoff has a kink: delta is the midpoint
    :math:`\phi e^{-qT}/2`, gamma is :math:`+\infty`, and at ``T = 0``
    theta is :math:`-\infty`. ``S = 0`` and ``K = 0`` are handled the same
    way; negative ``S``, ``K``, ``T`` or ``sigma`` raise :class:`ValueError`.

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity (years), risk-free rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    Greeks
        Each field broadcast like the inputs.

    Examples
    --------
    Hull, 10th ed., Example 19.1 (S = 49, K = 50, r = 5%, sigma = 20%,
    20 weeks):

    >>> g = black_scholes_greeks(49.0, 50.0, 0.3846, 0.05, 0.20)
    >>> round(g.delta, 3), round(g.gamma, 3), round(g.vega, 1), round(g.theta, 2), round(g.rho, 2)
    (0.522, 0.066, 12.1, -4.31, 8.91)
    """
    phi = check_option_type(option_type)
    S, K, T, r, sigma, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, T, r, sigma, q)))
    nonnegative(S=S, K=K, T=T, sigma=sigma)
    sqrtT = np.sqrt(T)
    v = sigma * sqrtT
    d1, d2 = black_d1_d2(S * np.exp((r - q) * T), K, sigma, T)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    nd1 = norm.pdf(d1)
    Nd1, Nd2 = norm.cdf(phi * d1), norm.cdf(phi * d2)
    vega = S * dq * nd1 * sqrtT
    # Terms with n(d1) vanish wherever d1 is infinite. Where sigma sqrt(T) = 0
    # they are only non-zero at the money (d1 = 0), and take their limits.
    live, regular = nd1 > 0, v > 0
    with np.errstate(divide="ignore"):
        gamma = np.where(live, dq * nd1 / np.where(live, S * v, 1.0), 0.0)
        decay = np.where(live & (sigma > 0), -S * dq * nd1 * sigma / np.where(live & (sigma > 0), 2 * sqrtT, 1.0), 0.0)
    safe_sigma = np.where(regular, sigma, 1.0)
    vanna = np.where(regular, -dq * nd1 * np.where(live, d2, 0.0) / safe_sigma, np.where(live, 0.5 * dq * nd1 * sqrtT, 0.0))
    volga = np.where(regular & live, vega * np.where(live, d1 * d2, 0.0) / safe_sigma, 0.0)
    return Greeks(
        delta=as_output(phi * dq * Nd1),
        gamma=as_output(gamma),
        vega=as_output(vega),
        theta=as_output(decay - phi * r * K * dr * Nd2 + phi * q * S * dq * Nd1),
        rho=as_output(phi * K * T * dr * Nd2),
        epsilon=as_output(-phi * S * T * dq * Nd1),
        vanna=as_output(vanna),
        volga=as_output(volga),
    )
