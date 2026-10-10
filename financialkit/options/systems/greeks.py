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

from financialkit.options.core.base import Greeks
from financialkit.options.utils.common import as_output, option_sign

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

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity (years, positive), risk-free rate and
        volatility (positive).
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
    phi = option_sign(option_type)
    S, K, T, r, sigma, q = (np.asarray(a, dtype=np.float64) for a in (S, K, T, r, sigma, q))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    nd1 = norm.pdf(d1)
    Nd1, Nd2 = norm.cdf(phi * d1), norm.cdf(phi * d2)
    vega = S * dq * nd1 * sqrtT
    return Greeks(
        delta=as_output(phi * dq * Nd1),
        gamma=as_output(dq * nd1 / (S * sigma * sqrtT)),
        vega=as_output(vega),
        theta=as_output(-S * dq * nd1 * sigma / (2 * sqrtT) - phi * r * K * dr * Nd2 + phi * q * S * dq * Nd1),
        rho=as_output(phi * K * T * dr * Nd2),
        epsilon=as_output(-phi * S * T * dq * Nd1),
        vanna=as_output(-dq * nd1 * d2 / sigma),
        volga=as_output(vega * d1 * d2 / sigma),
    )
