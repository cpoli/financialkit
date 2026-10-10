"""Barone-Adesi-Whaley quadratic approximation for American options.

The early-exercise premium :math:`\\varepsilon = V_{Am} - V_{Eu}` satisfies
the Black-Scholes PDE. Writing :math:`\\varepsilon = K(T) f(S, K(T))` with
:math:`K(T) = 1 - e^{-rT}` and dropping the small :math:`\\partial f/\\partial
K` term (MacMillan 1986) leaves an ordinary differential equation with
power solutions :math:`S^{q_{1,2}}`. Smooth pasting at the critical price
:math:`S^*` fixes the coefficient, so only one scalar root
(:func:`scipy.optimize.brentq`) is needed instead of a lattice:

.. math::

    C = c + A_2 (S/S^*)^{q_2} \\ (S < S^*), \\qquad
    P = p + A_1 (S/S^{**})^{q_1} \\ (S > S^{**}),

and the intrinsic value beyond the critical price. An American call on
an asset with ``q <= 0`` is never exercised early and equals the
European call.

References: G. Barone-Adesi and R. E. Whaley, *J. Finance* 42 (1987)
301-320; L. W. MacMillan, *Advances in Futures and Options Research* 1
(1986) 119-139; Haug, *The Complete Guide to Option Pricing Formulas*,
2nd ed., Section 3.10.1; Hull, *Options, Futures, and Other
Derivatives*, 10th ed., Technical Note 8.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.options.core.base import AmericanApproximationResult
from financialkit.options.systems.closed_form import black_scholes_price
from financialkit.options.utils.common import option_sign

__all__ = ["barone_adesi_whaley"]


def barone_adesi_whaley(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> AmericanApproximationResult:
    """Approximate American option value by Barone-Adesi and Whaley (1987).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity (years), risk-free rate and volatility.
    q : float
        Continuous dividend yield (cost of carry ``b = r - q``).
    option_type : {"call", "put"}

    Returns
    -------
    AmericanApproximationResult

    Examples
    --------
    An American put is worth more than its European counterpart:

    >>> res = barone_adesi_whaley(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put")
    >>> bool(res.price > res.european_price), round(res.critical_price, 2) < 100
    (True, True)
    """
    sign = option_sign(option_type)
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("S, K, T and sigma must all be positive")
    european = float(black_scholes_price(S, K, T, r, sigma, q, option_type))
    # no early exercise: a call with no dividend yield, or a put with a non-positive rate
    if (sign == 1 and q <= 0) or (sign == -1 and r <= 0):
        return AmericanApproximationResult(price=european, european_price=european, critical_price=np.inf if sign == 1 else 0.0)

    sig2 = sigma**2
    M, N, KT = 2 * r / sig2, 2 * (r - q) / sig2, -np.expm1(-r * T)
    root = np.sqrt((N - 1) ** 2 + 4 * M / KT)
    power = 0.5 * (-(N - 1) + sign * root)  # q2 > 1 for calls, q1 < 0 for puts
    vt = sigma * np.sqrt(T)

    def d1(x: float) -> float:
        return (np.log(x / K) + (r - q + 0.5 * sig2) * T) / vt

    def coeff(x: float) -> float:
        # A2 for calls, A1 for puts: sign * (x / power) * (1 - e^{-qT} N(sign d1))
        return sign * (x / power) * (1.0 - np.exp(-q * T) * norm.cdf(sign * d1(x)))

    def pasting(x: float) -> float:
        return sign * (x - K) - float(black_scholes_price(x, K, T, r, sigma, q, option_type)) - coeff(x)

    if sign == 1:
        hi = 2.0 * K
        while pasting(hi) < 0:
            hi *= 2.0
        s_crit = brentq(pasting, K, hi, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)
        early = S >= s_crit
    else:
        s_crit = brentq(pasting, 1e-12 * K, K, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)
        early = S <= s_crit
    if early:
        price = sign * (S - K)
    else:
        price = european + coeff(s_crit) * (S / s_crit) ** power
    return AmericanApproximationResult(price=float(price), european_price=european, critical_price=float(s_crit))
