"""Analytic approximations for American options: Barone-Adesi-Whaley and
Bjerksund-Stensland.

Barone-Adesi and Whaley approximate the early-exercise premium.

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

Bjerksund and Stensland (1993) instead value one exercise policy exactly:
exercise as soon as the spot reaches a flat trigger :math:`I`. The value
is then a combination of up-and-out claims with closed forms, and
:math:`I` is set between the perpetual boundary :math:`B_\\infty` and
the boundary at expiry :math:`B_0`. A fixed policy can only do worse
than the optimal one, so the result is a lower bound on the American
price. Puts follow from the put-call transformation
:math:`P(S, K, T, r, b) = C(K, S, T, r - b, -b)`.

References: G. Barone-Adesi and R. E. Whaley, *J. Finance* 42 (1987)
301-320; L. W. MacMillan, *Advances in Futures and Options Research* 1
(1986) 119-139; P. Bjerksund and G. Stensland, "Closed-Form Approximation
of American Options," *Scandinavian J. Management* 9 (1993) S87-S99; Haug, *The Complete Guide to Option Pricing Formulas*,
2nd ed., Sections 3.10.1-3.10.2; Hull, *Options, Futures, and Other
Derivatives*, 10th ed., Technical Note 8.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from financialkit._validation import option_type as check_option_type
from financialkit._validation import positive
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.options.core.base import AmericanApproximationResult
from financialkit.options.systems.closed_form import black_scholes_price

__all__ = ["barone_adesi_whaley", "bjerksund_stensland"]


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
    sign = check_option_type(option_type)
    positive(S=S, K=K, T=T, sigma=sigma)
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


def _bs_phi(S: float, T: float, gamma: float, H: float, I: float, r: float, b: float, sigma: float) -> float:
    """Bjerksund-Stensland's :math:`\\phi`: the value of :math:`S_T^\\gamma` paid if :math:`S_T \\le H`, knocked out at :math:`I`."""
    vt = sigma * np.sqrt(T)
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * sigma**2) * T
    d = -(np.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / vt
    kappa = 2 * b / sigma**2 + 2 * gamma - 1
    return float(np.exp(lam) * S**gamma * (norm.cdf(d) - (I / S) ** kappa * norm.cdf(d - 2 * np.log(I / S) / vt)))


def _bs_call(S: float, K: float, T: float, r: float, b: float, sigma: float) -> tuple[float, float]:
    """Bjerksund-Stensland (1993) American call with cost of carry ``b < r``; returns (price, trigger)."""
    sig2 = sigma**2
    beta = (0.5 - b / sig2) + np.sqrt((b / sig2 - 0.5) ** 2 + 2 * r / sig2)
    b_inf = beta / (beta - 1) * K
    b_0 = max(K, r / (r - b) * K)
    h = -(b * T + 2 * sigma * np.sqrt(T)) * b_0 / (b_inf - b_0)
    trigger = b_0 + (b_inf - b_0) * (1 - np.exp(h))
    if S >= trigger:
        return S - K, float(trigger)
    alpha = (trigger - K) * trigger**-beta
    price = (
        alpha * S**beta
        - alpha * _bs_phi(S, T, beta, trigger, trigger, r, b, sigma)
        + _bs_phi(S, T, 1, trigger, trigger, r, b, sigma)
        - _bs_phi(S, T, 1, K, trigger, r, b, sigma)
        - K * _bs_phi(S, T, 0, trigger, trigger, r, b, sigma)
        + K * _bs_phi(S, T, 0, K, trigger, r, b, sigma)
    )
    return float(price), float(trigger)


def bjerksund_stensland(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> AmericanApproximationResult:
    r"""Approximate American option value by Bjerksund and Stensland (1993).

    The holder exercises a call the first time the spot reaches the flat
    trigger

    .. math:: I = B_0 + (B_\infty - B_0)(1 - e^{h(T)}), \qquad
              h(T) = -(bT + 2\sigma\sqrt{T}) \frac{B_0}{B_\infty - B_0},

    with :math:`B_\infty = \beta K / (\beta - 1)` the perpetual boundary and
    :math:`B_0 = \max(K, rK/(r - b))` the boundary just before expiry, so
    the price is

    .. math:: C = \alpha S^\beta - \alpha\phi(S, T, \beta, I, I)
              + \phi(S, T, 1, I, I) - \phi(S, T, 1, K, I)
              - K\phi(S, T, 0, I, I) + K\phi(S, T, 0, K, I),

    with :math:`\alpha = (I - K) I^{-\beta}`. Because the policy is not
    optimal, the price is a lower bound on the true American value. A put
    is the call with spot and strike swapped, rate ``q`` and carry
    ``q - r`` (the put-call transformation).

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
        ``critical_price`` is the trigger: exercise at or above it for a
        call, at or below it for a put.

    Examples
    --------
    A lower bound, within a few cents of a fine tree:

    >>> from financialkit.options import CRRTree
    >>> res = bjerksund_stensland(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put")
    >>> tree = CRRTree(2000).price(100.0, 100.0, 1.0, 0.08, 0.25, option_type="put", american=True).price
    >>> bool(res.european_price < res.price < tree)
    True
    """
    sign = check_option_type(option_type)
    positive(S=S, K=K, T=T, sigma=sigma)
    european = float(black_scholes_price(S, K, T, r, sigma, q, option_type))
    if (sign == 1 and q <= 0) or (sign == -1 and r <= 0):
        return AmericanApproximationResult(price=european, european_price=european, critical_price=np.inf if sign == 1 else 0.0)
    if sign == 1:
        price, trigger = _bs_call(S, K, T, r, r - q, sigma)
        critical = trigger
    else:
        # put-call transformation P(S, K, T, r, b) = C(K, S, T, r - b, -b); the trigger scales with the strike S
        price, trigger = _bs_call(K, S, T, q, q - r, sigma)
        critical = K * S / trigger
    return AmericanApproximationResult(price=max(float(price), european), european_price=european, critical_price=float(critical))
