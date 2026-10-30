"""Implied volatility: inverting Black-Scholes, its at-the-money shortcut,
and the smile across strikes.

The Black-Scholes price is strictly increasing in :math:`\\sigma`, from the
lower no-arbitrage bound at :math:`\\sigma = 0` to the upper bound as
:math:`\\sigma \\to \\infty`, so a price strictly between Merton's bounds
has exactly one implied volatility. :func:`implied_volatility` checks the
bounds first, brackets the root and then solves every element at once by
a safeguarded Newton iteration on the log price, started at the
inflection point :math:`\\sigma^2 T = 2|\\ln(F/K)|` of Manaster and
Koehler (1982) and falling back to bisection whenever a step leaves the
bracket; ``method="brentq"`` keeps the per-element
:func:`scipy.optimize.brentq` loop, and ``method="jaeckel"`` runs
Jäckel's (2015) "Let's be rational" algorithm
(:mod:`~financialkit.options.systems.lets_be_rational`), which needs no
bracket and reaches machine precision in two iterations.
Latané and Rendleman (1976) were the first to read volatilities off
quoted option prices this way. Brenner and Subrahmanyam (1988) noticed
that at the money the formula is nearly linear in :math:`\\sigma`, giving
:func:`brenner_subrahmanyam_vol`. Inverting a whole strip of prices gives
the volatility smile (:func:`implied_vol_smile`), flat under
Black-Scholes and, since the 1987 crash, sharply skewed in equity markets.
:func:`bachelier_implied_volatility` inverts Bachelier's normal model
instead, which still works when the forward or the strike is negative, as
for WTI crude oil in April 2020.

References: H. A. Latané and R. J. Rendleman, *J. Finance* 31 (1976)
369-381; S. Manaster and G. Koehler, *J. Finance* 37 (1982) 227-230;
M. Brenner and M. G. Subrahmanyam, *Financial Analysts J.* 44(5)
(1988) 80-83; P. Jäckel, *Wilmott* 2015(75) 40-53; M. Rubinstein, *J. Finance* 49 (1994) 771-818; Hull,
*Options, Futures, and Other Derivatives*, 10th ed., Sections 15.11 and
20.1-20.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import ndtr

from financialkit._validation import option_type as check_option_type
from financialkit._validation import positive
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.options.core.base import SmileResult
from financialkit.options.systems.closed_form import bachelier_price, black_scholes_price  # noqa: F401  (doctests)
from financialkit.options.systems.lets_be_rational import normalised_implied_volatility
from financialkit.options.systems.parity import no_arbitrage_bounds
from financialkit.options.utils.common import as_output

__all__ = ["implied_volatility", "bachelier_implied_volatility", "brenner_subrahmanyam_vol", "implied_vol_smile"]

_SIGMA_LO = 1e-8
_SIGMA_HI = 5.0
_SIGMA_MAX = 1e4
_METHODS = ("newton", "brentq", "jaeckel")


def _price_and_vega(sigma: np.ndarray, F: np.ndarray, K: np.ndarray, T: np.ndarray, D: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Black price :math:`D\\theta[F N(\\theta d_1) - K N(\\theta d_2)]` and its vega, for ``sigma > 0``."""
    v = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v**2) / v
    price = D * sign * (F * ndtr(sign * d1) - K * ndtr(sign * (d1 - v)))
    vega = D * F * np.exp(-0.5 * d1**2) * np.sqrt(T / (2 * np.pi))
    return price, vega


def _newton(
    price: np.ndarray, F: np.ndarray, K: np.ndarray, T: np.ndarray, D: np.ndarray, sign: int, lo: np.ndarray, hi: np.ndarray, tol: float, max_iter: int
) -> np.ndarray:
    """Solve ``ln c(sigma) = ln price`` for every element at once.

    Each element keeps its own bracket ``[lo, hi]``, shrunk by the sign of
    every residual; a Newton step that leaves it is replaced by bisection,
    so convergence is guaranteed and is quadratic once Newton takes over.
    The log makes deep out-of-the-money prices, which vanish like
    :math:`e^{-x^2 / 2\\sigma^2 T}`, nearly linear in :math:`1/\\sigma`.
    """
    # Manaster-Koehler: the price is convex below and concave above sigma^2 T = 2|ln(F/K)|;
    # at the money that is 0, so the Brenner-Subrahmanyam guess takes over
    guess = np.maximum(np.sqrt(2 * np.abs(np.log(F / K)) / T), np.sqrt(2 * np.pi / T) * price / (D * np.sqrt(F * K)))
    sigma = np.clip(guess, lo, hi)
    target = np.log(price)
    active = np.arange(price.size)
    for _ in range(max_iter):
        s = sigma[active]
        c, vega = _price_and_vega(s, F[active], K[active], T[active], D[active], sign)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.log(c) - target[active]
            step = g * c / vega
        a, b = np.where(g < 0, s, lo[active]), np.where(g > 0, s, hi[active])
        lo[active], hi[active] = a, b
        new = s - step
        new = np.where(~np.isfinite(new) | (new <= a) | (new >= b), 0.5 * (a + b), new)
        sigma[active] = new
        active = active[(np.abs(new - s) > tol) & (g != 0) & (b - a > tol)]
        if active.size == 0:
            return sigma
    raise RuntimeError(f"implied volatility did not converge in {max_iter} iterations for {active.size} prices")


def implied_volatility(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    tol: float = DEFAULT_ATOL,
    max_iter: int = DEFAULT_MAX_ITER,
    method: str = "newton",
) -> np.ndarray | float:
    """Black-Scholes implied volatility, solved for all prices at once.

    For the Black-76 implied volatility of an option on a forward ``F``,
    pass ``S = F`` and ``q = r``.

    Parameters
    ----------
    price : array_like
        Observed European option prices.
    S, K, T, r, q : array_like
        Spot, strike, maturity, risk-free rate and dividend yield.
    option_type : {"call", "put"}
    tol : float
        Absolute tolerance on the volatility.
    max_iter : int
        Iteration cap of the solver.
    method : {"newton", "brentq", "jaeckel"}
        ``"newton"`` (the default) runs the safeguarded Newton iteration on
        the whole array at once; ``"brentq"`` loops
        :func:`scipy.optimize.brentq` over the elements. Both stop within
        ``tol`` of the root. ``"jaeckel"`` runs Jäckel's "Let's be
        rational" solver on the whole array: a rational initial guess and
        two Householder steps, accurate to machine precision, so ``tol``
        and ``max_iter`` are not used and prices too close to the bounds
        for the bracket of the other two methods are still resolved.

    Returns
    -------
    float or ndarray

    Raises
    ------
    ValueError
        If any maturity is not positive (at ``T = 0`` every volatility gives
        the intrinsic value, so none is implied), if any price lies outside
        Merton's open no-arbitrage interval, where no volatility can
        reproduce it, or if a price is so close to a bound that its
        volatility cannot be resolved in ``[1e-8, 1e4]`` (not checked by
        ``"jaeckel"``), or for an unknown ``method``.
    RuntimeError
        If the solver does not converge in ``max_iter`` iterations.

    Examples
    --------
    Hull, 10th ed., Section 15.11: a call worth 1.875 with S = 21,
    K = 20, r = 10% and three months has an implied volatility of 23.5%.

    >>> round(implied_volatility(1.875, 21.0, 20.0, 0.25, 0.10), 3)
    0.235
    >>> round(implied_volatility(1.875, 21.0, 20.0, 0.25, 0.10, method="jaeckel"), 3)
    0.235
    """
    sign = check_option_type(option_type)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    price, S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (price, S, K, T, r, q)))
    if np.any(T <= 0):
        t = T.ravel()[np.flatnonzero(T.ravel() <= 0)[0]]
        raise ValueError(f"maturity T = {t:.6g} must be positive; at T = 0 every volatility gives the intrinsic value, so none is implied")
    lower, upper = (np.asarray(b) for b in no_arbitrage_bounds(S, K, T, r, q, option_type))
    bad = ~((price > lower) & (price < upper))
    if np.any(bad):
        i = np.flatnonzero(bad.ravel())[0]
        raise ValueError(
            f"price {price.ravel()[i]:.6g} violates the no-arbitrage bounds ({lower.ravel()[i]:.6g}, {upper.ravel()[i]:.6g}); no implied volatility exists"
        )
    shape = price.shape
    price, S, K, T, r, q, lower, upper = (np.ravel(a) for a in (price, S, K, T, r, q, lower, upper))
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    if method == "jaeckel":
        # normalise to b = c / sqrt(FK) at x = ln(F/K); a put is a call at -x by put-call symmetry
        x = np.log(F / K)
        vols = normalised_implied_volatility(price / (D * np.sqrt(F * K)), sign * x) / np.sqrt(T)
        return as_output(vols.reshape(shape))

    def f(sigma: np.ndarray) -> np.ndarray:
        return _price_and_vega(sigma, F, K, T, D, sign)[0] - price

    lo, hi = np.full(price.shape, _SIGMA_LO), np.full(price.shape, _SIGMA_HI)
    while np.any(short := (f(hi) < 0) & (hi < _SIGMA_MAX)):
        hi[short] = np.minimum(4.0 * hi[short], _SIGMA_MAX)
    unresolved = (f(lo) >= 0) | (f(hi) < 0)
    if np.any(unresolved):
        i = np.flatnonzero(unresolved)[0]
        raise ValueError(
            f"price {price[i]:.6g} is too close to the no-arbitrage bounds ({lower[i]:.6g}, {upper[i]:.6g}) "
            f"for its volatility to be resolved in [{_SIGMA_LO:g}, {_SIGMA_MAX:g}]"
        )
    if method == "newton":
        vols = _newton(price, F, K, T, D, sign, lo, hi, tol, max_iter)
    else:
        vols = np.array(
            [
                brentq(lambda s, i=i: float(_price_and_vega(s, F[i], K[i], T[i], D[i], sign)[0] - price[i]), lo[i], hi[i], xtol=tol, maxiter=max_iter)
                for i in range(price.size)
            ]
        )
    vols = vols.reshape(shape)
    return as_output(vols)


def bachelier_implied_volatility(
    price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call", tol: float = DEFAULT_ATOL
) -> np.ndarray | float:
    r"""Normal (Bachelier) implied volatility, valid for negative forwards and strikes.

    Solves :func:`~financialkit.options.systems.closed_form.bachelier_price`
    for the absolute volatility :math:`\sigma_N`. The time value
    :math:`e^{rT}c - \max(F - K, 0) = \sigma_N\sqrt{T}\,[n(d) - |d|N(-|d|)]`
    is strictly increasing in :math:`\sigma_N` and unbounded, so every price
    above the discounted intrinsic value has exactly one normal
    volatility. For an option on a futures price ``F``, pass ``S = F`` and
    ``q = r``.

    Parameters
    ----------
    price : array_like
        Observed European option prices.
    S, K, T, r, q : array_like
        Spot (or futures price), strike, maturity, rate and dividend yield;
        ``S`` and ``K`` may be negative.
    option_type : {"call", "put"}
    tol : float
        Absolute tolerance on :math:`\sigma_N`, relative to its size.

    Returns
    -------
    float or ndarray

    Raises
    ------
    ValueError
        If a maturity is not positive or a price is not above the
        discounted intrinsic value.

    Examples
    --------
    CME's WTI options after 20 April 2020: futures at -10, a put struck
    at 0, normal volatility 15 dollars:

    >>> p = bachelier_price(-10.0, 0.0, 0.1, 0.0, 15.0, q=0.0, option_type="put")
    >>> round(bachelier_implied_volatility(p, -10.0, 0.0, 0.1, 0.0, option_type="put"), 10)
    15.0
    """
    sign = check_option_type(option_type)
    price, S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (price, S, K, T, r, q)))
    positive(T=T)
    F, D = S * np.exp((r - q) * T), np.exp(-r * T)
    time_value = price / D - np.maximum(sign * (F - K), 0.0)
    if np.any(time_value <= 0):
        i = np.flatnonzero(time_value.ravel() <= 0)[0]
        raise ValueError(f"price {price.ravel()[i]:.6g} is not above the discounted intrinsic value, so no normal volatility is implied")
    shape = price.shape
    F, K, T, tv = (np.ravel(a) for a in (F, K, T, time_value))
    m = np.abs(F - K)

    def g(v: float, i: int) -> float:
        # time value of the out-of-the-money option at total normal vol v, minus the target
        d = m[i] / v
        return float(v * (np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi) - d * ndtr(-d)) - tv[i])

    vols = np.empty(tv.size)
    for i in range(tv.size):
        lo = np.sqrt(2 * np.pi) * tv[i]  # the time value is at most v / sqrt(2 pi), so v >= this
        hi = 2 * lo + m[i]
        while g(hi, i) < 0:
            hi *= 2
        vols[i] = brentq(g, lo, hi, args=(i,), xtol=tol * lo, rtol=4 * np.finfo(float).eps, maxiter=DEFAULT_MAX_ITER) if g(lo, i) < 0 else lo
    return as_output((vols / np.sqrt(T)).reshape(shape))


def brenner_subrahmanyam_vol(price: ArrayLike, S: ArrayLike, T: ArrayLike, r: ArrayLike = 0.0, q: ArrayLike = 0.0) -> np.ndarray | float:
    r"""Brenner-Subrahmanyam (1988) approximation for at-the-money-forward options.

    For :math:`K = F`, :math:`c = S e^{-qT}[2N(\sigma\sqrt{T}/2) - 1]
    \approx S e^{-qT} \sigma \sqrt{T / (2\pi)}`, so

    .. math:: \sigma \approx \sqrt{\frac{2\pi}{T}} \frac{c}{S e^{-qT}},

    with a relative error of order :math:`\sigma^2 T / 24`. At the forward,
    calls and puts have the same price, so either can be passed.

    Parameters
    ----------
    price : array_like
        At-the-money-forward call (or put) price.
    S, T, r, q : array_like
        Spot, maturity, risk-free rate and dividend yield (``r`` is not used
        by the formula and is accepted for a uniform signature).

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> c = black_scholes_price(100.0, 100.0, 0.25, 0.0, 0.2)
    >>> round(brenner_subrahmanyam_vol(c, 100.0, 0.25), 4)
    0.1999
    """
    price, S, T, q = (np.asarray(a, dtype=np.float64) for a in (price, S, T, q))
    return as_output(np.sqrt(2 * np.pi / T) * price / (S * np.exp(-q * T)))


def implied_vol_smile(prices: ArrayLike, strikes: ArrayLike, S: float, T: float, r: float, q: float = 0.0, option_type: str = "call") -> SmileResult:
    """Implied volatilities across a strip of strikes at one maturity.

    Parameters
    ----------
    prices, strikes : array_like, shape (n,)
        Observed European option prices and their strikes.
    S, T, r, q : float
        Spot, maturity, risk-free rate and dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    SmileResult

    Examples
    --------
    Prices from a single Black-Scholes volatility give a flat smile:

    >>> import numpy as np
    >>> K = np.array([80.0, 100.0, 120.0])
    >>> smile = implied_vol_smile(black_scholes_price(100.0, K, 1.0, 0.03, 0.25), K, 100.0, 1.0, 0.03)
    >>> np.round(smile.vols, 6)
    array([0.25, 0.25, 0.25])
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    vols = np.asarray(implied_volatility(prices, S, strikes, T, r, q, option_type), dtype=np.float64)
    return SmileResult(strikes=strikes, vols=vols, forward=float(S * np.exp((r - q) * T)), T=float(T))
