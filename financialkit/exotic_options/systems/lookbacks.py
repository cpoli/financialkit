"""Lookback options: the payoff depends on the maximum or minimum price over the option's life.

* :func:`floating_lookback_price` -- Goldman, Sosin and Gatto (1979):
  "buy at the low, sell at the high". The call pays
  :math:`S_T - \\min_t S_t` and the put :math:`\\max_t S_t - S_T`.
* :func:`fixed_lookback_price` -- Conze and Viswanathan (1991): a call on
  the maximum, :math:`(\\max_t S_t - K)^+`, and a put on the minimum,
  :math:`(K - \\min_t S_t)^+`.
* :func:`lookback_option_mc` -- Monte Carlo with the extremum sampled
  exactly from the Brownian bridge between grid points, or read off the
  grid for a discretely monitored lookback.

All four closed forms reduce to two building blocks: the value of
:math:`\\max(M, \\max_t S_t) - M` and of :math:`m - \\min(m, \\min_t S_t)` for
levels :math:`M \\ge S` and :math:`m \\le S`. Their :math:`\\sigma^2/(2b)`
factor is singular at zero carry :math:`b = r - q`; there the limit is
taken as the average of :math:`b = \\pm 10^{-5}`, accurate to
:math:`10^{-10}`.

References: M. B. Goldman, H. B. Sosin and M. A. Gatto, "Path Dependent
Options: 'Buy at the Low, Sell at the High'," *J. Finance* 34 (1979)
1111-1127; A. Conze and Viswanathan, "Path Dependent Options: The
Case of Lookback Options," *J. Finance* 46 (1991) 1893-1907; E. G. Haug,
*The Complete Guide to Option Pricing Formulas*, 2nd ed. (McGraw-Hill,
2007), Sections 4.15.1-4.15.2.

With ``sigma = 0`` or at ``T = 0`` both closed forms return the payoff of
the riskless path :math:`S e^{bt}`, whose extremes are its end points.
Spots and observed extremes must be positive.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from financialkit._validation import nonnegative, positive
from financialkit._validation import option_type as check_option_type
from financialkit.exotic_options.utils.numerics import bridge_extremum, spot_paths
from financialkit.integrators import RNGLike
from financialkit.monte_carlo import MonteCarloResult, sample_mean
from financialkit.options.utils.common import as_output

__all__ = ["floating_lookback_price", "fixed_lookback_price", "lookback_option_mc"]

_ZERO_CARRY_STEP = 1e-5


def _blocks(S, level, T, r, b, sigma, upper: bool):
    """Value of ``max(level, running max) - level`` (``upper``) or ``level - min(level, running min)``."""

    def at(b):
        v = sigma * np.sqrt(T)
        d1 = (np.log(S / level) + (b + 0.5 * sigma**2) * T) / v
        d2 = d1 - v
        k = sigma**2 / (2.0 * b)
        reflect = (S / level) ** (-2.0 * b / sigma**2)
        if upper:
            return (
                S * np.exp((b - r) * T) * norm.cdf(d1)
                - level * np.exp(-r * T) * norm.cdf(d2)
                + S * np.exp(-r * T) * k * (-reflect * norm.cdf(d1 - 2.0 * b * np.sqrt(T) / sigma) + np.exp(b * T) * norm.cdf(d1))
            )
        return (
            level * np.exp(-r * T) * norm.cdf(-d2)
            - S * np.exp((b - r) * T) * norm.cdf(-d1)
            + S * np.exp(-r * T) * k * (reflect * norm.cdf(-d1 + 2.0 * b * np.sqrt(T) / sigma) - np.exp(b * T) * norm.cdf(-d1))
        )

    small = np.abs(b) < _ZERO_CARRY_STEP
    safe_b = np.where(small, _ZERO_CARRY_STEP, b)
    limit = 0.5 * (at(np.full_like(b, _ZERO_CARRY_STEP)) + at(np.full_like(b, -_ZERO_CARRY_STEP)))
    return np.where(small, limit, at(safe_b))


def _inputs(*args: ArrayLike) -> list[np.ndarray]:
    return list(np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in args)))


def _split(S: np.ndarray, X: np.ndarray, T: np.ndarray, r: np.ndarray, q: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, ...]:
    """Validate; return the riskless path's end point, the mask where the closed form applies, and safe ``T`` and ``sigma``."""
    nonnegative(T=T, sigma=sigma)
    positive(S=S, S_extreme=X)
    regular = sigma * np.sqrt(T) > 0
    return S * np.exp((r - q) * T), regular, np.where(regular, T, 1.0), np.where(regular, sigma, 1.0)


def floating_lookback_price(
    S: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0, option_type: str = "call", S_extreme: ArrayLike | None = None
) -> np.ndarray | float:
    """Goldman-Sosin-Gatto (1979) floating-strike lookback: :math:`S_T - \\min S` (call) or :math:`\\max S - S_T` (put).

    With :math:`a_1 = [\\ln(S/S_{\\min}) + (b + \\sigma^2/2)T]/(\\sigma\\sqrt T)`,
    :math:`a_2 = a_1 - \\sigma\\sqrt T` and :math:`b = r - q`, the call is

    .. math:: c = S e^{(b-r)T} N(a_1) - S_{\\min} e^{-rT} N(a_2)
              + S e^{-rT} \\frac{\\sigma^2}{2b}\\left[\\left(\\frac{S}{S_{\\min}}\\right)^{-2b/\\sigma^2}
              N\\!\\left(-a_1 + \\frac{2b\\sqrt T}{\\sigma}\\right) - e^{bT} N(-a_1)\\right],

    and the put is its mirror image in the maximum.

    Parameters
    ----------
    S, T, r, sigma : array_like
        Spot, remaining maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    S_extreme : array_like, optional
        Minimum (call) or maximum (put) observed so far; defaults to `S`
        for a newly written option.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.15.1: S = 120, S_min = 100, T = 0.5, r = 10%,
    b = 4%, sigma = 30%:

    >>> round(floating_lookback_price(120.0, 0.5, 0.10, 0.30, q=0.06, S_extreme=100.0), 3)
    25.353
    """
    phi = check_option_type(option_type)
    S, T, r, sigma, q, X = _inputs(S, T, r, sigma, q, S if S_extreme is None else S_extreme)
    end, regular, Ts, sig = _split(S, X, T, r, q, sigma)
    b = r - q
    if phi == 1:
        X = np.minimum(X, S)
        out = S * np.exp((b - r) * Ts) - X * np.exp(-r * Ts) + _blocks(S, X, Ts, r, b, sig, upper=False)
        deterministic = end - np.minimum(X, end)
    else:
        X = np.maximum(X, S)
        out = X * np.exp(-r * Ts) - S * np.exp((b - r) * Ts) + _blocks(S, X, Ts, r, b, sig, upper=True)
        deterministic = np.maximum(X, end) - end
    return as_output(np.where(regular, out, np.exp(-r * T) * deterministic))


def fixed_lookback_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    S_extreme: ArrayLike | None = None,
) -> np.ndarray | float:
    """Conze-Viswanathan (1991) fixed-strike lookback: :math:`(\\max S - K)^+` (call) or :math:`(K - \\min S)^+` (put).

    For the call with :math:`K \\ge S_{\\max}`,

    .. math:: c = S e^{(b-r)T} N(d_1) - K e^{-rT} N(d_2)
              + S e^{-rT} \\frac{\\sigma^2}{2b}\\left[-\\left(\\frac{S}{K}\\right)^{-2b/\\sigma^2}
              N\\!\\left(d_1 - \\frac{2b\\sqrt T}{\\sigma}\\right) + e^{bT} N(d_1)\\right];

    for :math:`K < S_{\\max}` the call locks in :math:`e^{-rT}(S_{\\max} - K)`
    and adds the same expression with :math:`S_{\\max}` in place of
    :math:`K`. The put mirrors it in the minimum.

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, remaining maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    S_extreme : array_like, optional
        Maximum (call) or minimum (put) observed so far; defaults to `S`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A lookback call is worth more than the vanilla call:

    >>> from financialkit.options import black_scholes_price
    >>> bool(fixed_lookback_price(100.0, 100.0, 1.0, 0.05, 0.2) > black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2))
    True
    """
    phi = check_option_type(option_type)
    S, K, T, r, sigma, q, X = _inputs(S, K, T, r, sigma, q, S if S_extreme is None else S_extreme)
    end, regular, Ts, sig = _split(S, X, T, r, q, sigma)
    nonnegative(K=K)
    b = r - q
    disc = np.exp(-r * Ts)
    if phi == 1:
        X = np.maximum(X, S)
        out = disc * np.maximum(X - K, 0.0) + _blocks(S, np.maximum(K, X), Ts, r, b, sig, upper=True)
        deterministic = np.maximum(np.maximum(X, end) - K, 0.0)
    else:
        X = np.minimum(X, S)
        # K = 0 makes the put worthless; any positive level keeps the closed form finite there
        out = disc * np.maximum(K - X, 0.0) + _blocks(S, np.where(K > 0, np.minimum(K, X), X), Ts, r, b, sig, upper=False)
        out = np.where(K > 0, out, 0.0)
        deterministic = np.maximum(K - np.minimum(X, end), 0.0)
    return as_output(np.where(regular, out, np.exp(-r * T) * deterministic))


def lookback_option_mc(
    S: float,
    T: float,
    r: float,
    sigma: float | None = None,
    q: float = 0.0,
    option_type: str = "call",
    K: float | None = None,
    n_steps: int = 50,
    n_paths: int = 100_000,
    continuous: bool = True,
    rng: RNGLike = None,
    confidence: float = 0.95,
    paths: ArrayLike | None = None,
    discount: ArrayLike | None = None,
) -> MonteCarloResult:
    """Monte Carlo price of a newly written lookback, floating (``K=None``) or fixed strike.

    Paths come from :func:`financialkit.integrators.gbm_paths`. With
    ``continuous=True`` the path extremum is sampled exactly from the
    Brownian bridge in each step
    (:func:`~financialkit.exotic_options.utils.numerics.bridge_extremum`), so
    the estimator is unbiased for continuous monitoring at any step size;
    with ``continuous=False`` it is the extremum over the grid dates,
    which prices a discretely monitored lookback. Any other model's spot
    paths can be given as `paths`; the bridge then assumes volatility
    `sigma` within each step.

    Parameters
    ----------
    S, T, r : float
        Spot, maturity and rate.
    sigma : float, optional
        Volatility of the GBM paths and of the bridge; required unless
        `paths` are given with ``continuous=False``.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    K : float, optional
        Fixed strike; ``None`` for a floating-strike lookback.
    n_steps : int
        Time steps (monitoring dates when ``continuous=False``); ignored with `paths`.
    n_paths : int
        Number of paths; ignored with `paths`.
    continuous : bool
        Continuous monitoring through the bridge, or discrete.
    rng : numpy.random.Generator or int, optional
        Generator or seed (for the GBM paths and the bridge samples).
    confidence : float
        Coverage of the confidence interval.
    paths : array_like, shape (n_paths, n_steps + 1), optional
        Risk-neutral spot paths starting at `S` on the grid :math:`t_i = iT/n`,
        in place of the GBM ones.
    discount : array_like, optional
        Discount factors :math:`D(0, t_i)` for `paths`, of shape
        ``(n_steps + 1,)`` or one row per path; only :math:`D(0, T)` is
        used. :math:`e^{-rT}` by default.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> res = lookback_option_mc(100.0, 1.0, 0.05, 0.2, n_steps=10, n_paths=40_000, rng=0)
    >>> bool(abs(res.estimate - floating_lookback_price(100.0, 1.0, 0.05, 0.2)) < 4 * res.std_error)
    True
    """
    phi = check_option_type(option_type)
    gen = np.random.default_rng(rng)
    spot, D = spot_paths(S, T, r - q, sigma, int(n_steps), int(n_paths), paths, discount, rng=gen)
    # the call needs the maximum for a fixed strike and the minimum for a floating one
    kind = "max" if (phi == 1) == (K is not None) else "min"
    if continuous:
        if sigma is None:
            raise ValueError("continuous monitoring needs sigma for the bridge")
        extreme = np.exp(bridge_extremum(np.log(spot), sigma, T / (spot.shape[1] - 1), kind, rng=gen))
    else:
        extreme = spot.max(axis=1) if kind == "max" else spot.min(axis=1)
    S_T = spot[:, -1]
    if K is None:
        payoff = S_T - extreme if phi == 1 else extreme - S_T
    else:
        payoff = np.maximum(phi * (extreme - K), 0.0)
    disc = np.exp(-r * T) if D is None else D[:, -1]
    return sample_mean(disc * payoff, confidence, "continuous lookback" if continuous else "discrete lookback")
