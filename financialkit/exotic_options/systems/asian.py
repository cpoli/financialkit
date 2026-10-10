"""Asian (average-rate) options: geometric closed forms, arithmetic Monte Carlo and moment matching.

* :func:`geometric_asian_price` -- Kemna and Vorst (1990): a geometric
  average of lognormal prices is lognormal, so its option is Black's
  formula with volatility :math:`\\sigma/\\sqrt3` (continuous averaging) or
  the exact discrete-fixing analogue.
* :func:`arithmetic_asian_mc` -- Kemna and Vorst's other contribution:
  the arithmetic average has no closed form, but the geometric option is
  an almost perfectly correlated control variate for it.
* :func:`turnbull_wakeman_price` -- Turnbull and Wakeman (1991): match
  the first two moments of the arithmetic average to a lognormal, with
  Levy's (1992) exact discrete-fixing moments.

Averages are over the fixings :math:`t_i = iT/n`, :math:`i = 1, \\dots, n`
(the spot at time 0 is not included), or continuous over
:math:`[0, T]` when ``n_fixings`` is ``None``.

References: A. G. Z. Kemna and A. C. F. Vorst, "A Pricing Method for
Options Based on Average Asset Values," *J. Banking and Finance* 14
(1990) 113-129; S. M. Turnbull and L. M. Wakeman, "A Quick Algorithm for
Pricing European Average Options," *J. Financial and Quantitative
Analysis* 26 (1991) 377-389; E. Levy, "Pricing European Average Rate
Currency Options," *J. Int. Money and Finance* 11 (1992) 474-491; E. G.
Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed.
(McGraw-Hill, 2007), Section 4.20.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import RNGLike, gbm_paths
from financialkit.monte_carlo import ControlVariateResult, MonteCarloResult, control_variate_estimate, sample_mean, vanilla_payoff
from financialkit.options import black76_price
from financialkit.options.utils.common import as_output

__all__ = ["geometric_asian_price", "arithmetic_asian_mc", "turnbull_wakeman_price"]


def _fixings(T: float, n: int) -> np.ndarray:
    return T * np.arange(1, int(n) + 1) / int(n)


def geometric_asian_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    n_fixings: int | None = None,
) -> np.ndarray | float:
    """Kemna-Vorst (1990) price of an option on the geometric average :math:`G`.

    :math:`\\ln G` is normal with mean :math:`m` and variance :math:`v`, so the
    option is Black's formula on the forward :math:`e^{m + v/2}` with
    volatility :math:`\\sqrt{v/T}`. For continuous averaging
    :math:`m = \\ln S + (b - \\sigma^2/2) T/2` and :math:`v = \\sigma^2 T/3`;
    for :math:`n` equally spaced fixings

    .. math:: m = \\ln S + (b - \\tfrac{\\sigma^2}{2}) \\frac{(n+1) T}{2n}, \\qquad
              v = \\sigma^2 T \\frac{(n+1)(2n+1)}{6 n^2}.

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity, rate and volatility.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_fixings : int, optional
        Number of fixings; ``None`` for continuous averaging.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Haug (2007), Section 4.20.1: S = 80, K = 85, T = 0.25, r = 5%, b = 8%,
    sigma = 20%:

    >>> round(geometric_asian_price(80.0, 85.0, 0.25, 0.05, 0.20, q=-0.03, option_type="put"), 4)
    4.6922
    """
    S, T, r, sigma, q = (np.asarray(a, dtype=np.float64) for a in (S, T, r, sigma, q))
    if n_fixings is None:
        mean_time, var_factor = 0.5, 1.0 / 3.0
    else:
        n = int(n_fixings)
        mean_time, var_factor = (n + 1) / (2.0 * n), (n + 1) * (2 * n + 1) / (6.0 * n**2)
    v = sigma**2 * T * var_factor
    F = S * np.exp((r - q - 0.5 * sigma**2) * T * mean_time + 0.5 * v)
    return black76_price(F, K, T, r, np.sqrt(v / T), option_type)


def arithmetic_asian_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_fixings: int = 12,
    n_paths: int = 50_000,
    control_variate: bool = True,
    rng: RNGLike = None,
    confidence: float = 0.95,
) -> ControlVariateResult | MonteCarloResult:
    """Arithmetic-average option by Monte Carlo, with Kemna and Vorst's geometric control variate.

    The paths come from :func:`financialkit.integrators.gbm_paths` on the
    fixing grid. The discounted geometric-average payoff on the same
    paths has the known mean :func:`geometric_asian_price` and a
    correlation with the arithmetic payoff above 0.99, so
    :func:`financialkit.monte_carlo.control_variate_estimate` removes almost
    all the variance.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_fixings : int
        Number of equally spaced fixings :math:`t_i = iT/n`.
    n_paths : int
        Number of paths.
    control_variate : bool
        Use the geometric control; otherwise plain Monte Carlo.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    ControlVariateResult or MonteCarloResult
        A :class:`~financialkit.monte_carlo.core.base.ControlVariateResult`
        (which also reports the crude estimate) with the control, a
        :class:`~financialkit.monte_carlo.core.base.MonteCarloResult` without.

    Examples
    --------
    >>> res = arithmetic_asian_mc(100.0, 100.0, 1.0, 0.05, 0.2, n_paths=20_000, rng=0)
    >>> bool(res.std_error < res.crude_std_error / 10)
    True
    """
    _, paths = gbm_paths(S, r - q, sigma, T, int(n_fixings), int(n_paths), rng=rng)
    fixings = paths[:, 1:]
    disc = np.exp(-r * T)
    y = disc * vanilla_payoff(fixings.mean(axis=1), K, option_type)
    if not control_variate:
        return sample_mean(y, confidence, "arithmetic Asian")
    x = disc * vanilla_payoff(np.exp(np.log(fixings).mean(axis=1)), K, option_type)
    mean_x = geometric_asian_price(S, K, T, r, sigma, q, option_type, n_fixings=int(n_fixings))
    res = control_variate_estimate(y, x, mean_x, confidence)
    res.method = "arithmetic Asian, geometric control"
    return res


def _continuous_average_moments(T, b, sigma):
    """:math:`E[A]/S` and :math:`E[A^2]/S^2` for the continuous average over :math:`[0, T]`.

    :math:`E[A^2]/S^2 = (2/T^2) \\int_0^T e^{bt} (e^{at} - 1)/a \\, dt` with
    :math:`a = b + \\sigma^2`. The closed form cancels catastrophically as
    :math:`a \\to 0`, so the smooth integrand (written with ``expm1``) is
    integrated by 64-point Gauss-Legendre quadrature instead.
    """
    safe_b = np.where(b == 0.0, 1.0, b)
    m1 = np.where(b == 0.0, 1.0, np.expm1(safe_b * T) / (safe_b * T))
    a = b + sigma**2
    safe_a = np.where(a == 0.0, 1.0, a)
    x, w = np.polynomial.legendre.leggauss(64)
    t = 0.5 * T[..., None] * (x + 1.0)
    bb, aa, sa = (np.asarray(z)[..., None] for z in (b, a, safe_a))
    integrand = np.exp(bb * t) * np.where(aa == 0.0, t, np.expm1(sa * t) / sa)
    m2 = 2.0 / T**2 * 0.5 * T * np.sum(w * integrand, axis=-1)
    return m1, m2


def turnbull_wakeman_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    n_fixings: int | None = None,
) -> np.ndarray | float:
    """Turnbull-Wakeman (1991) moment-matching approximation of an arithmetic-average option.

    The arithmetic average :math:`A` is replaced by a lognormal with the
    same first two moments :math:`M_1 = E[A]`, :math:`M_2 = E[A^2]`, and the
    option is Black's formula on the forward :math:`M_1` with volatility

    .. math:: \\sigma_A^2 = \\frac{1}{T} \\ln \\frac{M_2}{M_1^2}.

    For continuous averaging (Turnbull and Wakeman)
    :math:`M_1 = S (e^{bT} - 1)/(bT)` and :math:`M_2` is the double integral
    of :math:`E[S_t S_u]`; for discrete fixings the moments are Levy's
    (1992) exact sums :math:`M_1 = \\tfrac{S}{n}\\sum_i e^{b t_i}`,
    :math:`M_2 = \\tfrac{S^2}{n^2}\\sum_{i,j} e^{b(t_i + t_j) + \\sigma^2 \\min(t_i, t_j)}`.
    The approximation is excellent at moderate volatility and drifts as
    :math:`\\sigma^2 T` grows.

    Parameters
    ----------
    S, K, T, r, sigma : array_like
        Spot, strike, maturity, rate and volatility; with discrete fixings
        `T`, `r`, `sigma` and `q` must be scalars.
    q : array_like
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_fixings : int, optional
        Number of equally spaced fixings; ``None`` for continuous averaging.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A call struck at zero is worth the discounted mean of the average,
    which the approximation matches exactly:

    >>> import numpy as np
    >>> c = turnbull_wakeman_price(100.0, 1e-8, 1.0, 0.05, 0.3)
    >>> bool(abs(c - 100.0 * np.exp(-0.05) * np.expm1(0.05) / 0.05) < 1e-6)
    True
    """
    S, T, r, sigma, q = (np.asarray(a, dtype=np.float64) for a in (S, T, r, sigma, q))
    b = r - q
    if n_fixings is None:
        m1, m2 = _continuous_average_moments(T, b, sigma)
    else:
        t = _fixings(float(T), int(n_fixings))
        m1 = np.mean(np.exp(b * t))
        # sum_{i,j} e^{b(t_i + t_j) + sigma^2 min(t_i, t_j)}: the diagonal plus twice the
        # pairs j > i, whose inner sums are reverse cumulative sums -- O(n), not O(n^2).
        later = np.cumsum(np.exp(b * t)[::-1])[::-1] - np.exp(b * t)
        m2 = (np.sum(np.exp((2 * b + sigma**2) * t)) + 2.0 * np.sum(np.exp((b + sigma**2) * t) * later)) / t.size**2
    sigma_A = np.sqrt(np.log(m2 / m1**2) / T)
    return as_output(black76_price(S * m1, K, T, r, sigma_A, option_type))
