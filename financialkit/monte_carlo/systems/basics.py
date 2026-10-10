"""Crude Monte Carlo: integration with error bars, normal variates and option prices.

* :func:`monte_carlo_integrate` -- the method of Metropolis and Ulam
  (1949): an integral over the unit cube is the mean of the integrand at
  uniform points, with a :math:`1/\\sqrt{n}` standard error whatever the
  dimension.
* :func:`box_muller` and :func:`box_muller_normals` -- Box and Muller's
  (1958) exact transform of two uniforms into two independent normals.
* :func:`european_option_mc` -- Boyle (1977): an option is the discounted
  risk-neutral mean of its payoff.

References: N. Metropolis and S. Ulam, "The Monte Carlo Method," *JASA*
44 (1949) 335-341; G. E. P. Box and M. E. Muller, "A Note on the
Generation of Random Normal Deviates," *Ann. Math. Statist.* 29 (1958)
610-611; P. P. Boyle, "Options: A Monte Carlo Approach," *J. Financial
Economics* 4 (1977) 323-338; P. Glasserman, *Monte Carlo Methods in
Financial Engineering* (Springer, 2004), Chapters 1-2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import RNGLike
from financialkit.monte_carlo.core.base import MonteCarloResult
from financialkit.monte_carlo.utils.numerics import sample_mean, terminal_prices, vanilla_payoff

__all__ = ["monte_carlo_integrate", "box_muller", "box_muller_normals", "european_option_mc"]


def monte_carlo_integrate(f: Callable[[np.ndarray], ArrayLike], n: int, dim: int = 1, rng: RNGLike = None, confidence: float = 0.95) -> MonteCarloResult:
    """Crude Monte Carlo estimate of :math:`\\int_{[0,1)^d} f(u)\\,du` (Metropolis-Ulam 1949).

    Parameters
    ----------
    f : callable
        Vectorised integrand: takes points of shape ``(n, dim)`` and
        returns ``n`` values.
    n : int
        Number of uniform points.
    dim : int
        Dimension of the unit cube.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    The area of the quarter disc is :math:`\\pi / 4`:

    >>> import numpy as np
    >>> res = monte_carlo_integrate(lambda u: 4.0 * (u[:, 0] ** 2 + u[:, 1] ** 2 < 1), 100_000, dim=2, rng=0)
    >>> res.covers(np.pi)
    True
    """
    u = np.random.default_rng(rng).random((int(n), int(dim)))
    return sample_mean(np.asarray(f(u), dtype=np.float64), confidence)


def box_muller(u1: ArrayLike, u2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Box-Muller transform of two independent uniforms into two independent standard normals.

    .. math:: R = \\sqrt{-2 \\ln u_1}, \\quad z_1 = R \\cos(2\\pi u_2), \\quad z_2 = R \\sin(2\\pi u_2).

    :math:`R^2` is exponential with mean 2, the law of the squared radius
    of a standard bivariate normal, and the angle is uniform.

    Parameters
    ----------
    u1, u2 : array_like
        Uniforms on :math:`(0, 1]` and :math:`[0, 1)`.

    Returns
    -------
    z1, z2 : ndarray
    """
    u1, u2 = np.asarray(u1, dtype=np.float64), np.asarray(u2, dtype=np.float64)
    R = np.sqrt(-2.0 * np.log(u1))
    return R * np.cos(2.0 * np.pi * u2), R * np.sin(2.0 * np.pi * u2)


def box_muller_normals(n: int, rng: RNGLike = None) -> np.ndarray:
    """``n`` standard normals by the Box-Muller transform of fresh uniforms.

    Parameters
    ----------
    n : int
        Number of normals.
    rng : numpy.random.Generator or int, optional
        Generator or seed.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> z = box_muller_normals(200_000, rng=0)
    >>> round(float(z.mean()), 2) == 0.0, round(float(z.std()), 2)
    (True, 1.0)
    """
    m = (int(n) + 1) // 2
    u = np.random.default_rng(rng).random((2, m))
    z1, z2 = box_muller(1.0 - u[0], u[1])
    return np.concatenate([z1, z2])[: int(n)]


def european_option_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    rng: RNGLike = None,
    z: ArrayLike | None = None,
    confidence: float = 0.95,
) -> MonteCarloResult:
    """European option price by crude Monte Carlo under Black-Scholes (Boyle 1977).

    .. math:: V = e^{-rT} E^Q[\\text{payoff}(S_T)] \\approx \\frac{e^{-rT}}{n} \\sum_i \\text{payoff}(S_T^{(i)})

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, risk-free rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_paths : int
        Number of terminal prices; ignored if `z` is given.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    z : array_like, optional
        Standard normals to use instead of fresh ones.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    Hull, 10th ed., Example 15.6 (Black-Scholes value 4.76):

    >>> res = european_option_mc(42.0, 40.0, 0.5, 0.10, 0.20, n_paths=200_000, rng=0)
    >>> res.covers(4.7594)
    True
    """
    z = np.random.default_rng(rng).standard_normal(int(n_paths)) if z is None else np.asarray(z, dtype=np.float64)
    payoff = np.exp(-r * T) * vanilla_payoff(terminal_prices(S, T, r, sigma, z, q), K, option_type)
    return sample_mean(payoff, confidence)
