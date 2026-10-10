"""American options by simulation: the Longstaff-Schwartz least-squares method.

Simulation runs forward in time and early exercise is decided backward,
which kept American options out of Monte Carlo's reach until Longstaff
and Schwartz (2001) estimated the continuation value by regressing the
realized discounted cash flows on functions of the current price, using
only in-the-money paths. Working backward from expiry, a path is
exercised wherever the immediate payoff beats the fitted continuation
value, and the option's value is the mean discounted cash flow of that
policy. The algorithm is hand-written here: no allowed dependency
provides it.

References: F. A. Longstaff and E. S. Schwartz, "Valuing American Options
by Simulation: A Simple Least-Squares Approach," *Review of Financial
Studies* 14 (2001) 113-147; P. Glasserman, *Monte Carlo Methods in
Financial Engineering* (Springer, 2004), Section 8.6.
"""

from __future__ import annotations

import numpy as np

from financialkit.exotic_options.core.base import LongstaffSchwartzResult
from financialkit.integrators import RNGLike, antithetic_increments, brownian_increments, gbm_paths
from financialkit.monte_carlo import vanilla_payoff
from financialkit.options.utils.common import option_sign

__all__ = ["longstaff_schwartz", "regression_basis"]


def regression_basis(x: np.ndarray, degree: int = 3, basis: str = "laguerre") -> np.ndarray:
    """Regressors for the continuation value: a constant and `degree` basis functions of `x`.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Moneyness :math:`S/K` of the in-the-money paths.
    degree : int
        Number of non-constant basis functions.
    basis : {"laguerre", "monomial"}
        Longstaff and Schwartz's weighted Laguerre polynomials
        :math:`e^{-x/2} L_k(x)`, :math:`k = 0, \\dots, \\text{degree} - 1`, or
        the powers :math:`x, \\dots, x^{\\text{degree}}`.

    Returns
    -------
    ndarray, shape (n, degree + 1)

    Examples
    --------
    >>> regression_basis([0.0, 1.0], degree=2).round(4).tolist()
    [[1.0, 1.0, 1.0], [1.0, 0.6065, 0.0]]
    """
    x = np.asarray(x, dtype=np.float64)
    if basis == "laguerre":
        cols = np.exp(-0.5 * x)[:, None] * np.polynomial.laguerre.lagvander(x, int(degree) - 1)
    elif basis == "monomial":
        cols = np.polynomial.polynomial.polyvander(x, int(degree))[:, 1:]
    else:
        raise ValueError(f"basis must be 'laguerre' or 'monomial', got {basis!r}")
    return np.column_stack([np.ones_like(x), cols])


def longstaff_schwartz(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "put",
    n_exercise: int = 50,
    n_paths: int = 50_000,
    degree: int = 3,
    basis: str = "laguerre",
    antithetic: bool = True,
    rng: RNGLike = None,
    confidence: float = 0.95,
) -> LongstaffSchwartzResult:
    """American (Bermudan) option value by least-squares Monte Carlo (Longstaff-Schwartz 2001).

    Exercise is allowed on `n_exercise` equally spaced dates up to
    :math:`T`. At each date :math:`t_i`, working backward, the discounted
    future cash flow :math:`Y` of each in-the-money path is regressed on
    :func:`regression_basis` of :math:`S_{t_i}/K`, and the path is exercised
    where the payoff exceeds the fitted :math:`E[Y \\mid S_{t_i}]`. The paths
    come from :func:`financialkit.integrators.gbm_paths`, optionally with
    antithetic increments.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, maturity, rate and volatility.
    q : float
        Continuous dividend yield.
    option_type : {"call", "put"}
    n_exercise : int
        Number of exercise dates (time steps).
    n_paths : int
        Number of paths (rounded up to even with `antithetic`).
    degree : int
        Number of non-constant regressors.
    basis : {"laguerre", "monomial"}
    antithetic : bool
        Simulate antithetic pairs; the standard error then uses pair means.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    confidence : float
        Coverage of the confidence interval.

    Returns
    -------
    LongstaffSchwartzResult

    Examples
    --------
    Longstaff and Schwartz (2001), Table 1, first row: S = 36, K = 40,
    r = 6%, sigma = 20%, T = 1, 50 exercise dates; the finite-difference
    value is 4.478:

    >>> res = longstaff_schwartz(36.0, 40.0, 1.0, 0.06, 0.2, n_paths=20_000, rng=0)
    >>> bool(abs(res.estimate - 4.478) < 4 * res.std_error + 0.02)
    True
    """
    phi = option_sign(option_type)
    n, m = int(n_exercise), int(n_paths)
    if antithetic:
        m += m % 2
        dW = antithetic_increments(T, n, m, rng=rng)
    else:
        dW = brownian_increments(T, n, m, rng=rng)
    times, paths = gbm_paths(S, r - q, sigma, T, n, m, dW=dW)
    dt = T / n
    growth = np.exp(-r * dt)
    cash = vanilla_payoff(paths[:, -1], K, option_type)
    boundary = np.full(n + 1, np.nan)
    if cash.any():
        boundary[-1] = K
    exercised_early = np.zeros(m, dtype=bool)
    for i in range(n - 1, 0, -1):
        cash *= growth
        exercise_value = vanilla_payoff(paths[:, i], K, option_type)
        itm = np.nonzero(exercise_value > 0)[0]
        if itm.size <= degree + 1:
            continue
        X = regression_basis(paths[itm, i] / K, degree, basis)
        coef, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        exercise = exercise_value[itm] > X @ coef
        idx = itm[exercise]
        cash[idx] = exercise_value[idx]
        exercised_early[idx] = True
        grid = np.linspace(paths[itm, i].min(), paths[itm, i].max(), 256)
        policy = vanilla_payoff(grid, K, option_type) > regression_basis(grid / K, degree, basis) @ coef
        if policy.any():
            boundary[i] = grid[policy].max() if phi == -1 else grid[policy].min()
    cash *= growth
    samples = 0.5 * (cash[: m // 2] + cash[m // 2 :]) if antithetic else cash
    estimate = float(cash.mean())
    se = float(samples.std(ddof=1) / np.sqrt(samples.size))
    intrinsic = float(vanilla_payoff(S, K, option_type))
    if intrinsic > estimate:  # exercising at once beats the simulated policy
        estimate, se = intrinsic, 0.0
    return LongstaffSchwartzResult(estimate, se, m, confidence, "longstaff_schwartz", times, boundary, float(exercised_early.mean()))
