"""Estimation-aware optimization: Black-Litterman (1992), Michaud resampling (1998) and Goldfarb-Iyengar robust portfolios (2003).

Mean-variance weights are wildly sensitive to expected returns, which
are the hardest inputs to estimate. Black and Litterman start instead
from the returns that make the market portfolio optimal,
:math:`\\Pi = \\delta\\Sigma w_{mkt}`, treat them as a prior with
covariance :math:`\\tau\\Sigma`, and update them with the investor's views
:math:`P\\mu = Q + \\varepsilon`, :math:`\\varepsilon \\sim N(0, \\Omega)`.
The posterior is

.. math::

   \\mu_{BL} = \\left[(\\tau\\Sigma)^{-1} + P'\\Omega^{-1}P\\right]^{-1}
              \\left[(\\tau\\Sigma)^{-1}\\Pi + P'\\Omega^{-1}Q\\right],

and the optimal portfolio departs from market weights only in the
directions the views name. Michaud instead averages the frontiers of
many inputs resampled from the estimates, which spreads weight across
assets that the classical frontier treats as interchangeable. Goldfarb
and Iyengar take the opposite stance to averaging: they let the inputs
range over an uncertainty set and optimize the worst case. With a factor
model :math:`r = \\mu + Bf + \\varepsilon`, a box around the mean,
:math:`|\\mu_i - \\hat\\mu_i| \\le \\gamma_i`, and a ball around each
asset's loadings, :math:`\\|B_i - \\hat B_i\\|_F \\le \\rho_i` in the norm
:math:`\\|v\\|_F = \\sqrt{v'Fv}`, the worst cases are explicit:

.. math::

   \\min_{\\mu} w'\\mu = w'\\hat\\mu - \\gamma'|w|, \\qquad
   \\max_{B} w'BFB'w = \\left(\\|F^{1/2}\\hat B'w\\| + \\rho'|w|\\right)^2,

so the robust problem stays convex: errors in the inputs act as a
penalty on the gross position in each asset.

References: F. Black and R. Litterman, "Global Portfolio Optimization,"
*Financial Analysts J.* 48 (1992) 28-43; G. He and R. Litterman, "The
Intuition Behind Black-Litterman Model Portfolios," Goldman Sachs
(1999); R. O. Michaud, *Efficient Asset Management*, Harvard Business
School Press (1998), Ch. 6; D. Goldfarb and G. Iyengar, "Robust Portfolio
Selection Problems," *Math. Oper. Res.* 28 (2003) 1-38.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

from financialkit.constants import DEFAULT_MAX_ITER
from financialkit.portfolio.core.base import BlackLittermanResult, ResampledFrontierResult, RobustPortfolioResult
from financialkit.portfolio.systems.mean_variance import efficient_frontier
from financialkit.portfolio.utils.optimize import as_mean_cov

__all__ = ["implied_equilibrium_returns", "black_litterman", "resampled_frontier", "robust_portfolio"]


def implied_equilibrium_returns(cov: ArrayLike, market_weights: ArrayLike, delta: float = 2.5) -> np.ndarray:
    """Reverse optimization: the excess returns that make the market portfolio optimal.

    Parameters
    ----------
    cov : array_like, shape (d, d)
    market_weights : array_like, shape (d,)
        Capitalisation weights.
    delta : float
        Market risk aversion, the market's excess return over its variance.

    Returns
    -------
    ndarray, shape (d,)
        :math:`\\Pi = \\delta\\Sigma w_{mkt}`.

    Examples
    --------
    >>> implied_equilibrium_returns([[0.04, 0.01], [0.01, 0.09]], [0.6, 0.4], 2.0)
    array([0.056, 0.084])
    """
    _, C = as_mean_cov(None, cov)
    return delta * C @ np.asarray(market_weights, dtype=np.float64)


def black_litterman(
    cov: ArrayLike,
    market_weights: ArrayLike,
    P: ArrayLike,
    Q: ArrayLike,
    omega: Optional[ArrayLike] = None,
    tau: float = 0.05,
    delta: float = 2.5,
) -> BlackLittermanResult:
    """Black-Litterman posterior returns and optimal weights.

    Parameters
    ----------
    cov : array_like, shape (d, d)
        Covariance of excess returns.
    market_weights : array_like, shape (d,)
    P : array_like, shape (k, d)
        Each row is a view portfolio (absolute or relative).
    Q : array_like, shape (k,)
        Expected excess return of each view portfolio.
    omega : array_like, shape (k, k), optional
        Covariance of the view errors. Defaults to He and Litterman's
        :math:`\\mathrm{diag}(P\\tau\\Sigma P')`, which gives each view the
        same confidence as the prior.
    tau : float
        Scale of the prior's uncertainty.
    delta : float
        Risk aversion.

    Returns
    -------
    BlackLittermanResult

    Examples
    --------
    He and Litterman (1999): the view that German equities outperform the
    rest of Europe by 5% raises Germany's expected return to about 11%.
    See :func:`implied_equilibrium_returns` and the tests for the full
    seven-country example.

    >>> res = black_litterman([[0.04, 0.0], [0.0, 0.04]], [0.5, 0.5], [[1.0, 0.0]], [0.10])
    >>> np.round(res.posterior_returns, 4)
    array([0.075, 0.05 ])
    """
    _, C = as_mean_cov(None, cov)
    w = np.asarray(market_weights, dtype=np.float64)
    Pm = np.atleast_2d(np.asarray(P, dtype=np.float64))
    q = np.atleast_1d(np.asarray(Q, dtype=np.float64))
    if Pm.shape != (q.size, C.shape[0]):
        raise ValueError(f"P must have shape ({q.size}, {C.shape[0]}), got {Pm.shape}")
    pi = implied_equilibrium_returns(C, w, delta)
    tS = tau * C
    Om = np.diag(np.diag(Pm @ tS @ Pm.T)) if omega is None else np.atleast_2d(np.asarray(omega, dtype=np.float64))
    # Theil's mixed estimator in its update form: avoids inverting tau * Sigma.
    gain = tS @ Pm.T @ np.linalg.inv(Pm @ tS @ Pm.T + Om)
    mu_bl = pi + gain @ (q - Pm @ pi)
    M = tS - gain @ Pm @ tS
    post_cov = C + M
    return BlackLittermanResult(
        prior_returns=pi,
        posterior_returns=mu_bl,
        posterior_cov=post_cov,
        weights=np.linalg.solve(delta * post_cov, mu_bl),
        market_weights=w,
        tau=float(tau),
        delta=float(delta),
    )


def resampled_frontier(
    mu: ArrayLike,
    cov: ArrayLike,
    n_obs: int,
    n_resamples: int = 50,
    n_points: int = 15,
    long_only: bool = True,
    rng: np.random.Generator | int | None = None,
) -> ResampledFrontierResult:
    """Michaud's resampled efficient frontier.

    Draws ``n_resamples`` histories of ``n_obs`` normal returns with the
    given moments, computes each history's frontier at ``n_points``
    equally spaced ranks (from its minimum-variance portfolio to its
    highest-return one) and averages the weights rank by rank. The
    averaged portfolios are then evaluated under the true inputs.

    Parameters
    ----------
    mu : array_like, shape (d,)
    cov : array_like, shape (d, d)
        Point estimates, treated as the truth for resampling.
    n_obs : int
        Length of each simulated history, the amount of data the
        estimates are as reliable as.
    n_resamples, n_points : int
    long_only : bool
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    ResampledFrontierResult
    """
    m, C = as_mean_cov(mu, cov)
    assert m is not None
    gen = np.random.default_rng(rng)
    samples = np.empty((n_resamples, n_points, m.size))
    for s in range(n_resamples):
        X = gen.multivariate_normal(m, C, size=n_obs)
        ms, Cs = X.mean(axis=0), np.cov(X, rowvar=False)
        samples[s] = efficient_frontier(ms, Cs, n_points, long_only, max_return=ms.max() if long_only else None).weights
    W = samples.mean(axis=0)
    return ResampledFrontierResult(
        weights=W,
        sample_weights=samples,
        returns=W @ m,
        volatilities=np.sqrt(np.einsum("ij,jk,ik->i", W, C, W)),
        frontier=efficient_frontier(m, C, n_points, long_only),
    )


def robust_portfolio(
    mu: ArrayLike,
    exposures: ArrayLike,
    factor_cov: ArrayLike,
    specific_var: ArrayLike,
    risk_aversion: float,
    mu_halfwidth: ArrayLike = 0.0,
    loading_radius: ArrayLike = 0.0,
    long_only: bool = True,
) -> RobustPortfolioResult:
    """Goldfarb and Iyengar's robust mean-variance portfolio under a factor model.

    Maximizes the worst-case mean-variance utility over the uncertainty
    set of the module docstring,

    .. math::

       \\max_{\\mathbf 1'w = 1}\\; w'\\hat\\mu - \\gamma_\\mu'|w|
       - \\frac{\\gamma}{2}\\left[\\left(\\|F^{1/2}\\hat B'w\\| + \\rho'|w|\\right)^2
       + w'\\bar D w\\right],

    where ``specific_var`` is the upper bound :math:`\\bar D` of the
    specific variances. Goldfarb and Iyengar solve it as a second-order
    cone program; here :func:`scipy.optimize.minimize` (SLSQP) solves it
    with :math:`w` split into long and short parts, so that :math:`|w|`
    is smooth. With no uncertainty it is ordinary mean-variance
    optimization with a budget constraint.

    Parameters
    ----------
    mu : array_like, shape (d,)
        Point estimate :math:`\\hat\\mu` of expected returns.
    exposures : array_like, shape (d, k)
        Point estimate :math:`\\hat B` of the factor loadings.
    factor_cov : array_like, shape (k, k)
        Factor covariance :math:`F`, taken as known.
    specific_var : array_like, shape (d,)
        Upper bound of each asset's specific variance.
    risk_aversion : float
        Coefficient :math:`\\gamma > 0`.
    mu_halfwidth : float or array_like, shape (d,)
        Half-width :math:`\\gamma_{\\mu,i}` of each mean's confidence interval.
    loading_radius : float or array_like, shape (d,)
        Radius :math:`\\rho_i` of each asset's loading uncertainty, in the
        :math:`F`-norm, so that it is measured in units of volatility.
    long_only : bool

    Returns
    -------
    RobustPortfolioResult

    Examples
    --------
    Two uncorrelated assets with the same estimated mean and risk, one
    estimated less precisely: the robust investor tilts away from it.

    >>> res = robust_portfolio([0.08, 0.08], [[0.0], [0.0]], [[1.0]], [0.04, 0.04], 2.0, mu_halfwidth=[0.0, 0.02])
    >>> np.round(res.weights, 4), round(res.worst_case_return, 4)
    (array([0.625, 0.375]), 0.0725)
    """
    m, _ = as_mean_cov(mu, np.diag(np.asarray(specific_var, dtype=np.float64)))
    assert m is not None
    d = m.size
    B = np.atleast_2d(np.asarray(exposures, dtype=np.float64))
    F = np.atleast_2d(np.asarray(factor_cov, dtype=np.float64))
    D = np.asarray(specific_var, dtype=np.float64)
    if B.shape[0] != d or F.shape != (B.shape[1], B.shape[1]):
        raise ValueError(f"exposures must have shape ({d}, k) and factor_cov (k, k), got {B.shape} and {F.shape}")
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    g = np.broadcast_to(np.asarray(mu_halfwidth, dtype=np.float64), (d,))
    rho = np.broadcast_to(np.asarray(loading_radius, dtype=np.float64), (d,))
    if np.any(g < 0) or np.any(rho < 0):
        raise ValueError("mu_halfwidth and loading_radius must be non-negative")
    # Columns of H' w are the factor exposures in units of volatility: ||H' w|| = sqrt(w' B F B' w).
    H = B @ np.linalg.cholesky(F)
    lam = float(risk_aversion)

    def worst(w):
        s = np.sqrt((H.T @ w) @ (H.T @ w))
        return w @ m - g @ np.abs(w), s + rho @ np.abs(w), s

    # z = (w+, w-); |w| = w+ + w- at the optimum because both penalties are non-negative.
    n_var = d if long_only else 2 * d
    sign = np.ones(n_var) if long_only else np.r_[np.ones(d), -np.ones(d)]

    def objective(z):
        w = z[:d] - (0.0 if long_only else z[d:])
        y = H.T @ w
        s = np.sqrt(y @ y)
        a = z[:d] + (0.0 if long_only else z[d:])
        t = s + rho @ a
        val = -(w @ m - g @ a) + 0.5 * lam * (t**2 + w**2 @ D)
        ds = H @ y / s if s > 0 else np.zeros(d)
        grad_w = -m + lam * (t * ds + w * D)
        grad_a = g + lam * t * rho
        grad = np.tile(grad_w, n_var // d) * sign + np.tile(grad_a, n_var // d)
        return val, grad

    res = optimize.minimize(
        objective,
        np.r_[np.full(d, 1.0 / d), np.zeros(n_var - d)],
        jac=True,
        method="SLSQP",
        bounds=[(0.0, None)] * n_var,
        constraints=[{"type": "eq", "fun": lambda z: sign @ z - 1.0, "jac": lambda z: sign}],
        options={"ftol": 1e-15, "maxiter": DEFAULT_MAX_ITER},
    )
    if not res.success:
        raise RuntimeError(f"robust optimization failed: {res.message}")
    w = res.x[:d] - (0.0 if long_only else res.x[d:])
    wc_ret, wc_factor_vol, s = worst(w)
    return RobustPortfolioResult(
        weights=w,
        worst_case_return=float(wc_ret),
        worst_case_volatility=float(np.sqrt(wc_factor_vol**2 + w**2 @ D)),
        nominal_return=float(w @ m),
        nominal_volatility=float(np.sqrt(s**2 + w**2 @ D)),
        risk_aversion=lam,
    )
