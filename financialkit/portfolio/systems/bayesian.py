"""Estimation-aware optimization: Black-Litterman (1992) and Michaud resampling (1998).

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
assets that the classical frontier treats as interchangeable.

References: F. Black and R. Litterman, "Global Portfolio Optimization,"
*Financial Analysts J.* 48 (1992) 28-43; G. He and R. Litterman, "The
Intuition Behind Black-Litterman Model Portfolios," Goldman Sachs
(1999); R. O. Michaud, *Efficient Asset Management*, Harvard Business
School Press (1998), Ch. 6.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from financialkit.portfolio.core.base import BlackLittermanResult, ResampledFrontierResult
from financialkit.portfolio.systems.mean_variance import efficient_frontier
from financialkit.portfolio.utils.optimize import as_mean_cov

__all__ = ["implied_equilibrium_returns", "black_litterman", "resampled_frontier"]


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
