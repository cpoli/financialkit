"""Mean-variance portfolio selection: Markowitz (1952), Tobin (1958) and Merton (1972).

Markowitz proposed that an investor care about the mean and variance of
a portfolio's return and hold only *efficient* portfolios, those with
the least variance for their mean. Tobin added a riskless asset: every
investor then holds the same tangency portfolio of risky assets, mixed
with cash according to taste. Merton derived the whole unconstrained
frontier in closed form from four scalars. With short sales excluded,
there is no closed form and each point is a quadratic program, solved
here with :func:`scipy.optimize.minimize` (SLSQP).

References: H. Markowitz, "Portfolio Selection," *J. Finance* 7 (1952)
77-91; J. Tobin, "Liquidity Preference as Behavior Towards Risk," *Rev.
Econ. Stud.* 25 (1958) 65-86; R. C. Merton, "An Analytic Derivation of
the Efficient Portfolio Frontier," *J. Financial Quant. Anal.* 7 (1972)
1851-1872.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from financialkit.portfolio.core.base import FrontierConstants, FrontierResult, PortfolioResult
from financialkit.portfolio.utils.optimize import as_mean_cov, portfolio_moments, solve_qp
from financialkit.risk.systems.safety_first import safety_first_portfolio

__all__ = [
    "frontier_constants",
    "min_variance_portfolio",
    "efficient_portfolio",
    "efficient_frontier",
    "tangency_portfolio",
    "two_fund_allocation",
]


def _result(w: np.ndarray, mu: Optional[np.ndarray], cov: np.ndarray, method: str) -> PortfolioResult:
    mean, vol = portfolio_moments(w, mu, cov)
    return PortfolioResult(weights=w, expected_return=mean, volatility=vol, method=method)


def frontier_constants(mu: ArrayLike, cov: ArrayLike) -> FrontierConstants:
    """Merton's constants :math:`A, B, C, D` of the unconstrained frontier.

    Parameters
    ----------
    mu : array_like, shape (d,)
        Expected returns; not all equal.
    cov : array_like, shape (d, d)
        Positive definite covariance.

    Returns
    -------
    FrontierConstants

    Examples
    --------
    >>> fc = frontier_constants([0.1, 0.2], [[0.04, 0.0], [0.0, 0.09]])
    >>> round(fc.gmv_return, 4), round(fc.gmv_volatility, 4)
    (0.1308, 0.1664)
    """
    m, C = as_mean_cov(mu, cov)
    assert m is not None
    ones = np.ones_like(m)
    x, y = np.linalg.solve(C, m), np.linalg.solve(C, ones)
    A, B, Cc = float(ones @ x), float(m @ x), float(ones @ y)
    D = B * Cc - A * A
    if D <= 0:
        raise ValueError("the frontier is degenerate: all expected returns are equal")
    return FrontierConstants(A=A, B=B, C=Cc, D=D)


def min_variance_portfolio(cov: ArrayLike, mu: Optional[ArrayLike] = None, long_only: bool = False) -> PortfolioResult:
    """The global minimum-variance portfolio.

    Unconstrained, :math:`w = \\Sigma^{-1}\\mathbf 1 / \\mathbf 1'\\Sigma^{-1}\\mathbf 1`.

    Parameters
    ----------
    cov : array_like, shape (d, d)
    mu : array_like, shape (d,), optional
        Expected returns, used only to report the portfolio's mean.
    long_only : bool
        Exclude short sales.

    Returns
    -------
    PortfolioResult

    Examples
    --------
    With uncorrelated assets the weights are proportional to inverse variances.

    >>> res = min_variance_portfolio([[0.04, 0.0], [0.0, 0.01]])
    >>> res.weights, round(res.volatility, 4)
    (array([0.2, 0.8]), 0.0894)
    """
    m, C = as_mean_cov(mu, cov)
    if long_only:
        w = solve_qp(C, np.ones((1, C.shape[0])), [1.0])
    else:
        y = np.linalg.solve(C, np.ones(C.shape[0]))
        w = y / y.sum()
    return _result(w, m, C, "min_variance")


def efficient_portfolio(mu: ArrayLike, cov: ArrayLike, target_return: float, long_only: bool = False) -> PortfolioResult:
    """The least-variance portfolio with a given expected return.

    Unconstrained, Merton's closed form
    :math:`w = \\Sigma^{-1}[(C m - A)\\mu + (B - A m)\\mathbf 1] / D`;
    long-only, a quadratic program.

    Parameters
    ----------
    mu : array_like, shape (d,)
    cov : array_like, shape (d, d)
    target_return : float
        Required mean return; long-only it must lie within ``[min(mu), max(mu)]``.
    long_only : bool

    Returns
    -------
    PortfolioResult

    Examples
    --------
    >>> res = efficient_portfolio([0.1, 0.2], [[0.04, 0.0], [0.0, 0.09]], 0.15)
    >>> res.weights
    array([0.5, 0.5])
    """
    m, C = as_mean_cov(mu, cov)
    assert m is not None
    if long_only:
        if not m.min() - 1e-12 <= target_return <= m.max() + 1e-12:
            raise ValueError("a long-only target return must lie between the smallest and largest expected return")
        d = m.size
        w = solve_qp(C, np.vstack([np.ones(d), m]), [1.0, target_return])
    else:
        fc = frontier_constants(m, C)
        w = np.linalg.solve(C, (fc.C * target_return - fc.A) * m + (fc.B - fc.A * target_return) * np.ones_like(m)) / fc.D
    return _result(w, m, C, "efficient")


def efficient_frontier(
    mu: ArrayLike,
    cov: ArrayLike,
    n_points: int = 50,
    long_only: bool = False,
    max_return: Optional[float] = None,
) -> FrontierResult:
    """Efficient portfolios from the minimum-variance portfolio upwards.

    Parameters
    ----------
    mu : array_like, shape (d,)
    cov : array_like, shape (d, d)
    n_points : int
        Number of target returns, equally spaced.
    long_only : bool
        Exclude short sales; the frontier then ends at the asset with the
        highest expected return.
    max_return : float, optional
        Highest target. Defaults to ``max(mu)`` long-only and to
        ``1.5 * max(mu)`` otherwise.

    Returns
    -------
    FrontierResult

    Examples
    --------
    >>> fr = efficient_frontier([0.05, 0.1, 0.15], [[0.01, 0, 0], [0, 0.04, 0], [0, 0, 0.09]], 5)
    >>> bool(np.all(np.diff(fr.volatilities) > 0))
    True
    """
    m, C = as_mean_cov(mu, cov)
    assert m is not None
    gmv = min_variance_portfolio(C, m, long_only)
    top = max_return if max_return is not None else (m.max() if long_only else 1.5 * m.max())
    targets = np.linspace(gmv.expected_return, top, n_points)
    W = np.empty((n_points, m.size))
    for i, t in enumerate(targets):
        W[i] = gmv.weights if i == 0 else efficient_portfolio(m, C, t, long_only).weights
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, C, W))
    return FrontierResult(returns=W @ m, volatilities=vols, weights=W, mu=m, cov=C, long_only=long_only)


def tangency_portfolio(mu: ArrayLike, cov: ArrayLike, rf: float = 0.0, long_only: bool = False) -> PortfolioResult:
    """Tobin's tangency portfolio, the fully invested maximum-Sharpe portfolio.

    Unconstrained, :math:`w \\propto \\Sigma^{-1}(\\mu - r_f\\mathbf 1)`, the
    same computation as Roy's safety-first portfolio with the disaster
    level set to the riskless rate, so it calls
    :func:`financialkit.risk.safety_first_portfolio`. Long-only, it
    solves :math:`\\min y'\\Sigma y` subject to :math:`(\\mu - r_f)'y = 1`,
    :math:`y \\ge 0`, and rescales :math:`y` to sum to one.

    Parameters
    ----------
    mu : array_like, shape (d,)
    cov : array_like, shape (d, d)
    rf : float
        Riskless rate, below the minimum-variance portfolio's mean.
    long_only : bool

    Returns
    -------
    PortfolioResult

    Examples
    --------
    >>> res = tangency_portfolio([0.1, 0.2], [[0.04, 0.0], [0.0, 0.09]], rf=0.02)
    >>> np.round(res.weights, 4), round(res.sharpe_ratio(0.02), 4)
    (array([0.5, 0.5]), 0.7211)
    """
    m, C = as_mean_cov(mu, cov)
    assert m is not None
    if long_only:
        ex = m - rf
        if np.all(ex <= 0):
            raise ValueError("no asset earns more than the riskless rate")
        pos = (ex > 0).astype(np.float64)
        y = solve_qp(C, ex[None, :], [1.0], bounds=(0.0, np.inf), x0=pos / (pos @ ex))
        w = y / y.sum()
    else:
        w = safety_first_portfolio(m, C, rf)
    return _result(w, m, C, "tangency")


def two_fund_allocation(mu: ArrayLike, cov: ArrayLike, rf: float, risk_aversion: float) -> PortfolioResult:
    """Tobin's optimal mix of risky assets and cash for a mean-variance investor.

    Maximizing :math:`w'(\\mu - r_f) - \\tfrac{\\gamma}{2} w'\\Sigma w` gives
    :math:`w = \\Sigma^{-1}(\\mu - r_f\\mathbf 1)/\\gamma`: the tangency
    portfolio scaled by the investor's risk aversion, with
    :math:`1 - \\mathbf 1'w` in cash. Separation means the risky mix does
    not depend on :math:`\\gamma`.

    Parameters
    ----------
    mu : array_like, shape (d,)
    cov : array_like, shape (d, d)
    rf : float
        Riskless rate.
    risk_aversion : float
        Coefficient :math:`\\gamma > 0`.

    Returns
    -------
    PortfolioResult
        Risky weights (not summing to one); ``expected_return`` includes
        the riskless rate earned on the cash.

    Examples
    --------
    >>> res = two_fund_allocation([0.1, 0.2], [[0.04, 0.0], [0.0, 0.09]], 0.02, 5.0)
    >>> np.round(res.weights, 4), round(float(1 - res.weights.sum()), 4)
    (array([0.4, 0.4]), 0.2)
    """
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    m, C = as_mean_cov(mu, cov)
    assert m is not None
    w = np.linalg.solve(C, m - rf) / risk_aversion
    _, vol = portfolio_moments(w, m, C)
    return PortfolioResult(weights=w, expected_return=float(rf + w @ (m - rf)), volatility=vol, method="two_fund")
