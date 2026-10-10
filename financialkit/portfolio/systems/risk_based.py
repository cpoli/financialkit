"""Risk-based allocation: CVaR optimization, maximum diversification, risk parity and HRP.

These rules need no expected returns, or treat them only as a
constraint. Rockafellar and Uryasev showed that minimizing conditional
value at risk over scenarios is a linear program,

.. math::

   \\min_{w,\\,\\zeta,\\,u}\\; \\zeta + \\frac{1}{(1-\\alpha) n}\\sum_k u_k,
   \\qquad u_k \\ge -r_k'w - \\zeta,\\; u_k \\ge 0,

whose optimal :math:`\\zeta` is the value at risk. Choueifaty and Coignard
maximize the diversification ratio
:math:`D(w) = w'\\sigma / \\sqrt{w'\\Sigma w}`. Maillard, Roncalli and
Teiletche characterized the equal risk contribution portfolio, in which
every asset's Euler contribution :math:`w_i(\\Sigma w)_i` to the variance
is the same, computed here through Spinu's (2013) convex program. Lopez
de Prado's hierarchical risk parity clusters the correlation matrix,
orders assets by the dendrogram and splits weight top-down by inverse
cluster variance, never inverting the covariance matrix.

References: R. T. Rockafellar and S. Uryasev, "Optimization of
Conditional Value-at-Risk," *J. Risk* 2 (2000) 21-41; Y. Choueifaty and
Y. Coignard, "Toward Maximum Diversification," *J. Portfolio Mgmt.* 35
(2008) 40-51; S. Maillard, T. Roncalli and J. Teiletche, "The
Properties of Equally Weighted Risk Contribution Portfolios," *J.
Portfolio Mgmt.* 36 (2010) 60-70; F. Spinu, "An Algorithm for Computing
Risk Parity Weights," SSRN 2297383 (2013); M. Lopez de Prado, "Building
Diversified Portfolios that Outperform Out of Sample," *J. Portfolio
Mgmt.* 42 (2016) 59-69.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.portfolio.core.base import CVaRResult, HRPResult, PortfolioResult
from financialkit.portfolio.systems.mean_variance import min_variance_portfolio
from financialkit.portfolio.utils.optimize import as_mean_cov, portfolio_moments
from financialkit.risk.utils.tail import check_alpha

__all__ = [
    "cvar_optimization",
    "diversification_ratio",
    "most_diversified_portfolio",
    "inverse_volatility_portfolio",
    "risk_parity",
    "hierarchical_risk_parity",
]


def cvar_optimization(
    scenarios: ArrayLike,
    alpha: float = 0.95,
    target_return: Optional[float] = None,
    long_only: bool = True,
) -> CVaRResult:
    """Rockafellar-Uryasev minimum-CVaR portfolio by linear programming.

    Parameters
    ----------
    scenarios : array_like, shape (n, d)
        Equally likely joint return scenarios.
    alpha : float
        Confidence level.
    target_return : float, optional
        Lower bound on the mean scenario return.
    long_only : bool
        Exclude short sales (otherwise weights lie in ``[-1, 2]``, keeping
        the program bounded).

    Returns
    -------
    CVaRResult

    Examples
    --------
    Two scenarios, one asset safe in each: hedging both equally removes all loss.

    >>> res = cvar_optimization([[0.1, -0.1], [-0.1, 0.1]], alpha=0.5)
    >>> np.round(res.weights, 6), round(res.cvar, 6) + 0.0
    (array([0.5, 0.5]), 0.0)
    """
    R = np.asarray(scenarios, dtype=np.float64)
    n, d = R.shape
    a = check_alpha(alpha)
    # Variables: w (d), zeta (1), u (n).
    c = np.concatenate([np.zeros(d), [1.0], np.full(n, 1.0 / ((1.0 - a) * n))])
    A_ub = np.hstack([-R, -np.ones((n, 1)), -np.eye(n)])
    b_ub = np.zeros(n)
    if target_return is not None:
        A_ub = np.vstack([A_ub, np.concatenate([-R.mean(axis=0), [0.0], np.zeros(n)])])
        b_ub = np.append(b_ub, -target_return)
    A_eq = np.concatenate([np.ones(d), [0.0], np.zeros(n)])[None, :]
    wb = (0.0, 1.0) if long_only else (-1.0, 2.0)
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[wb] * d + [(None, None)] + [(0.0, None)] * n, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR linear program failed: {res.message}")
    w = res.x[:d]
    return CVaRResult(weights=w, var=float(res.x[d]), cvar=float(res.fun), expected_return=float(R.mean(axis=0) @ w), alpha=a)


def diversification_ratio(weights: ArrayLike, cov: ArrayLike) -> float:
    """Choueifaty and Coignard's ratio of weighted average volatility to portfolio volatility.

    Parameters
    ----------
    weights : array_like, shape (d,)
    cov : array_like, shape (d, d)

    Returns
    -------
    float
        At least one for long-only weights; ``sqrt(d)`` for ``d``
        uncorrelated equal-volatility assets held equally.

    Examples
    --------
    >>> diversification_ratio([0.25] * 4, np.eye(4))
    2.0
    """
    _, C = as_mean_cov(None, cov)
    w = np.asarray(weights, dtype=np.float64)
    return float(w @ np.sqrt(np.diag(C)) / np.sqrt(w @ C @ w))


def most_diversified_portfolio(cov: ArrayLike, mu: Optional[ArrayLike] = None, long_only: bool = True) -> PortfolioResult:
    """The portfolio with the largest diversification ratio.

    Maximizing :math:`D(w)` is the minimum-variance problem on the
    correlation matrix followed by dividing each weight by the asset's
    volatility and renormalizing (Choueifaty and Coignard 2008); the
    unconstrained answer is :math:`w \\propto \\Sigma^{-1}\\sigma`.

    Parameters
    ----------
    cov : array_like, shape (d, d)
    mu : array_like, shape (d,), optional
        Used only to report the mean.
    long_only : bool

    Returns
    -------
    PortfolioResult

    Examples
    --------
    Uncorrelated assets: weights proportional to inverse volatility.

    >>> np.round(most_diversified_portfolio(np.diag([0.01, 0.04])).weights, 6)
    array([0.666667, 0.333333])
    """
    m, C = as_mean_cov(mu, cov)
    s = np.sqrt(np.diag(C))
    y = min_variance_portfolio(C / np.outer(s, s), long_only=long_only).weights / s
    w = y / y.sum()
    mean, vol = portfolio_moments(w, m, C)
    return PortfolioResult(weights=w, expected_return=mean, volatility=vol, method="most_diversified")


def inverse_volatility_portfolio(cov: ArrayLike, mu: Optional[ArrayLike] = None) -> PortfolioResult:
    """Weights proportional to :math:`1/\\sigma_i`, the naive risk parity rule.

    Parameters
    ----------
    cov : array_like, shape (d, d)
    mu : array_like, shape (d,), optional

    Returns
    -------
    PortfolioResult

    Examples
    --------
    >>> np.round(inverse_volatility_portfolio(np.diag([0.01, 0.04])).weights, 6)
    array([0.666667, 0.333333])
    """
    m, C = as_mean_cov(mu, cov)
    y = 1.0 / np.sqrt(np.diag(C))
    w = y / y.sum()
    mean, vol = portfolio_moments(w, m, C)
    return PortfolioResult(weights=w, expected_return=mean, volatility=vol, method="inverse_volatility")


def risk_parity(cov: ArrayLike, budgets: Optional[ArrayLike] = None, mu: Optional[ArrayLike] = None) -> PortfolioResult:
    """Equal (or budgeted) risk contribution portfolio.

    Solves Spinu's strictly convex program
    :math:`\\min_{y > 0} \\tfrac12 y'\\Sigma y - \\sum_i b_i \\log y_i`,
    whose first-order condition :math:`y_i(\\Sigma y)_i = b_i` makes each
    risk contribution proportional to its budget, by his damped Newton
    iteration (hand-written: SciPy's trust-region methods stall on this
    problem at rounding level), then normalizes :math:`w = y / \\mathbf 1'y`.

    Parameters
    ----------
    cov : array_like, shape (d, d)
    budgets : array_like, shape (d,), optional
        Positive risk budgets; equal by default.
    mu : array_like, shape (d,), optional

    Returns
    -------
    PortfolioResult

    Examples
    --------
    With two assets, risk parity is inverse volatility whatever the correlation.

    >>> cov = np.array([[0.04, 0.018], [0.018, 0.09]])
    >>> np.round(risk_parity(cov).weights, 6)
    array([0.6, 0.4])
    """
    m, C = as_mean_cov(mu, cov)
    d = C.shape[0]
    b = np.full(d, 1.0 / d) if budgets is None else np.asarray(budgets, dtype=np.float64)
    if np.any(b <= 0):
        raise ValueError("risk budgets must be positive")
    b = b / b.sum()
    # Spinu's damped Newton method: the objective is self-concordant, so the
    # damped step 1 / (1 + lambda) keeps y positive and converges globally,
    # and the full Newton step converges quadratically once the Newton
    # decrement lambda is below (3 - sqrt 5) / 2.
    Cs = C / np.mean(np.diag(C))
    y = 1.0 / np.sqrt(np.diag(Cs))
    y /= np.sqrt(y @ Cs @ y)
    for _ in range(DEFAULT_MAX_ITER):
        g = Cs @ y - b / y
        step = np.linalg.solve(Cs + np.diag(b / y**2), g)
        lam = float(np.sqrt(g @ step))
        y = y - (step / (1.0 + lam) if lam > 0.95 * (3.0 - np.sqrt(5.0)) / 2.0 else step)
        if lam < DEFAULT_ATOL:
            break
    else:
        raise RuntimeError("risk parity did not converge")
    w = y / y.sum()
    mean, vol = portfolio_moments(w, m, C)
    return PortfolioResult(weights=w, expected_return=mean, volatility=vol, method="risk_parity")


def hierarchical_risk_parity(cov: ArrayLike, method: str = "single") -> HRPResult:
    """Lopez de Prado's hierarchical risk parity.

    1. Cluster the assets with :func:`scipy.cluster.hierarchy.linkage` on
       the distance :math:`\\sqrt{(1 - \\rho_{ij})/2}`.
    2. Reorder them by the dendrogram's leaves (quasi-diagonalization).
    3. Bisect the ordered list recursively, splitting each parent's weight
       between its halves in inverse proportion to the variance of each
       half's inverse-variance portfolio.

    Parameters
    ----------
    cov : array_like, shape (d, d)
    method : str
        Linkage method (``"single"`` in the original paper).

    Returns
    -------
    HRPResult

    Examples
    --------
    With a diagonal covariance, HRP reduces to inverse-variance weights.

    >>> res = hierarchical_risk_parity(np.diag([0.01, 0.04, 0.04, 0.16]))
    >>> np.round(res.weights * 25, 6)
    array([16.,  4.,  4.,  1.])
    """
    _, C = as_mean_cov(None, cov)
    d = C.shape[0]
    s = np.sqrt(np.diag(C))
    rho = np.clip(C / np.outer(s, s), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = hierarchy.linkage(squareform(dist, checks=False), method=method)
    order = hierarchy.leaves_list(link)

    def cluster_var(idx: np.ndarray) -> float:
        sub = C[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = np.ones(d)
    stack = [order]
    while stack:
        items = stack.pop()
        if items.size < 2:
            continue
        left, right = items[: items.size // 2], items[items.size // 2 :]
        vl, vr = cluster_var(left), cluster_var(right)
        a = 1.0 - vl / (vl + vr)
        w[left] *= a
        w[right] *= 1.0 - a
        stack += [left, right]
    return HRPResult(weights=w, order=order, linkage=link)
