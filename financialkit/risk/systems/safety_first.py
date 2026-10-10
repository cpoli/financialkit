"""Roy's (1952) safety-first criterion.

An investor who fears above all that wealth falls below a disaster
level :math:`d` should minimise :math:`P(R < d)`. Knowing only the mean
and standard deviation, Chebyshev's inequality bounds that probability
by :math:`1 / s^2` with :math:`s = (\\mu - d) / \\sigma`, so the best
portfolio maximises the safety-first ratio :math:`s`, a Sharpe ratio
with the disaster level in place of the risk-free rate. Among risky
portfolios it is the tangency portfolio from :math:`d`,
:math:`w \\propto \\Sigma^{-1}(\\mu - d \\mathbf{1})`.

References: A. D. Roy, "Safety First and the Holding of Assets,"
*Econometrica* 20 (1952) 431-449.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

__all__ = ["safety_first_ratio", "shortfall_probability", "safety_first_portfolio"]


def safety_first_ratio(mu: ArrayLike, sigma: ArrayLike, threshold: ArrayLike = 0.0) -> np.ndarray | float:
    """Roy's ratio :math:`(\\mu - d) / \\sigma`.

    Parameters
    ----------
    mu, sigma : array_like
        Mean and standard deviation of the return.
    threshold : array_like
        Disaster level :math:`d`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> safety_first_ratio(0.08, 0.15, 0.02)
    0.4
    """
    out = (np.asarray(mu, dtype=np.float64) - np.asarray(threshold, dtype=np.float64)) / np.asarray(sigma, dtype=np.float64)
    return float(out) if out.ndim == 0 else out


def shortfall_probability(mu: ArrayLike, sigma: ArrayLike, threshold: ArrayLike = 0.0, bound: str = "chebyshev") -> np.ndarray | float:
    """Probability that the return falls below the disaster level.

    Parameters
    ----------
    mu, sigma, threshold : array_like
    bound : {"chebyshev", "normal"}
        ``"chebyshev"`` is Roy's distribution-free upper bound
        :math:`\\min(1, 1/s^2)` (for :math:`s > 0`); ``"normal"`` is the
        exact value :math:`\\Phi(-s)` for normal returns.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(shortfall_probability(0.08, 0.15, 0.02, "normal"), 4)
    0.3446
    >>> shortfall_probability(0.08, 0.03, 0.02)
    0.25
    """
    s = np.asarray(safety_first_ratio(mu, sigma, threshold))
    if bound == "normal":
        out = stats.norm.cdf(-s)
    elif bound == "chebyshev":
        with np.errstate(divide="ignore"):
            out = np.where(s > 0, np.minimum(1.0, 1.0 / s**2), 1.0)
    else:
        raise ValueError(f"bound must be 'chebyshev' or 'normal', got {bound!r}")
    return float(out) if out.ndim == 0 else out


def safety_first_portfolio(mu: ArrayLike, cov: ArrayLike, threshold: float = 0.0) -> np.ndarray:
    """Fully invested weights maximising Roy's ratio.

    Parameters
    ----------
    mu : array_like, shape (d,)
        Expected returns.
    cov : array_like, shape (d, d)
        Covariance of returns.
    threshold : float
        Disaster level; it must lie below the global minimum-variance
        portfolio's mean, or no risky portfolio beats it.

    Returns
    -------
    ndarray, shape (d,)
        Weights summing to one.
    """
    mu = np.asarray(mu, dtype=np.float64)
    C = np.asarray(cov, dtype=np.float64)
    ones = np.ones_like(mu)
    v = np.linalg.solve(C, mu - threshold * ones)
    g = np.linalg.solve(C, ones)
    if threshold >= mu @ g / g.sum():
        raise ValueError("threshold must lie below the minimum-variance portfolio's mean return")
    return v / v.sum()
