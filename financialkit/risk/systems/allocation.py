"""Euler allocation of portfolio risk (Tasche 1999).

A risk measure that is positively homogeneous in the position sizes
:math:`w` satisfies Euler's theorem,
:math:`\\rho(w) = \\sum_i w_i\\, \\partial \\rho / \\partial w_i`, so the
terms :math:`w_i\\, \\partial \\rho / \\partial w_i` split the total
exactly. Tasche showed this is the only allocation consistent with
performance measurement: a position whose return on its contribution
beats the portfolio's adds value at the margin. For normal returns the
contributions are :math:`w_i (\\Sigma w)_i / \\sigma_p` times the
quantile factor; for expected shortfall on scenarios, the contribution
of position :math:`i` is minus its average outcome in the portfolio's
tail.

References: D. Tasche, "Risk Contributions and Performance
Measurement," working paper, TU München (1999); McNeil, Frey and
Embrechts, *Quantitative Risk Management*, revised ed. (2015), Sec. 8.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.risk.core.base import AllocationResult
from financialkit.risk.utils.tail import check_alpha

__all__ = ["euler_allocation", "scenario_es_allocation"]


def euler_allocation(weights: ArrayLike, cov: ArrayLike, alpha: float = 0.99, measure: str = "var") -> AllocationResult:
    """Euler contributions to the volatility, normal VaR or normal ES of a portfolio.

    Parameters
    ----------
    weights : array_like, shape (d,)
        Money exposures.
    cov : array_like, shape (d, d)
        Covariance of one-period returns.
    alpha : float
        Confidence level (ignored for ``"volatility"``).
    measure : {"var", "es", "volatility"}

    Returns
    -------
    AllocationResult

    Examples
    --------
    Hull, 10th ed., Ch. 22's Microsoft/AT&T portfolio: the one-day 99%
    VaR of about 512,300 splits into component VaRs that sum to it.

    >>> import numpy as np
    >>> cov = np.array([[0.02**2, 0.3 * 0.02 * 0.01], [0.3 * 0.02 * 0.01, 0.01**2]])
    >>> res = euler_allocation([10e6, 5e6], cov, 0.99)
    >>> np.round(res.contributions), round(res.total)
    (array([454226.,  58099.]), 512325)
    """
    w = np.asarray(weights, dtype=np.float64)
    marginal = np.asarray(cov, dtype=np.float64) @ w
    sigma_p = float(np.sqrt(w @ marginal))
    if measure == "volatility":
        factor = 1.0
    elif measure == "var":
        factor = float(stats.norm.ppf(check_alpha(alpha)))
    elif measure == "es":
        factor = float(stats.norm.pdf(stats.norm.ppf(check_alpha(alpha))) / (1.0 - alpha))
    else:
        raise ValueError(f"measure must be 'var', 'es' or 'volatility', got {measure!r}")
    contributions = factor * w * marginal / sigma_p
    return AllocationResult(contributions=contributions, total=factor * sigma_p, measure=measure)


def scenario_es_allocation(pnl: ArrayLike, alpha: float = 0.99) -> AllocationResult:
    """Euler contributions to expected shortfall estimated from scenarios.

    Position :math:`i`'s contribution is minus its mean profit and loss
    over the :math:`k = \\lceil n(1 - \\alpha) \\rceil` scenarios in which
    the portfolio does worst; the contributions add up to the portfolio's
    tail mean.

    Parameters
    ----------
    pnl : array_like, shape (n, d)
        Profit and loss of each position (column) in each scenario (row).
    alpha : float
        Confidence level.

    Returns
    -------
    AllocationResult
    """
    X = np.asarray(pnl, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("pnl must have shape (n_scenarios, n_positions)")
    n = X.shape[0]
    k = max(int(np.ceil(n * (1.0 - check_alpha(alpha)) * (1.0 - 1e-12))), 1)
    tail = np.argsort(X.sum(axis=1), kind="stable")[:k]
    contributions = -X[tail].mean(axis=0)
    return AllocationResult(contributions=contributions, total=float(contributions.sum()), measure="es")
