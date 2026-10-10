"""Naive diversification and out-of-sample evaluation: DeMiguel, Garlappi and Uppal (2009).

DeMiguel, Garlappi and Uppal compared fourteen optimized portfolio rules
with the equally weighted portfolio, re-estimating each on a rolling
window and holding it for one period. None beat 1/N consistently in
Sharpe ratio: the gain from optimization is eaten by estimation error
unless the window is far longer than any available history (thousands of
months for 25 assets, by their calculation).

References: V. DeMiguel, L. Garlappi and R. Uppal, "Optimal Versus Naive
Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Rev.
Financial Stud.* 22 (2009) 1915-1953.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from financialkit.portfolio.core.base import OutOfSampleResult

__all__ = ["equal_weight", "out_of_sample_returns"]


def equal_weight(n_assets: int) -> np.ndarray:
    """The naive 1/N portfolio.

    Parameters
    ----------
    n_assets : int

    Returns
    -------
    ndarray, shape (n_assets,)

    Examples
    --------
    >>> equal_weight(4)
    array([0.25, 0.25, 0.25, 0.25])
    """
    if n_assets < 1:
        raise ValueError(f"n_assets must be positive, got {n_assets}")
    return np.full(n_assets, 1.0 / n_assets)


def out_of_sample_returns(returns: ArrayLike, rule: Callable[[np.ndarray], ArrayLike], window: int, name: str = "") -> OutOfSampleResult:
    """Roll a portfolio rule through history, estimating on ``window`` periods and holding for one.

    Parameters
    ----------
    returns : array_like, shape (n, d)
        Asset returns.
    rule : callable
        Maps the ``(window, d)`` block of past returns to weights of shape ``(d,)``.
    window : int
        Estimation window length.
    name : str
        Label stored on the result.

    Returns
    -------
    OutOfSampleResult

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[0.01, 0.03], [0.02, -0.01], [0.0, 0.02]])
    >>> out_of_sample_returns(R, lambda past: equal_weight(2), window=1).returns
    array([0.005, 0.01 ])
    """
    R = np.asarray(returns, dtype=np.float64)
    n, d = R.shape
    if not 1 <= window < n:
        raise ValueError(f"window must lie in [1, {n - 1}], got {window}")
    W = np.array([np.asarray(rule(R[t - window : t]), dtype=np.float64) for t in range(window, n)])
    return OutOfSampleResult(returns=np.einsum("ij,ij->i", W, R[window:]), weights=W, name=name)
