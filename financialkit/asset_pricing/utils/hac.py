"""Newey and West's heteroskedasticity and autocorrelation consistent covariance.

GMM needs the long-run covariance of the moment conditions,
:math:`S = \\sum_{j=-\\infty}^{\\infty} E[u_t u_{t-j}']`. Newey and West
(1987) truncated the sum at ``lags`` and weighted the autocovariances
with Bartlett's triangle, which keeps the estimate positive
semi-definite.

Reference: W. K. Newey and K. D. West, "A Simple, Positive
Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
Covariance Matrix," *Econometrica* 55 (1987) 703-708.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.asset_pricing.utils.sorts import as_panel

__all__ = ["newey_west"]


def newey_west(u: ArrayLike, lags: int = 0) -> np.ndarray:
    """Bartlett-weighted long-run covariance of the rows of ``u``.

    :math:`\\hat S = \\hat\\Gamma_0 + \\sum_{j=1}^{L} \\left(1 - \\tfrac{j}{L+1}\\right)(\\hat\\Gamma_j + \\hat\\Gamma_j')`
    with :math:`\\hat\\Gamma_j = \\tfrac1T \\sum_{t > j} u_t u_{t-j}'`.
    The rows are used as given: demean them first if they do not have
    mean zero.

    Parameters
    ----------
    u : array_like, shape (T,) or (T, q)
    lags : int
        Number of autocovariances, ``0`` for White's heteroskedasticity-robust
        covariance.

    Returns
    -------
    ndarray, shape (q, q)

    Examples
    --------
    >>> newey_west([1.0, -1.0, 1.0, -1.0], lags=1).tolist()
    [[0.25]]
    """
    if lags < 0:
        raise ValueError(f"lags must be nonnegative, got {lags}")
    U = as_panel(u)
    T = U.shape[0]
    S = U.T @ U / T
    for j in range(1, lags + 1):
        G = U[j:].T @ U[:-j] / T
        S += (1.0 - j / (lags + 1.0)) * (G + G.T)
    return S
