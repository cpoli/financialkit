"""Least squares with standard errors, lag matrices and a numerical Hessian."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["OLSResult", "ols", "lag_matrix", "numerical_hessian"]


@dataclass
class OLSResult:
    """Ordinary least squares :math:`y = X b + e`."""

    coef: np.ndarray
    """ndarray, shape (k,): Estimated coefficients :math:`\\hat b`."""

    std_errors: np.ndarray
    """ndarray, shape (k,): Classical standard errors."""

    t_values: np.ndarray
    """ndarray, shape (k,): Coefficients divided by their standard errors."""

    residuals: np.ndarray
    """ndarray, shape (n,): :math:`y - X\\hat b`."""

    sigma2: float
    """float: Residual variance with ``n - k`` degrees of freedom."""

    r_squared: float
    """float: Centred :math:`R^2` (uncentred when `X` has no constant column)."""


def ols(y, X) -> OLSResult:
    """Least squares by :func:`numpy.linalg.lstsq`, with classical standard errors.

    Parameters
    ----------
    y : array_like, shape (n,)
    X : array_like, shape (n, k)

    Returns
    -------
    OLSResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(5.0)
    >>> res = ols(1.0 + 2.0 * x, np.column_stack([np.ones(5), x]))
    >>> res.coef.round(12).tolist(), res.r_squared
    ([1.0, 2.0], 1.0)
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    sigma2 = float(e @ e / (n - k))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    has_const = bool(np.any(np.all(X == X[0], axis=0) & (X[0] != 0)))
    tss = np.sum((y - y.mean()) ** 2) if has_const else np.sum(y**2)
    r2 = 1.0 - float(e @ e) / float(tss) if tss > 0 else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / se
    return OLSResult(coef=coef, std_errors=se, t_values=t, residuals=e, sigma2=sigma2, r_squared=r2)


def lag_matrix(x, lags: int) -> np.ndarray:
    """Columns ``x[t-1], ..., x[t-lags]`` for ``t = lags .. n-1``.

    Examples
    --------
    >>> lag_matrix([1.0, 2.0, 3.0, 4.0], 2).tolist()
    [[2.0, 1.0], [3.0, 2.0]]
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    return np.column_stack([x[lags - i : n - i] for i in range(1, lags + 1)]) if lags > 0 else np.empty((n, 0))


def numerical_hessian(f: Callable[[np.ndarray], float], x, rel_step: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function at `x`.

    Examples
    --------
    >>> import numpy as np
    >>> numerical_hessian(lambda v: v[0] ** 2 + 3 * v[0] * v[1], np.array([1.0, 2.0])).round(6).tolist()
    [[2.0, 3.0], [3.0, 0.0]]
    """
    x = np.asarray(x, dtype=np.float64)
    k = x.size
    h = rel_step * np.maximum(np.abs(x), 1e-2)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H
