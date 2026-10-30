"""Vector autoregressions: Sims's VAR, impulse responses and variance decompositions.

Sims (1980) argued that the identifying restrictions of the large
macroeconometric models were "incredible", and proposed instead to fit an
unrestricted vector autoregression in which every variable depends on lags
of all of them. The model is summarized by its impulse responses to
orthogonalized shocks, obtained from a Cholesky factorization of the
residual covariance, and by the decomposition of each variable's
forecast-error variance into the shares due to each shock. Every equation
has the same regressors, so equation-by-equation least squares is the
Gaussian maximum-likelihood estimate (:func:`numpy.linalg.lstsq` solves all
of them at once). References: C. A. Sims, "Macroeconomics and Reality,"
Econometrica 48(1) (1980), 1-48; H. Lütkepohl, *New Introduction to
Multiple Time Series Analysis* (Springer, 2005), Chs. 2-3; Hamilton, *Time
Series Analysis* (Princeton, 1994), Ch. 11.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from financialkit.time_series.core.base import VectorAutoregressionResult

__all__ = ["simulate_var", "fit_var"]


def _coefs(coefs) -> np.ndarray:
    A = np.asarray(coefs, dtype=np.float64)
    if A.ndim == 2:
        A = A[None]
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("coefs must have shape (k, k) or (p, k, k)")
    return A


def simulate_var(coefs, n: int, sigma=None, intercept=None, burn: int = 500, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Simulate :math:`y_t = c + \\sum_{i=1}^p A_i y_{t-i} + u_t`, :math:`u_t \\sim N(0, \\Sigma)`.

    Parameters
    ----------
    coefs : array_like, shape (k, k) or (p, k, k)
        :math:`A_1..A_p`.
    n : int
    sigma : array_like, shape (k, k), optional
        Innovation covariance (identity by default).
    intercept : array_like, shape (k,), optional
    burn : int
        Start-up values discarded.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n, k)
    """
    rng = np.random.default_rng(rng)
    A = _coefs(coefs)
    p, k, _ = A.shape
    L = np.linalg.cholesky(np.eye(k) if sigma is None else np.asarray(sigma, dtype=np.float64))
    c = np.zeros(k) if intercept is None else np.asarray(intercept, dtype=np.float64)
    u = rng.standard_normal((n + burn, k)) @ L.T
    y = np.zeros((n + burn + p, k))
    for t in range(p, n + burn + p):
        y[t] = c + u[t - p] + sum(A[i] @ y[t - 1 - i] for i in range(p))
    return y[p + burn :]


def fit_var(y, p: int, include_mean: bool = True, start: Optional[int] = None) -> VectorAutoregressionResult:
    """Fit a VAR(`p`) by least squares, which is Gaussian maximum likelihood given the first values.

    Parameters
    ----------
    y : array_like, shape (n, k)
    p : int
        Lag order.
    include_mean : bool
        Estimate an intercept :math:`c`.
    start : int, optional
        Observations conditioned on (default `p`); use a common value to
        compare orders by :attr:`~financialkit.time_series.core.base.VectorAutoregressionResult.aic`
        on the same sample.

    Returns
    -------
    VectorAutoregressionResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0.5, 0.1], [0.4, 0.3]])
    >>> res = fit_var(simulate_var(A, 20_000, rng=0), 1)
    >>> res.coefs[0].round(1).tolist(), res.is_stable()
    ([[0.5, 0.1], [0.4, 0.3]], True)
    """
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("y must have shape (n, k)")
    n, k = Y.shape
    start = p if start is None else start
    if p < 1 or start < p:
        raise ValueError("p must be at least 1 and start at least p")
    T = n - start
    if T <= k * p + int(include_mean):
        raise ValueError("series too short for this order")
    lags = [Y[start - i : n - i] for i in range(1, p + 1)]
    X = np.hstack(([np.ones((T, 1))] if include_mean else []) + lags)
    B, *_ = np.linalg.lstsq(X, Y[start:], rcond=None)
    U = Y[start:] - X @ B
    sigma = U.T @ U / T
    c = B[0] if include_mean else np.zeros(k)
    A = B[int(include_mean) :].reshape(p, k, k).transpose(0, 2, 1)
    _, logdet = np.linalg.slogdet(sigma)
    loglik = -0.5 * T * (k * math.log(2 * math.pi) + logdet + k)
    return VectorAutoregressionResult(coefs=A, intercept=c, sigma=sigma, residuals=U, loglik=float(loglik), n_obs=T, n_params=int(B.size), data=Y)
