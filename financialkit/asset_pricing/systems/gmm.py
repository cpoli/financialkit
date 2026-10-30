"""Hansen's generalized method of moments.

An asset pricing model says that some functions of the data and the
parameters have mean zero, for example the pricing errors
:math:`E[m_{t+1}(\\theta) R_{t+1} - 1] = 0`. Hansen (1982) estimated
:math:`\\theta` by making the sample means
:math:`g_T(\\theta) = \\frac1T\\sum_t u_t(\\theta)` as small as possible in
the metric :math:`g_T' W g_T`, and gave the distribution theory under
nothing more than stationarity: the estimates are asymptotically
normal with covariance

.. math::

   \\frac1T (D'WD)^{-1} D'WSWD (D'WD)^{-1}, \\qquad D = \\frac{\\partial g_T}{\\partial \\theta'},
   \\quad S = \\sum_j E[u_t u_{t-j}'],

which is smallest when :math:`W = S^{-1}`. With more moments ``q`` than
parameters ``p``, the minimised objective :math:`J = T g_T' S^{-1} g_T`
is :math:`\\chi^2_{q-p}` when the model is right: a test of the model
that comes free with the estimate. OLS, instrumental variables and
maximum likelihood are all special cases.

References: L. P. Hansen, "Large Sample Properties of Generalized Method
of Moments Estimators," *Econometrica* 50 (1982) 1029-1054;
J. H. Cochrane, *Asset Pricing*, rev. ed., Princeton (2005), ch. 10-11.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from financialkit.asset_pricing.core.base import GMMResult
from financialkit.asset_pricing.utils.hac import newey_west
from financialkit.asset_pricing.utils.sorts import as_panel
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER

__all__ = ["gmm"]


def _jacobian(gbar: Callable[[np.ndarray], np.ndarray], theta: np.ndarray) -> np.ndarray:
    """Central-difference Jacobian of the mean moments, shape ``(q, p)``."""
    cols = []
    for k in range(theta.size):
        h = 1e-6 * max(1.0, abs(theta[k]))
        e = np.zeros_like(theta)
        e[k] = h
        cols.append((gbar(theta + e) - gbar(theta - e)) / (2.0 * h))
    return np.column_stack(cols)


def _long_run_cov(u: np.ndarray, lags: int) -> np.ndarray:
    return newey_west(u - u.mean(axis=0), lags)


def gmm(
    moments: Callable[[np.ndarray], ArrayLike],
    theta0: ArrayLike,
    weighting: str | ArrayLike = "two-step",
    lags: int = 0,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_ATOL,
) -> GMMResult:
    """Estimate parameters by Hansen's generalized method of moments.

    Parameters
    ----------
    moments : callable
        ``moments(theta)`` returns the moment conditions, one row per
        observation, shape ``(T, q)`` with ``q >= p``; their expectation is
        zero at the true parameters.
    theta0 : array_like, shape (p,)
        Starting values.
    weighting : {"two-step", "iterated", "identity"} or array_like
        ``"identity"`` minimises the plain sum of squared mean moments;
        ``"two-step"`` starts there and re-estimates once with the
        efficient weighting :math:`\\hat S^{-1}`; ``"iterated"`` repeats
        until the estimates settle. An array is a fixed ``(q, q)``
        weighting matrix.
    lags : int
        Newey-West lags in :math:`\\hat S`; ``0`` when the moments are
        serially uncorrelated.
    max_iter : int
        Largest number of weighting updates for ``"iterated"``.
    tol : float
        Convergence tolerance on the parameters for ``"iterated"``.

    Returns
    -------
    GMMResult
        The covariance is the sandwich formula evaluated at the
        estimates. The J statistic is :math:`T g_T' W g_T` under efficient
        weighting and Hansen's generalised-inverse form otherwise, so it
        is :math:`\\chi^2_{q-p}` under the null in both cases.

    Examples
    --------
    The mean of a sample, from the moment ``x - theta``:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 4.0, 5.0])
    >>> res = gmm(lambda th: (x - th[0])[:, None], [0.0])
    >>> round(float(res.params[0]), 9), round(float(res.std_errors[0]), 9)
    (3.0, 0.790569415)
    """
    theta = np.atleast_1d(np.asarray(theta0, dtype=np.float64))
    u0 = as_panel(moments(theta))
    T, q = u0.shape
    p = theta.size
    if q < p:
        raise ValueError(f"need at least as many moments as parameters, got {q} moments and {p} parameters")

    def gbar(th: np.ndarray) -> np.ndarray:
        return as_panel(moments(th)).mean(axis=0)

    def minimise(W: np.ndarray, start: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(W)
        fit = least_squares(lambda th: L.T @ gbar(th), start, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200 * (p + 1))
        return fit.x

    efficient = isinstance(weighting, str) and weighting in ("two-step", "iterated")
    if isinstance(weighting, str):
        if weighting not in ("two-step", "iterated", "identity"):
            raise ValueError(f"weighting must be 'two-step', 'iterated', 'identity' or a matrix, got {weighting!r}")
        W = np.eye(q)
    else:
        W = np.asarray(weighting, dtype=np.float64)
        if W.shape != (q, q):
            raise ValueError(f"weighting must have shape ({q}, {q}), got {W.shape}")
    theta = minimise(W, theta)
    if efficient:
        for _ in range(1 if weighting == "two-step" else max_iter):
            W = np.linalg.inv(_long_run_cov(as_panel(moments(theta)), lags))
            W = 0.5 * (W + W.T)
            new = minimise(W, theta)
            done = np.max(np.abs(new - theta)) <= tol * max(1.0, float(np.max(np.abs(theta))))
            theta = new
            if done:
                break

    u = as_panel(moments(theta))
    g = u.mean(axis=0)
    S = _long_run_cov(u, lags)
    D = _jacobian(gbar, theta)
    A = np.linalg.inv(D.T @ W @ D)
    cov = A @ D.T @ W @ S @ W @ D @ A / T
    if efficient:
        j_stat = float(T * g @ W @ g)
    else:
        P = np.eye(q) - D @ A @ D.T @ W
        j_stat = float(T * g @ np.linalg.pinv(P @ S @ P.T, rcond=1e-10, hermitian=True) @ g)
    return GMMResult(params=theta, cov=cov, j_stat=j_stat, df=q - p, moments=g, weighting=W, n_obs=T)
