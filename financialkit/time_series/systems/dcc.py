"""Dynamic conditional correlation: Engle's DCC-GARCH.

Multivariate GARCH models with a free covariance matrix have parameters
that grow with the fourth power of the number of assets. Bollerslev (1990)
kept the correlations constant; Engle (2002) let them move with a scalar
GARCH-like recursion on the standardized residuals, so the estimation
splits in two. Step 1 fits a univariate GARCH to each series, and step 2
estimates the two correlation parameters :math:`(a, b)` by maximizing the
correlation part of the Gaussian likelihood, with the intercept fixed by
*correlation targeting* at the sample covariance of the standardized
residuals. Correlations that rise in sell-offs are the finding that made
the model a standard tool of risk management. References: R. Engle,
"Dynamic Conditional Correlation: A Simple Class of Multivariate
Generalized Autoregressive Conditional Heteroskedasticity Models," Journal
of Business & Economic Statistics 20(3) (2002), 339-350; R. F. Engle and
K. Sheppard, NBER Working Paper 8554 (2001); T. Bollerslev, Review of
Economics and Statistics 72(3) (1990), 498-505.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from financialkit.time_series.core.base import DCCFit, VolatilityModel
from financialkit.time_series.systems.garch import GARCH

__all__ = ["dcc_correlations", "dcc_loglikelihood", "fit_dcc", "simulate_dcc"]


def _q_path(z: np.ndarray, a: float, b: float, Qbar: np.ndarray) -> np.ndarray:
    # Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, Q_0 = Qbar: one linear filter per entry.
    n, k = z.shape
    zz = np.einsum("ti,tj->tij", z[:-1], z[:-1]).reshape(n - 1, k * k)
    u = (1.0 - a - b) * Qbar.reshape(1, -1) + a * zz
    rest = lfilter([1.0], [1.0, -b], u, axis=0, zi=b * Qbar.reshape(1, -1))[0]
    return np.vstack([Qbar.reshape(1, -1), rest]).reshape(n, k, k)


def dcc_correlations(z, a: float, b: float, Qbar=None) -> np.ndarray:
    """Conditional correlations :math:`R_t` of standardized residuals `z` under DCC(1,1).

    .. math::

       Q_t = (1 - a - b)\\bar Q + a\\,z_{t-1}z_{t-1}' + b\\,Q_{t-1}, \\qquad
       R_t = \\operatorname{diag}(Q_t)^{-1/2} Q_t \\operatorname{diag}(Q_t)^{-1/2}

    Parameters
    ----------
    z : array_like, shape (n, k)
    a, b : float
        Nonnegative with :math:`a + b < 1`.
    Qbar : array_like, shape (k, k), optional
        Defaults to :math:`z'z/n` (correlation targeting).

    Returns
    -------
    ndarray, shape (n, k, k)

    Examples
    --------
    >>> import numpy as np
    >>> z = np.random.default_rng(0).standard_normal((5, 2))
    >>> R = dcc_correlations(z, 0.05, 0.9)
    >>> np.allclose(R[:, 0, 0], 1.0), bool(np.all(np.abs(R[:, 0, 1]) < 1))
    (True, True)
    """
    Z = np.asarray(z, dtype=np.float64)
    Qb = Z.T @ Z / Z.shape[0] if Qbar is None else np.asarray(Qbar, dtype=np.float64)
    Q = _q_path(Z, a, b, Qb)
    d = 1.0 / np.sqrt(np.einsum("tii->ti", Q))
    return d[:, :, None] * Q * d[:, None, :]


def dcc_loglikelihood(z, a: float, b: float, Qbar=None) -> float:
    """Correlation part of the DCC log-likelihood, :math:`-\\tfrac12\\sum_t(\\ln|R_t| + z_t'R_t^{-1}z_t - z_t'z_t)`."""
    Z = np.asarray(z, dtype=np.float64)
    R = dcc_correlations(Z, a, b, Qbar)
    sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0):
        return -np.inf
    quad = np.einsum("ti,ti->t", Z, np.linalg.solve(R, Z[:, :, None])[:, :, 0])
    return float(-0.5 * np.sum(logdet + quad - np.einsum("ti,ti->t", Z, Z)))


def fit_dcc(returns, model: type[VolatilityModel] = GARCH, start: Optional[tuple[float, float]] = None) -> DCCFit:
    """Engle's two-step estimate of a DCC-GARCH(1,1) model.

    Step 1 fits `model` (GARCH(1,1) by default) to each column by maximum
    likelihood and standardizes the returns. Step 2 maximizes
    :func:`dcc_loglikelihood` over :math:`(a, b)` with
    :func:`scipy.optimize.minimize` (SLSQP, :math:`a, b \\ge 0`,
    :math:`a + b < 1`) and :math:`\\bar Q` fixed at the sample covariance
    of the standardized residuals. The two-step estimate is consistent; its
    efficiency loss against full maximum likelihood is small (Engle 2002).

    Parameters
    ----------
    returns : array_like, shape (n, k)
    model : VolatilityModel subclass
        Univariate model of step 1.
    start : (float, float), optional
        Starting :math:`(a, b)`, ``(0.03, 0.95)`` by default.

    Returns
    -------
    DCCFit

    Examples
    --------
    >>> import numpy as np
    >>> g = GARCH(omega=0.02, alpha=0.08, beta=0.9)
    >>> r, R = simulate_dcc([g, g], a=0.05, b=0.93, Rbar=[[1.0, 0.5], [0.5, 1.0]], n=4000, rng=0)
    >>> fit = fit_dcc(r)
    >>> bool(abs(fit.a - 0.05) < 0.02), bool(abs(fit.b - 0.93) < 0.03)
    (True, True)
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("returns must have shape (n, k) with k >= 2")
    fits = [model.fit(r[:, i]) for i in range(r.shape[1])]
    z = np.column_stack([f.model.standardized_residuals(r[:, i]) for i, f in enumerate(fits)])
    Qbar = z.T @ z / z.shape[0]

    def objective(v):
        ll = dcc_loglikelihood(z, v[0], v[1], Qbar)
        return -ll / z.shape[0] if np.isfinite(ll) else 1e10

    res = minimize(
        objective,
        np.asarray((0.03, 0.95) if start is None else start, dtype=np.float64),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda v: 0.9999 - v[0] - v[1]}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    a, b = (float(c) for c in res.x)
    corr_ll = dcc_loglikelihood(z, a, b, Qbar)
    return DCCFit(
        a=a,
        b=b,
        Qbar=Qbar,
        univariate=fits,
        correlation=dcc_correlations(z, a, b, Qbar),
        standardized_residuals=z,
        loglik=float(sum(f.loglik for f in fits) + corr_ll),
        correlation_loglik=corr_ll,
        n_obs=r.shape[0],
        converged=bool(res.success),
    )


def simulate_dcc(models, a: float, b: float, Rbar, n: int, burn: int = 500, rng: np.random.Generator | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate returns from univariate GARCH-type `models` joined by DCC(1,1) correlations.

    Each step draws :math:`z_t = L_t \\epsilon_t` with :math:`L_t L_t' = R_t`
    and :math:`\\epsilon_t \\sim N(0, I)`, sets each return to
    :math:`\\mu_i + \\sigma_{t,i} z_{t,i}`, and updates the variances by each
    model's news impact curve and :math:`Q_t` by the DCC recursion with
    :math:`\\bar Q = \\bar R`.

    Parameters
    ----------
    models : sequence of VolatilityModel
    a, b : float
    Rbar : array_like, shape (k, k)
        Unconditional correlation matrix.
    n, burn : int
    rng : numpy.random.Generator, int or None

    Returns
    -------
    returns : ndarray, shape (n, k)
    correlation : ndarray, shape (n, k, k)
        The true :math:`R_t`.
    """
    rng = np.random.default_rng(rng)
    Rb = np.asarray(Rbar, dtype=np.float64)
    k = len(models)
    eps = rng.standard_normal((n + burn, k))
    var = np.array([m.unconditional_variance for m in models], dtype=np.float64)
    var = np.where(np.isfinite(var), var, 1.0)
    mean = np.array([m.mean for m in models])
    Q = Rb.copy()
    out = np.empty((n + burn, k))
    corr = np.empty((n + burn, k, k))
    for t in range(n + burn):
        d = 1.0 / np.sqrt(np.diag(Q))
        R = d[:, None] * Q * d[None, :]
        z = np.linalg.cholesky(R) @ eps[t]
        e = np.sqrt(var) * z
        out[t], corr[t] = mean + e, R
        var = np.array([float(m.news_impact(e[i], var[i])) for i, m in enumerate(models)])
        Q = (1.0 - a - b) * Rb + a * np.outer(z, z) + b * Q
    return out[burn:], corr[burn:]
