"""Supporting numerics for the short-rate models: SDE callbacks for
:mod:`financialkit.integrators`, and pathwise discounting.

The drift and diffusion callbacks follow the integrators' ``f(x, t,
params)`` convention; every model parameter travels in ``params``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

__all__ = [
    "path_discount_factors",
    "cir_drift",
    "cir_diffusion",
    "affine_drift",
    "affine_diffusion",
    "g2_drift",
    "g2_diffusion",
    "hjm_drift_callback",
    "hjm_diffusion_callback",
]


def path_discount_factors(times, rates) -> np.ndarray:
    r"""Pathwise discount factors :math:`e^{-\int_0^{t_i} r(s)\,ds}` by the trapezoidal rule.

    Parameters
    ----------
    times : array_like, shape (n + 1,)
        Simulation grid starting at 0.
    rates : array_like, shape (n_paths, n + 1)
        Short-rate paths on that grid.

    Returns
    -------
    ndarray, shape (n_paths, n + 1)
        Equal to 1 in the first column.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.0, 2.0, 5)
    >>> float(path_discount_factors(t, np.full((1, 5), 0.05))[0, -1].round(6))
    0.904837
    """
    rates = np.atleast_2d(np.asarray(rates, dtype=np.float64))
    return np.exp(-cumulative_trapezoid(rates, np.asarray(times, dtype=np.float64), axis=1, initial=0.0))


def cir_drift(x, t, p):
    """``kappa (theta - x)`` with ``p = [kappa, theta, sigma]``; full truncation uses ``max(x, 0)``."""
    return p[0] * (p[1] - np.maximum(x, 0.0))


def cir_diffusion(x, t, p):
    """``sigma sqrt(max(x, 0))`` with ``p = [kappa, theta, sigma]``."""
    return p[2] * np.sqrt(np.maximum(x, 0.0))


def affine_drift(x, t, p):
    """``K0 + K1 x`` for each path; ``p = [d, K0, K1^T (row-major), sigma, alpha, beta^T]``."""
    d = int(p[0])
    K0 = p[1 : 1 + d]
    K1T = np.ascontiguousarray(p[1 + d : 1 + d + d * d]).reshape((d, d))
    return K0 + x @ K1T


def affine_diffusion(x, t, p):
    """``sigma_k sqrt(alpha_k + beta_k . x)`` for each factor ``k``."""
    d = int(p[0])
    o = 1 + d + d * d
    sigma = p[o : o + d]
    alpha = p[o + d : o + 2 * d]
    betaT = np.ascontiguousarray(p[o + 2 * d : o + 2 * d + d * d]).reshape((d, d))
    return sigma * np.sqrt(np.maximum(alpha + x @ betaT, 0.0))


def g2_drift(x, t, p):
    """``(-a x, -b y)`` with ``p = [a, b, sigma, eta]``."""
    out = np.empty_like(x)
    out[:, 0] = -p[0] * x[:, 0]
    out[:, 1] = -p[1] * x[:, 1]
    return out


def g2_diffusion(x, t, p):
    """``(sigma, eta)`` with ``p = [a, b, sigma, eta]``."""
    out = np.empty_like(x)
    out[:, 0] = p[2]
    out[:, 1] = p[3]
    return out


def _hjm_vol(t, p):
    tau = np.maximum(p[3:] - t, 0.0)
    return tau, p[0] * np.exp(-p[1] * tau)


def hjm_drift_callback(x, t, p):
    r"""HJM drift :math:`\sigma(t,T)\int_t^T \sigma(t,u)\,du` for :math:`\sigma(t,T) = \sigma e^{-a(T-t)}`.

    ``p = [sigma, a, use_drift, T_0, ..., T_n]``; the drift is multiplied by
    ``use_drift`` so the same callback shows the arbitrage without it.
    """
    tau, s = _hjm_vol(t, p)
    if p[1] > 0.0:
        integral = p[0] * (1.0 - np.exp(-p[1] * tau)) / p[1]
    else:
        integral = p[0] * tau
    return x * 0.0 + p[2] * s * integral


def hjm_diffusion_callback(x, t, p):
    r""":math:`\sigma(t, T_j) = \sigma e^{-a(T_j - t)}` for every maturity on the grid."""
    tau, s = _hjm_vol(t, p)
    return x * 0.0 + s
