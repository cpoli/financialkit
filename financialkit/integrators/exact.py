"""Exact sampling of geometric Brownian motion and the Ornstein-Uhlenbeck process.

Both processes have Gaussian transition laws, so they are simulated on
the grid with no discretisation error at all. That makes them the
reference against which :func:`~financialkit.integrators.euler_maruyama`
and :func:`~financialkit.integrators.milstein` are measured.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financialkit.integrators.brownian import RNGLike, _check_grid, brownian_increments

__all__ = ["gbm_paths", "ou_paths"]


def gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: RNGLike = None,
    dW: Optional[ArrayLike] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact paths of geometric Brownian motion ``dS = mu S dt + sigma S dW``.

    Uses the closed-form solution (Samuelson 1965; Black-Scholes 1973)

    .. math:: S_t = S_0 \\exp\\left((\\mu - \\tfrac12 \\sigma^2) t + \\sigma W_t\\right).

    Under the risk-neutral measure, pass ``mu = r - q``.

    Parameters
    ----------
    s0 : float
        Initial price.
    mu : float
        Drift (continuously compounded, per year).
    sigma : float
        Volatility (per square-root year).
    T : float
        Horizon in years.
    n_steps : int
        Number of uniform steps.
    n_paths : int
        Number of paths.
    rng : numpy.random.Generator or int, optional
        Generator or seed; ignored if `dW` is given.
    dW : array_like, shape (n_paths, n_steps), optional
        Brownian increments to use, e.g. the same ones fed to
        :func:`~financialkit.integrators.euler_maruyama` for a pathwise
        comparison.

    Returns
    -------
    times : ndarray of float, shape (n_steps + 1,)
    paths : ndarray of float, shape (n_paths, n_steps + 1)

    Examples
    --------
    >>> import numpy as np
    >>> times, S = gbm_paths(100.0, 0.05, 0.2, T=1.0, n_steps=12, n_paths=100_000, rng=0)
    >>> bool(abs(S[:, -1].mean() / (100 * np.exp(0.05)) - 1) < 0.005)
    True
    """
    _check_grid(T, n_steps, n_paths)
    if dW is None:
        dW = brownian_increments(T, n_steps, n_paths, rng=rng)
    dW = np.asarray(dW, dtype=float)
    if dW.shape != (n_paths, n_steps):
        raise ValueError(f"dW must have shape {(n_paths, n_steps)}, got {dW.shape}")
    times = np.linspace(0.0, T, n_steps + 1)
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    return times, s0 * np.exp((mu - 0.5 * sigma**2) * times + sigma * W)


def ou_paths(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: RNGLike = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact paths of the Ornstein-Uhlenbeck process ``dX = kappa (theta - X) dt + sigma dW``.

    Samples the Gaussian transition (Uhlenbeck-Ornstein 1930; the Vasicek
    1977 short rate)

    .. math:: X_{t+\\Delta t} = \\theta + (X_t - \\theta) e^{-\\kappa \\Delta t} + \\sigma \\sqrt{\\frac{1 - e^{-2\\kappa\\Delta t}}{2\\kappa}}\\, Z,

    which reduces to arithmetic Brownian motion when ``kappa = 0``.

    Parameters
    ----------
    x0 : float
        Initial value.
    kappa : float
        Mean-reversion speed (per year).
    theta : float
        Long-run mean.
    sigma : float
        Volatility.
    T, n_steps, n_paths, rng
        As in :func:`gbm_paths`.

    Returns
    -------
    times : ndarray of float, shape (n_steps + 1,)
    paths : ndarray of float, shape (n_paths, n_steps + 1)
    """
    dt = _check_grid(T, n_steps, n_paths)
    decay = np.exp(-kappa * dt)
    var = dt if kappa == 0 else -np.expm1(-2.0 * kappa * dt) / (2.0 * kappa)
    z = np.random.default_rng(rng).standard_normal((n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for i in range(n_steps):
        paths[:, i + 1] = theta + (paths[:, i] - theta) * decay + sigma * np.sqrt(var) * z[:, i]
    return np.linspace(0.0, T, n_steps + 1), paths
