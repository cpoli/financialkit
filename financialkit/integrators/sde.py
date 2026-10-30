"""Euler-Maruyama and Milstein schemes for Ito SDEs with diagonal noise.

Both schemes integrate

.. math:: dX_i = a_i(X, t)\\,dt + b_i(X, t)\\,dW_i, \\qquad i = 1, \\dots, d,

where each component has its own Brownian driver ``W_i`` (correlate the
drivers with :func:`~financialkit.integrators.correlated_increments`).

Callbacks follow the ``f(x, t, params) -> ndarray`` convention shared with
mathematicskit's ODE integrators, with one difference: ``x`` is the whole
batch of paths, shape ``(n_paths, d)``, and the callback returns an array
of that shape. A step is therefore a handful of vectorised array
operations, so the time loop runs in plain Python and NumPy does the
work: the kernels are not compiled with Numba, which measured no faster
on them (see the docs' Benchmarks page). Any callable works as a
callback, including closures, bound methods and existing ``@njit``
dispatchers; ``params`` is passed through as a ``float64`` array.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financialkit.integrators.brownian import RNGLike, _check_grid, brownian_increments

__all__ = ["SDEFunc", "euler_maruyama", "milstein"]

#: Signature of every drift / diffusion callback: ``f(x, t, params) -> ndarray``
#: with ``x`` of shape ``(n_paths, d)``.
SDEFunc = Callable[[NDArray[np.float64], float, NDArray[np.float64]], NDArray[np.float64]]


def _euler_maruyama_kernel(drift, diffusion, x0, t0, dt, dW, params):
    n_steps = dW.shape[0]
    out = np.empty((n_steps + 1,) + x0.shape)
    out[0] = x0
    x = x0
    for i in range(n_steps):
        t = t0 + i * dt
        x = x + drift(x, t, params) * dt + diffusion(x, t, params) * dW[i]
        out[i + 1] = x
    return out


def _milstein_kernel(drift, diffusion, diffusion_dx, x0, t0, dt, dW, params):
    n_steps = dW.shape[0]
    out = np.empty((n_steps + 1,) + x0.shape)
    out[0] = x0
    x = x0
    for i in range(n_steps):
        t = t0 + i * dt
        b = diffusion(x, t, params)
        x = x + drift(x, t, params) * dt + b * dW[i] + 0.5 * b * diffusion_dx(x, t, params) * (dW[i] * dW[i] - dt)
        out[i + 1] = x
    return out


def _prepare(
    x0: ArrayLike,
    T: float,
    n_steps: int,
    n_paths: int,
    params: Optional[ArrayLike],
    t0: float,
    rng: RNGLike,
    dW: Optional[ArrayLike],
) -> tuple[bool, NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalise inputs to the kernels' layout: ``x0`` (n_paths, d), ``dW`` (n_steps, n_paths, d)."""
    dt = _check_grid(T, n_steps, n_paths)
    x0_arr = np.asarray(x0, dtype=float)
    if x0_arr.ndim > 1:
        raise ValueError(f"x0 must be a scalar or a 1-D state vector, got shape {x0_arr.shape}")
    scalar = x0_arr.ndim == 0
    dim = 1 if scalar else x0_arr.shape[0]
    if dW is None:
        dW_arr = brownian_increments(T, n_steps, n_paths, dim=dim, rng=rng)
    else:
        dW_arr = np.asarray(dW, dtype=float)
        if scalar and dW_arr.ndim == 2:
            dW_arr = dW_arr[:, :, None]
        if dW_arr.shape != (n_paths, n_steps, dim):
            expected = (n_paths, n_steps) if scalar else (n_paths, n_steps, dim)
            raise ValueError(f"dW must have shape {expected}, got {np.shape(dW)}")
    x0_batch = np.ascontiguousarray(np.broadcast_to(x0_arr.reshape(dim), (n_paths, dim)))
    dW_steps = np.ascontiguousarray(dW_arr.transpose(1, 0, 2))
    p = np.asarray([] if params is None else params, dtype=float).reshape(-1)
    times = np.linspace(t0, t0 + T, n_steps + 1)
    return scalar, x0_batch, dt, dW_steps, p, times


def _finish(scalar: bool, times: NDArray[np.float64], out: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    paths = out.transpose(1, 0, 2)
    return times, (paths[:, :, 0] if scalar else paths)


def euler_maruyama(
    drift: SDEFunc,
    diffusion: SDEFunc,
    x0: ArrayLike,
    T: float,
    n_steps: int,
    n_paths: int,
    params: Optional[ArrayLike] = None,
    t0: float = 0.0,
    rng: RNGLike = None,
    dW: Optional[ArrayLike] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate SDE paths with the Euler-Maruyama scheme (Maruyama 1955).

    .. math:: X_{n+1} = X_n + a(X_n, t_n)\\,\\Delta t + b(X_n, t_n)\\,\\Delta W_n

    Strong order 1/2 and weak order 1 for smooth coefficients
    (Kloeden-Platen 1992, Thms. 10.2.2 and 14.5.1).

    Parameters
    ----------
    drift, diffusion : callable
        ``f(x, t, params) -> ndarray`` with ``x`` of shape ``(n_paths, d)``;
        see the module docstring.
    x0 : float or array_like, shape (d,)
        Initial state, shared by every path.
    T : float
        Horizon in years.
    n_steps : int
        Number of uniform steps.
    n_paths : int
        Number of paths.
    params : array_like of float, optional
        Parameter vector passed through to the callbacks.
    t0 : float, default 0.0
        Start time.
    rng : numpy.random.Generator or int, optional
        Generator or seed for the increments; ignored if `dW` is given.
    dW : array_like, optional
        Brownian increments to use instead of fresh ones, shape
        ``(n_paths, n_steps)`` for scalar `x0` or ``(n_paths, n_steps, d)``
        (e.g. from :func:`~financialkit.integrators.antithetic_increments`).

    Returns
    -------
    times : ndarray of float, shape (n_steps + 1,)
    paths : ndarray of float, shape (n_paths, n_steps + 1) or (n_paths, n_steps + 1, d)
        Scalar `x0` drops the trailing state axis.

    Examples
    --------
    Geometric Brownian motion ``dS = mu S dt + sigma S dW``:

    >>> import numpy as np
    >>> from financialkit.integrators import euler_maruyama
    >>> def drift(x, t, p):
    ...     return p[0] * x
    >>> def diffusion(x, t, p):
    ...     return p[1] * x
    >>> times, paths = euler_maruyama(drift, diffusion, 100.0, T=1.0, n_steps=50, n_paths=20_000, params=[0.05, 0.2], rng=0)
    >>> paths.shape
    (20000, 51)
    >>> bool(abs(paths[:, -1].mean() - 100 * np.exp(0.05)) < 0.5)
    True
    """
    scalar, x0_batch, dt, dW_steps, p, times = _prepare(x0, T, n_steps, n_paths, params, t0, rng, dW)
    return _finish(scalar, times, _euler_maruyama_kernel(drift, diffusion, x0_batch, float(t0), dt, dW_steps, p))


def milstein(
    drift: SDEFunc,
    diffusion: SDEFunc,
    diffusion_dx: SDEFunc,
    x0: ArrayLike,
    T: float,
    n_steps: int,
    n_paths: int,
    params: Optional[ArrayLike] = None,
    t0: float = 0.0,
    rng: RNGLike = None,
    dW: Optional[ArrayLike] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate SDE paths with the Milstein scheme (Milstein 1974).

    .. math:: X_{n+1} = X_n + a\\,\\Delta t + b\\,\\Delta W_n + \\tfrac12\\, b\\, \\partial_x b\\,(\\Delta W_n^2 - \\Delta t)

    Strong order 1 (Kloeden-Platen 1992, Thm. 10.3.5). The correction
    uses only ``diffusion_dx``, the derivative of each ``b_i`` with respect
    to its own ``x_i``, so it is exact Milstein when each ``b_i`` depends
    on ``x_i`` alone (diagonal noise); otherwise the Levy-area cross terms
    are dropped.

    Parameters
    ----------
    diffusion_dx : callable
        ``f(x, t, params) -> ndarray``: ``d b_i / d x_i`` for each component.
    drift, diffusion, x0, T, n_steps, n_paths, params, t0, rng, dW
        As in :func:`euler_maruyama`.

    Returns
    -------
    times, paths
        As in :func:`euler_maruyama`.
    """
    scalar, x0_batch, dt, dW_steps, p, times = _prepare(x0, T, n_steps, n_paths, params, t0, rng, dW)
    return _finish(scalar, times, _milstein_kernel(drift, diffusion, diffusion_dx, x0_batch, float(t0), dt, dW_steps, p))
