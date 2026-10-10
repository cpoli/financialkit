"""Brownian increments: plain, correlated, antithetic and scrambled Sobol.

Every generator returns increments ``dW`` on a uniform grid of ``n_steps``
steps over ``[0, T]``, shaped ``(n_paths, n_steps)`` for a single Brownian
motion or ``(n_paths, n_steps, dim)`` for several, ready to pass to
:func:`~financialkit.integrators.euler_maruyama`,
:func:`~financialkit.integrators.milstein` or
:func:`~financialkit.integrators.gbm_paths`. Random numbers are always drawn
here, in Python, from ``np.random.default_rng(rng)``; the compiled kernels
never call :mod:`numpy.random`, so results are identical with and without
Numba.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, qmc

__all__ = [
    "RNGLike",
    "brownian_increments",
    "correlated_increments",
    "antithetic_increments",
    "sobol_increments",
    "brownian_bridge",
]

#: Anything :func:`numpy.random.default_rng` accepts as a seed.
RNGLike = Optional[np.random.Generator | int]


def _check_grid(T: float, n_steps: int, n_paths: int) -> float:
    """Validate a uniform time grid and return its step ``dt = T / n_steps``."""
    if not T > 0:
        raise ValueError(f"T must be positive, got {T}")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps}")
    if int(n_paths) != n_paths or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths}")
    return T / n_steps


def brownian_increments(T: float, n_steps: int, n_paths: int, dim: Optional[int] = None, rng: RNGLike = None) -> NDArray[np.float64]:
    """Independent Brownian increments ``dW ~ N(0, dt)``.

    Parameters
    ----------
    T : float
        Horizon in years.
    n_steps : int
        Number of time steps, so ``dt = T / n_steps``.
    n_paths : int
        Number of sample paths.
    dim : int, optional
        Number of independent Brownian motions. ``None`` (default) returns
        a single one without the trailing axis.
    rng : numpy.random.Generator or int, optional
        Generator or seed.

    Returns
    -------
    ndarray of float, shape (n_paths, n_steps) or (n_paths, n_steps, dim)
    """
    dt = _check_grid(T, n_steps, n_paths)
    shape = (n_paths, n_steps) if dim is None else (n_paths, n_steps, dim)
    return np.sqrt(dt) * np.random.default_rng(rng).standard_normal(shape)


def correlated_increments(corr: ArrayLike, T: float, n_steps: int, n_paths: int, rng: RNGLike = None) -> NDArray[np.float64]:
    """Correlated Brownian increments by Cholesky factorisation.

    With ``corr = L L^T`` and independent standard normals ``Z``, the
    increments ``sqrt(dt) * L Z`` have covariance ``corr * dt``.

    Parameters
    ----------
    corr : array_like, shape (dim, dim)
        Correlation matrix: symmetric, unit diagonal, positive definite.
    T, n_steps, n_paths, rng
        As in :func:`brownian_increments`.

    Returns
    -------
    ndarray of float, shape (n_paths, n_steps, dim)

    Raises
    ------
    ValueError
        If `corr` is not a valid correlation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> dW = correlated_increments([[1.0, -0.7], [-0.7, 1.0]], T=1.0, n_steps=1, n_paths=200_000, rng=0)
    >>> round(float(np.corrcoef(dW[:, 0, 0], dW[:, 0, 1])[0, 1]), 2)
    -0.7
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
        raise ValueError("corr must be a symmetric square matrix with unit diagonal")
    try:
        L = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("corr must be positive definite") from exc
    z = brownian_increments(T, n_steps, n_paths, dim=c.shape[0], rng=rng)
    return z @ L.T


def antithetic_increments(T: float, n_steps: int, n_paths: int, dim: Optional[int] = None, rng: RNGLike = None) -> NDArray[np.float64]:
    """Antithetic Brownian increments (Hammersley-Morton 1956).

    The first ``n_paths // 2`` paths are drawn as in
    :func:`brownian_increments`; the second half are their negatives, so
    every path is paired with its mirror image and the sample mean of
    ``dW`` is zero up to rounding.

    Parameters
    ----------
    T, n_steps, dim, rng
        As in :func:`brownian_increments`.
    n_paths : int
        Total number of paths; must be even.

    Returns
    -------
    ndarray of float, shape (n_paths, n_steps) or (n_paths, n_steps, dim)
    """
    _check_grid(T, n_steps, n_paths)
    if n_paths % 2:
        raise ValueError(f"n_paths must be even for antithetic pairs, got {n_paths}")
    half = brownian_increments(T, n_steps, n_paths // 2, dim=dim, rng=rng)
    return np.concatenate([half, -half])


def sobol_increments(T: float, n_steps: int, n_paths: int, bridge: bool = True, rng: RNGLike = None) -> NDArray[np.float64]:
    """Quasi-random Brownian increments from a scrambled Sobol sequence.

    Each path uses one ``n_steps``-dimensional point of an Owen-scrambled
    Sobol sequence (:class:`scipy.stats.qmc.Sobol`), mapped to standard
    normals by the inverse normal CDF. With ``bridge=True`` the normals
    are assembled by :func:`brownian_bridge`, so the first, best
    distributed Sobol coordinates fix the coarse shape of each path
    (Caflisch-Moskowitz 1995); otherwise coordinate ``i`` drives step ``i``.

    Parameters
    ----------
    T, n_steps, rng
        As in :func:`brownian_increments`; `rng` seeds the scrambling.
    n_paths : int
        Number of paths. A power of two keeps the Sobol points balanced
        (scipy warns otherwise).
    bridge : bool, default True
        Use Brownian-bridge ordering.

    Returns
    -------
    ndarray of float, shape (n_paths, n_steps)
    """
    dt = _check_grid(T, n_steps, n_paths)
    try:
        sampler = qmc.Sobol(d=n_steps, scramble=True, rng=np.random.default_rng(rng))
    except TypeError:  # pragma: no cover - scipy < 1.15 names the argument `seed`
        sampler = qmc.Sobol(d=n_steps, scramble=True, seed=np.random.default_rng(rng))
    z = norm.ppf(sampler.random(n_paths))
    if bridge:
        return np.diff(brownian_bridge(z, T), axis=1)
    return np.sqrt(dt) * z


def _bridge_order(n_steps: int) -> list[tuple[int, int, int]]:
    """Breadth-first bisection order ``(left, mid, right)`` of grid indices 1..n_steps-1."""
    order = []
    queue = [(0, n_steps)]
    while queue:
        left, right = queue.pop(0)
        if right - left < 2:
            continue
        mid = (left + right) // 2
        order.append((left, mid, right))
        queue += [(left, mid), (mid, right)]
    return order


def brownian_bridge(z: ArrayLike, T: float) -> NDArray[np.float64]:
    """Build Brownian paths from standard normals by Brownian-bridge construction.

    Column 0 of `z` sets the terminal value ``W(T) = sqrt(T) z_0``; each
    later column fills the midpoint of an interval whose ends are already
    known, from the bridge law

    .. math::

        W(t_m) \\mid W(t_l), W(t_r) \\sim
        N\\left(\\frac{(t_r - t_m) W(t_l) + (t_m - t_l) W(t_r)}{t_r - t_l},\\;
        \\frac{(t_m - t_l)(t_r - t_m)}{t_r - t_l}\\right),

    bisecting breadth-first (Glasserman 2004, Sec. 3.1). The result has
    the law of Brownian motion for i.i.d. normal `z`; the point of the
    ordering is that the first columns carry most of the variance, which
    is what quasi-Monte Carlo needs.

    Parameters
    ----------
    z : array_like, shape (n_paths, n_steps)
        Standard normal draws (pseudo- or quasi-random).
    T : float
        Horizon in years.

    Returns
    -------
    ndarray of float, shape (n_paths, n_steps + 1)
        ``W`` on the uniform grid ``0, T/n_steps, ..., T``, with ``W(0) = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> W = brownian_bridge(np.random.default_rng(0).standard_normal((100_000, 8)), T=2.0)
    >>> bool(np.allclose(W.var(axis=0), np.linspace(0.0, 2.0, 9), atol=0.05))
    True
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    n_paths, n_steps = z.shape
    _check_grid(T, n_steps, n_paths)
    t = np.linspace(0.0, T, n_steps + 1)
    W = np.zeros((n_paths, n_steps + 1))
    W[:, -1] = np.sqrt(T) * z[:, 0]
    for k, (l, m, r) in enumerate(_bridge_order(n_steps), start=1):
        a, b = t[m] - t[l], t[r] - t[m]
        W[:, m] = (b * W[:, l] + a * W[:, r]) / (a + b) + np.sqrt(a * b / (a + b)) * z[:, k]
    return W
