"""Empirical strong and weak convergence orders of the SDE schemes.

A scheme converges *strongly* with order :math:`\\gamma` if
:math:`E|X_T - X^{\\Delta t}_T| \\le C \\Delta t^{\\gamma}` (pathwise
accuracy, what matters for hedging a path), and *weakly* with order
:math:`\\beta` if :math:`|E f(X_T) - E f(X^{\\Delta t}_T)| \\le C \\Delta t^{\\beta}`
(accuracy of expectations, what matters for pricing). Both orders are
estimated as the slope of a least-squares line through
``(log dt, log error)`` (Kloeden-Platen 1992, Sec. 9.3).

Every step size reuses the same Brownian motion: increments are drawn on
the finest grid and summed onto coarser ones, so the comparison isolates
the discretisation error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financialkit.integrators.brownian import RNGLike, _check_grid, brownian_increments
from financialkit.integrators.sde import SDEFunc, euler_maruyama, milstein

__all__ = ["ConvergenceResult", "strong_convergence", "weak_convergence"]

Scheme = Literal["euler_maruyama", "milstein"]


@dataclass(frozen=True)
class ConvergenceResult:
    """Errors of a scheme at several step sizes and the fitted order."""

    dts: NDArray[np.float64]
    """ndarray of float: Step sizes, coarsest first."""

    errors: NDArray[np.float64]
    """ndarray of float: Monte Carlo estimate of the error at each step size."""

    standard_errors: NDArray[np.float64]
    """ndarray of float: Standard error of each entry of :attr:`errors`."""

    order: float
    """float: Slope of ``log(errors)`` against ``log(dts)``."""


def _terminal_values(
    drift: SDEFunc,
    diffusion: SDEFunc,
    x0: ArrayLike,
    T: float,
    n_steps_list: Sequence[int],
    n_paths: int,
    params: Optional[ArrayLike],
    scheme: Scheme,
    diffusion_dx: Optional[SDEFunc],
    rng: RNGLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Return ``dts``, the terminal Brownian value ``W_T`` and the scheme's ``X_T`` per step size."""
    steps = sorted(int(n) for n in n_steps_list)
    if len(steps) < 2:
        raise ValueError("n_steps_list needs at least two step counts to fit an order")
    finest = steps[-1]
    _check_grid(T, finest, n_paths)
    if any(finest % n for n in steps):
        raise ValueError(f"every entry of n_steps_list must divide the finest, {finest}")
    if scheme not in ("euler_maruyama", "milstein"):
        raise ValueError(f"scheme must be 'euler_maruyama' or 'milstein', got {scheme!r}")
    if scheme == "milstein" and diffusion_dx is None:
        raise ValueError("the milstein scheme needs diffusion_dx")
    scalar = np.ndim(x0) == 0
    dim = 1 if scalar else np.shape(x0)[0]
    dW_fine = brownian_increments(T, finest, n_paths, dim=dim, rng=rng)
    terminals = []
    for n in steps:
        dW = dW_fine.reshape(n_paths, n, finest // n, dim).sum(axis=2)
        dW = dW[:, :, 0] if scalar else dW
        if scheme == "milstein":
            assert diffusion_dx is not None
            _, paths = milstein(drift, diffusion, diffusion_dx, x0, T, n, n_paths, params=params, dW=dW)
        else:
            _, paths = euler_maruyama(drift, diffusion, x0, T, n, n_paths, params=params, dW=dW)
        terminals.append(paths[:, -1])
    W_T = dW_fine.sum(axis=1)
    return T / np.asarray(steps, dtype=float), (W_T[:, 0] if scalar else W_T), terminals


def _fit(dts: NDArray[np.float64], errors: NDArray[np.float64], ses: NDArray[np.float64]) -> ConvergenceResult:
    order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
    return ConvergenceResult(dts=dts, errors=errors, standard_errors=ses, order=order)


def strong_convergence(
    drift: SDEFunc,
    diffusion: SDEFunc,
    x0: ArrayLike,
    T: float,
    exact: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_steps_list: Sequence[int] = (8, 16, 32, 64, 128),
    n_paths: int = 2000,
    params: Optional[ArrayLike] = None,
    scheme: Scheme = "euler_maruyama",
    diffusion_dx: Optional[SDEFunc] = None,
    rng: RNGLike = None,
) -> ConvergenceResult:
    """Estimate a scheme's strong order against a known pathwise solution.

    Parameters
    ----------
    drift, diffusion, x0, T, params
        As in :func:`~financialkit.integrators.euler_maruyama`.
    exact : callable
        Exact solution at ``T`` as a function of the terminal Brownian value:
        ``exact(W_T) -> X_T``, with ``W_T`` of shape ``(n_paths,)`` (scalar
        `x0`) or ``(n_paths, d)``.
    n_steps_list : sequence of int
        Step counts to compare; each must divide the largest.
    n_paths : int
        Number of paths.
    scheme : {"euler_maruyama", "milstein"}
        Scheme to test.
    diffusion_dx : callable, optional
        Required for ``scheme="milstein"``.
    rng : numpy.random.Generator or int, optional
        Generator or seed.

    Returns
    -------
    ConvergenceResult
        ``errors`` are mean absolute (Euclidean, for ``d > 1``) terminal errors.

    Examples
    --------
    >>> import numpy as np
    >>> from financialkit.integrators import njit, strong_convergence
    >>> @njit
    ... def drift(x, t, p):
    ...     return p[0] * x
    >>> @njit
    ... def diffusion(x, t, p):
    ...     return p[1] * x
    >>> exact = lambda W: np.exp((1.0 - 0.5 * 0.5**2) + 0.5 * W)
    >>> res = strong_convergence(drift, diffusion, 1.0, 1.0, exact, params=[1.0, 0.5], rng=0)
    >>> bool(abs(res.order - 0.5) < 0.15)  # strong order 1/2
    True
    """
    dts, W_T, terminals = _terminal_values(drift, diffusion, x0, T, n_steps_list, n_paths, params, scheme, diffusion_dx, rng)
    x_exact = np.asarray(exact(W_T), dtype=float)
    errs = [np.abs(x - x_exact) if x.ndim == 1 else np.linalg.norm(x - x_exact, axis=1) for x in terminals]
    errors = np.array([e.mean() for e in errs])
    ses = np.array([e.std(ddof=1) / np.sqrt(n_paths) for e in errs])
    return _fit(dts, errors, ses)


def weak_convergence(
    drift: SDEFunc,
    diffusion: SDEFunc,
    x0: ArrayLike,
    T: float,
    exact_mean: float,
    payoff: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    n_steps_list: Sequence[int] = (4, 8, 16, 32),
    n_paths: int = 100_000,
    params: Optional[ArrayLike] = None,
    scheme: Scheme = "euler_maruyama",
    diffusion_dx: Optional[SDEFunc] = None,
    rng: RNGLike = None,
) -> ConvergenceResult:
    """Estimate a scheme's weak order against a known expectation.

    The weak error is usually small next to the Monte Carlo noise, so use
    coarse steps and many paths, and check that ``errors`` stand well
    clear of ``standard_errors`` before trusting ``order``.

    Parameters
    ----------
    exact_mean : float
        Exact value of ``E[payoff(X_T)]``.
    payoff : callable, optional
        ``payoff(X_T) -> ndarray of shape (n_paths,)``; the identity by
        default (scalar SDEs).
    drift, diffusion, x0, T, n_steps_list, n_paths, params, scheme, diffusion_dx, rng
        As in :func:`strong_convergence`.

    Returns
    -------
    ConvergenceResult
        ``errors`` are ``|mean(payoff(X_T)) - exact_mean|``.
    """
    dts, _, terminals = _terminal_values(drift, diffusion, x0, T, n_steps_list, n_paths, params, scheme, diffusion_dx, rng)
    values = [np.asarray(x if payoff is None else payoff(x), dtype=float) for x in terminals]
    errors = np.array([abs(v.mean() - exact_mean) for v in values])
    ses = np.array([v.std(ddof=1) / np.sqrt(n_paths) for v in values])
    return _fit(dts, errors, ses)
