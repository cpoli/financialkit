"""Plotting helpers for financialkit.pde_methods: value profiles and
surfaces, exercise boundaries, amplification factors, grid convergence,
grid spacing, heat-equation solutions and two-asset value surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from financialkit.pde_methods.core.base import ADIResult, FDResult, HeatEquationResult, HestonPDEResult, VonNeumannResult

__all__ = [
    "plot_value_profile",
    "plot_value_surface",
    "plot_exercise_boundary",
    "plot_amplification",
    "plot_grid_convergence",
    "plot_grid",
    "plot_heat_solution",
    "plot_adi_surface",
    "plot_heston_surface",
]


def plot_value_profile(result: FDResult, exact=None, ax=None, label: str = "", xlim=None):
    """Today's grid values against the spot, optionally with exact values on the same grid.

    Parameters
    ----------
    result : FDResult
    exact : array_like, shape (M + 1,), optional
        Reference values at ``result.S``.
    ax : matplotlib.axes.Axes, optional
    label : str
        Legend label of the grid values (default: the method).
    xlim : tuple of float, optional
        Spot range to show.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if exact is not None:
        ax.plot(result.S, np.asarray(exact), "k", lw=1, label="exact")
    ax.plot(result.S, result.V, ".", ms=3, label=label or result.method)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("spot S")
    ax.set_ylabel("value V(S, 0)")
    ax.legend()
    return ax


def plot_value_surface(result: FDResult, ax=None, s_range=None, levels: int = 20):
    """Filled contours of the value over spot and calendar time.

    Parameters
    ----------
    result : FDResult
    ax : matplotlib.axes.Axes, optional
    s_range : tuple of float, optional
        Spot range to show.
    levels : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    keep = np.ones(result.S.size, dtype=bool) if s_range is None else (result.S >= s_range[0]) & (result.S <= s_range[1])
    cs = ax.contourf(result.S[keep], result.times, result.surface[:, keep], levels=levels, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="V")
    ax.set_xlabel("spot S")
    ax.set_ylabel("time t")
    return ax


def plot_exercise_boundary(result: FDResult, ax=None, label: str = "exercise boundary"):
    """Early-exercise boundary of an American valuation against time.

    Parameters
    ----------
    result : FDResult
        From an American valuation.
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if result.exercise_boundary.size == 0:
        raise ValueError("result has no exercise boundary (European valuation)")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.times, result.exercise_boundary, label=label)
    ax.set_xlabel("time t")
    ax.set_ylabel("critical spot S*(t)")
    ax.legend()
    return ax


def plot_amplification(results: VonNeumannResult | Sequence[VonNeumannResult], labels: Sequence[str] | None = None, ax=None):
    """Modulus of the amplification factor against the phase angle, with the stability line.

    Parameters
    ----------
    results : VonNeumannResult or sequence of VonNeumannResult
    labels : sequence of str, optional
        Default: ``theta = ...``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if isinstance(results, VonNeumannResult):
        results = [results]
    if ax is None:
        _, ax = plt.subplots()
    for i, res in enumerate(results):
        ax.plot(res.xi, np.abs(res.g), label=labels[i] if labels else f"theta = {res.theta:g}")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"phase angle $\xi = k \Delta x$")
    ax.set_ylabel(r"$|g(\xi)|$")
    ax.legend()
    return ax


def plot_grid_convergence(n, errors, ax=None, label: str = "", rates: Sequence[float] = (1.0, 2.0)):
    """Absolute errors against the number of grid points on log-log axes, with reference slopes.

    Parameters
    ----------
    n : array_like
        Grid sizes (spot intervals or time steps).
    errors : array_like
        Absolute errors, one per grid size.
    ax : matplotlib.axes.Axes, optional
    label : str
    rates : sequence of float
        Draw :math:`n^{-p}` guides through the first point for each ``p``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    n, errors = np.asarray(n, dtype=np.float64), np.abs(np.asarray(errors, dtype=np.float64))
    if ax is None:
        _, ax = plt.subplots()
    ax.loglog(n, errors, "o-", label=label or None)
    for p in rates:
        ax.loglog(n, errors[0] * (n / n[0]) ** -p, ":", color="0.4", label=f"$n^{{-{p:g}}}$")
    ax.set_xlabel("grid points n")
    ax.set_ylabel("absolute error")
    ax.legend()
    return ax


def plot_grid(S, ax=None, K: float | None = None, label: str = ""):
    """Local spacing of a spot grid against the spot.

    Parameters
    ----------
    S : array_like, shape (M + 1,)
    ax : matplotlib.axes.Axes, optional
    K : float, optional
        Strike, marked by a vertical line.
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    S = np.asarray(S, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(0.5 * (S[1:] + S[:-1]), np.diff(S), ".-", ms=3, label=label or None)
    if K is not None:
        ax.axvline(K, color="k", lw=0.8, ls="--")
    ax.set_xlabel("spot S")
    ax.set_ylabel(r"spacing $\Delta S$")
    if label:
        ax.legend()
    return ax


def plot_heat_solution(result: HeatEquationResult, n_curves: int = 5, ax=None):
    """Heat-equation solution :math:`u(x, \\tau)` at a few scaled times.

    Parameters
    ----------
    result : HeatEquationResult
    n_curves : int
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for k in np.linspace(0, result.tau.size - 1, n_curves).astype(int):
        ax.plot(result.x, result.u[k], label=rf"$\tau = {result.tau[k]:.3f}$")
    ax.set_xlabel(r"$x = \ln(S/K)$")
    ax.set_ylabel(r"$u(x, \tau)$")
    ax.legend()
    return ax


def plot_adi_surface(result: ADIResult, ax=None, levels: int = 20, log: bool = False):
    """Filled contours of a two-asset value surface.

    Parameters
    ----------
    result : ADIResult
    ax : matplotlib.axes.Axes, optional
    levels : int
    log : bool
        Use logarithmic spot axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    cs = ax.contourf(result.S1, result.S2, result.V.T, levels=levels, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="V")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    return ax


def plot_heston_surface(result: HestonPDEResult, ax=None, levels: int = 20, s_range: tuple[float, float] | None = None, v_max: float | None = 1.0):
    """Filled contours of a Heston value surface over spot and variance.

    Parameters
    ----------
    result : HestonPDEResult
    ax : matplotlib.axes.Axes, optional
    levels : int
    s_range : (float, float), optional
        Spot window to show.
    v_max : float, optional
        Largest variance to show.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    cs = ax.contourf(result.S, result.v, result.V.T, levels=levels, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="V")
    if s_range is not None:
        ax.set_xlim(*s_range)
    if v_max is not None:
        ax.set_ylim(0.0, v_max)
    ax.set_xlabel("S")
    ax.set_ylabel("variance v")
    return ax
