"""Plotting helpers for financialkit.stochastic_volatility: smiles, local
volatility surfaces, Heston paths and return densities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility.core.base import HestonPaths, LocalVolSurface

__all__ = ["plot_smile", "plot_local_vol", "plot_heston_paths", "plot_density"]


def plot_smile(strikes, vols, forward: float | None = None, ax=None, label: str = "implied vol", **kwargs):
    """Implied volatility against strike, or against log-moneyness if ``forward`` is given.

    Parameters
    ----------
    strikes, vols : array_like, shape (n,)
    forward : float, optional
        Plot against :math:`\\ln(K/F)` instead of :math:`K`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    K = np.asarray(strikes, dtype=np.float64)
    x = K if forward is None else np.log(K / forward)
    ax.plot(x, vols, label=label, **kwargs)
    ax.set_xlabel("strike K" if forward is None else "log-moneyness ln(K/F)")
    ax.set_ylabel("implied volatility")
    ax.legend()
    return ax


def plot_local_vol(surface: LocalVolSurface, maturities=None, ax=None):
    """Local volatility against strike, one curve per maturity.

    Parameters
    ----------
    surface : LocalVolSurface
    maturities : array_like, optional
        Maturities to draw (nearest grid rows); five spread over the grid by default.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    T = surface.maturities
    if maturities is None:
        rows = np.unique(np.linspace(0, T.size - 1, min(5, T.size)).round().astype(int))
    else:
        rows = np.unique([int(np.abs(T - t).argmin()) for t in np.atleast_1d(maturities)])
    for i in rows:
        ax.plot(surface.strikes, surface.vol[i], label=f"T = {T[i]:.2f}")
    ax.set_xlabel("spot / strike")
    ax.set_ylabel("local volatility")
    ax.legend()
    return ax


def plot_heston_paths(paths: HestonPaths, n_show: int = 20, axes=None):
    """Spot and volatility paths side by side.

    Parameters
    ----------
    paths : HestonPaths
    n_show : int
        Number of paths drawn.
    axes : sequence of two matplotlib.axes.Axes, optional

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes = np.asarray(axes).ravel()
    n = min(n_show, paths.spot.shape[0])
    axes[0].plot(paths.times, paths.spot[:n].T, lw=0.7)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("spot S")
    axes[1].plot(paths.times, np.sqrt(np.maximum(paths.variance[:n], 0.0)).T, lw=0.7)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("volatility sqrt(v)")
    axes[0].set_title(f"Heston paths ({paths.scheme})")
    return axes


def plot_density(x, density, ax=None, label: str = "density", log: bool = False, **kwargs):
    """A density of log returns, optionally on a log scale to show the tails.

    Parameters
    ----------
    x, density : array_like, shape (n,)
    ax : matplotlib.axes.Axes, optional
    label : str
    log : bool
        Logarithmic vertical axis.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, density, label=label, **kwargs)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("log return")
    ax.set_ylabel("density")
    ax.legend()
    return ax
