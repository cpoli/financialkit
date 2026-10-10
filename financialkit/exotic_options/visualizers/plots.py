"""Plotting helpers for financialkit.exotic_options: price profiles against
a vanilla benchmark, simulated paths against barriers, the
Longstaff-Schwartz exercise boundary and a variance-swap replication strip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from financialkit.exotic_options.core.base import LongstaffSchwartzResult, VarianceSwapResult

__all__ = ["plot_price_profile", "plot_barrier_paths", "plot_exercise_boundary", "plot_variance_strip"]


def plot_price_profile(
    x: ArrayLike, curves: Mapping[str, ArrayLike], ax=None, xlabel: str = "spot S", ylabel: str = "option value", markers: Sequence[float] = ()
):
    """Several pricing curves over a common axis, with dashed vertical markers (strikes, barriers).

    Parameters
    ----------
    x : array_like
        Common abscissa (usually the spot).
    curves : mapping of str to array_like
        Label and values of each curve.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    xlabel, ylabel : str
        Axis labels.
    markers : sequence of float
        Positions of dashed vertical lines.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for label, y in curves.items():
        ax.plot(x, np.asarray(y, dtype=np.float64), label=label)
    for m in markers:
        ax.axvline(m, color="0.5", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def plot_barrier_paths(times: ArrayLike, paths: ArrayLike, barriers: Sequence[float], ax=None, max_paths: int = 40):
    """Simulated price paths, red where they touch a barrier on the grid and blue where they survive.

    Parameters
    ----------
    times : array_like, shape (n_steps + 1,)
    paths : array_like, shape (n_paths, n_steps + 1)
    barriers : sequence of float
        Barrier levels; a path is knocked out if it reaches any of them
        from the starting side.
    ax : matplotlib.axes.Axes, optional
    max_paths : int
        Number of paths drawn.

    Returns
    -------
    matplotlib.axes.Axes
    """
    t, P = np.asarray(times, dtype=np.float64), np.asarray(paths, dtype=np.float64)[: int(max_paths)]
    if ax is None:
        _, ax = plt.subplots()
    S0 = P[0, 0]
    hit = np.zeros(P.shape[0], dtype=bool)
    for H in barriers:
        hit |= np.any(P <= H, axis=1) if H < S0 else np.any(P >= H, axis=1)
    for path, h in zip(P, hit, strict=True):
        ax.plot(t, path, color="C3" if h else "C0", lw=0.8, alpha=0.7)
    for H in barriers:
        ax.axhline(H, color="k", lw=1.5)
    ax.set_xlabel("time t")
    ax.set_ylabel("price $S_t$")
    ax.set_title(f"{int(hit.sum())} of {hit.size} paths knocked out")
    return ax


def plot_exercise_boundary(result: LongstaffSchwartzResult, ax=None, label: str = "Longstaff-Schwartz"):
    """The early-exercise boundary of a :class:`~financialkit.exotic_options.core.base.LongstaffSchwartzResult`.

    Parameters
    ----------
    result : LongstaffSchwartzResult
    ax : matplotlib.axes.Axes, optional
    label : str
        Legend label.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.times, result.exercise_boundary, "o", ms=3, label=label)
    ax.set_xlabel("time t")
    ax.set_ylabel("critical price")
    ax.legend()
    return ax


def plot_variance_strip(result: VarianceSwapResult, ax=None):
    """Each option's contribution :math:`w_i Q(K_i)` to a variance swap's fair strike, puts left of :math:`K_0` and calls right.

    Parameters
    ----------
    result : VarianceSwapResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    K, c = result.strikes, result.contributions
    width = np.gradient(K)
    puts = K < result.K0
    ax.bar(K[puts], c[puts], width=width[puts], color="C3", label="puts")
    ax.bar(K[~puts], c[~puts], width=width[~puts], color="C0", label="calls")
    ax.axvline(result.forward, color="k", ls="--", lw=1, label="forward")
    ax.set_xlabel("strike K")
    ax.set_ylabel(r"contribution $w_i Q(K_i)$")
    ax.set_title(rf"fair volatility {result.vol_strike:.2%}")
    ax.legend()
    return ax
