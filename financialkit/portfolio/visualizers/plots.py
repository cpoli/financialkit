"""Plotting helpers for financialkit.portfolio: efficient frontiers,
weight compositions, CPPI paths and out-of-sample wealth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from financialkit.portfolio.core.base import CPPIResult, FrontierResult, OutOfSampleResult

__all__ = ["plot_frontier", "plot_weights", "plot_cppi", "plot_out_of_sample"]


def plot_frontier(frontier: FrontierResult, ax=None, labels: Optional[Sequence[str]] = None, rf: Optional[float] = None, **kwargs):
    """The frontier in the volatility-mean plane, with the assets and, optionally, the capital market line.

    Parameters
    ----------
    frontier : FrontierResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    labels : sequence of str, optional
        Asset names to annotate.
    rf : float, optional
        Riskless rate: draws the line from it tangent to the frontier.
    **kwargs
        Passed to the frontier's ``ax.plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    kwargs.setdefault("label", "long-only frontier" if frontier.long_only else "efficient frontier")
    ax.plot(frontier.volatilities, frontier.returns, lw=2, **kwargs)
    sig = frontier.asset_volatilities
    ax.plot(sig, frontier.mu, "o", color="0.3", ms=5)
    if labels is not None:
        for name, x, y in zip(labels, sig, frontier.mu, strict=False):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
    if rf is not None:
        sr = (frontier.returns - rf) / frontier.volatilities
        k = int(np.argmax(sr))
        xs = np.array([0.0, 1.2 * max(sig.max(), frontier.volatilities.max())])
        ax.plot(xs, rf + sr[k] * xs, "--", color="C3", lw=1, label="capital market line")
        ax.plot(frontier.volatilities[k], frontier.returns[k], "*", color="C3", ms=12)
    ax.set_xlim(left=0)
    ax.set_xlabel("volatility")
    ax.set_ylabel("expected return")
    ax.legend(fontsize=8)
    return ax


def plot_weights(weights, labels: Optional[Sequence[str]] = None, x=None, ax=None):
    """Bar chart of one portfolio, or a stacked area chart of a family of portfolios.

    Parameters
    ----------
    weights : array_like, shape (d,) or (n, d)
    labels : sequence of str, optional
    x : array_like, shape (n,), optional
        Horizontal coordinate of each portfolio in the 2-D case (e.g. its volatility).
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    W = np.asarray(weights, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    names = list(labels) if labels is not None else [f"asset {i}" for i in range(W.shape[-1])]
    if W.ndim == 1:
        ax.bar(names, W, color=[f"C{i % 10}" for i in range(W.size)])
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_ylabel("weight")
    else:
        xs = np.arange(W.shape[0]) if x is None else np.asarray(x)
        ax.stackplot(xs, W.T, labels=names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("weight")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return ax


def plot_cppi(result: CPPIResult, ax=None, n_paths: int = 20):
    """CPPI wealth paths against the floor, gap-risk breaches highlighted.

    Parameters
    ----------
    result : CPPIResult
    ax : matplotlib.axes.Axes, optional
    n_paths : int
        Number of paths drawn.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    k = min(n_paths, result.wealth.shape[0])
    breached = result.breached
    for i in range(k):
        ax.plot(result.times, result.wealth[i], color="C3" if breached[i] else "C0", lw=0.8, alpha=0.7)
    ax.plot(result.times, result.floor, "k--", lw=1.5, label="floor")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("wealth")
    ax.set_title(f"CPPI, m = {result.multiplier:g}: {breached.mean():.1%} of paths breach the floor")
    ax.legend()
    return ax


def plot_out_of_sample(results: Sequence[OutOfSampleResult], ax=None):
    """Cumulative wealth of several rolling-window strategies.

    Parameters
    ----------
    results : sequence of OutOfSampleResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for res in results:
        wealth = np.concatenate([[1.0], np.cumprod(1.0 + res.returns)])
        ax.plot(wealth, label=f"{res.name}: SR = {res.sharpe:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("wealth")
    ax.legend(fontsize=8)
    return ax
