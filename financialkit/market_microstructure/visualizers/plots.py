"""Plotting helpers for financialkit.market_microstructure: quotes and
trades, Glosten-Milgrom learning, dealer inventories, market making,
execution schedules, trade-impact responses and VPIN.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure.core.base import ExecutionResult, GarmanResult, GlostenMilgromResult, MarketMakingResult, TradeImpactResult, VPINResult

__all__ = ["plot_quotes", "plot_glosten_milgrom", "plot_garman", "plot_market_making", "plot_execution", "plot_impulse_response", "plot_vpin"]


def plot_quotes(prices, bid, ask, side=None, times=None, ax=None):
    """Bid and ask as steps with trades as dots, coloured by side if given.

    Parameters
    ----------
    prices, bid, ask : array_like, shape (n,)
    side : array_like, shape (n,), optional
        ``+1`` buys are drawn in C2, ``-1`` sells in C3, others grey.
    times : array_like, shape (n,), optional
        Trade index if omitted.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    p = np.asarray(prices, dtype=np.float64)
    t = np.arange(p.size) if times is None else np.asarray(times)
    ax.step(t, bid, where="post", color="C0", lw=1, label="bid")
    ax.step(t, ask, where="post", color="C1", lw=1, label="ask")
    if side is None:
        ax.plot(t, p, ".", color="0.3", ms=4, label="trades")
    else:
        q = np.asarray(side)
        for val, color, name in [(1, "C2", "buys"), (-1, "C3", "sells"), (0, "0.5", "unclassified")]:
            m = q == val
            if m.any():
                ax.plot(t[m], p[m], ".", color=color, ms=5, label=name)
    ax.set_xlabel("time" if times is not None else "trade")
    ax.set_ylabel("price")
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_glosten_milgrom(result: GlostenMilgromResult, ax=None):
    """Glosten-Milgrom bid and ask converging to the true value.

    Parameters
    ----------
    result : GlostenMilgromResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    n = result.trades.size
    i = np.arange(n)
    ax.fill_between(i, result.bid, result.ask, step="post", color="C0", alpha=0.25, label="spread")
    ax.step(i, result.ask, where="post", color="C1", lw=1, label="ask")
    ax.step(i, result.bid, where="post", color="C0", lw=1, label="bid")
    ax.axhline(result.true_value, color="k", ls="--", lw=1, label="true value")
    ax.set_xlabel("trade")
    ax.set_ylabel("price")
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_garman(result: GarmanResult, n_show: int = 20, ax=None):
    """Inventory paths of Garman's dealer, failures ending in a cross.

    Parameters
    ----------
    result : GarmanResult
    n_show : int
        Number of paths drawn.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for j in range(min(n_show, result.stock.shape[0])):
        color = "C3" if result.failed[j] else "C0"
        ax.step(result.times[j], result.stock[j], where="post", color=color, lw=0.8, alpha=0.7)
        if result.failed[j]:
            k = np.searchsorted(result.times[j], result.failure_time[j])
            ax.plot(result.failure_time[j], result.stock[j, k], "x", color="C3")
    ax.set_xlabel("time")
    ax.set_ylabel("dealer inventory (shares)")
    ax.set_title(f"{result.failed.mean():.0%} of {result.failed.size} dealers failed")
    return ax


def plot_market_making(result: MarketMakingResult, path: int = 0, axes=None):
    """One market-making path: mid and quotes above, inventory below.

    Parameters
    ----------
    result : MarketMakingResult
    path : int
    axes : sequence of two matplotlib.axes.Axes, optional

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5), gridspec_kw={"height_ratios": [2, 1]})
    t = result.times
    axes[0].plot(t, result.mid[path], color="k", lw=1, label="mid")
    axes[0].plot(t[:-1], result.ask[path], color="C1", lw=0.8, label="ask")
    axes[0].plot(t[:-1], result.bid[path], color="C0", lw=0.8, label="bid")
    axes[0].set_ylabel("price")
    axes[0].legend(loc="best", fontsize="small")
    axes[1].step(t, result.inventory[path], where="post", color="C2")
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_ylabel("inventory")
    axes[1].set_xlabel("time")
    return np.asarray(axes)


def plot_execution(results, labels=None, ax=None):
    """Holdings over time for one or more execution schedules.

    Parameters
    ----------
    results : ExecutionResult or sequence of ExecutionResult
    labels : sequence of str, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if isinstance(results, ExecutionResult):
        results = [results]
    for i, r in enumerate(results):
        ax.plot(r.times, r.holdings, "o-", ms=3, label=None if labels is None else labels[i])
    ax.set_xlabel("time")
    ax.set_ylabel("shares still held")
    if labels is not None:
        ax.legend(loc="best", fontsize="small")
    return ax


def plot_impulse_response(result: TradeImpactResult, ax=None):
    """Cumulative quote revision after a unit trade innovation, with its permanent level.

    Parameters
    ----------
    result : TradeImpactResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.arange(result.impulse.size), result.impulse, "o-", ms=3, color="C0", label="cumulative response")
    ax.axhline(result.permanent_impact, color="C3", ls="--", label=f"permanent impact {result.permanent_impact:.4g}")
    ax.set_xlabel("trades after the shock")
    ax.set_ylabel("quote revision")
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_vpin(result: VPINResult, ax=None):
    """Bucket order imbalances and the rolling VPIN.

    Parameters
    ----------
    result : VPINResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    k = np.arange(result.vpin.size)
    ax.bar(k, result.imbalance, width=1.0, color="0.8", label="bucket imbalance")
    ax.plot(k, result.vpin, color="C3", lw=1.5, label="VPIN")
    ax.set_xlabel("volume bucket")
    ax.set_ylabel(r"$|V^B - V^S| / V$")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize="small")
    return ax
