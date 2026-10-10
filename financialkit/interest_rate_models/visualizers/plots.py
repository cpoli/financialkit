"""Plotting helpers for financialkit.interest_rate_models: short-rate paths,
model yield curves, Monte Carlo bond prices, rate trees, caplet strips,
Jamshidian's decomposition and HJM forward curves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from financialkit.interest_rate_models.core.base import BondMonteCarloResult, CapFloorResult, HJMResult, JamshidianResult, RateTree, ShortRateModel

__all__ = [
    "plot_short_rate_paths",
    "plot_model_yields",
    "plot_bond_monte_carlo",
    "plot_rate_tree",
    "plot_caplets",
    "plot_jamshidian",
    "plot_forward_curves",
]


def plot_short_rate_paths(times, rates, n_show: int = 20, ax=None, label: str = "", quantiles: bool = True):
    """A sample of short-rate paths, in percent, with the 5-95% band and the mean.

    Parameters
    ----------
    times : array_like, shape (n + 1,)
    rates : array_like, shape (n_paths, n + 1)
    n_show : int
        Number of individual paths drawn.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str
        Legend label for the mean.
    quantiles : bool
        Shade the 5-95% band across paths.

    Returns
    -------
    matplotlib.axes.Axes
    """
    times = np.asarray(times, dtype=np.float64)
    rates = np.atleast_2d(np.asarray(rates, dtype=np.float64))
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(times, 100 * rates[:n_show].T, lw=0.6, alpha=0.5)
    (line,) = ax.plot(times, 100 * rates.mean(axis=0), "k", lw=2, label=f"{label} mean".strip())
    if quantiles and rates.shape[0] > 1:
        lo, hi = np.quantile(rates, [0.05, 0.95], axis=0)
        ax.fill_between(times, 100 * lo, 100 * hi, color=line.get_color(), alpha=0.1, label="5-95%")
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("short rate (%)")
    ax.legend()
    return ax


def plot_model_yields(model: ShortRateModel, t_max: float = 30.0, ax=None, label: str = ""):
    """A model's time-0 zero curve, in percent.

    Parameters
    ----------
    model : ShortRateModel
    t_max : float
        Longest maturity.
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    T = np.linspace(0.05, t_max, 300)
    ax.plot(T, 100 * model.yields(T), label=label or None)
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("zero yield (%)")
    if label:
        ax.legend()
    return ax


def plot_bond_monte_carlo(result: BondMonteCarloResult, ax=None, label: str = "Monte Carlo"):
    """Monte Carlo bond prices with 95% error bars against the closed form.

    Parameters
    ----------
    result : BondMonteCarloResult
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.maturities, result.exact, "k-", lw=1, label="closed form")
    ax.errorbar(result.maturities, result.prices, yerr=1.96 * result.std_errors, fmt="o", ms=3, capsize=2, label=label)
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("zero-coupon bond price")
    ax.legend()
    return ax


def plot_rate_tree(tree: RateTree, ax=None, max_steps: int = 12):
    """The nodes of a binomial short-rate tree, in percent, joined by their branches.

    Parameters
    ----------
    tree : RateTree
    ax : matplotlib.axes.Axes, optional
    max_steps : int
        Draw only the first `max_steps` steps of a large tree.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    n = min(tree.n_steps, max_steps)
    for i in range(n):
        r = 100 * tree.rates[i, : i + 1]
        if i + 1 < n:
            nxt = 100 * tree.rates[i + 1, : i + 2]
            for j in range(i + 1):
                ax.plot([tree.times[i], tree.times[i + 1]], [r[j], nxt[j]], color="0.7", lw=0.7)
                ax.plot([tree.times[i], tree.times[i + 1]], [r[j], nxt[j + 1]], color="0.7", lw=0.7)
        ax.plot(np.full(i + 1, tree.times[i]), r, "o", color="C0", ms=3)
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("short rate (%)")
    return ax


def plot_caplets(result: CapFloorResult, ax=None, label: str = ""):
    """Caplet (floorlet) prices by fixing date, with the forward rates on a twin axis.

    Parameters
    ----------
    result : CapFloorResult
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    width = 0.8 * np.min(result.payment_times - result.reset_times)
    ax.bar(result.reset_times, result.caplets, width=width, align="edge", alpha=0.7, label=label or f"{result.kind}lets")
    ax.set_xlabel("fixing date (years)")
    ax.set_ylabel(f"{result.kind}let price")
    twin = ax.twinx()
    twin.plot(result.reset_times, 100 * result.forwards, "k.-", lw=1, label="forward rate")
    twin.axhline(100 * result.strike, color="k", ls=":", lw=1)
    twin.set_ylabel("forward rate (%)")
    ax.legend(loc="upper left")
    return ax


def plot_jamshidian(result: JamshidianResult, ax=None):
    """Jamshidian's decomposition: the zero-coupon option behind each payment.

    Parameters
    ----------
    result : JamshidianResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(np.arange(result.components.size), result.components, color="C1")
    ax.set_xticks(np.arange(result.components.size), [f"{t:g}" for t in result.payment_times], rotation=45)
    ax.set_xlabel("payment date (years)")
    ax.set_ylabel("zero-bond option value")
    ax.set_title(f"total {result.price:.5f}, r* = {100 * result.critical_rate:.3f}%")
    return ax


def plot_forward_curves(result: HJMResult, path: int = 0, every: int = 5, ax=None):
    """Snapshots of one simulated forward curve :math:`T \\mapsto f(t, T)` as time passes.

    Parameters
    ----------
    result : HJMResult
    path : int
        Which simulated path to draw.
    every : int
        Draw every `every`-th simulation date.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    idx = range(0, result.times.size - 1, every)
    for i in idx:
        ax.plot(result.times[i:], 100 * result.forwards[path, i, i:], color=cmap(i / result.times.size), lw=1)
    ax.plot(result.times, 100 * result.initial_curve, "k--", lw=1, label="f(0, T)")
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("forward rate (%)")
    ax.legend()
    return ax
