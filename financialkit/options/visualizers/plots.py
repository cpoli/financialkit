"""Plotting helpers for financialkit.options: value and Greek profiles, tree
convergence, exercise boundaries, smiles, densities and hedging P&L.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from financialkit.options.core.base import DensityResult, Greeks, HedgeResult, SmileResult, TreeResult
from financialkit.options.utils.common import payoff

__all__ = [
    "plot_option_value",
    "plot_greeks",
    "plot_tree_convergence",
    "plot_exercise_boundary",
    "plot_smile",
    "plot_density",
    "plot_hedge_pnl",
]


def plot_option_value(S, values, K: float, option_type: str = "call", ax=None, label: str = "value"):
    """Option value against spot, with the payoff at expiry for reference.

    Parameters
    ----------
    S, values : array_like, shape (n,)
        Spot grid and option values on it.
    K : float
        Strike.
    option_type : {"call", "put"}
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    S = np.asarray(S, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(S, payoff(S, K, option_type), "k--", lw=1, label="payoff")
    ax.plot(S, values, label=label)
    ax.set_xlabel("spot S")
    ax.set_ylabel(f"{option_type} value")
    ax.legend()
    return ax


def plot_greeks(S, greeks: Greeks, names=("delta", "gamma", "vega", "theta", "rho", "vanna"), axes=None):
    """One panel per Greek against spot.

    Parameters
    ----------
    S : array_like, shape (n,)
        Spot grid the Greeks were evaluated on.
    greeks : Greeks
        From :func:`~financialkit.options.systems.greeks.black_scholes_greeks`.
    names : sequence of str
        Fields of :class:`~financialkit.options.core.base.Greeks` to plot.
    axes : array of matplotlib.axes.Axes, optional
        At least ``len(names)`` axes; a grid is created if omitted.

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        ncols = 3
        nrows = int(np.ceil(len(names) / ncols))
        _, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows), squeeze=False)
    axes = np.asarray(axes).ravel()
    for ax, name in zip(axes, names, strict=False):
        ax.plot(S, np.broadcast_to(getattr(greeks, name), np.shape(S)))
        ax.set_title(name)
        ax.set_xlabel("spot S")
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    return axes


def plot_tree_convergence(n_steps, prices, reference: float, ax=None, label: str = "tree"):
    """Absolute pricing error of a lattice against a reference price, log-log.

    Parameters
    ----------
    n_steps, prices : array_like, shape (m,)
        Step counts and the lattice prices obtained with them.
    reference : float
        Exact (or converged) price.
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    err = np.abs(np.asarray(prices, dtype=np.float64) - reference)
    ax.loglog(n_steps, np.where(err > 0, err, np.nan), marker="o", ms=3, label=label)
    ax.set_xlabel("time steps n")
    ax.set_ylabel("|tree - reference|")
    ax.legend()
    return ax


def plot_exercise_boundary(result: TreeResult, ax=None, label: str = "exercise boundary"):
    """Early-exercise boundary of an American option on a lattice.

    Parameters
    ----------
    result : TreeResult
        An American valuation (``american=True``).
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if result.exercise_boundary.size == 0:
        raise ValueError("result has no exercise boundary; price with american=True")
    if ax is None:
        _, ax = plt.subplots()
    ax.step(result.times, result.exercise_boundary, where="post", label=label)
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("critical spot S*(t)")
    ax.legend()
    return ax


def plot_smile(result: SmileResult, ax=None, label=None, moneyness: bool = False):
    """Implied volatility against strike (or log-moneyness).

    Parameters
    ----------
    result : SmileResult
    ax : matplotlib.axes.Axes, optional
    label : str, optional
    moneyness : bool
        Plot against :math:`\\ln(K/F)` instead of the strike.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    x = result.log_moneyness if moneyness else result.strikes
    ax.plot(x, result.vols, marker="o", ms=3, label=label or f"T = {result.T:g}")
    ax.set_xlabel("log-moneyness ln(K/F)" if moneyness else "strike K")
    ax.set_ylabel("implied volatility")
    ax.legend()
    return ax


def plot_density(result: DensityResult, ax=None, label: str = "Breeden-Litzenberger"):
    """A risk-neutral density recovered from option prices.

    Parameters
    ----------
    result : DensityResult
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.strikes, result.density, label=label)
    ax.set_xlabel("terminal price S_T")
    ax.set_ylabel("risk-neutral density")
    ax.legend()
    return ax


def plot_hedge_pnl(result: HedgeResult, ax=None, bins: int = 50, label=None):
    """Histogram of the terminal P&L of a delta-hedged short option.

    Parameters
    ----------
    result : HedgeResult
    ax : matplotlib.axes.Axes, optional
    bins : int
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    n = result.times.size - 1
    ax.hist(result.pnl, bins=bins, density=True, alpha=0.6, label=label or f"{n} rebalances")
    ax.set_xlabel("hedged P&L at T")
    ax.set_ylabel("density")
    ax.legend()
    return ax
