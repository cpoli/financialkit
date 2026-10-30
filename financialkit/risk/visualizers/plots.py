"""Plotting helpers for financialkit.risk: scenario distributions with VaR
and ES, VaR backtests, extreme-value fits, copula samples, risk
allocations, drawdowns and the FRTB liquidity-horizon cascade.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.risk.core.base import AllocationResult, DrawdownResult, FRTBResult, GEVResult, GPDResult, VaRResult
from financialkit.risk.utils.tail import var_exceptions

__all__ = ["plot_var", "plot_backtest", "plot_gev_fit", "plot_pot_tail", "plot_copula", "plot_allocation", "plot_drawdowns", "plot_frtb_cascade"]


def plot_var(result: VaRResult, ax=None, bins: int = 60):
    """Histogram of the scenario profit and loss, weighted, with VaR and ES marked.

    Parameters
    ----------
    result : VaRResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    bins : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(result.scenarios, bins=bins, weights=result.weights, color="0.7")
    pct = f"{100 * result.alpha:g}%"
    ax.axvline(-result.var, color="C0", lw=2, label=f"{pct} VaR = {result.var:.4g}")
    ax.axvline(-result.es, color="C3", lw=2, ls="--", label=f"{pct} ES = {result.es:.4g}")
    ax.set_xlabel("profit and loss")
    ax.set_ylabel("probability")
    ax.set_title(result.method.replace("_", " "))
    ax.legend()
    return ax


def plot_backtest(pnl, var, ax=None):
    """Daily profit and loss against minus the VaR forecast, exceptions marked.

    Parameters
    ----------
    pnl : array_like, shape (n,)
    var : array_like, shape (n,) or scalar
        VaR forecasts aligned with ``pnl``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    pnl = np.asarray(pnl, dtype=np.float64)
    var = np.broadcast_to(np.asarray(var, dtype=np.float64), pnl.shape)
    hits = var_exceptions(pnl, var)
    t = np.arange(pnl.size)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, pnl, color="0.6", lw=0.7, label="P&L")
    ax.plot(t, -var, color="C0", lw=1.5, label="-VaR")
    ax.plot(t[hits], pnl[hits], "o", color="C3", ms=4, label=f"{int(hits.sum())} exceptions")
    ax.set_xlabel("day")
    ax.legend(loc="lower left")
    return ax


def plot_gev_fit(fit: GEVResult, ax=None, bins: int = 30):
    """Histogram of block maxima with the fitted GEV density.

    Parameters
    ----------
    fit : GEVResult
    ax : matplotlib.axes.Axes, optional
    bins : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(fit.maxima, bins=bins, density=True, color="0.75", label=f"{fit.maxima.size} block maxima")
    x = np.linspace(fit.maxima.min(), fit.maxima.max(), 300)
    ax.plot(x, stats.genextreme.pdf(x, -fit.xi, loc=fit.mu, scale=fit.sigma), "C3", lw=2, label=f"GEV, xi = {fit.xi:.2f}")
    ax.set_xlabel(f"largest loss per {fit.block_size} periods")
    ax.set_ylabel("density")
    ax.legend()
    return ax


def plot_pot_tail(fit: GPDResult, ax=None):
    """Empirical tail of the losses above the threshold against the fitted GPD, on log axes.

    Parameters
    ----------
    fit : GPDResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    rate = fit.n_exceed / fit.n_obs
    y = np.sort(fit.excesses)
    emp = rate * (1.0 - np.arange(y.size) / y.size)
    ax.loglog(fit.threshold + y, emp, ".", color="0.4", label="empirical")
    x = np.geomspace(max(y[0], 1e-12 * fit.beta), y[-1] * 3.0, 200)
    ax.loglog(fit.threshold + x, rate * stats.genpareto.sf(x, fit.xi, scale=fit.beta), "C3", lw=2, label=f"GPD, xi = {fit.xi:.2f}")
    ax.set_xlabel("loss")
    ax.set_ylabel("P(L > x)")
    ax.legend()
    return ax


def plot_copula(u, ax=None, title: str = ""):
    """Scatter of a bivariate copula sample on the unit square.

    Parameters
    ----------
    u : array_like, shape (n, 2)
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    u = np.asarray(u, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(u[:, 0], u[:, 1], s=2, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    if title:
        ax.set_title(title)
    return ax


def plot_allocation(result: AllocationResult, labels=None, ax=None):
    """Bar chart of risk contributions, with the portfolio total.

    Parameters
    ----------
    result : AllocationResult
    labels : sequence of str, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    c = result.contributions
    labels = list(labels) if labels is not None else [f"asset {i}" for i in range(c.size)]
    if ax is None:
        _, ax = plt.subplots()
    ax.bar(labels, c, color=["C3" if v < 0 else "C0" for v in c])
    ax.axhline(result.total, color="k", ls="--", lw=1, label=f"total {result.measure} = {result.total:.4g}")
    ax.set_ylabel(f"contribution to {result.measure}")
    ax.legend()
    return ax


def plot_drawdowns(result: DrawdownResult, axes=None):
    """Wealth with its running peak (top) and the drawdown (bottom).

    Parameters
    ----------
    result : DrawdownResult
    axes : pair of matplotlib.axes.Axes, optional

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5), gridspec_kw={"height_ratios": [2, 1]})
    t = np.arange(result.wealth.size)
    axes[0].plot(t, result.wealth, label="wealth")
    axes[0].plot(t, result.peak, color="0.5", lw=1, label="running peak")
    axes[0].plot([result.peak_index, result.trough_index], result.wealth[[result.peak_index, result.trough_index]], "o", color="C3")
    axes[0].legend()
    axes[1].fill_between(t, -result.drawdown, color="C3", alpha=0.5)
    axes[1].set_ylabel("drawdown")
    axes[1].set_xlabel("period")
    return np.asarray(axes)


def plot_frtb_cascade(result: FRTBResult, ax=None):
    """The FRTB cascade: each horizon's ten-day ES and its scaled term, with the total.

    Parameters
    ----------
    result : FRTBResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.arange(result.horizons.size)
    ax.bar(x - 0.2, result.es_by_horizon, 0.4, color="0.7", label="ten-day ES, factors with LH $\\geq$ horizon")
    ax.bar(x + 0.2, result.scaled, 0.4, color="C0", label="scaled by $\\sqrt{(LH_j - LH_{j-1})/10}$")
    ax.axhline(result.es, color="k", ls="--", lw=1, label=f"liquidity-adjusted ES = {result.es:.4g}")
    ax.set_xticks(x, [f"{h:g}d" for h in result.horizons])
    ax.set_xlabel("liquidity horizon")
    ax.set_ylabel(f"{100 * result.alpha:g}% expected shortfall")
    ax.legend()
    return ax
