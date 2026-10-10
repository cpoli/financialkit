"""Plotting helpers for financialkit.monte_carlo: error convergence,
confidence intervals, estimator comparisons, point sets, stratified
allocations and multilevel diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from financialkit.monte_carlo.core.base import MLMCResult, MonteCarloResult, StratifiedResult

__all__ = [
    "plot_error_convergence",
    "plot_confidence_intervals",
    "plot_estimator_comparison",
    "plot_point_set",
    "plot_allocation",
    "plot_mlmc",
]


def plot_error_convergence(n, errors, ax=None, label: str = "", rates: Sequence[float] = (0.5,)):
    """Errors (or standard errors) against sample size on log-log axes, with reference slopes.

    Parameters
    ----------
    n : array_like
        Sample sizes.
    errors : array_like
        Absolute errors or standard errors, one per sample size.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str
        Legend label of the data.
    rates : sequence of float
        Draw :math:`n^{-a}` guides through the first point for each ``a``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    n, errors = np.asarray(n, dtype=np.float64), np.asarray(errors, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    ax.loglog(n, errors, "o-", label=label or None)
    for a in rates:
        ax.loglog(n, errors[0] * (n / n[0]) ** -a, ":", color="0.4", label=f"$n^{{-{a:g}}}$")
    ax.set_xlabel("samples n")
    ax.set_ylabel("error")
    ax.legend()
    return ax


def plot_confidence_intervals(results: Sequence[MonteCarloResult], exact: float | None = None, ax=None):
    """Confidence intervals of repeated estimates, missed ones in red, with the exact value.

    Parameters
    ----------
    results : sequence of MonteCarloResult
    exact : float, optional
        True value; intervals that miss it are highlighted.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for i, res in enumerate(results):
        miss = exact is not None and not res.covers(exact)
        ax.errorbar(i, res.estimate, yerr=res.half_width, fmt="o", ms=3, color="C3" if miss else "C0", lw=1)
    if exact is not None:
        ax.axhline(exact, color="k", lw=1, label="exact")
        hits = sum(r.covers(exact) for r in results)
        ax.set_title(f"{hits}/{len(results)} intervals cover the exact value")
        ax.legend()
    ax.set_xlabel("replication")
    ax.set_ylabel("estimate")
    return ax


def plot_estimator_comparison(results: Sequence[MonteCarloResult], baseline: MonteCarloResult | None = None, exact: float | None = None, ax=None):
    """Estimates with confidence intervals side by side, annotated with variance reduction factors.

    Parameters
    ----------
    results : sequence of MonteCarloResult
        Labelled by their ``method``.
    baseline : MonteCarloResult, optional
        Reference for :meth:`~financialkit.monte_carlo.core.base.MonteCarloResult.variance_ratio`;
        defaults to the first result.
    exact : float, optional
        True value, drawn as a horizontal line.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    base = baseline if baseline is not None else results[0]
    x = np.arange(len(results))
    ax.errorbar(x, [r.estimate for r in results], yerr=[r.half_width for r in results], fmt="o", capsize=4)
    for i, r in enumerate(results):
        ax.annotate(f"x{r.variance_ratio(base):.3g}", (i, r.ci[1]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    if exact is not None:
        ax.axhline(exact, color="k", lw=1, ls="--", label="exact")
        ax.legend()
    ax.set_xticks(x, [r.method for r in results], rotation=20)
    ax.set_ylabel("estimate (95% CI); variance reduction")
    return ax


def plot_point_set(points, ax=None, title: str = "", grid: int = 0, dims: tuple[int, int] = (0, 1)):
    """A two-dimensional projection of a point set in the unit square.

    Parameters
    ----------
    points : array_like, shape (n, d)
    ax : matplotlib.axes.Axes, optional
    title : str
    grid : int
        Draw a ``grid x grid`` lattice of cells (e.g. to show stratification).
    dims : tuple of int
        Coordinates to project onto.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x = np.asarray(points, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(x[:, dims[0]], x[:, dims[1]], s=6)
    for k in range(1, grid):
        ax.axhline(k / grid, color="0.8", lw=0.5)
        ax.axvline(k / grid, color="0.8", lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"$u_{{{dims[0] + 1}}}$")
    ax.set_ylabel(f"$u_{{{dims[1] + 1}}}$")
    ax.set_title(title)
    return ax


def plot_allocation(result: StratifiedResult, ax=None):
    """Samples per stratum next to the within-stratum standard deviations.

    Parameters
    ----------
    result : StratifiedResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    h = np.arange(result.allocation.size)
    ax.bar(h, result.allocation, color="C0", alpha=0.7, label="samples $n_h$")
    ax.set_xlabel("stratum h")
    ax.set_ylabel("samples")
    ax2 = ax.twinx()
    ax2.plot(h, result.stratum_stds, "o-", color="C3", label=r"$\sigma_h$")
    ax2.set_ylabel(r"within-stratum std $\sigma_h$")
    ax.set_title(result.method)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper center")
    return ax


def plot_mlmc(result: MLMCResult, axes=None):
    """Giles's diagnostics: level variances and means (log2) and samples per level.

    Parameters
    ----------
    result : MLMCResult
    axes : sequence of two matplotlib.axes.Axes, optional

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    l = result.levels
    axes[0].plot(l, np.log2(result.level_variances), "o-", label=r"$\log_2 V_l$")
    axes[0].plot(l[1:], np.log2(np.abs(result.level_means[1:])), "s--", label=r"$\log_2 |E[P_l - P_{l-1}]|$")
    axes[0].set_xlabel("level l")
    axes[0].legend()
    axes[1].semilogy(l, result.level_samples, "o-")
    axes[1].set_xlabel("level l")
    axes[1].set_ylabel("samples $N_l$")
    axes[1].set_title(f"eps = {result.target_rmse:g}")
    return np.asarray(axes)
