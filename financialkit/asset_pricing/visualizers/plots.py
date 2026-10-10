"""Plotting helpers for financialkit.asset_pricing: the security market line,
pricing errors, Fama-MacBeth slopes, Shiller's variance bound and the
Hansen-Jagannathan region.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing.core.base import CAPMEquilibrium, ConsumptionMoments, ExcessVolatilityResult, FactorRegressionResult, FamaMacBethResult, HJBound

__all__ = ["plot_sml", "plot_alphas", "plot_fama_macbeth", "plot_excess_volatility", "plot_hj_bound"]


def plot_sml(eq: CAPMEquilibrium, ax=None, labels: Sequence[str] | None = None):
    """The assets in (beta, expected return) space with the security market line.

    Parameters
    ----------
    eq : CAPMEquilibrium
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    labels : sequence of str, optional
        Asset names.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    b = np.linspace(0.0, max(1.5, 1.1 * float(eq.betas.max())), 2)
    ax.plot(b, eq.risk_free + b * eq.market_premium, "C0", lw=1.5, label="security market line")
    ax.plot(eq.betas, eq.expected_returns, "o", color="C1", label="assets")
    ax.plot(1.0, eq.market_return, "s", color="C3", ms=8, label="market")
    for i, name in enumerate(labels or []):
        ax.annotate(name, (eq.betas[i], eq.expected_returns[i]), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("beta")
    ax.set_ylabel("expected return")
    ax.legend()
    return ax


def plot_alphas(result: FactorRegressionResult, ax=None, label: str = "alpha", offset: float = 0.0, width: float = 0.8):
    """Intercepts of a set of time-series regressions with two-standard-error bars.

    Call twice with ``offset`` and ``width`` to compare two models on the same assets.

    Parameters
    ----------
    result : FactorRegressionResult
    ax : matplotlib.axes.Axes, optional
    label : str
    offset, width : float
        Horizontal shift and width of the bars.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.arange(result.alphas.size) + offset
    ax.bar(x, result.alphas, width=width, yerr=2 * result.alpha_se, capsize=2, label=label, alpha=0.8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("test portfolio")
    ax.set_ylabel("alpha per period")
    ax.legend()
    return ax


def plot_fama_macbeth(result: FamaMacBethResult, index: int = 1, ax=None, true_value: float | None = None):
    """Time series of one Fama-MacBeth slope with its mean and two-standard-error band.

    Parameters
    ----------
    result : FamaMacBethResult
    index : int
        Column of ``gammas``: 0 for the intercept, 1 for the first factor.
    ax : matplotlib.axes.Axes, optional
    true_value : float, optional
        Marked with a dashed line if given.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    g = result.gammas[:, index]
    m, se = result.premia[index], result.std_errors[index]
    ax.plot(g, color="0.6", lw=0.7, label="period-by-period slope")
    ax.axhline(m, color="C0", lw=2, label=f"mean = {m:.4f} (t = {m / se:.1f})")
    ax.axhspan(m - 2 * se, m + 2 * se, color="C0", alpha=0.2)
    if true_value is not None:
        ax.axhline(true_value, color="C3", ls="--", label=f"true = {true_value:.4f}")
    ax.set_xlabel("period")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_excess_volatility(result: ExcessVolatilityResult, ax=None, label: str = "price"):
    """A price against Shiller's ex-post rational price.

    Parameters
    ----------
    result : ExcessVolatilityResult
    ax : matplotlib.axes.Axes, optional
    label : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(result.rational_price, "k", lw=2, label=f"ex-post rational p*, sd = {result.std_rational:.2f}")
    ax.plot(result.price, color="C3", lw=1, label=f"{label}, sd = {result.std_price:.2f}")
    ax.set_xlabel("period")
    ax.legend(fontsize=8)
    return ax


def plot_hj_bound(bound: HJBound, models: dict[str, ConsumptionMoments] | None = None, ax=None):
    """The Hansen-Jagannathan frontier, with the moments of model discount factors.

    Parameters
    ----------
    bound : HJBound
    models : dict of str to ConsumptionMoments, optional
        Each is drawn at ``(sdf_mean, sdf_vol)``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(bound.sdf_means, bound.sdf_stds, "k", lw=2, label="HJ bound")
    ax.fill_between(bound.sdf_means, bound.sdf_stds, bound.sdf_stds.max() * 1.2, color="0.85", label="admissible")
    for i, (name, m) in enumerate((models or {}).items()):
        ax.plot(m.sdf_mean, m.sdf_vol, "o", color=f"C{i % 10}", label=name)
    ax.set_xlabel("E[m]")
    ax.set_ylabel("sigma(m)")
    ax.legend(fontsize=8)
    return ax
