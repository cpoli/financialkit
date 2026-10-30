"""Plotting helpers for financialkit.credit: survival curves, loss
distributions, transition matrices, CreditMetrics value distributions,
tranche expected losses and counterparty exposure profiles.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit.core.base import CreditMetricsResult, ExposureProfile, LossDistribution, SurvivalCurve

__all__ = ["plot_survival", "plot_loss_distribution", "plot_transition_matrix", "plot_value_distribution", "plot_tranche_losses", "plot_exposure_profile"]


def plot_survival(curve: SurvivalCurve, T: float = 10.0, ax=None, label: str | None = None, hazard: bool = True):
    """Survival probability against time, with the hazard rate on a twin axis.

    Parameters
    ----------
    curve : SurvivalCurve
    T : float
        Horizon of the plot.
    ax : matplotlib.axes.Axes, optional
    label : str, optional
    hazard : bool
        Also draw the hazard rate (right axis).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    t = np.linspace(0.0, T, 400)
    (line,) = ax.plot(t, curve.survival(t), lw=2, label=label or "survival")
    ax.set_xlabel("years")
    ax.set_ylabel("survival probability")
    if hazard:
        ax2 = ax.twinx()
        ax2.step(t, curve.hazard(t), where="post", color=line.get_color(), ls="--", lw=1)
        ax2.set_ylabel("hazard rate (dashed)")
    return ax


def plot_loss_distribution(dist: LossDistribution, alpha: float = 0.999, ax=None, log: bool = False):
    """Loss distribution with the expected loss and a quantile marked.

    Parameters
    ----------
    dist : LossDistribution
    alpha : float
        Quantile to mark.
    ax : matplotlib.axes.Axes, optional
    log : bool
        Log scale on the probability axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    width = float(np.min(np.diff(dist.losses))) if dist.losses.size > 1 else 1.0
    ax.bar(dist.losses, dist.pmf, width=width, color="0.6")
    ax.axvline(dist.mean, color="C0", lw=2, label=f"expected loss = {dist.mean:.4g}")
    ax.axvline(dist.quantile(alpha), color="C3", lw=2, ls="--", label=f"{100 * alpha:g}% quantile = {dist.quantile(alpha):.4g}")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("loss")
    ax.set_ylabel("probability")
    ax.set_title(dist.method)
    ax.legend()
    return ax


def plot_transition_matrix(P, labels=None, ax=None, fmt: str = "{:.2f}", scale: float = 100.0):
    """Heat map of a transition matrix (or generator), annotated in percent.

    Parameters
    ----------
    P : array_like, shape (k, k)
    labels : sequence of str, optional
    ax : matplotlib.axes.Axes, optional
    fmt : str
        Format of each annotation.
    scale : float
        Factor applied to the annotations (100 for percent).

    Returns
    -------
    matplotlib.axes.Axes
    """
    P = np.asarray(P, dtype=np.float64)
    k = P.shape[0]
    labels = list(labels) if labels is not None else [str(i) for i in range(k)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.log10(np.abs(P) + 1e-5), cmap="Blues")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, fmt.format(scale * P[i, j]), ha="center", va="center", fontsize=7, color="C3" if P[i, j] < 0 else "k")
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("to")
    ax.set_ylabel("from")
    return ax


def plot_value_distribution(result: CreditMetricsResult, alpha: float = 0.99, ax=None, bins: int = 60, labels=None):
    """CreditMetrics value distribution with its mean and credit VaR.

    Rating-state results (few distinct values) are drawn as bars, Monte
    Carlo results as a histogram.

    Parameters
    ----------
    result : CreditMetricsResult
    alpha : float
    ax : matplotlib.axes.Axes, optional
    bins : int
    labels : sequence of str, optional
        Bar labels for a rating-state result.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if result.values.size <= 20:
        x = np.arange(result.values.size)
        ax.bar(x, result.probabilities, color="0.6")
        ax.set_xticks(x, labels if labels is not None else [f"{v:.4g}" for v in result.values])
        ax.set_yscale("log")
        ax.set_ylabel("probability")
        ax.set_xlabel("end-of-year state")
        ax.set_title(f"mean {result.mean:.2f}, std {result.std:.2f}")
        return ax
    ax.hist(result.values, bins=bins, weights=result.probabilities, color="0.6")
    ax.axvline(result.mean, color="C0", lw=2, label=f"mean = {result.mean:.4g}")
    q = result.percentile(1.0 - alpha)
    ax.axvline(q, color="C3", lw=2, ls="--", label=f"{100 * (1 - alpha):g}% percentile, VaR = {result.mean - q:.4g}")
    ax.set_xlabel("portfolio value")
    ax.set_ylabel("probability")
    ax.legend()
    return ax


def plot_tranche_losses(results, ax=None):
    """Expected loss of each tranche against time.

    Parameters
    ----------
    results : sequence of TrancheResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for res in results:
        ax.plot(res.times, res.expected_loss, marker=".", label=f"{100 * res.attachment:g}-{100 * res.detachment:g}%: {1e4 * res.par_spread:.0f} bp")
    ax.set_xlabel("years")
    ax.set_ylabel("expected tranche loss")
    ax.legend()
    return ax


def plot_exposure_profile(profile: ExposureProfile, ax=None, pfe_level: float | None = 0.95, n_paths: int = 0, label: str | None = None):
    """Discounted expected positive and negative exposure against time.

    Parameters
    ----------
    profile : ExposureProfile
    ax : matplotlib.axes.Axes, optional
    pfe_level : float or None
        Also draw the potential future exposure at this level (undiscounted).
    n_paths : int
        Number of mark-to-market paths to draw in the background.
    label : str, optional
        Suffix for the legend entries.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    t = profile.times
    suffix = f" ({label})" if label else ""
    for path in profile.values[:n_paths]:
        ax.plot(t, path, color="0.85", lw=0.6, zorder=0)
    ax.plot(t, profile.epe, lw=2, label="EPE, discounted" + suffix)
    ax.plot(t, -profile.ene, lw=2, label="-ENE, discounted" + suffix)
    if pfe_level is not None:
        ax.plot(t, profile.pfe(pfe_level), ls="--", label=f"PFE {100 * pfe_level:g}%" + suffix)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("years")
    ax.set_ylabel("exposure")
    ax.legend()
    return ax
