"""Plotting helpers for financialkit.time_series: correlograms, tails, R/S, variance ratios and volatility."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series.core.base import HurstResult, TailIndexResult, VarianceRatioResult, VolatilityModel
from financialkit.time_series.systems.efficiency import autocorrelation, bartlett_band
from financialkit.time_series.systems.linear_models import partial_autocorrelation
from financialkit.time_series.systems.stylized_facts import tail_survival

__all__ = ["plot_acf", "plot_tail", "plot_hill", "plot_rescaled_range", "plot_variance_ratio", "plot_conditional_volatility", "plot_news_impact"]


def plot_acf(x, nlags: int = 20, partial: bool = False, ax=None, label: str = ""):
    """Correlogram of `x` (lags 1..`nlags`) with the 95% white-noise band.

    Parameters
    ----------
    x : array_like
    nlags : int
    partial : bool
        Plot partial autocorrelations instead.
    ax : matplotlib.axes.Axes, optional
    label : str
        Legend label.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.asarray(x, dtype=np.float64)
    rho = partial_autocorrelation(x, nlags) if partial else autocorrelation(x, nlags)
    k = np.arange(1, nlags + 1)
    ax.vlines(k, 0, rho[1:], lw=2, label=label or None)
    ax.plot(k, rho[1:], "o", ms=4, color=ax.lines[-1].get_color() if ax.lines else None)
    band = bartlett_band(x.size)
    ax.axhspan(-band, band, color="0.85", zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("partial autocorrelation" if partial else "autocorrelation")
    if label:
        ax.legend()
    return ax


def plot_tail(x, ax=None, label: str = "", side: str = "both"):
    """Empirical survival function :math:`P(|X| > u)` on log-log axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    u, p = tail_survival(x, side)
    keep = u > 0
    ax.loglog(u[keep], p[keep], ".", ms=3, label=label or None)
    ax.set_xlabel("|return|" if side == "both" else "return magnitude")
    ax.set_ylabel("P(|X| > u)" if side == "both" else "P(X > u)")
    if label:
        ax.legend()
    return ax


def plot_hill(result: TailIndexResult, true_alpha: Optional[float] = None, ax=None, label: str = ""):
    """Hill plot: :math:`\\hat\\alpha_k` against `k` with a two-standard-error band.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    (line,) = ax.plot(result.k, result.alpha, label=label or None)
    ax.fill_between(result.k, result.alpha - 2 * result.std_error, result.alpha + 2 * result.std_error, color=line.get_color(), alpha=0.2)
    if true_alpha is not None:
        ax.axhline(true_alpha, color="k", ls="--", lw=1)
    ax.set_xlabel("k (upper order statistics)")
    ax.set_ylabel(r"Hill estimate $\hat\alpha_k$")
    if label:
        ax.legend()
    return ax


def plot_rescaled_range(result: HurstResult, ax=None, label: str = ""):
    """:math:`(R/S)_n` against :math:`n` on log-log axes, with the fitted slope and the Anis-Lloyd benchmark.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    n = result.window
    (pts,) = ax.loglog(n, result.rs, "o", label=f"{label} H = {result.hurst:.2f}".strip())
    c = np.exp(np.mean(np.log(result.rs) - result.hurst * np.log(n)))
    ax.loglog(n, c * n**result.hurst, color=pts.get_color(), lw=1)
    ax.loglog(n, result.expected_rs, "k--", lw=1, label="E[R/S], independent")
    ax.set_xlabel("window n")
    ax.set_ylabel("R/S")
    ax.legend()
    return ax


def plot_variance_ratio(result: VarianceRatioResult, ax=None, label: str = ""):
    """Variance ratios against horizon `q`, with 95% bands under the homoskedastic random walk.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    q = result.q.astype(np.float64)
    ax.plot(q, result.ratio, "o-", label=label or None)
    se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * result.nobs))
    ax.fill_between(q, 1 - 1.96 * se, 1 + 1.96 * se, color="0.85", zorder=0)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("horizon q")
    ax.set_ylabel("VR(q)")
    if label:
        ax.legend()
    return ax


def plot_conditional_volatility(returns, variance, ax=None, periods_per_year: Optional[int] = None):
    """Returns with a :math:`\\pm 2\\sigma_t` envelope of the conditional volatility.

    Parameters
    ----------
    returns, variance : array_like, shape (n,)
    ax : matplotlib.axes.Axes, optional
    periods_per_year : int, optional
        If given, also draw the annualized volatility on a twin axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    r = np.asarray(returns, dtype=np.float64)
    sd = np.sqrt(np.asarray(variance, dtype=np.float64))
    t = np.arange(r.size)
    ax.plot(t, r, color="0.6", lw=0.6, label="return")
    ax.plot(t, 2 * sd, color="C3", lw=1, label=r"$\pm 2\sigma_t$")
    ax.plot(t, -2 * sd, color="C3", lw=1)
    ax.set_xlabel("period")
    ax.set_ylabel("return")
    ax.legend(loc="upper left")
    if periods_per_year:
        ax2 = ax.twinx()
        ax2.plot(t, 100 * sd * np.sqrt(periods_per_year), alpha=0)
        ax2.set_ylabel("annualized volatility (%)")
    return ax


def plot_news_impact(models: dict, var_prev: float, eps_range: Optional[float] = None, ax=None):
    """News impact curves: next period's variance against today's shock, from the same `var_prev`.

    Parameters
    ----------
    models : dict of str to VolatilityModel
    var_prev : float
        Today's conditional variance.
    eps_range : float, optional
        Shocks span ``[-eps_range, eps_range]`` (default five standard deviations).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    eps_range = 5 * np.sqrt(var_prev) if eps_range is None else eps_range
    eps = np.linspace(-eps_range, eps_range, 401)
    for name, model in models.items():
        if not isinstance(model, VolatilityModel):
            raise TypeError("models must be VolatilityModel instances")
        ax.plot(eps / np.sqrt(var_prev), model.news_impact(eps, var_prev) / var_prev, label=name)
    ax.set_xlabel(r"shock $\varepsilon_{t-1} / \sigma_{t-1}$")
    ax.set_ylabel(r"$\sigma_t^2 / \sigma_{t-1}^2$")
    ax.legend()
    return ax
