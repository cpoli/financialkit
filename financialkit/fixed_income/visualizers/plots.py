"""Plotting helpers for financialkit.fixed_income: curves, price-yield, cash flows and risk profiles."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income.core.base import CashFlows, CurveLike, CurvePCAResult, ImmunizationResult, KeyRateDurationResult, YieldCurve, as_curve
from financialkit.fixed_income.systems.bonds import price_from_yield
from financialkit.fixed_income.systems.duration import bond_risk

__all__ = ["plot_yield_curve", "plot_price_yield", "plot_cash_flows", "plot_key_rate_durations", "plot_pca_loadings", "plot_surplus"]


def plot_yield_curve(curve: YieldCurve, t_max: float = 30.0, ax=None, label: str = "", show_forward: bool = True):
    """Zero curve, and optionally its instantaneous forward curve, in percent.

    Parameters
    ----------
    curve : YieldCurve
    t_max : float
        Longest maturity plotted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str
        Legend label prefix.
    show_forward : bool
        Also draw the instantaneous forward curve (dashed).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    t = np.linspace(0.01, t_max, 300)
    (line,) = ax.plot(t, 100 * curve.zero_rate(t), label=f"{label} zero".strip())
    if show_forward:
        ax.plot(t, 100 * curve.instantaneous_forward(t), ls="--", color=line.get_color(), label=f"{label} forward".strip())
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("rate (%)")
    ax.legend()
    return ax


def plot_price_yield(cash_flows: CashFlows, y0: float, compounding: Optional[int] = None, dy: float = 0.04, ax=None):
    """Price-yield curve with its duration tangent and duration-plus-convexity parabola at `y0`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    risk = bond_risk(cash_flows, y0, compounding)
    y = np.linspace(y0 - dy, y0 + dy, 200)
    ax.plot(100 * y, price_from_yield(cash_flows, y, compounding), lw=2, label="price")
    ax.plot(100 * y, risk.price + risk.price_change(y - y0, order=1), ls="--", label="duration")
    ax.plot(100 * y, risk.price + risk.price_change(y - y0, order=2), ls=":", label="duration + convexity")
    ax.plot([100 * y0], [risk.price], "ko")
    ax.set_xlabel("yield (%)")
    ax.set_ylabel("price")
    ax.legend()
    return ax


def plot_cash_flows(cash_flows: CashFlows, curve: Optional[CurveLike] = None, ax=None):
    """Bar chart of a cash-flow stream, overlaid with present values if a curve is given.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    width = 0.6 * (np.min(np.diff(cash_flows.times)) if cash_flows.times.size > 1 else 0.5)
    ax.bar(cash_flows.times, cash_flows.amounts, width=width, color="0.75", label="cash flow")
    if curve is not None:
        pv = cash_flows.amounts * as_curve(curve).discount(cash_flows.times)
        ax.bar(cash_flows.times, pv, width=width, label="present value")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("amount")
    ax.legend()
    return ax


def plot_key_rate_durations(result: KeyRateDurationResult, ax=None, label: Optional[str] = None):
    """Bar chart of key rate durations against key maturity.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    pos = np.arange(result.key_tenors.size)
    ax.bar(pos, result.durations, label=label)
    ax.set_xticks(pos, [f"{k:g}y" for k in result.key_tenors])
    ax.set_xlabel("key rate")
    ax.set_ylabel("key rate duration")
    ax.set_title(f"sum {result.durations.sum():.3f} vs parallel {result.parallel_duration:.3f}")
    return ax


def plot_pca_loadings(result: CurvePCAResult, n_components: int = 3, ax=None):
    """Loadings of the leading principal components against maturity.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    names = ["level", "slope", "curvature"]
    ratio = result.explained_variance_ratio
    for k in range(min(n_components, result.loadings.shape[1])):
        name = names[k] if k < len(names) else f"PC{k + 1}"
        ax.plot(result.tenors, result.loadings[:, k], marker="o", label=f"{name} ({100 * ratio[k]:.1f}%)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("loading")
    ax.legend()
    return ax


def plot_surplus(result: ImmunizationResult, shifts=None, ax=None):
    """Surplus (assets minus liabilities) against a parallel shift of the curve, in basis points.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    shifts = np.linspace(-0.03, 0.03, 61) if shifts is None else np.asarray(shifts, dtype=np.float64)
    ax.plot(1e4 * shifts, result.surplus(shifts))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("parallel shift (bp)")
    ax.set_ylabel("surplus")
    return ax
