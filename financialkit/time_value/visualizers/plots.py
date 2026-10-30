"""Plotting helpers for financialkit.time_value: cash-flow timelines,
compounding growth, NPV profiles and amortization schedules.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_value.systems.compounding import future_value

__all__ = ["plot_cash_flows", "plot_compounding_growth", "plot_npv_profile", "plot_amortization"]


def plot_cash_flows(cash_flows, times=None, ax=None, title: str = "Cash flows"):
    """Bar timeline of a cash-flow stream, inflows up and outflows down.

    Takes plain arrays of amounts and times. To plot a
    :class:`~financialkit.fixed_income.core.base.CashFlows` stream with its
    present values under a curve, use
    :func:`financialkit.fixed_income.plot_discounted_cash_flows`, which
    draws its bars with this function.

    Parameters
    ----------
    cash_flows : array-like, shape (n,)
    times : array-like, shape (n,), optional
        Defaults to ``0, 1, ..., n - 1``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    c = np.asarray(cash_flows, dtype=np.float64)
    t = np.arange(c.size) if times is None else np.asarray(times, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    width = 0.6 * (np.min(np.diff(t)) if t.size > 1 else 1.0)
    ax.bar(t, c, width=width, color=np.where(c >= 0, "tab:green", "tab:red"))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("cash flow")
    ax.set_title(title)
    return ax


def plot_compounding_growth(rate: float, T: float, compoundings=(1, 4, 12, None), ax=None):
    """Growth of 1 at `rate` under several compounding frequencies.

    Parameters
    ----------
    rate : float
        Annual rate.
    T : float
        Horizon in years.
    compoundings : sequence of int or None
        Periods per year; ``None`` is continuous.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for m in compoundings:
        if m is None:
            t = np.linspace(0.0, T, 400)
            ax.plot(t, future_value(1.0, rate, t), label="continuous")
        else:
            # discrete compounding credits interest only at period ends: a step function
            t = np.linspace(0.0, T, 400)
            ax.step(t, future_value(1.0, rate, np.floor(t * m) / m, m), where="post", label=f"m = {m}")
    ax.set_xlabel("years")
    ax.set_ylabel("value of 1")
    ax.set_title(f"Compounding at {100 * rate:g}%")
    ax.legend()
    return ax


def plot_npv_profile(profile, ax=None, label: Optional[str] = None):
    """NPV against the discount rate, with each IRR marked.

    Parameters
    ----------
    profile : NPVProfile
        From :func:`~financialkit.time_value.systems.irr.npv_profile`.
    ax : matplotlib.axes.Axes, optional
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(100 * profile.rates, profile.npv, label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    lo, hi = profile.rates.min(), profile.rates.max()
    shown = profile.irrs[(profile.irrs >= lo) & (profile.irrs <= hi)]
    ax.plot(100 * shown, np.zeros_like(shown), "o", color="tab:red", label="IRR" if shown.size else None)
    ax.set_xlabel("discount rate (%)")
    ax.set_ylabel("NPV")
    ax.set_title("NPV profile")
    return ax


def plot_amortization(schedule, ax=None):
    """Stacked interest and principal per payment, with the balance on a twin axis.

    Parameters
    ----------
    schedule : AmortizationSchedule
        From :func:`~financialkit.time_value.systems.amortization.amortization_schedule`.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    years = schedule.periods / schedule.periods_per_year
    width = 1.0 / schedule.periods_per_year
    ax.bar(years, schedule.interest, width=width, color="tab:orange", label="interest")
    ax.bar(years, schedule.principal, width=width, bottom=schedule.interest, color="tab:blue", label="principal")
    ax.set_xlabel("years")
    ax.set_ylabel("payment split")
    ax.legend(loc="center left")
    twin = ax.twinx()
    twin.plot(years, schedule.balance, color="k", lw=1.5)
    twin.set_ylabel("outstanding balance")
    ax.set_title("Amortization schedule")
    return ax
