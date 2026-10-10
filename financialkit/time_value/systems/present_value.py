"""Present value, net present value and Fisher's real rate of interest.

Discounting a stream of payments to one comparable number goes back to
Fibonacci's *Liber Abaci* (1202), whose Chapter 12 compares payment
schedules by discounting them at a per-period rate (Goetzmann, "Fibonacci
and the Financial Revolution", in *The Origins of Value*, 2005). Irving
Fisher made the net present value the criterion for choosing between
income streams in *The Rate of Interest* (1907), and in *Appreciation and
Interest* (1896) separated the nominal rate into a real rate and expected
inflation, :math:`1 + i = (1 + r)(1 + \\pi)`. Brealey, Myers and Allen,
*Principles of Corporate Finance*, 13th ed., 2020, Chs. 2 and 5, is the
textbook reference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.time_value.core.base import _finalize
from financialkit.time_value.systems.compounding import discount_factor
from financialkit.time_value.utils.cash_flows import as_cash_flows

__all__ = ["present_value", "npv", "fisher_real_rate", "fisher_nominal_rate"]


def present_value(cash_flows, times, rate, compounding: Optional[int] = None):
    """Discounted value today of payments ``cash_flows`` made at ``times``.

    Parameters
    ----------
    cash_flows : array-like, shape (n,)
    times : array-like, shape (n,)
        Payment times in years.
    rate : float or array-like
        Annual discount rate; an array gives one present value per rate.
    compounding : int or None
        Periods per year, or ``None`` (default) for continuous compounding.

    Returns
    -------
    float or ndarray
        Same shape as `rate`.

    Examples
    --------
    >>> round(present_value([100.0, 100.0], [1.0, 2.0], 0.10, compounding=1), 4)
    173.5537
    """
    c, t = as_cash_flows(cash_flows, times)
    r = np.asarray(rate, dtype=np.float64)
    df = np.asarray(discount_factor(r[..., None], t, compounding))
    return _finalize((c * df).sum(axis=-1))


def npv(rate, cash_flows, times=None, compounding: Optional[int] = 1):
    """Fisher's net present value of a project, outflows negative.

    Unlike :func:`present_value` the defaults follow capital budgeting:
    cash flows at the end of periods ``0, 1, ..., n - 1`` and an annually
    compounded `rate`.

    Parameters
    ----------
    rate : float or array-like
        Discount rate (the opportunity cost of capital).
    cash_flows : array-like, shape (n,)
        ``cash_flows[0]`` is usually the (negative) initial investment.
    times : array-like, shape (n,), optional
        Payment times; defaults to ``0, 1, ..., n - 1``.
    compounding : int or None
        Periods per year of `rate`; default 1.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Brealey-Myers-Allen (13th ed.), Sec. 2.1: an office block costing
    370,000 that sells for 420,000 in a year, at 5%:

    >>> round(npv(0.05, [-370_000, 420_000]))
    30000
    """
    c, t = as_cash_flows(cash_flows, times)
    return present_value(c, t, rate, compounding)


def fisher_real_rate(nominal, inflation, exact: bool = True):
    """Real interest rate from a nominal rate and inflation (Fisher, 1896).

    Parameters
    ----------
    nominal : float or array-like
        Nominal rate per period.
    inflation : float or array-like
        Inflation rate per period.
    exact : bool
        ``True`` for :math:`(1 + i)/(1 + \\pi) - 1`, ``False`` for the
        first-order approximation :math:`i - \\pi`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(fisher_real_rate(0.10, 0.06), 6), round(fisher_real_rate(0.10, 0.06, exact=False), 6)
    (0.037736, 0.04)
    """
    i = np.asarray(nominal, dtype=np.float64)
    p = np.asarray(inflation, dtype=np.float64)
    return _finalize((1.0 + i) / (1.0 + p) - 1.0 if exact else i - p)


def fisher_nominal_rate(real, inflation):
    """Nominal rate implied by a real rate and inflation, :math:`(1 + r)(1 + \\pi) - 1`.

    Parameters
    ----------
    real : float or array-like
    inflation : float or array-like

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(fisher_nominal_rate(0.02, 0.03), 4)
    0.0506
    """
    return _finalize((1.0 + np.asarray(real, dtype=np.float64)) * (1.0 + np.asarray(inflation, dtype=np.float64)) - 1.0)
