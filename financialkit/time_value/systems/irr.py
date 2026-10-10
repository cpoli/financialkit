"""Internal rate of return, multiple IRRs and the modified IRR.

The internal rate of return is the discount rate at which a project's NPV
is zero -- Irving Fisher's "rate of return over cost" (*The Theory of
Interest*, 1930, Ch. 7). :func:`irr` finds it with
:func:`scipy.optimize.brentq` after bracketing a sign change of the NPV
on a grid. When the cash flows change sign more than once the NPV
polynomial can have several roots (Lorie and Savage, "Three Problems in
Rationing Capital", *Journal of Business* 28, 1955); :func:`irr_all`
finds every one of them with :func:`numpy.roots` and :func:`irr` warns
with :class:`~financialkit.time_value.core.base.MultipleIRRWarning`.
Baldwin's modified IRR ("How to Assess Investment Proposals", *Harvard
Business Review* 37, 1959) removes the ambiguity by compounding inflows
at a reinvestment rate and discounting outflows at a finance rate.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER, DEFAULT_RTOL
from financialkit.time_value.core.base import MultipleIRRWarning, NPVProfile
from financialkit.time_value.systems.present_value import npv
from financialkit.time_value.utils.cash_flows import as_cash_flows

__all__ = ["irr", "irr_all", "mirr", "npv_profile"]


def irr_all(cash_flows) -> np.ndarray:
    """Every real IRR above -100% of cash flows at periods ``0, 1, ..., n``.

    With :math:`x = 1/(1 + r)` the NPV is the polynomial
    :math:`\\sum_k c_k x^k`; its positive real roots give the IRRs.

    Parameters
    ----------
    cash_flows : array-like, shape (n + 1,)

    Returns
    -------
    ndarray
        IRRs per period, ascending (empty if there is none).

    Examples
    --------
    Lorie and Savage's oil pump (1955): 25% and 400%.

    >>> irr_all([-1600, 10000, -10000]).round(6).tolist()
    [0.25, 4.0]
    """
    c, _ = as_cash_flows(cash_flows)
    c = np.trim_zeros(c, "b")
    if c.size < 2:
        return np.empty(0)
    roots = np.roots(c[::-1])
    tol = 1e-9 * np.maximum(1.0, np.abs(roots))
    x = roots[(np.abs(roots.imag) < tol) & (roots.real > 0)].real
    return np.unique(np.round(1.0 / x - 1.0, 12))


def irr(cash_flows, times=None, compounding: Optional[int] = 1, bracket: tuple[float, float] = (-0.99, 10.0), n_grid: int = 2001) -> float:
    """Internal rate of return: the `rate` at which :func:`npv` is zero.

    Parameters
    ----------
    cash_flows : array-like, shape (n,)
    times : array-like, shape (n,), optional
        Payment times; defaults to ``0, 1, ..., n - 1``.
    compounding : int or None
        Compounding of the returned rate; default annual.
    bracket : (float, float)
        Search interval for the rate.
    n_grid : int
        Grid points used to bracket sign changes before root polishing.

    Returns
    -------
    float
        The lowest IRR in `bracket`.

    Raises
    ------
    ValueError
        If the NPV does not change sign in `bracket`.

    Warns
    -----
    MultipleIRRWarning
        If the NPV has more than one root in `bracket`.

    Examples
    --------
    >>> round(irr([-100.0, 60.0, 60.0]), 6)
    0.130662
    """
    c, t = as_cash_flows(cash_flows, times)
    grid = np.linspace(bracket[0], bracket[1], n_grid)
    values = np.asarray(npv(grid, c, t, compounding))
    s = np.sign(values)
    exact = np.flatnonzero(s == 0)
    crossings = np.flatnonzero(s[:-1] * s[1:] < 0)
    roots = [float(grid[k]) for k in exact]
    for k in crossings:
        roots.append(
            brentq(
                lambda r: float(npv(r, c, t, compounding)),
                grid[k],
                grid[k + 1],
                xtol=DEFAULT_ATOL,
                rtol=max(DEFAULT_RTOL, 4 * np.finfo(float).eps),
                maxiter=DEFAULT_MAX_ITER,
            )
        )
    if not roots:
        raise ValueError(f"NPV does not change sign for rates in {bracket}; no IRR there")
    roots.sort()
    if len(roots) > 1:
        warnings.warn(f"cash flows have {len(roots)} IRRs {np.round(roots, 6).tolist()}; returning the lowest", MultipleIRRWarning, stacklevel=2)
    return roots[0]


def mirr(cash_flows, finance_rate: float, reinvest_rate: float) -> float:
    """Baldwin's modified internal rate of return.

    Inflows are compounded to the horizon at `reinvest_rate`, outflows are
    discounted to today at `finance_rate`, and the MIRR is the single rate
    linking the two: :math:`(FV^+ / |PV^-|)^{1/n} - 1`.

    Parameters
    ----------
    cash_flows : array-like, shape (n + 1,)
        Cash flows at periods ``0, ..., n``, with at least one of each sign.
    finance_rate : float
        Rate paid on the money invested (outflows).
    reinvest_rate : float
        Rate earned on reinvested inflows.

    Returns
    -------
    float

    Examples
    --------
    >>> round(mirr([-1600, 10000, -10000], 0.10, 0.10), 6)
    0.05599
    """
    c, _ = as_cash_flows(cash_flows)
    n = c.size - 1
    k = np.arange(c.size)
    fv_in = np.sum(np.where(c > 0, c, 0.0) * (1.0 + reinvest_rate) ** (n - k))
    pv_out = np.sum(np.where(c < 0, c, 0.0) / (1.0 + finance_rate) ** k)
    if n < 1 or fv_in == 0.0 or pv_out == 0.0:
        raise ValueError("mirr needs at least one positive and one negative cash flow")
    return float((fv_in / -pv_out) ** (1.0 / n) - 1.0)


def npv_profile(cash_flows, rates) -> NPVProfile:
    """NPV across a range of discount rates, with every IRR marked.

    Parameters
    ----------
    cash_flows : array-like, shape (n + 1,)
        Cash flows at periods ``0, ..., n``.
    rates : array-like
        Annually compounded discount rates.

    Returns
    -------
    NPVProfile

    Examples
    --------
    >>> profile = npv_profile([-100.0, 60.0, 60.0], [0.0, 0.1, 0.2])
    >>> profile.npv.round(2).tolist()
    [20.0, 4.13, -8.33]
    """
    r = np.asarray(rates, dtype=np.float64)
    return NPVProfile(rates=r, npv=np.asarray(npv(r, cash_flows)), irrs=irr_all(cash_flows))
