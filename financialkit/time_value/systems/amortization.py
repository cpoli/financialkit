"""Fully amortizing loans and the annual percentage rate.

A loan of ``L`` at nominal annual rate ``r`` repaid by ``n`` level
payments, ``m`` a year, has payment :math:`L / a_{\\overline{n}|i}` with
:math:`i = r/m`; each payment first covers the interest on the balance
and repays principal with the rest, so the debt is gone at the last
payment. The U.S. Home Owners' Loan Corporation (1933) and the FHA (1934)
made this long-term level-payment mortgage the American standard (Green
and Wachter, "The American Mortgage in Historical and International
Context", *Journal of Economic Perspectives* 19, 2005). The Truth in
Lending Act (1968, Regulation Z, 12 CFR 1026, Appendix J) requires
lenders to disclose the APR: the nominal rate ``m * i`` at which the
payments are worth exactly the amount financed, net of fees, which is
found here with :func:`scipy.optimize.brentq`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.time_value.core.base import AmortizationSchedule
from financialkit.time_value.systems.annuities import annuity_payment, annuity_present_value

__all__ = ["loan_payment", "amortization_schedule", "annual_percentage_rate"]


def loan_payment(principal, annual_rate, n_payments, periods_per_year: int = 12):
    """Level payment of a fully amortizing loan.

    Parameters
    ----------
    principal : float or array-like
    annual_rate : float or array-like
        Nominal annual rate, compounded `periods_per_year` times a year.
    n_payments : int or array-like
    periods_per_year : int

    Returns
    -------
    float or ndarray

    Examples
    --------
    A 30-year, 200,000 mortgage at 6%:

    >>> round(loan_payment(200_000.0, 0.06, 360), 2)
    1199.1
    """
    return annuity_payment(principal, np.asarray(annual_rate, dtype=np.float64) / periods_per_year, n_payments)


def amortization_schedule(principal: float, annual_rate: float, n_payments: int, periods_per_year: int = 12) -> AmortizationSchedule:
    """Split each level payment into interest and principal.

    Parameters
    ----------
    principal : float
    annual_rate : float
        Nominal annual rate, compounded `periods_per_year` times a year.
    n_payments : int
    periods_per_year : int

    Returns
    -------
    AmortizationSchedule

    Examples
    --------
    >>> s = amortization_schedule(1000.0, 0.12, 12)
    >>> round(s.payment, 2), float(s.interest[0]), abs(float(s.balance[-1])) < 1e-9
    (88.85, 10.0, True)
    """
    i = annual_rate / periods_per_year
    payment = float(loan_payment(principal, annual_rate, n_payments, periods_per_year))
    k = np.arange(1, n_payments + 1)
    # balance after k payments = L (1+i)^k - P s_k, i.e. the PV of the remaining payments
    balance = payment * np.asarray(annuity_present_value(1.0, i, n_payments - k))
    previous = np.concatenate(([principal], balance[:-1]))
    interest = previous * i
    return AmortizationSchedule(
        periods=k,
        payment=payment,
        interest=interest,
        principal=payment - interest,
        balance=balance,
        periods_per_year=periods_per_year,
    )


def annual_percentage_rate(amount_financed: float, payment: float, n_payments: int, periods_per_year: int = 12) -> float:
    """Truth-in-Lending APR of a level-payment loan.

    Parameters
    ----------
    amount_financed : float
        Loan principal less prepaid finance charges (fees, points).
    payment : float
        Level payment per period.
    n_payments : int
    periods_per_year : int

    Returns
    -------
    float
        Nominal annual rate ``periods_per_year * i``, where ``i`` solves
        ``payment * a_n(i) == amount_financed``.

    Examples
    --------
    A 1,000 loan at 12% with a 20 fee deducted up front:

    >>> payment = loan_payment(1000.0, 0.12, 12)
    >>> round(annual_percentage_rate(1000.0, payment, 12), 6)
    0.12
    >>> round(annual_percentage_rate(980.0, payment, 12), 4)
    0.1585
    """
    if payment * n_payments <= amount_financed:
        raise ValueError("total payments must exceed the amount financed for a positive APR")
    f = lambda i: float(annuity_present_value(payment, i, n_payments)) - amount_financed
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
    i = brentq(f, 0.0, hi, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)
    return float(periods_per_year * i)
