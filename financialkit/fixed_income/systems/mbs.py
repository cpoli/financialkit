"""Mortgage pass-throughs under the PSA prepayment benchmark.

Homeowners can repay their mortgages early, so a pass-through's cash
flows depend on a prepayment assumption. In 1985 the Public Securities
Association adopted a standard one: the conditional prepayment rate
(CPR) rises by 0.2% a year each month to 6% at month 30 and stays there.
That is 100 PSA; 200 PSA is twice as fast. Each month's CPR becomes a
single monthly mortality :math:`SMM = 1 - (1 - CPR)^{1/12}`, the share
of the balance left after the scheduled principal that is prepaid.
References: F. J. Fabozzi, *Bond Markets, Analysis, and Strategies*, 9th
ed. (Pearson, 2015), the chapter on agency pass-throughs; Public
Securities Association, *Uniform Practices for the Clearance and
Settlement of Mortgage-Backed Securities*, Ch. SF (1985 and later).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.fixed_income.core.base import MBSCashFlows

__all__ = ["psa_cpr", "single_monthly_mortality", "mbs_cash_flows"]


def psa_cpr(age, psa: float = 100.0):
    """Annual conditional prepayment rate in month `age` of a mortgage under `psa` percent of the PSA benchmark; broadcasts.

    Examples
    --------
    >>> psa_cpr([1, 15, 30, 200]).round(4).tolist()
    [0.002, 0.03, 0.06, 0.06]
    >>> round(float(psa_cpr(10, psa=150)), 4)
    0.03
    """
    age = np.asarray(age, dtype=np.float64)
    if np.any(age < 1):
        raise ValueError("age must be at least 1 month")
    if psa < 0:
        raise ValueError("psa must be nonnegative")
    return np.minimum(1.0, 0.06 * np.minimum(age, 30.0) / 30.0 * psa / 100.0)


def single_monthly_mortality(cpr):
    """Monthly prepayment rate equivalent to an annual CPR: :math:`1 - (1 - CPR)^{1/12}`; broadcasts.

    Examples
    --------
    >>> round(float(single_monthly_mortality(0.06)), 6)
    0.005143
    """
    cpr = np.asarray(cpr, dtype=np.float64)
    if np.any((cpr < 0) | (cpr > 1)):
        raise ValueError("cpr must be in [0, 1]")
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def mbs_cash_flows(
    balance: float,
    mortgage_rate: float,
    term_months: int = 360,
    pass_through_rate: Optional[float] = None,
    psa: float = 100.0,
    age: int = 0,
) -> MBSCashFlows:
    """Monthly cash flows of a pass-through on a pool of level-payment mortgages.

    Each month the pool pays the level payment that amortizes its balance
    over the remaining term at `mortgage_rate` (the WAC); the scheduled
    principal is that payment less the interest, and the SMM of the
    month's loan age is then prepaid from what is left. Investors receive
    interest at `pass_through_rate` (the WAC less servicing and guarantee
    fees) and all the principal.

    Parameters
    ----------
    balance : float
        Pool balance today.
    mortgage_rate : float
        Annual weighted average coupon of the mortgages.
    term_months : int
        Remaining term (WAM), in months.
    pass_through_rate : float, optional
        Annual rate paid to investors; defaults to `mortgage_rate`.
    psa : float
        Prepayment speed in percent of the PSA benchmark (0 for none).
    age : int
        Loan age in months today, so the first month is age ``age + 1``.

    Returns
    -------
    MBSCashFlows

    Examples
    --------
    Fabozzi's pass-through: $400 million, WAC 8.125%, pass-through rate
    7.5%, 357 months left (three months seasoned) at 100 PSA. Month 1
    pays 2,500,000 of interest, 267,535 of scheduled principal and
    267,470 of prepayments.

    >>> cf = mbs_cash_flows(400e6, 0.08125, 357, 0.075, psa=100, age=3)
    >>> [round(float(x)) for x in (cf.interest[0], cf.scheduled_principal[0], cf.prepayment[0], cf.cash_flow[0])]
    [2500000, 267535, 267470, 3035005]
    """
    if balance <= 0:
        raise ValueError("balance must be positive")
    if term_months < 1 or age < 0:
        raise ValueError("term_months must be at least 1 and age nonnegative")
    i = mortgage_rate / 12.0
    j = (mortgage_rate if pass_through_rate is None else pass_through_rate) / 12.0
    smm = single_monthly_mortality(psa_cpr(np.arange(age + 1, age + term_months + 1), psa))
    B = np.empty(term_months)
    sched = np.empty(term_months)
    prepay = np.empty(term_months)
    b = float(balance)
    for k in range(term_months):
        left = term_months - k
        payment = b / left if i == 0.0 else b * i / (1.0 - (1.0 + i) ** -left)
        B[k] = b
        sched[k] = payment - b * i
        prepay[k] = smm[k] * (b - sched[k])
        b -= sched[k] + prepay[k]
    return MBSCashFlows(np.arange(1, term_months + 1), B, B * j, sched, prepay, smm)
