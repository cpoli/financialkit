"""Result containers and warnings for financialkit.time_value.

Every schedule or table in this domain is returned as a small dataclass
rather than a bare tuple, so that :mod:`financialkit.time_value.visualizers`
has a stable interface. Scalar valuations (a present value, an internal
rate of return, a year fraction) are returned as plain floats or arrays.

Rate conventions used throughout the domain:

* ``compounding=None`` means continuous compounding, the package default;
  a positive integer ``m`` means ``m`` compounding periods per year.
* Annuity, sinking-fund and IRR functions work in *periods*: their
  ``rate`` is the effective rate per period, as in the classical tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CompoundingTable",
    "NPVProfile",
    "AmortizationSchedule",
    "SinkingFundSchedule",
    "MultipleIRRWarning",
]


@dataclass
class CompoundingTable:
    """A compound-interest table for one rate, in the layout of Witt (1613)."""

    rate: float
    """float: Effective interest rate per period."""

    periods: np.ndarray
    """ndarray, shape (n,): Period numbers ``1, ..., n``."""

    amount: np.ndarray
    """ndarray, shape (n,): Accumulation factors :math:`(1 + i)^k`."""

    discount: np.ndarray
    """ndarray, shape (n,): Discount factors :math:`v^k = (1 + i)^{-k}`."""

    annuity_present: np.ndarray
    """ndarray, shape (n,): Present value of 1 per period, :math:`a_{\\overline{k}|}`."""

    annuity_future: np.ndarray
    """ndarray, shape (n,): Accumulated value of 1 per period, :math:`s_{\\overline{k}|}`."""


@dataclass
class NPVProfile:
    """Net present value as a function of the discount rate."""

    rates: np.ndarray
    """ndarray, shape (k,): Discount rates at which the NPV was evaluated."""

    npv: np.ndarray
    """ndarray, shape (k,): Net present value at each rate."""

    irrs: np.ndarray
    """ndarray: Every real internal rate of return (roots of the NPV), ascending."""


@dataclass
class AmortizationSchedule:
    """A level-payment loan schedule, one row per payment."""

    periods: np.ndarray
    """ndarray, shape (n,): Payment numbers ``1, ..., n``."""

    payment: float
    """float: The level payment per period."""

    interest: np.ndarray
    """ndarray, shape (n,): Interest part of each payment."""

    principal: np.ndarray
    """ndarray, shape (n,): Principal repaid by each payment."""

    balance: np.ndarray
    """ndarray, shape (n,): Outstanding balance after each payment (ends at zero)."""

    periods_per_year: int = 12
    """int: Payments per year."""

    @property
    def total_interest(self) -> float:
        """float: Interest paid over the life of the loan."""
        return float(self.interest.sum())


@dataclass
class SinkingFundSchedule:
    """Level deposits accumulating at interest to repay a debt at maturity."""

    periods: np.ndarray
    """ndarray, shape (n,): Deposit numbers ``1, ..., n``."""

    deposit: float
    """float: The level deposit per period."""

    interest: np.ndarray
    """ndarray, shape (n,): Interest earned by the fund during each period."""

    balance: np.ndarray
    """ndarray, shape (n,): Fund balance after each deposit (ends at the target)."""


class MultipleIRRWarning(UserWarning):
    """Raised when a cash-flow stream has more than one internal rate of return.

    By Descartes' rule of signs a stream with more than one sign change can
    have several IRRs (Lorie and Savage, 1955), and then no single one of
    them is a meaningful rate of return.
    """


def _finalize(x):
    """Return a Python float for 0-d results and an ndarray otherwise, so that
    scalar calls give scalars and broadcast calls give arrays."""
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x
