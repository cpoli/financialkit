"""Interest-rate and currency swaps, valued as a long and a short bond.

The 1981 IBM-World Bank swap exchanged the payments on Swiss franc and
Deutsche mark debt for payments on US dollar debt: a fixed-for-fixed
currency swap, worth the difference between a foreign bond converted at
the spot rate and a domestic bond. The plain-vanilla interest-rate swap
that followed is the same idea in one currency, with a floating-rate leg
worth par on each reset date. References: J. C. Hull, *Options, Futures,
and Other Derivatives*, 9th ed. (Pearson, 2015), Ch. 7; C. W. Smith,
C. W. Smithson and L. M. Wakeman, "The Evolving Market for Swaps,"
Midland Corporate Finance Journal 3(4) (1986), 20-32.

Every swap function also takes a dated
:class:`~financialkit.time_value.Schedule` in place of `maturity` and
`freq`. Payments then fall on the schedule's business-day-adjusted dates,
each period accrues ``year_fraction(start, end, day_count)`` between
adjusted dates (2006 ISDA Definitions, Sec. 4.16), and the curve is read
at ACT/365F times from `valuation_date`.

Since SOFR's first publication on 3 April 2018, floating legs have been
moving from LIBOR, fixed at the start of each period, to an overnight
rate compounded *in arrears* over the period and known only at its end.
:func:`compounded_in_arrears` applies the ARRC conventions (lookback,
observation shift, lockout), and :func:`overnight_forwards` gives the
overnight rates a curve implies, whose compounded rate is the curve's
simple forward rate over the period. That is why a SOFR leg is worth
:math:`P(t_0) - P(t_n)`, as the dated swap functions above assume.
References: Alternative Reference Rates Committee, *A User's Guide to
SOFR* (April 2019) and *SOFR Floating Rate Notes Conventions Matrix*
(2019); Federal Reserve Bank of New York, *Statement Regarding the
Publication of the Secured Overnight Financing Rate* (2018).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.constants import DayCount
from financialkit.fixed_income.core.base import CompoundedRate, CurveLike, SwapValuation, YieldCurve, as_curve, coupon_times
from financialkit.fixed_income.systems.bonds import _maturity, bond_cash_flows
from financialkit.time_value import Schedule, year_fraction
from financialkit.time_value.utils.dates import as_dates

__all__ = ["swap_annuity", "swap_rate", "interest_rate_swap_value", "currency_swap_value", "compounded_in_arrears", "overnight_forwards"]


class _DatedLeg:
    """The periods of a dated schedule still to pay after `valuation_date`, priced on one curve."""

    def __init__(self, curve: YieldCurve, schedule: Schedule, valuation_date, day_count):
        v = as_dates(schedule.dates[0] if valuation_date is None else valuation_date)[()]
        d = schedule.dates
        if v >= d[-1]:
            raise ValueError("valuation_date must be before the schedule's last payment")
        self.i = int(np.searchsorted(d, v, side="right"))
        self.seasoned = self.i > 0 and v > d[0]
        self.on_reset = v == d[self.i - 1] if self.i > 0 else False
        alpha = np.asarray(year_fraction(d[:-1], d[1:], day_count), dtype=np.float64)
        self.alpha = alpha[max(self.i - 1, 0) :]
        self.current_alpha = float(alpha[self.i - 1]) if self.i > 0 else 0.0
        t = np.asarray(year_fraction(v, d, DayCount.ACT_365F), dtype=np.float64)
        self.P = curve.discount(np.maximum(t, 0.0))
        self.P_start = float(self.P[max(self.i - 1, 0)])
        self.P_pay = self.P[max(self.i, 1) :]
        self.annuity = float(np.sum(self.alpha * self.P_pay))
        self.P_end = float(self.P[-1])

    def fixed_leg(self, notional: float, rate: float) -> float:
        """The fixed coupons plus the notional at the end, as a bond."""
        return float(notional * (rate * self.annuity + self.P_end))


def _dated_leg(curve: CurveLike, schedule: Schedule, valuation_date, day_count) -> _DatedLeg:
    return _DatedLeg(as_curve(curve), schedule, valuation_date, day_count)


def swap_annuity(
    curve: CurveLike,
    maturity: Optional[float] = None,
    freq: int = 2,
    *,
    schedule: Optional[Schedule] = None,
    valuation_date=None,
    day_count=DayCount.ACT_360,
) -> float:
    r"""Present value of 1 per year paid ``freq`` times a year to `maturity`: the fixed leg's PV01 times 10,000.

    With a dated `schedule` it is :math:`\sum_i \alpha_i P(t_i)` over the
    periods paying after `valuation_date` (default: the schedule's first
    date), with :math:`\alpha_i` in the `day_count` convention.

    Examples
    --------
    >>> from financialkit.time_value import coupon_schedule
    >>> s = coupon_schedule("2024-01-15", "2026-01-15", convention="modified following")
    >>> round(swap_annuity(0.0, schedule=s), 6)  # 731 accrued days on ACT/360
    2.030556
    """
    maturity = _maturity(maturity, schedule)
    if schedule is not None:
        return _dated_leg(curve, schedule, valuation_date, day_count).annuity
    return float(np.sum(as_curve(curve).discount(coupon_times(maturity, freq))) / freq)


def swap_rate(
    curve: CurveLike,
    maturity=None,
    freq: int = 2,
    *,
    schedule: Optional[Schedule] = None,
    valuation_date=None,
    day_count=DayCount.ACT_360,
):
    r"""Par swap rate: the fixed rate that makes a new swap worth zero.

    .. math:: K = \frac{1 - P(0, T)}{\frac{1}{m}\sum_i P(0, t_i)}

    It equals the par yield of a bond with the same schedule. With a dated
    `schedule` it is :math:`(P(t_0) - P(t_n)) / \sum_i \alpha_i P(t_i)`,
    which also prices a forward-starting swap; `valuation_date` must be on
    or before the schedule's first date.

    Examples
    --------
    >>> from financialkit.fixed_income.core.base import FlatCurve
    >>> round(swap_rate(FlatCurve(np.log(1.03)), 5.0, freq=1), 10)
    0.03
    """
    if schedule is None:
        if maturity is None:
            raise ValueError("pass either maturity or schedule")
        return as_curve(curve).par_yield(maturity, freq)
    _maturity(maturity, schedule)
    leg = _dated_leg(curve, schedule, valuation_date, day_count)
    if leg.seasoned:
        raise ValueError("a dated swap_rate needs valuation_date on or before the schedule's first date")
    return (leg.P_start - leg.P_end) / leg.annuity


def interest_rate_swap_value(
    curve: CurveLike,
    notional: float,
    fixed_rate: float,
    maturity: Optional[float] = None,
    freq: int = 2,
    pay_fixed: bool = True,
    floating_rate: Optional[float] = None,
    *,
    schedule: Optional[Schedule] = None,
    valuation_date=None,
    day_count=DayCount.ACT_360,
) -> SwapValuation:
    """Value of a fixed-for-floating interest-rate swap.

    The fixed leg is a bond with coupon `fixed_rate`. The floating leg is
    worth par immediately after a reset, so today it is worth the notional
    plus the already-fixed next coupon, discounted from the next payment
    date. Payments on both legs fall on ``maturity - k/freq``.

    Parameters
    ----------
    curve : YieldCurve or float
        Discount curve.
    notional : float
    fixed_rate : float
        Annual fixed rate.
    maturity : float, optional
        Remaining life (years); omit when passing `schedule`.
    freq : int
        Payments per year on both legs.
    pay_fixed : bool
        Value to the fixed-rate payer (``True``) or receiver.
    floating_rate : float, optional
        Annual rate already fixed for the next floating payment. May be
        omitted only on a reset date (maturity a whole number of periods),
        where the floating leg is worth exactly the notional, or, with a
        `schedule`, on or before its first date.
    schedule : Schedule, optional
        Dated payment schedule for both legs, from
        :func:`financialkit.time_value.coupon_schedule`.
    valuation_date : date-like, optional
        With `schedule`; defaults to the schedule's first date.
    day_count : DayCount or str
        With `schedule`, the accrual convention on both legs.

    Returns
    -------
    SwapValuation

    Examples
    --------
    Hull (9th ed., Example 7.1): receive 8% and pay LIBOR on $100 million,
    1.25 years left, LIBOR zero rates 10%, 10.5% and 11% at 3, 9 and 15
    months, and 10.2% fixed for the next payment.

    >>> from financialkit.fixed_income.core.base import ZeroCurve
    >>> curve = ZeroCurve([0.25, 0.75, 1.25], [0.10, 0.105, 0.11])
    >>> v = interest_rate_swap_value(curve, 100.0, 0.08, 1.25, pay_fixed=False, floating_rate=0.102)
    >>> round(v.receive_leg, 3), round(v.pay_leg, 3), round(v.value, 3)
    (98.238, 102.505, -4.267)
    """
    maturity = _maturity(maturity, schedule)
    curve = as_curve(curve)
    if schedule is not None:
        leg = _dated_leg(curve, schedule, valuation_date, day_count)
        fixed_leg = leg.fixed_leg(notional, fixed_rate)
        if not leg.seasoned or leg.on_reset:
            floating_leg = float(notional * leg.P_start)
        elif floating_rate is None:
            raise ValueError("floating_rate is required between reset dates")
        else:
            floating_leg = float(notional * (1.0 + floating_rate * leg.current_alpha) * leg.P_pay[0])
    else:
        fixed_leg, floating_leg = _undated_legs(curve, notional, fixed_rate, maturity, freq, floating_rate)
    if pay_fixed:
        return SwapValuation(floating_leg - fixed_leg, floating_leg, fixed_leg, "pay_fixed")
    return SwapValuation(fixed_leg - floating_leg, fixed_leg, floating_leg, "receive_fixed")


def _undated_legs(curve: YieldCurve, notional: float, fixed_rate: float, maturity: float, freq: int, floating_rate: Optional[float]) -> tuple[float, float]:
    fixed_leg = bond_cash_flows(notional, fixed_rate, maturity, freq).present_value(curve)
    t1 = coupon_times(maturity, freq)[0]
    if floating_rate is None:
        if abs(t1 - 1.0 / freq) > 1e-9:
            raise ValueError("floating_rate is required between reset dates")
        floating_leg = float(notional)
    else:
        floating_leg = float(notional * (1.0 + floating_rate / freq) * curve.discount(t1))
    return fixed_leg, floating_leg


def currency_swap_value(
    domestic_curve: CurveLike,
    foreign_curve: CurveLike,
    spot: float,
    domestic_notional: float,
    foreign_notional: float,
    domestic_rate: float,
    foreign_rate: float,
    maturity: Optional[float] = None,
    freq: int = 1,
    receive_foreign: bool = True,
    *,
    schedule: Optional[Schedule] = None,
    valuation_date=None,
    day_count=DayCount.ACT_360,
) -> SwapValuation:
    """Value, in domestic currency, of a fixed-for-fixed currency swap with exchange of principal.

    Each leg is a bond in its own currency discounted on its own curve; the
    foreign bond is converted at the spot rate.

    Parameters
    ----------
    domestic_curve, foreign_curve : YieldCurve or float
    spot : float
        Domestic currency per unit of foreign currency.
    domestic_notional, foreign_notional : float
    domestic_rate, foreign_rate : float
        Annual fixed coupon rates on each leg.
    maturity : float, optional
        Omit when passing `schedule`.
    freq : int
    receive_foreign : bool
        Value to the party receiving the foreign leg (``True``) or paying it.
    schedule : Schedule, optional
        Dated payment schedule for both legs.
    valuation_date : date-like, optional
        With `schedule`; defaults to the schedule's first date.
    day_count : DayCount or str
        With `schedule`, the accrual convention on both legs.

    Returns
    -------
    SwapValuation

    Examples
    --------
    Hull (9th ed., Example 7.3): receive 5% on 1,200 million yen and pay
    8% on $10 million annually for 3 years, with flat rates of 4% (yen) and
    9% (dollar) and 110 yen per dollar.

    >>> v = currency_swap_value(0.09, 0.04, 1 / 110, 10.0, 1200.0, 0.08, 0.05, 3.0)
    >>> round(v.value, 3)
    1.543
    """
    maturity = _maturity(maturity, schedule)
    if schedule is not None:
        domestic = _dated_leg(domestic_curve, schedule, valuation_date, day_count).fixed_leg(domestic_notional, domestic_rate)
        foreign = spot * _dated_leg(foreign_curve, schedule, valuation_date, day_count).fixed_leg(foreign_notional, foreign_rate)
    else:
        domestic = bond_cash_flows(domestic_notional, domestic_rate, maturity, freq).present_value(as_curve(domestic_curve))
        foreign = spot * bond_cash_flows(foreign_notional, foreign_rate, maturity, freq).present_value(as_curve(foreign_curve))
    if receive_foreign:
        return SwapValuation(foreign - domestic, foreign, domestic, "receive_foreign")
    return SwapValuation(domestic - foreign, domestic, foreign, "pay_foreign")


def _business_days(start, end, holidays) -> tuple[np.ndarray, np.ndarray]:
    """Business days in ``[start, end)`` and the calendar days each one's rate applies for."""
    days = np.arange(start, end, dtype="datetime64[D]")
    days = days[np.is_busday(days, holidays=holidays)]
    if days.size == 0:
        raise ValueError("the period contains no business day")
    return days, np.diff(np.append(days, end)).astype(np.int64)


def compounded_in_arrears(
    fixing_dates,
    fixings,
    start,
    end,
    *,
    lookback: int = 0,
    observation_shift: bool = False,
    lockout: int = 0,
    holidays=None,
    basis: int = 360,
) -> CompoundedRate:
    r"""An overnight rate (SOFR, SONIA, €STR) compounded in arrears over an interest period.

    .. math:: R = \left[\prod_i \left(1 + \frac{r_i n_i}{B}\right) - 1\right] \frac{B}{N}

    where :math:`r_i` is the fixing for business day :math:`i`, applied
    for the :math:`n_i` calendar days to the next business day, and
    :math:`N = \sum_i n_i`. Payment is due at the end of the period, so the
    conventions give the payer time to know the amount:

    * a *lookback* of :math:`p` business days uses the fixing from
      :math:`p` business days before each day, with the interest period's
      day weights;
    * an *observation shift* moves the whole observation period back by
      the lookback and weights the rates by its own days;
    * a *lockout* of :math:`k` days repeats the fixing of the
      :math:`(k+1)`-th last day for the last :math:`k` days.

    Parameters
    ----------
    fixing_dates : array-like of date-likes
        Business days with a published fixing, ascending.
    fixings : array_like
        The overnight rate for each of `fixing_dates`, as a decimal.
    start, end : date-like
        Business days that start and end the interest period.
    lookback : int
        Business days of lookback (ARRC's usual choices are 2 to 5).
    observation_shift : bool
        Shift the observation period by `lookback` instead of looking back day by day.
    lockout : int
        Business days at the end of the period that reuse an earlier fixing.
    holidays : array-like of date-likes, optional
        Holidays on top of Saturdays and Sundays.
    basis : int
        Day basis, 360 for SOFR and €STR and 365 for SONIA.

    Returns
    -------
    CompoundedRate

    Examples
    --------
    A week at 5.30% on business days: Friday's rate counts three times.

    >>> days = np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]")
    >>> days = days[np.is_busday(days)]
    >>> r = compounded_in_arrears(days, [0.053] * 5, "2024-01-01", "2024-01-08")
    >>> r.weights.tolist(), round(r.rate, 8) == round(((1 + 0.053 / 360) ** 4 * (1 + 0.053 * 3 / 360) - 1) * 360 / 7, 8)
    ([1, 1, 1, 1, 3], True)
    """
    if lookback < 0 or lockout < 0:
        raise ValueError("lookback and lockout must be nonnegative")
    hol = np.array([], dtype="datetime64[D]") if holidays is None else as_dates(holidays).ravel()
    s, e = as_dates(start)[()], as_dates(end)[()]
    if e <= s:
        raise ValueError("end must be after start")
    if not (np.is_busday(s, holidays=hol) and np.is_busday(e, holidays=hol)):
        raise ValueError("start and end must be business days")
    if observation_shift:
        days, weights = _business_days(np.busday_offset(s, -lookback, holidays=hol), np.busday_offset(e, -lookback, holidays=hol), hol)
        observed = days.copy()
    else:
        days, weights = _business_days(s, e, hol)
        observed = np.busday_offset(days, -lookback, holidays=hol)
    if lockout:
        if lockout >= days.size:
            raise ValueError("lockout must be shorter than the period")
        observed[-lockout:] = observed[-lockout - 1]
    known = as_dates(fixing_dates).ravel()
    values = np.asarray(fixings, dtype=np.float64).ravel()
    if known.shape != values.shape or known.size == 0:
        raise ValueError("fixing_dates and fixings must be nonempty and of the same length")
    idx = np.clip(np.searchsorted(known, observed), 0, known.size - 1)
    missing = known[idx] != observed
    if np.any(missing):
        raise ValueError(f"no fixing for {observed[np.argmax(missing)]}")
    rates = values[idx]
    growth = float(np.prod(1.0 + rates * weights / basis))
    rate = (growth - 1.0) * basis / float(np.sum(weights))
    return CompoundedRate(rate, float((e - s).astype(np.int64)) / basis, days, observed, rates, weights, basis)


def overnight_forwards(curve: CurveLike, start, end, valuation_date=None, holidays=None, basis: int = 360) -> tuple[np.ndarray, np.ndarray]:
    r"""The overnight rates a curve implies for each business day in ``[start, end)``.

    Day :math:`i`'s forward rate is
    :math:`(P(t_i)/P(t_{i+1}) - 1) B / n_i`, with times in ACT/365F years
    from `valuation_date` (default `start`). Compounding them telescopes
    to :math:`P(t_0)/P(t_n)`, so the expected compounded-in-arrears coupon
    is the curve's simple forward rate over the period.

    Parameters
    ----------
    curve : YieldCurve or float
    start, end : date-like
        The period; `end` must be a business day.
    valuation_date : date-like, optional
    holidays : array-like of date-likes, optional
    basis : int

    Returns
    -------
    dates : ndarray of datetime64[D]
        The business days.
    rates : ndarray
        Their implied overnight rates, ready for :func:`compounded_in_arrears`.

    Examples
    --------
    >>> from financialkit.fixed_income.core.base import FlatCurve
    >>> curve = FlatCurve(0.04)
    >>> r = compounded_in_arrears(*overnight_forwards(curve, "2025-01-02", "2025-04-02"), "2025-01-02", "2025-04-02")
    >>> P = curve.discount(np.array([0.0, 90 / 365]))
    >>> bool(abs(r.rate - (P[0] / P[1] - 1) * 360 / 90) < 1e-14)
    True
    """
    hol = np.array([], dtype="datetime64[D]") if holidays is None else as_dates(holidays).ravel()
    s, e = as_dates(start)[()], as_dates(end)[()]
    days, weights = _business_days(s, e, hol)
    v = s if valuation_date is None else as_dates(valuation_date)[()]
    t = np.asarray(year_fraction(v, np.append(days, e), DayCount.ACT_365F), dtype=np.float64)
    P = as_curve(curve).discount(t)
    return days, (P[:-1] / P[1:] - 1.0) * basis / weights
