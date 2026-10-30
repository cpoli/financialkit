"""Mortgage pass-throughs valued under a short-rate model, with Richard and Roll's prepayment model.

A homeowner can repay the mortgage at any time, and does so mostly when
rates have fallen far enough to refinance. The investor in a pass-through
is therefore short a prepayment option, and its cash flows depend on the
path of rates. Richard and Roll (1989), at Goldman Sachs, split the
monthly prepayment rate into four effects, multiplied together:

* refinancing incentive: an S-shaped function of the gap between the
  pool's mortgage rate (WAC) and the current refinancing rate;
* seasoning: new loans prepay slowly, ramping up over 30 months;
* seasonality: more moves, and so more prepayments, in late summer;
* burnout: once the most rate-sensitive borrowers have refinanced, the
  rest prepay less, measured by the pool factor.

:func:`richard_roll_cpr` uses the arctangent parameterisation of these
effects that textbooks and the Numerix model use. :func:`mbs_price`
simulates the short rate with any one-factor
:class:`~financialkit.interest_rate_models.core.base.ShortRateModel`, takes
the refinancing rate to be the model's 10-year zero yield on each path,
runs the pool's amortisation month by month, and discounts each path's
cash flows with its own short rates (Schwartz and Torous 1989).

References: S. F. Richard and R. Roll, "Prepayments on Fixed-Rate
Mortgage-Backed Securities," Journal of Portfolio Management 15(3) (1989)
73-82; E. S. Schwartz and W. N. Torous, "Prepayment and the Valuation of
Mortgage-Backed Securities," Journal of Finance 44(2) (1989) 375-392;
F. J. Fabozzi (ed.), *The Handbook of Mortgage-Backed Securities*, 7th ed.
(Oxford, 2016), Ch. 23.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from financialkit._validation import in_interval, nonnegative, positive
from financialkit.integrators import RNGLike
from financialkit.interest_rate_models.core.base import MBSResult, ShortRateModel
from financialkit.interest_rate_models.utils.numerics import path_discount_factors

__all__ = ["SEASONALITY", "richard_roll_cpr", "mbs_price"]

SEASONALITY = np.array([0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.10, 1.18, 1.22, 1.23, 0.98])
"""Monthly seasonality multipliers, January to December."""


def richard_roll_cpr(wac, refi_rate, age, month, pool_factor=1.0):
    r"""Richard and Roll's (1989) conditional prepayment rate (CPR, annualised).

    .. math:: \text{CPR} = \underbrace{\left[0.28 + 0.14\arctan\big(-8.57 + 430(\text{WAC} - R)\big)\right]}_{\text{refinancing}}
              \cdot \underbrace{\min(1, t/30)}_{\text{seasoning}}
              \cdot \underbrace{s_m}_{\text{seasonality}}
              \cdot \underbrace{(0.3 + 0.7 B_{t-1}/B_0)}_{\text{burnout}}

    with :math:`R` the refinancing (10-year) rate, :math:`t` the loan age
    in months, :math:`s_m` the :data:`SEASONALITY` multiplier of calendar
    month :math:`m` and :math:`B_{t-1}/B_0` the pool factor. The incentive
    curve rises steeply around :math:`\text{WAC} - R = 2\%`, a typical
    spread of mortgage rates over the 10-year rate, from about 7% CPR
    (turnover) to 50% when refinancing is very profitable.

    Parameters
    ----------
    wac : array_like
        Weighted-average coupon of the mortgages (decimal).
    refi_rate : array_like
        Refinancing rate (decimal).
    age : array_like
        Loan age in months.
    month : array_like of int
        Calendar month, 1 (January) to 12.
    pool_factor : array_like
        Remaining principal as a fraction of the original.

    Returns
    -------
    float or ndarray

    Examples
    --------
    A seasoned, untouched pool in January whose WAC is 2% above the
    10-year rate, the middle of the S-curve:

    >>> round(float(richard_roll_cpr(0.08, 0.06, 60, 1)), 4)
    0.2671
    """
    wac, refi_rate, age, pool_factor = (np.asarray(x, dtype=np.float64) for x in (wac, refi_rate, age, pool_factor))
    nonnegative(age=age, pool_factor=pool_factor)
    incentive = 0.28 + 0.14 * np.arctan(-8.57 + 430.0 * (wac - refi_rate))
    seasoning = np.minimum(1.0, age / 30.0)
    seasonality = SEASONALITY[(np.asarray(month, dtype=np.int64) - 1) % 12]
    burnout = 0.3 + 0.7 * pool_factor
    out = incentive * seasoning * seasonality * burnout
    return float(out) if out.ndim == 0 else out


def mbs_price(
    model: ShortRateModel,
    wac: float,
    coupon: float,
    term_months: int = 360,
    n_paths: int = 2000,
    face: float = 100.0,
    age: int = 0,
    start_month: int = 1,
    refi_tenor: float = 10.0,
    spread: float = 0.0,
    prepayment: Callable = richard_roll_cpr,
    rng: RNGLike = None,
) -> MBSResult:
    r"""Value a mortgage pass-through by Monte Carlo over short-rate paths.

    Each month :math:`k` the pool of balance :math:`B` pays the level
    mortgage payment on its remaining term at rate :math:`c = \text{WAC}/12`,

    .. math:: \text{scheduled principal} = B\,\frac{c}{1 - (1 + c)^{-(N - k + 1)}} - cB,

    and borrowers prepay a fraction
    :math:`\text{SMM} = 1 - (1 - \text{CPR})^{1/12}` of what remains. The
    investor receives interest at the pass-through `coupon` (the rest is
    the servicing fee) plus all principal. The CPR is set at the start of
    the month from the model's ``refi_tenor``-year zero yield on that path,
    and each cash flow is discounted by
    :math:`e^{-\int_0^t (r_s + s)\,ds}` along its own path.

    Parameters
    ----------
    model : ShortRateModel
        A one-factor model; its state is the short rate.
    wac : float
        Mortgage rate paid by the borrowers.
    coupon : float
        Pass-through rate paid to the investor.
    term_months : int
        Remaining term in months.
    n_paths : int
    face : float
        Current principal.
    age : int
        Loan age in months today, for seasoning.
    start_month : int
        Calendar month of the first payment, 1 to 12.
    refi_tenor : float
        Maturity of the zero yield used as the refinancing rate.
    spread : float
        Spread over the model's short rate in the discounting (an OAS).
    prepayment : callable
        ``prepayment(wac, refi_rate, age, month, pool_factor)`` giving the
        CPR, :func:`richard_roll_cpr` by default; a constant gives a
        deterministic schedule.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    MBSResult

    Examples
    --------
    With no prepayment the cash flows are a level annuity and the price is
    their present value on the model's curve:

    >>> from financialkit.interest_rate_models.systems.curve_fitted import HullWhiteModel
    >>> res = mbs_price(HullWhiteModel(0.05, 0.1, 0.01), 0.06, 0.06, prepayment=lambda *a: 0.0, rng=0)
    >>> annuity = 100 * 0.005 / (1 - 1.005**-360) * np.exp(-0.05 * np.arange(1, 361) / 12).sum()
    >>> bool(abs(res.price - annuity) < 4 * res.std_error)
    True
    """
    if model.n_factors != 1:
        raise ValueError("mbs_price needs a one-factor model")
    if int(term_months) != term_months or term_months < 1:
        raise ValueError("term_months must be a positive integer")
    positive(face=face, refi_tenor=refi_tenor)
    in_interval(1, 12, closed="both", start_month=start_month)
    N = int(term_months)
    times, rates = model.simulate(N / 12.0, N, n_paths, rng=rng)
    discount = path_discount_factors(times, rates)[:, 1:] * np.exp(-spread * times[1:])
    c = wac / 12.0
    balance = np.full(rates.shape[0], float(face))
    flows = np.empty((rates.shape[0], N))
    cprs = np.empty((rates.shape[0], N))
    balances = np.empty(N + 1)
    balances[0] = face
    for k in range(1, N + 1):
        t = times[k - 1]
        refi = model.yields(t + refi_tenor, t, rates[:, k - 1])
        cpr = np.broadcast_to(prepayment(wac, refi, age + k, start_month + k - 1, balance / face), balance.shape)
        smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0)
        remaining = N - k + 1
        scheduled = balance * (c / -np.expm1(-remaining * np.log1p(c)) - c) if c > 0 else balance / remaining
        prepaid = (balance - scheduled) * smm
        flows[:, k - 1] = coupon / 12.0 * balance + scheduled + prepaid
        cprs[:, k - 1] = cpr
        balance = balance - scheduled - prepaid
        balances[k] = balance.mean()
    values = np.sum(flows * discount, axis=1)
    return MBSResult(
        price=float(values.mean()),
        std_error=float(values.std(ddof=1) / np.sqrt(values.size)),
        times=times[1:],
        expected_cash_flows=flows.mean(axis=0),
        expected_cpr=cprs.mean(axis=0),
        expected_balance=balances,
        n_paths=int(values.size),
    )
