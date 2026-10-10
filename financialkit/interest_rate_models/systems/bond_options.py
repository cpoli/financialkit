"""Interest-rate options: zero-coupon and coupon bond options, caps and
floors, and European swaptions.

* :func:`zero_coupon_bond_option`: Jamshidian's (1989) closed form for a
  European option on a zero-coupon bond in any Gaussian short-rate model.
  The bond's forward price is lognormal under the option-expiry forward
  measure, so the price is Black-76 (from :mod:`financialkit.options`)
  with the model's total bond volatility.
* :func:`coupon_bond_option` and :func:`jamshidian_swaption`: Jamshidian's
  decomposition. In a one-factor model every bond price falls with the
  short rate, so an option on a coupon bond is a portfolio of options on
  its zero-coupon pieces, struck at their values at the critical rate
  :math:`r^*` (:func:`scipy.optimize.brentq`). A payer swaption is a put
  on a coupon bond.
* :func:`cap_floor_price` and :func:`model_cap_floor_price`: a cap is a
  strip of caplets, each priced by Black's formula on its simple forward
  rate (the market model of Brace, Gatarek and Musiela, 1997), or, in a
  Gaussian short-rate model, as puts on zero-coupon bonds.
* :func:`black_swaption_price`: Jamshidian's (1997) swap market model, in
  which the forward swap rate is lognormal under the annuity measure and
  a swaption is Black-76 on it, times the annuity.

References: F. Jamshidian, "An Exact Bond Option Formula," Journal of
Finance 44(1) (1989) 205-209; A. Brace, D. Gatarek and M. Musiela,
"The Market Model of Interest Rate Dynamics," Mathematical Finance 7(2)
(1997) 127-155; F. Jamshidian, "LIBOR and Swap Market Models and
Measures," Finance and Stochastics 1(4) (1997) 293-330; J. C. Hull,
*Options, Futures, and Other Derivatives*, 10th ed. (Pearson, 2018),
Ch. 29 and Sec. 32.4; D. Brigo and F. Mercurio, *Interest Rate Models --
Theory and Practice*, 2nd ed. (Springer, 2006), Ch. 3.3 and 6.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CurveLike, as_curve, coupon_times
from financialkit.interest_rate_models.core.base import CapFloorResult, GaussianShortRateModel, JamshidianResult
from financialkit.options import black76_price

__all__ = [
    "zero_coupon_bond_option",
    "coupon_bond_option",
    "swaption_schedule",
    "forward_swap_rate",
    "forward_swap_annuity",
    "jamshidian_swaption",
    "black_swaption_price",
    "cap_floor_price",
    "model_cap_floor_price",
]


def zero_coupon_bond_option(model: GaussianShortRateModel, T, S, K, option_type: str = "call"):
    r"""European option, expiring at ``T``, on a unit zero-coupon bond maturing at ``S``.

    .. math:: c = P(0, S) N(h) - K P(0, T) N(h - \sigma_P), \quad
              h = \frac{1}{\sigma_P}\ln\frac{P(0, S)}{K P(0, T)} + \frac{\sigma_P}{2},

    which is Black-76 on the forward bond price :math:`P(0, S)/P(0, T)`
    with volatility :math:`\sigma_P/\sqrt{T}`, discounted by :math:`P(0, T)`.
    :math:`\sigma_P` is the model's
    :meth:`~financialkit.interest_rate_models.core.base.GaussianShortRateModel.bond_option_volatility`
    (Jamshidian 1989 for Vasicek; the same formula holds for Ho-Lee,
    Hull-White and G2++).

    Parameters
    ----------
    model : GaussianShortRateModel
    T, S : array_like
        Option expiry and bond maturity, ``0 < T < S``.
    K : array_like
        Strike per unit face.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    A 1-year option on a 5-year bond under Vasicek; call minus put is the
    forward value :math:`P(0, 5) - K P(0, 1)`:

    >>> from financialkit.interest_rate_models.systems.short_rate import VasicekModel
    >>> m = VasicekModel(0.1, 0.1, 0.1, 0.02)
    >>> c = zero_coupon_bond_option(m, 1.0, 5.0, 0.67, "call")
    >>> p = zero_coupon_bond_option(m, 1.0, 5.0, 0.67, "put")
    >>> bool(np.isclose(c - p, m.zero_coupon_bond(5.0) - 0.67 * m.zero_coupon_bond(1.0)))
    True
    """
    T, S, K = (np.asarray(x, dtype=np.float64) for x in (T, S, K))
    if np.any(T <= 0) or np.any(S <= T):
        raise ValueError("need 0 < T < S")
    PT = np.asarray(model.zero_coupon_bond(T), dtype=np.float64)
    PS = np.asarray(model.zero_coupon_bond(S), dtype=np.float64)
    vol = np.asarray(model.bond_option_volatility(T, S), dtype=np.float64) / np.sqrt(T)
    return black76_price(PS / PT, K, T, -np.log(PT) / T, vol, option_type)


def _critical_rate(model: GaussianShortRateModel, T: float, times: np.ndarray, flows: np.ndarray, K: float) -> float:
    def value(r):
        return float(np.sum(flows * model.zero_coupon_bond(times, t=T, state=r))) - K

    lo, hi = -0.5, 0.5
    while value(lo) < 0:
        lo *= 2
    while value(hi) > 0:
        hi *= 2
    return brentq(value, lo, hi, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)


def coupon_bond_option(model: GaussianShortRateModel, T: float, payment_times, cash_flows, K: float, option_type: str = "call") -> JamshidianResult:
    r"""European option on a coupon bond by Jamshidian's (1989) decomposition.

    In a one-factor model the bond's value at expiry,
    :math:`\sum_i c_i P(T, t_i; r)`, falls with :math:`r`. Let :math:`r^*`
    solve :math:`\sum_i c_i P(T, t_i; r^*) = K` and :math:`K_i = P(T, t_i; r^*)`.
    The bond is above the strike exactly when every zero is above its
    :math:`K_i`, so

    .. math:: \max\left(\sum_i c_i P(T, t_i) - K, 0\right) = \sum_i c_i \max(P(T, t_i) - K_i, 0),

    and the option is a portfolio of zero-coupon bond options.

    Parameters
    ----------
    model : GaussianShortRateModel
        A one-factor model (Merton, Vasicek, Ho-Lee or Hull-White).
    T : float
        Option expiry.
    payment_times, cash_flows : array_like, shape (n,)
        Bond payments after ``T``.
    K : float
        Strike (on the bond's dirty price).
    option_type : {"call", "put"}

    Returns
    -------
    JamshidianResult
    """
    if model.n_factors != 1:
        raise ValueError("Jamshidian's decomposition needs a one-factor model")
    times = np.asarray(payment_times, dtype=np.float64)
    flows = np.asarray(cash_flows, dtype=np.float64)
    if times.shape != flows.shape or np.any(times <= T):
        raise ValueError("payment_times and cash_flows must match and fall after T")
    r_star = _critical_rate(model, T, times, flows, K)
    strikes = np.asarray(model.zero_coupon_bond(times, t=T, state=r_star), dtype=np.float64)
    components = flows * np.asarray(zero_coupon_bond_option(model, T, times, strikes, option_type), dtype=np.float64)
    return JamshidianResult(
        price=float(components.sum()),
        critical_rate=float(r_star),
        payment_times=times,
        cash_flows=flows,
        strikes=strikes,
        components=components,
    )


def swaption_schedule(expiry: float, tenor: float, freq: int = 2) -> np.ndarray:
    """Fixed-leg payment dates of a swap starting at `expiry` and running for `tenor` years.

    Examples
    --------
    >>> swaption_schedule(1.0, 2.0, 2).tolist()
    [1.5, 2.0, 2.5, 3.0]
    """
    return expiry + coupon_times(tenor, freq)


def forward_swap_annuity(curve: CurveLike, expiry: float, tenor: float, freq: int = 2) -> float:
    r"""Annuity :math:`A = \sum_i \tau_i P(0, t_i)` of a forward-starting swap's fixed leg."""
    times = swaption_schedule(expiry, tenor, freq)
    accrual = np.diff(np.concatenate([[expiry], times]))
    return float(np.sum(accrual * as_curve(curve).discount(times)))


def forward_swap_rate(curve: CurveLike, expiry: float, tenor: float, freq: int = 2) -> float:
    r"""Forward par swap rate :math:`S = [P(0, T_0) - P(0, T_n)] / A`.

    Examples
    --------
    >>> from financialkit.fixed_income import FlatCurve
    >>> round(forward_swap_rate(FlatCurve(0.04), 1.0, 5.0, 1), 6) == round(float(np.expm1(0.04)), 6)
    True
    """
    c = as_curve(curve)
    times = swaption_schedule(expiry, tenor, freq)
    return float((c.discount(expiry) - c.discount(times[-1])) / forward_swap_annuity(c, expiry, tenor, freq))


def jamshidian_swaption(
    model: GaussianShortRateModel, expiry: float, tenor: float, strike: float, freq: int = 2, payer: bool = True, notional: float = 1.0
) -> JamshidianResult:
    r"""European swaption in a one-factor model, by Jamshidian's decomposition.

    At expiry, receiving fixed at :math:`K` on a swap is worth
    :math:`\sum_i c_i P(T_0, t_i) - 1` with :math:`c_i = K\tau_i` (plus 1 at
    the end), so a receiver swaption is a call, and a payer swaption a
    put, on that coupon bond struck at par.

    Parameters
    ----------
    model : GaussianShortRateModel
        One-factor model.
    expiry, tenor : float
        Option expiry and length of the underlying swap.
    strike : float
        Fixed rate.
    freq : int
        Fixed-leg payments per year.
    payer : bool
        Payer (right to pay fixed) or receiver swaption.
    notional : float

    Returns
    -------
    JamshidianResult
        Scaled by `notional`.
    """
    times = swaption_schedule(expiry, tenor, freq)
    accrual = np.diff(np.concatenate([[expiry], times]))
    flows = strike * accrual
    flows[-1] += 1.0
    res = coupon_bond_option(model, expiry, times, flows, 1.0, "put" if payer else "call")
    res.price *= notional
    res.components = res.components * notional
    return res


def black_swaption_price(curve: CurveLike, expiry: float, tenor: float, strike, vol, freq: int = 2, payer: bool = True, notional: float = 1.0):
    r"""Black's formula for a European swaption (Jamshidian's 1997 swap market model).

    Under the annuity measure the forward swap rate :math:`S` is a
    martingale; taking it lognormal gives

    .. math:: V_{\text{payer}} = A\,[S N(d_1) - K N(d_2)], \quad
              V_{\text{receiver}} = A\,[K N(-d_2) - S N(-d_1)],

    Black-76 with zero rate times the annuity :math:`A`.

    Parameters
    ----------
    curve : YieldCurve or float
    expiry, tenor : float
        Option expiry and swap length.
    strike, vol : array_like
        Fixed rate and Black volatility of the swap rate.
    freq : int
    payer : bool
    notional : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Example 29.5: a 5 x 3 payer swaption struck at 6.2%,
    semiannual, flat 6% curve with continuous compounding, 20% volatility,
    principal 100.

    >>> round(black_swaption_price(0.06, 5.0, 3.0, 0.062, 0.2, 2, notional=100.0), 2)
    2.07
    """
    c = as_curve(curve)
    A = forward_swap_annuity(c, expiry, tenor, freq)
    S = forward_swap_rate(c, expiry, tenor, freq)
    return notional * A * black76_price(S, strike, expiry, 0.0, vol, "call" if payer else "put")


def _cap_schedule(maturity: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    times = coupon_times(maturity, freq)
    if times.size < 2:
        raise ValueError("a cap needs at least two periods (the first, already fixed, is excluded)")
    return times[:-1], times[1:]


def cap_floor_price(curve: CurveLike, strike: float, maturity: float, vol, freq: int = 4, kind: str = "cap", notional: float = 1.0) -> CapFloorResult:
    r"""A cap or floor priced caplet by caplet with Black's formula.

    Caplet :math:`k` pays :math:`\tau_k \max(L_k - K, 0)` at :math:`t_{k+1}`
    on the rate :math:`L_k` fixed at :math:`t_k`. Under the
    :math:`t_{k+1}`-forward measure the simple forward rate
    :math:`F_k = [P(0, t_k)/P(0, t_{k+1}) - 1]/\tau_k` is a martingale,
    and taking it lognormal (Brace-Gatarek-Musiela 1997) gives

    .. math:: \text{caplet}_k = \tau_k P(0, t_{k+1})\,[F_k N(d_1) - K N(d_2)].

    The first period, whose rate is already fixed, is excluded.

    Parameters
    ----------
    curve : YieldCurve or float
    strike : float
        Cap (floor) rate.
    maturity : float
        End of the last period.
    vol : float or array_like
        Black volatility: one flat volatility, or one per caplet.
    freq : int
        Resets per year.
    kind : {"cap", "floor"}
    notional : float

    Returns
    -------
    CapFloorResult

    Examples
    --------
    On a curve flat at 7% with quarterly compounding every forward is 7%;
    cap minus floor at the same strike is the forward-starting swap:

    >>> curve = 4 * np.log(1.0175)
    >>> cap = cap_floor_price(curve, 0.07, 2.0, 0.20, freq=4)
    >>> floor = cap_floor_price(curve, 0.07, 2.0, 0.20, freq=4, kind="floor")
    >>> cap.caplets.size, round(float(cap.forwards[0]), 4), bool(np.isclose(cap.price, floor.price))
    (7, 0.07, True)
    """
    if kind not in ("cap", "floor"):
        raise ValueError(f'kind must be "cap" or "floor", got {kind!r}')
    c = as_curve(curve)
    resets, pays = _cap_schedule(maturity, freq)
    tau = pays - resets
    P_reset, P_pay = c.discount(resets), c.discount(pays)
    F = (P_reset / P_pay - 1.0) / tau
    vol = np.broadcast_to(np.asarray(vol, dtype=np.float64), resets.shape)
    r_eff = -np.log(P_pay) / resets
    caplets = notional * tau * np.asarray(black76_price(F, strike, resets, r_eff, vol, "call" if kind == "cap" else "put"), dtype=np.float64)
    return CapFloorResult(price=float(caplets.sum()), caplets=caplets, forwards=F, reset_times=resets, payment_times=pays, strike=float(strike), kind=kind)


def model_cap_floor_price(
    model: GaussianShortRateModel, strike: float, maturity: float, freq: int = 4, kind: str = "cap", notional: float = 1.0
) -> CapFloorResult:
    r"""A cap or floor in a Gaussian short-rate model, as options on zero-coupon bonds.

    A caplet paying at :math:`t_{k+1}` is worth, at its fixing :math:`t_k`,
    :math:`(1 + K\tau)\max(1/(1 + K\tau) - P(t_k, t_{k+1}), 0)`: a put on a
    zero-coupon bond (Hull, 10th ed., Sec. 32.4). A floorlet is the call.

    Parameters
    ----------
    model : GaussianShortRateModel
    strike, maturity : float
    freq : int
    kind : {"cap", "floor"}
    notional : float

    Returns
    -------
    CapFloorResult
    """
    if kind not in ("cap", "floor"):
        raise ValueError(f'kind must be "cap" or "floor", got {kind!r}')
    resets, pays = _cap_schedule(maturity, freq)
    tau = pays - resets
    P_reset = np.asarray(model.zero_coupon_bond(resets), dtype=np.float64)
    P_pay = np.asarray(model.zero_coupon_bond(pays), dtype=np.float64)
    F = (P_reset / P_pay - 1.0) / tau
    scale = 1.0 + strike * tau
    zbo = zero_coupon_bond_option(model, resets, pays, 1.0 / scale, "put" if kind == "cap" else "call")
    caplets = notional * scale * np.asarray(zbo, dtype=np.float64)
    return CapFloorResult(price=float(caplets.sum()), caplets=caplets, forwards=F, reset_times=resets, payment_times=pays, strike=float(strike), kind=kind)
