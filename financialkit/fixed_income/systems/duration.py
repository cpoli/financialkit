"""Duration, convexity and key rate durations.

Four generations of interest-rate sensitivity, all computed in closed form
from a :class:`~financialkit.fixed_income.core.base.CashFlows` stream:

* Macaulay's (1938) present-value-weighted average time to payment;
* Hicks's (1939) elasticity, which becomes *modified duration*
  :math:`-P'/P = D_{mac} / (1 + y/m)` for a periodically compounded yield;
* convexity, :math:`P''/P`, the second-order term popularised by
  Bierwag, Kaufman and Toevs (1983);
* Fisher and Weil's (1971) duration on a full zero curve, and Ho's (1992)
  key rate durations, which split that sensitivity across maturities.

References: F. R. Macaulay, *Some Theoretical Problems Suggested by the
Movements of Interest Rates, Bond Yields and Stock Prices in the United
States since 1856* (NBER, 1938); J. R. Hicks, *Value and Capital*
(Clarendon, 1939); L. Fisher and R. L. Weil, Journal of Business
44(4) (1971), 408-431; G. O. Bierwag, G. G. Kaufman and A. Toevs,
Financial Analysts Journal 39(4) (1983), 15-35; T. S. Y. Ho, Journal of
Fixed Income 2(2) (1992), 29-44; J. C. Hull, *Options, Futures, and Other
Derivatives*, 9th ed. (Pearson, 2015), Sec. 4.8-4.9.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from financialkit.fixed_income.core.base import BondRisk, CashFlows, CurveLike, KeyRateDurationResult, as_curve
from financialkit.fixed_income.utils.rates import yield_discount_derivatives
from financialkit.time_value import discount_factor

__all__ = [
    "bond_risk",
    "macaulay_duration",
    "modified_duration",
    "convexity",
    "fisher_weil_duration",
    "fisher_weil_convexity",
    "effective_duration",
    "key_rate_weights",
    "key_rate_durations",
]


def bond_risk(cash_flows: CashFlows, y: float, compounding: Optional[int] = None) -> BondRisk:
    """Price, Macaulay and modified duration, and convexity at a single yield.

    Parameters
    ----------
    cash_flows : CashFlows
    y : float
        Yield (decimal).
    compounding : int or None
        Compounding frequency of `y`; ``None`` for continuous.

    Returns
    -------
    BondRisk

    Examples
    --------
    Hull (9th ed., Table 4.6 and Sec. 4.9): a 3-year 10% semiannual bond at
    a 12% continuously compounded yield.

    >>> from financialkit.fixed_income.systems.bonds import bond_cash_flows
    >>> risk = bond_risk(bond_cash_flows(100.0, 0.10, 3.0), 0.12)
    >>> round(risk.price, 3), round(risk.macaulay_duration, 3), round(risk.convexity, 2)
    (94.213, 2.653, 7.57)
    """
    t, c = cash_flows.times, cash_flows.amounts
    df = discount_factor(y, t, compounding)
    d1, d2 = yield_discount_derivatives(y, t, compounding)
    price = float(np.sum(c * df))
    return BondRisk(
        price=price,
        macaulay_duration=float(np.sum(t * c * df) / price),
        modified_duration=float(-np.sum(c * d1) / price),
        convexity=float(np.sum(c * d2) / price),
        yield_=float(y),
        compounding=compounding,
    )


def macaulay_duration(cash_flows: CashFlows, y: float, compounding: Optional[int] = None) -> float:
    r"""Macaulay duration :math:`D = \sum_i t_i \, c_i \, d(t_i) / P` at yield `y`.

    Examples
    --------
    >>> from financialkit.fixed_income.systems.bonds import bond_cash_flows
    >>> macaulay_duration(bond_cash_flows(100.0, 0.0, 7.0), 0.05)  # a zero's duration is its maturity
    7.0
    """
    return bond_risk(cash_flows, y, compounding).macaulay_duration


def modified_duration(cash_flows: CashFlows, y: float, compounding: Optional[int] = None) -> float:
    r"""Modified duration :math:`-\frac{1}{P}\frac{dP}{dy}`: Hicks's elasticity in yield units.

    Equals :math:`D_{mac} / (1 + y/m)` for a yield compounded `m` times a
    year, and :math:`D_{mac}` itself under continuous compounding.

    Examples
    --------
    >>> from financialkit.fixed_income.systems.bonds import bond_cash_flows
    >>> cf = bond_cash_flows(100.0, 0.06, 5.0)
    >>> d_mac = macaulay_duration(cf, 0.06, compounding=2)
    >>> bool(abs(modified_duration(cf, 0.06, compounding=2) - d_mac / 1.03) < 1e-12)
    True
    """
    return bond_risk(cash_flows, y, compounding).modified_duration


def convexity(cash_flows: CashFlows, y: float, compounding: Optional[int] = None) -> float:
    r"""Convexity :math:`\frac{1}{P}\frac{d^2P}{dy^2}` at yield `y`."""
    return bond_risk(cash_flows, y, compounding).convexity


def fisher_weil_duration(cash_flows: CashFlows, curve: CurveLike) -> float:
    r"""Fisher-Weil duration: Macaulay's weights computed on a zero curve.

    .. math:: D_{FW} = \frac{1}{P} \sum_i t_i \, c_i \, P(0, t_i),
              \qquad P = \sum_i c_i \, P(0, t_i)

    It is exactly :math:`-\frac{1}{P}\frac{dP}{ds}` for a parallel shift
    :math:`s` of the continuously compounded zero curve.

    Examples
    --------
    >>> from financialkit.fixed_income.core.base import ZeroCurve
    >>> from financialkit.fixed_income.systems.bonds import bond_cash_flows
    >>> cf = bond_cash_flows(100.0, 0.05, 3.0, freq=1)
    >>> round(fisher_weil_duration(cf, ZeroCurve([1, 2, 3], [0.02, 0.03, 0.04])), 4)
    2.8588
    """
    disc = as_curve(curve).discount(cash_flows.times)
    pv = cash_flows.amounts * disc
    return float(np.sum(cash_flows.times * pv) / np.sum(pv))


def fisher_weil_convexity(cash_flows: CashFlows, curve: CurveLike) -> float:
    r"""Convexity under parallel shifts of a zero curve, :math:`\frac{1}{P}\sum_i t_i^2 c_i P(0, t_i)`."""
    disc = as_curve(curve).discount(cash_flows.times)
    pv = cash_flows.amounts * disc
    return float(np.sum(cash_flows.times**2 * pv) / np.sum(pv))


def effective_duration(cash_flows: CashFlows, curve: CurveLike, bump: float = 1e-4) -> float:
    """Duration by repricing under parallel curve shifts of ``+/- bump`` (central difference).

    Works for any curve and agrees with :func:`fisher_weil_duration` to
    :math:`O(\\text{bump}^2)`.
    """
    curve = as_curve(curve)
    up = cash_flows.present_value(curve.shifted(bump))
    down = cash_flows.present_value(curve.shifted(-bump))
    return float((down - up) / (2.0 * bump * cash_flows.present_value(curve)))


def key_rate_weights(t, key_tenors) -> np.ndarray:
    """Ho's triangular key-rate shock shapes evaluated at maturities `t`.

    Shock ``k`` is 1 at ``key_tenors[k]``, falls linearly to 0 at the
    neighbouring keys and is flat beyond the first and last keys. The shapes
    sum to 1 at every maturity, so the key rate durations add up to the
    parallel duration.

    Parameters
    ----------
    t : array_like, shape (n,)
    key_tenors : array_like, shape (k,)
        Strictly increasing key maturities.

    Returns
    -------
    ndarray, shape (k, n)

    Examples
    --------
    >>> key_rate_weights([1.0, 3.5, 20.0], [2.0, 5.0, 10.0]).round(3).tolist()
    [[1.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    keys = np.asarray(key_tenors, dtype=np.float64)
    if keys.ndim != 1 or np.any(np.diff(keys) <= 0):
        raise ValueError("key_tenors must be strictly increasing")
    eye = np.eye(keys.size)
    return np.array([np.interp(t, keys, eye[k]) for k in range(keys.size)])


def key_rate_durations(cash_flows: CashFlows, curve: CurveLike, key_tenors, bump: float = 1e-4) -> KeyRateDurationResult:
    """Ho's key rate durations: sensitivity to each key maturity's zero rate.

    Each key rate is shifted by ``+/- bump`` with the triangular shape of
    :func:`key_rate_weights`, and the bond is repriced on the shocked curve.

    Parameters
    ----------
    cash_flows : CashFlows
    curve : YieldCurve or float
    key_tenors : array_like
        Key maturities (years), strictly increasing.
    bump : float
        Size of each shock in rate units.

    Returns
    -------
    KeyRateDurationResult

    Examples
    --------
    >>> from financialkit.fixed_income.systems.bonds import bond_cash_flows
    >>> res = key_rate_durations(bond_cash_flows(100.0, 0.05, 10.0), 0.04, [2.0, 5.0, 10.0, 30.0])
    >>> bool(abs(res.durations.sum() - res.parallel_duration) < 1e-6)
    True
    """
    curve = as_curve(curve)
    keys = np.asarray(key_tenors, dtype=np.float64)
    price = cash_flows.present_value(curve)
    durations = np.empty(keys.size)
    for k in range(keys.size):
        shape = lambda s, k=k: key_rate_weights(s, keys)[k].reshape(np.shape(s))
        up = cash_flows.present_value(curve.shifted(lambda s: bump * shape(s)))
        down = cash_flows.present_value(curve.shifted(lambda s: -bump * shape(s)))
        durations[k] = (down - up) / (2.0 * bump * price)
    return KeyRateDurationResult(key_tenors=keys, durations=durations, parallel_duration=effective_duration(cash_flows, curve, bump), price=price)
