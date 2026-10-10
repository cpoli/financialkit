"""Immunization: protecting a surplus against interest-rate moves.

Redington (1952) showed that matching the present value and duration of
assets and liabilities, with asset convexity exceeding liability
convexity, makes the surplus a local minimum under parallel shifts. Fong
and Vasicek (1984) bounded the loss under non-parallel shifts by the
dispersion :math:`M^2` of the portfolio's cash flows around the horizon,
so among immunized portfolios the one with the smallest :math:`M^2` (the
bullet, not the barbell) is the safest. References: F. M. Redington,
"Review of the Principles of Life-Office Valuations," Journal of the
Institute of Actuaries 78(3) (1952), 286-340; H. G. Fong and O. A.
Vasicek, "A Risk Minimizing Strategy for Portfolio Immunization," Journal
of Finance 39(5) (1984), 1541-1546.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from financialkit.fixed_income.core.base import CashFlows, CurveLike, ImmunizationResult, as_curve
from financialkit.fixed_income.systems.duration import fisher_weil_convexity, fisher_weil_duration

__all__ = ["redington_immunize", "m_squared", "fong_vasicek_bound", "horizon_value", "linear_forward_shock"]


def redington_immunize(liabilities: CashFlows, assets: list, curve: CurveLike) -> ImmunizationResult:
    """Hold two assets in the amounts that match the liabilities' present value and duration.

    Solves the 2x2 linear system
    :math:`\\sum_j h_j PV_j = PV_L` and :math:`\\sum_j h_j PV_j D_j = PV_L D_L`
    (Fisher-Weil durations on `curve`). Check
    :attr:`~financialkit.fixed_income.core.base.ImmunizationResult.redington_satisfied`
    for Redington's third condition, asset convexity above liability
    convexity; it holds when the assets' maturities straddle the liabilities'.

    Parameters
    ----------
    liabilities : CashFlows
    assets : list of two CashFlows
        Cash flows per unit of each asset.
    curve : YieldCurve or float

    Returns
    -------
    ImmunizationResult

    Examples
    --------
    >>> from financialkit.fixed_income.systems.bonds import bond_cash_flows
    >>> L = CashFlows([10.0], [1000.0])
    >>> res = redington_immunize(L, [bond_cash_flows(1.0, 0.0, 5.0), bond_cash_flows(1.0, 0.0, 20.0)], 0.05)
    >>> res.holdings.round(2).tolist(), res.redington_satisfied
    ([519.2, 549.57], True)
    """
    if len(assets) != 2:
        raise ValueError("Redington matching uses exactly two assets")
    curve = as_curve(curve)
    pv = np.array([a.present_value(curve) for a in assets])
    dur = np.array([fisher_weil_duration(a, curve) for a in assets])
    pv_l = liabilities.present_value(curve)
    d_l = fisher_weil_duration(liabilities, curve)
    A = np.array([pv, pv * dur])
    if abs(np.linalg.det(A)) < 1e-12 * np.abs(A).max() ** 2:
        raise ValueError("assets must have different durations")
    holdings = np.linalg.solve(A, [pv_l, pv_l * d_l])
    res = ImmunizationResult(holdings=holdings, assets=list(assets), liabilities=liabilities, curve=curve)
    port = res.portfolio
    res.pv_assets = port.present_value(curve)
    res.pv_liabilities = pv_l
    res.duration_assets = fisher_weil_duration(port, curve)
    res.duration_liabilities = d_l
    res.convexity_assets = fisher_weil_convexity(port, curve)
    res.convexity_liabilities = fisher_weil_convexity(liabilities, curve)
    return res


def m_squared(cash_flows: CashFlows, horizon: float, curve: CurveLike) -> float:
    r"""Fong and Vasicek's dispersion :math:`M^2 = \sum_i w_i (t_i - H)^2`, with present-value weights :math:`w_i`.

    Examples
    --------
    >>> m_squared(CashFlows([5.0], [1.0]), 5.0, 0.04)  # a zero maturing at the horizon
    0.0
    """
    disc = as_curve(curve).discount(cash_flows.times)
    w = cash_flows.amounts * disc
    w = w / w.sum()
    return float(np.sum(w * (cash_flows.times - horizon) ** 2))


def fong_vasicek_bound(m2: float, K: float) -> float:
    r"""Fong-Vasicek lower bound :math:`-\tfrac12 K M^2` on the relative change in horizon value.

    Holds for a portfolio whose Fisher-Weil duration equals the horizon
    when the shock to instantaneous forward rates has slope at most `K`.
    """
    return -0.5 * K * m2


def horizon_value(cash_flows: CashFlows, curve: CurveLike, horizon: float) -> float:
    """Value of `cash_flows` carried forward to `horizon`: present value divided by :math:`P(0, H)`."""
    curve = as_curve(curve)
    return float(cash_flows.present_value(curve) / curve.discount(horizon))


def linear_forward_shock(level: float, slope: float, horizon: float) -> Callable[[np.ndarray], np.ndarray]:
    r"""Zero-rate shift produced by the forward-rate shock :math:`\Delta f(s) = a + b (s - H)`.

    Integrating gives :math:`\Delta r(t) = a + b (t/2 - H)`; pass the
    result to :meth:`~financialkit.fixed_income.core.base.YieldCurve.shifted`.

    Examples
    --------
    >>> from financialkit.fixed_income.core.base import CashFlows, FlatCurve
    >>> cf = CashFlows([2.0, 8.0], [50.0, 50.0])
    >>> H = 5.0  # equal weights on 2 and 8 at a zero rate: duration exactly 5
    >>> base, shocked = FlatCurve(0.0), FlatCurve(0.0).shifted(linear_forward_shock(0.0, 0.002, H))
    >>> change = horizon_value(cf, shocked, H) / horizon_value(cf, base, H) - 1
    >>> bool(change >= fong_vasicek_bound(m_squared(cf, H, base), 0.002))
    True
    """
    return lambda t: level + slope * (np.asarray(t, dtype=np.float64) / 2.0 - horizon)
