"""Leland's endogenous default: capital structure with taxes and bankruptcy costs.

Leland (1994) priced perpetual debt paying a coupon :math:`C` on a firm
whose unlevered assets follow geometric Brownian motion. Coupons are tax
deductible at rate :math:`\\tau` and bankruptcy destroys a fraction
:math:`\\alpha` of the assets. Shareholders default when equity hits
zero, and choosing that point to maximise equity (smooth pasting,
:math:`E'(V_B) = 0`) gives, with :math:`X = 2r/\\sigma^2`,

.. math:: V_B = \\frac{(1 - \\tau) C}{r + \\sigma^2/2}, \\qquad p_B = (V/V_B)^{-X},

.. math:: D = \\frac{C}{r} + \\left[(1 - \\alpha) V_B - \\frac{C}{r}\\right] p_B, \\qquad
          v = V + \\frac{\\tau C}{r}(1 - p_B) - \\alpha V_B p_B.

The coupon that maximises firm value :math:`v` trades the tax shield
against expected bankruptcy costs and fixes the optimal leverage.
The model is stationary (perpetual debt, time-independent claims), so
the riskless rate is a single flat ``r`` by construction; unlike the
other credit pricers it takes no yield curve.

References: H. E. Leland, "Corporate Debt Value, Bond Covenants, and
Optimal Capital Structure," *J. Finance* 49 (1994) 1213-1252.
"""

from __future__ import annotations

from scipy import optimize

from financialkit.credit.core.base import LelandResult

__all__ = ["leland_model", "leland_optimal_coupon"]


def leland_model(
    V: float, coupon: float, r: float, sigma: float, tax_rate: float = 0.35, bankruptcy_cost: float = 0.5, default_barrier: float | None = None
) -> LelandResult:
    """Debt, equity and firm value in Leland's (1994) model.

    Parameters
    ----------
    V : float
        Unlevered asset value.
    coupon : float
        Perpetual coupon per year.
    r : float
        Risk-free rate.
    sigma : float
        Asset volatility.
    tax_rate : float
        Corporate tax rate :math:`\\tau`.
    bankruptcy_cost : float
        Fraction :math:`\\alpha` of assets lost in bankruptcy.
    default_barrier : float, optional
        Exogenous barrier; the endogenous :math:`V_B` if omitted.

    Returns
    -------
    LelandResult

    Examples
    --------
    At the endogenous barrier equity is exactly worthless:

    >>> res = leland_model(100.0, 5.0, 0.06, 0.2)
    >>> round(res.default_barrier, 4)
    40.625
    >>> leland_model(res.default_barrier, 5.0, 0.06, 0.2).equity < 1e-12
    True
    """
    if V <= 0 or r <= 0 or sigma <= 0 or coupon < 0:
        raise ValueError("V, r and sigma must be positive and the coupon non-negative")
    X = 2.0 * r / sigma**2
    VB = (1.0 - tax_rate) * coupon / (r + 0.5 * sigma**2) if default_barrier is None else float(default_barrier)
    pB = min((V / VB) ** -X, 1.0) if VB > 0 else 0.0
    debt = coupon / r + ((1.0 - bankruptcy_cost) * VB - coupon / r) * pB
    firm = V + tax_rate * coupon / r * (1.0 - pB) - bankruptcy_cost * VB * pB
    equity = firm - debt
    return LelandResult(
        default_barrier=float(VB),
        debt=float(debt),
        equity=float(equity),
        firm_value=float(firm),
        leverage=float(debt / firm) if firm > 0 else float("nan"),
        credit_spread=float(coupon / debt - r) if debt > 0 else float("nan"),
        default_probability_weight=float(pB),
    )


def leland_optimal_coupon(V: float, r: float, sigma: float, tax_rate: float = 0.35, bankruptcy_cost: float = 0.5) -> tuple[float, LelandResult]:
    """The coupon that maximises firm value, by :func:`scipy.optimize.minimize_scalar`.

    Parameters
    ----------
    V, r, sigma, tax_rate, bankruptcy_cost : float
        As in :func:`leland_model`.

    Returns
    -------
    coupon : float
    result : LelandResult
        The firm at the optimal coupon.
    """
    hi = 2.0 * r * V / (1.0 - tax_rate)
    sol = optimize.minimize_scalar(
        lambda c: -leland_model(V, c, r, sigma, tax_rate, bankruptcy_cost).firm_value, bounds=(1e-9 * V, hi), method="bounded", options={"xatol": 1e-10 * V}
    )
    return float(sol.x), leland_model(V, float(sol.x), r, sigma, tax_rate, bankruptcy_cost)
