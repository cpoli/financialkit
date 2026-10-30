"""Valuation adjustments from exposure profiles: CVA, DVA and FVA.

Each adjustment integrates a discounted exposure from
:class:`~financialkit.credit.core.base.ExposureProfile` against a default
or funding curve. Rates and defaults are taken independent (no
wrong-way risk), so the integrals separate. On the profile's dates
:math:`t_0 < \\dots < t_m`, a default in :math:`(t_{i-1}, t_i]` is settled
on the exposure at :math:`t_i`, Brigo and Masetti's discretization.

* Unilateral CVA (Brigo and Masetti 2005): the expected loss from the
  counterparty's default, :math:`(1 - R_C) \\sum_i EPE(t_i)\\,
  [Q_C(t_{i-1}) - Q_C(t_i)]`. For a swap the discounted positive
  exposure at a payment date is a swaption, so CVA is a strip of
  swaptions (:func:`swap_cva_swaptions`).
* Bilateral CVA and DVA (Gregory 2009): our own default is a gain on
  the negative exposure. Counting each default only if the other party
  survived it (first to default) makes the two parties agree on the
  price: one side's DVA is the other's CVA.
* FVA (Burgard and Kjaer 2011): uncollateralised positive exposure must
  be funded at a borrowing spread over the risk-free rate (cost, FCA),
  and negative exposure can be lent at a lending spread (benefit, FBA),
  weighted by both parties' survival. When the funding spread is our
  own credit spread :math:`(1 - R_B)\\lambda_B`, the benefit equals the
  DVA, which is why DVA and FBA must not both be counted.

References: D. Brigo and M. Masetti, "Risk Neutral Pricing of
Counterparty Risk," in M. Pykhtin (ed.), *Counterparty Credit Risk
Modelling* (Risk Books, 2005); J. Gregory, "Being Two-Faced over
Counterparty Credit Risk," *Risk* 22(2) (2009) 86-90; C. Burgard and
M. Kjaer, "Partial Differential Equation Representations of Derivatives
with Bilateral Counterparty Risk and Funding Costs," *J. Credit Risk*
7(3) (2011) 75-93; J. Gregory, *Counterparty Credit Risk and Credit
Value Adjustment*, 2nd ed. (Wiley, 2012), Ch. 12-14.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from financialkit.credit.core.base import BilateralCVAResult, ExposureProfile, FVAResult, SurvivalCurve
from financialkit.interest_rate_models import GaussianShortRateModel, jamshidian_swaption

__all__ = ["unilateral_cva", "swap_cva_swaptions", "bilateral_cva", "funding_valuation_adjustment"]


def _default_buckets(curve: SurvivalCurve, t: np.ndarray) -> np.ndarray:
    """:math:`Q(t_{i-1}) - Q(t_i)` for every bucket of the dates `t`."""
    return -np.diff(np.asarray(curve.survival(t), dtype=np.float64))


def _survival(curve: SurvivalCurve | None, t: np.ndarray) -> np.ndarray:
    return np.ones_like(t) if curve is None else np.asarray(curve.survival(t), dtype=np.float64)


def _check_recovery(**recoveries: float) -> None:
    for name, R in recoveries.items():
        if not 0.0 <= R <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {R}")


def unilateral_cva(profile: ExposureProfile, counterparty: SurvivalCurve, recovery: float = 0.4) -> float:
    """Unilateral credit valuation adjustment (Brigo and Masetti 2005).

    .. math:: \\text{CVA} = (1 - R) \\sum_{i=1}^{m} E[D(0, t_i) V(t_i)^+]\\,[Q_C(t_{i-1}) - Q_C(t_i)],

    the expected discounted loss if the counterparty defaults, rates and
    default independent.

    Parameters
    ----------
    profile : ExposureProfile
    counterparty : SurvivalCurve
        The counterparty's risk-neutral survival curve.
    recovery : float
        Recovery as a fraction of the exposure.

    Returns
    -------
    float
        CVA, a positive number subtracted from the risk-free value.

    Examples
    --------
    A constant exposure of 1 loses :math:`(1 - R)` times the default probability:

    >>> from financialkit.credit import ExposureProfile, HazardCurve
    >>> t = np.linspace(0.0, 5.0, 11)
    >>> prof = ExposureProfile(t, np.ones((2, 11)), np.ones((2, 11)), "exact")
    >>> curve = HazardCurve.flat(0.02)
    >>> bool(np.isclose(unilateral_cva(prof, curve), 0.6 * curve.default_probability(5.0)))
    True
    """
    _check_recovery(recovery=recovery)
    return float((1.0 - recovery) * np.sum(profile.epe[1:] * _default_buckets(counterparty, profile.times)))


def swap_cva_swaptions(
    model: GaussianShortRateModel,
    fixed_rate: float,
    maturity: float,
    counterparty: SurvivalCurve,
    recovery: float = 0.4,
    freq: int = 2,
    notional: float = 1.0,
    payer: bool = True,
) -> float:
    """Unilateral CVA of a swap as a strip of swaptions (Brigo and Masetti 2005).

    After the payment at :math:`T_i`, a payer swap is worth the remaining
    swap, so its discounted positive exposure there is a payer swaption
    expiring at :math:`T_i` on the swap to :math:`T_n`. With defaults in
    :math:`(T_{i-1}, T_i]` settled at :math:`T_i`,

    .. math:: \\text{CVA} = (1 - R) \\sum_{i=1}^{n-1} [Q_C(T_{i-1}) - Q_C(T_i)]\\,
              \\text{Swaption}(T_i, T_n; K),

    each swaption priced in closed form by
    :func:`~financialkit.interest_rate_models.jamshidian_swaption`.

    Parameters
    ----------
    model : GaussianShortRateModel
        One-factor model, e.g. :class:`~financialkit.interest_rate_models.HullWhiteModel`.
    fixed_rate : float
    maturity : float
        A whole number of periods.
    counterparty : SurvivalCurve
    recovery : float
    freq : int
        Payments per year on both legs.
    notional : float
    payer : bool
        Pay fixed (exposure is a payer swaption) or receive fixed.

    Returns
    -------
    float

    Examples
    --------
    >>> from financialkit.credit import HazardCurve
    >>> from financialkit.interest_rate_models import HullWhiteModel
    >>> cva = swap_cva_swaptions(HullWhiteModel(0.03, 0.1, 0.01), 0.03, 5.0, HazardCurve.flat(0.02))
    >>> bool(0.0 < cva < 1e-3)
    True
    """
    _check_recovery(recovery=recovery)
    n = int(round(maturity * freq))
    if n < 1 or not np.isclose(n, maturity * freq):
        raise ValueError("maturity must be a whole number of payment periods")
    T = np.arange(n + 1) / freq
    swaptions = np.array([jamshidian_swaption(model, T[i], T[n] - T[i], fixed_rate, freq, payer, notional).price for i in range(1, n)])
    return float((1.0 - recovery) * np.sum(_default_buckets(counterparty, T)[:-1] * swaptions))


def bilateral_cva(
    profile: ExposureProfile,
    counterparty: SurvivalCurve,
    own: SurvivalCurve,
    recovery_counterparty: float = 0.4,
    recovery_own: float = 0.4,
    first_to_default: bool = True,
) -> BilateralCVAResult:
    """Bilateral CVA and DVA (Gregory 2009).

    .. math:: \\text{CVA} = (1 - R_C) \\sum_i EPE(t_i)\\,Q_B(t_i)\\,[Q_C(t_{i-1}) - Q_C(t_i)],

    .. math:: \\text{DVA} = (1 - R_B) \\sum_i ENE(t_i)\\,Q_C(t_i)\\,[Q_B(t_{i-1}) - Q_B(t_i)],

    with :math:`Q_B` our own survival and the two defaults independent.
    The survival factors count a default only if the other party has not
    defaulted first; without them (``first_to_default=False``) both
    defaults are counted and the adjustment is overstated. With them,
    the counterparty's DVA on the same trade equals our CVA, so both
    sides agree on the price.

    Parameters
    ----------
    profile : ExposureProfile
        Values from our side.
    counterparty, own : SurvivalCurve
    recovery_counterparty, recovery_own : float
    first_to_default : bool

    Returns
    -------
    BilateralCVAResult

    Examples
    --------
    >>> from financialkit.credit import ExposureProfile, HazardCurve
    >>> t = np.linspace(0.0, 5.0, 11)
    >>> V = np.array([np.ones(11), -np.ones(11)])
    >>> prof = ExposureProfile(t, V, np.ones((2, 11)), "exact")
    >>> res = bilateral_cva(prof, HazardCurve.flat(0.02), HazardCurve.flat(0.02))
    >>> bool(np.isclose(res.cva, res.dva)), res.bva == res.dva - res.cva
    (True, True)
    """
    _check_recovery(recovery_counterparty=recovery_counterparty, recovery_own=recovery_own)
    t = profile.times
    survive_own = _survival(own, t[1:]) if first_to_default else 1.0
    survive_cpty = _survival(counterparty, t[1:]) if first_to_default else 1.0
    cva = (1.0 - recovery_counterparty) * np.sum(profile.epe[1:] * survive_own * _default_buckets(counterparty, t))
    dva = (1.0 - recovery_own) * np.sum(profile.ene[1:] * survive_cpty * _default_buckets(own, t))
    return BilateralCVAResult(cva=float(cva), dva=float(dva), first_to_default=first_to_default)


def funding_valuation_adjustment(
    profile: ExposureProfile,
    borrowing_spread: ArrayLike,
    lending_spread: ArrayLike | None = None,
    counterparty: SurvivalCurve | None = None,
    own: SurvivalCurve | None = None,
) -> FVAResult:
    """Funding cost and benefit adjustments (Burgard and Kjaer 2011).

    .. math:: \\text{FCA} = \\sum_i s_b(t_i)\\,EPE(t_i)\\,Q_B(t_i) Q_C(t_i)\\,\\Delta t_i,
              \\qquad
              \\text{FBA} = \\sum_i s_l(t_i)\\,ENE(t_i)\\,Q_B(t_i) Q_C(t_i)\\,\\Delta t_i.

    An uncollateralised positive value must be financed at the bank's
    borrowing spread :math:`s_b` over the risk-free rate, and a negative
    one earns the lending spread :math:`s_l`, for as long as neither
    party has defaulted. In Burgard and Kjaer's semi-replication, funding
    with the bank's own bonds makes :math:`s_b = (1 - R_B)\\lambda_B`, and
    then a symmetric FBA is the DVA.

    Parameters
    ----------
    profile : ExposureProfile
    borrowing_spread : array_like
        Spread over the risk-free rate paid on funding, a number or one per profile date.
    lending_spread : array_like, optional
        Spread earned on surplus cash; the borrowing spread if omitted.
    counterparty, own : SurvivalCurve, optional
        Survival curves that stop the funding at either default; none if omitted.

    Returns
    -------
    FVAResult

    Examples
    --------
    A constant exposure of 1 funded at 100 bp for 5 years, nobody defaulting:

    >>> from financialkit.credit import ExposureProfile
    >>> t = np.linspace(0.0, 5.0, 11)
    >>> prof = ExposureProfile(t, np.ones((2, 11)), np.ones((2, 11)), "exact")
    >>> res = funding_valuation_adjustment(prof, 0.01)
    >>> round(res.fca, 10), res.fba
    (0.05, 0.0)
    """
    t = profile.times
    s_b = np.broadcast_to(np.asarray(borrowing_spread, dtype=np.float64), t.shape)
    s_l = s_b if lending_spread is None else np.broadcast_to(np.asarray(lending_spread, dtype=np.float64), t.shape)
    weight = _survival(counterparty, t[1:]) * _survival(own, t[1:]) * np.diff(t)
    fca = np.sum(s_b[1:] * profile.epe[1:] * weight)
    fba = np.sum(s_l[1:] * profile.ene[1:] * weight)
    return FVAResult(fca=float(fca), fba=float(fba))
