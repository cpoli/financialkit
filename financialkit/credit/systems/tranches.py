"""Synthetic CDO tranches in the one-factor Gaussian copula, without Monte Carlo.

A tranche with attachment :math:`A` and detachment :math:`D` absorbs
portfolio losses between those two fractions of the notional:

.. math:: L_{[A, D]} = \\frac{\\min(L, D) - \\min(L, A)}{D - A}.

Hull and White (2004) observed that, conditional on the common factor
:math:`M` of Li's copula, defaults are independent, so the loss
distribution of a homogeneous pool is binomial given :math:`M`, and
expected tranche losses at each payment date follow from one
integral over :math:`M` (Gauss-Hermite quadrature here). In the large
homogeneous pool limit the loss given :math:`M` is the deterministic
:math:`(1 - R)\\,p(M)`, and :math:`E[\\min(L, K)]` has a closed form in
the bivariate normal distribution function. Both legs then price like a CDS on the
tranche's outstanding notional. Raising correlation moves expected loss
from the equity tranche to the senior ones.

References: J. Hull and A. White, "Valuation of a CDO and an n-th to
Default CDS Without Monte Carlo Simulation," *J. Derivatives* 12 (2004)
8-23; D. O'Kane, *Modelling Single-name and Multi-name Credit
Derivatives* (Wiley, 2008), Ch. 12-16.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.credit.core.base import SurvivalCurve, TrancheResult
from financialkit.credit.systems.vasicek import conditional_default_probability
from financialkit.credit.utils.numerics import normal_factor_quadrature

__all__ = ["tranche_loss", "expected_tranche_loss", "tranche_spread"]


def tranche_loss(L: ArrayLike, attachment: float, detachment: float) -> np.ndarray | float:
    """Tranche loss as a fraction of tranche notional.

    Parameters
    ----------
    L : array_like
        Portfolio loss as a fraction of portfolio notional.
    attachment, detachment : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> tranche_loss([0.01, 0.05, 0.2], 0.03, 0.07).tolist()
    [0.0, 0.5, 1.0]
    """
    if not 0.0 <= attachment < detachment <= 1.0:
        raise ValueError("need 0 <= attachment < detachment <= 1")
    L = np.asarray(L, dtype=np.float64)
    out = (np.minimum(L, detachment) - np.minimum(L, attachment)) / (detachment - attachment)
    return float(out) if out.ndim == 0 else out


def expected_tranche_loss(
    attachment: float, detachment: float, p: float, rho: float, recovery: float = 0.4, n_names: int | None = None, n_nodes: int = 96
) -> float:
    """Expected tranche loss (fraction of tranche notional) at one horizon.

    Parameters
    ----------
    attachment, detachment : float
    p : float
        Default probability of each name to the horizon.
    rho : float
        Copula correlation.
    recovery : float
    n_names : int, optional
        Pool size; the large homogeneous pool limit if omitted.
    n_nodes : int
        Gauss-Hermite nodes for a finite pool.

    Returns
    -------
    float

    Examples
    --------
    A [0, 100%] tranche loses the pool's expected loss whatever the correlation:

    >>> round(expected_tranche_loss(0.0, 1.0, 0.05, 0.3, 0.4), 10)
    0.03
    """
    if p <= 0:
        return 0.0
    lgd = 1.0 - recovery
    if n_names is not None:
        x, w = normal_factor_quadrature(n_nodes)
        pM = np.asarray(conditional_default_probability(p, rho, x))
        k = np.arange(n_names + 1)
        tl = np.asarray(tranche_loss(lgd * k / n_names, attachment, detachment))
        return float(w @ (stats.binom.pmf(k[None, :], n_names, pM[:, None]) @ tl))
    if rho <= 0:
        return float(tranche_loss(lgd * p, attachment, detachment))
    zp = float(stats.norm.ppf(p))
    sr = float(np.sqrt(rho))
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, sr], [sr, 1.0]])

    def capped(K):
        # E[min(L, K)] for L = (1 - R) p(M). With m* the factor value where L = K,
        # E[p(M) 1{M > m*}] = P(Z < N^-1(p), M > m*) = p - Phi2(N^-1(p), m*; sqrt(rho)).
        if K <= 0:
            return 0.0
        if K >= lgd:
            return lgd * p
        m = float((zp - np.sqrt(1.0 - rho) * stats.norm.ppf(K / lgd)) / sr)
        return lgd * (p - float(mvn.cdf([zp, m]))) + K * float(stats.norm.cdf(m))

    return (capped(detachment) - capped(attachment)) / (detachment - attachment)


def tranche_spread(
    attachment: float,
    detachment: float,
    curve: SurvivalCurve,
    rho: float,
    maturity: float,
    r: float,
    recovery: float = 0.4,
    frequency: int = 4,
    n_names: int | None = None,
) -> TrancheResult:
    """Par spread of a synthetic CDO tranche (Hull and White 2004).

    Premiums accrue on the expected outstanding tranche notional,
    averaged over each period, and protection pays the increase in
    expected tranche loss, discounted from mid-period.

    Parameters
    ----------
    attachment, detachment : float
    curve : SurvivalCurve
        Survival curve shared by every name in the pool.
    rho : float
        Copula correlation.
    maturity : float
    r : float
        Flat risk-free rate.
    recovery : float
    frequency : int
        Premium payments per year.
    n_names : int, optional
        Pool size; the large homogeneous pool if omitted.

    Returns
    -------
    TrancheResult
    """
    n = max(int(np.ceil(maturity * frequency - 1e-9)), 1)
    t = np.linspace(0.0, maturity, n + 1)
    p = np.asarray(curve.default_probability(t), dtype=np.float64)
    el = np.array([expected_tranche_loss(attachment, detachment, float(pi), rho, recovery, n_names) for pi in p])
    dt = np.diff(t)
    z_end = np.exp(-r * t[1:])
    z_mid = np.exp(-r * 0.5 * (t[1:] + t[:-1]))
    rpv01 = float(np.sum(dt * z_end * (1.0 - 0.5 * (el[1:] + el[:-1]))))
    protection = float(np.sum(z_mid * np.diff(el)))
    return TrancheResult(
        attachment=attachment,
        detachment=detachment,
        par_spread=protection / rpv01,
        expected_loss=el[1:],
        times=t[1:],
        protection_leg=protection,
        rpv01=rpv01,
    )
