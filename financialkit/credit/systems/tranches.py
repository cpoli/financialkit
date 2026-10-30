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

The market quoted tranches by the flat correlation that reprices each
one, its *compound* correlation. A mezzanine tranche's spread is not
monotonic in correlation, so some quotes have two compound correlations
and some have none. McGinty, Beinstein, Ahluwalia and Watts (2004)
quoted instead the *base* correlations of the equity tranches
:math:`[0, K]`, whose expected loss falls monotonically with
correlation. A tranche :math:`[A, D]` is the difference of two of them,

.. math:: D\\,L_{[0, D]}(\\rho_D) - A\\,L_{[0, A]}(\\rho_A) = (D - A)\\,L_{[A, D]},

so the skew :math:`\\rho(K)` is bootstrapped from the most junior
tranche up, one root-find each, and bespoke tranches are priced by
interpolating it.

References: J. Hull and A. White, "Valuation of a CDO and an n-th to
Default CDS Without Monte Carlo Simulation," *J. Derivatives* 12 (2004)
8-23; L. McGinty, E. Beinstein, R. Ahluwalia and M. Watts, *Introducing
Base Correlations* (J.P. Morgan Credit Derivatives Strategy, 2004);
D. O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*
(Wiley, 2008), Ch. 12-17.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import brentq

from financialkit._deprecation import renamed_parameter
from financialkit.credit.core.base import BaseCorrelationResult, SurvivalCurve, TrancheResult
from financialkit.credit.systems.vasicek import conditional_default_probability
from financialkit.credit.utils.discounting import discount_factors
from financialkit.credit.utils.numerics import normal_factor_quadrature
from financialkit.fixed_income import CurveLike

__all__ = ["tranche_loss", "expected_tranche_loss", "tranche_spread", "base_correlation_spread", "base_correlation", "compound_correlations"]

_RHO_MIN, _RHO_MAX = 1e-6, 0.999


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
        # scipy's binomial overflows for p near the smallest normal double; below 1e-300 it is a point mass at 0
        pM[pM < 1e-300] = 0.0
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


@renamed_parameter("frequency", "freq", "financialkit.credit.tranche_spread")
def tranche_spread(
    attachment: float,
    detachment: float,
    curve: SurvivalCurve,
    rho: float,
    maturity: float,
    r: CurveLike,
    recovery: float = 0.4,
    freq: int = 4,
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
    r : float or YieldCurve
        Continuously compounded risk-free rate, or a ``fixed_income`` discount curve.
    recovery : float
    freq : int
        Premium payments per year.
    n_names : int, optional
        Pool size; the large homogeneous pool if omitted.

    Returns
    -------
    TrancheResult
    """
    t, p = _payment_grid(curve, maturity, freq)
    el = np.array([expected_tranche_loss(attachment, detachment, float(pi), rho, recovery, n_names) for pi in p])
    return _tranche_result(attachment, detachment, t, el, r)


def _payment_grid(curve: SurvivalCurve, maturity: float, freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Premium dates from 0 to `maturity` and each name's default probability at them."""
    n = max(int(np.ceil(maturity * freq - 1e-9)), 1)
    t = np.linspace(0.0, maturity, n + 1)
    return t, np.asarray(curve.default_probability(t), dtype=np.float64)


def _tranche_result(attachment: float, detachment: float, t: np.ndarray, el: np.ndarray, r: CurveLike) -> TrancheResult:
    """Both legs from the expected tranche loss `el` at the dates `t`."""
    dt = np.diff(t)
    z_end = discount_factors(r, t[1:])
    z_mid = discount_factors(r, 0.5 * (t[1:] + t[:-1]))
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


def _equity_loss(K: float, p: np.ndarray, rho: float, recovery: float, n_names: int | None) -> np.ndarray:
    """Expected loss of the equity tranche ``[0, K]`` at each probability in `p`, as a fraction of the pool."""
    if K <= 0:
        return np.zeros_like(p)
    return K * np.array([expected_tranche_loss(0.0, K, float(pi), rho, recovery, n_names) for pi in p])


@renamed_parameter("frequency", "freq", "financialkit.credit.base_correlation_spread")
def base_correlation_spread(
    attachment: float,
    detachment: float,
    rho_attachment: float,
    rho_detachment: float,
    curve: SurvivalCurve,
    maturity: float,
    r: CurveLike,
    recovery: float = 0.4,
    freq: int = 4,
    n_names: int | None = None,
) -> TrancheResult:
    """Par spread of the tranche ``[attachment, detachment]`` from two base correlations.

    The tranche's expected loss is the difference of two equity tranches,
    each in the one-factor Gaussian copula at its own correlation:
    :math:`[D L_{[0, D]}(\\rho_D) - A L_{[0, A]}(\\rho_A)] / (D - A)`. With
    ``rho_attachment == rho_detachment`` this is :func:`tranche_spread`.

    Parameters
    ----------
    attachment, detachment : float
    rho_attachment, rho_detachment : float
        Base correlations at the attachment and detachment points.
    curve : SurvivalCurve
        Survival curve shared by every name in the pool.
    maturity : float
    r : float or YieldCurve
        Continuously compounded risk-free rate, or a ``fixed_income`` discount curve.
    recovery : float
    freq : int
        Premium payments per year.
    n_names : int, optional
        Pool size; the large homogeneous pool if omitted.

    Returns
    -------
    TrancheResult

    Examples
    --------
    >>> from financialkit.credit import HazardCurve
    >>> curve = HazardCurve.flat(0.01)
    >>> a = base_correlation_spread(0.03, 0.07, 0.3, 0.3, curve, 5.0, 0.03).par_spread
    >>> bool(np.isclose(a, tranche_spread(0.03, 0.07, curve, 0.3, 5.0, 0.03).par_spread))
    True
    """
    if not 0.0 <= attachment < detachment <= 1.0:
        raise ValueError("need 0 <= attachment < detachment <= 1")
    t, p = _payment_grid(curve, maturity, freq)
    upper = _equity_loss(detachment, p, rho_detachment, recovery, n_names)
    lower = _equity_loss(attachment, p, rho_attachment, recovery, n_names)
    return _tranche_result(attachment, detachment, t, (upper - lower) / (detachment - attachment), r)


@renamed_parameter("frequency", "freq", "financialkit.credit.base_correlation")
def base_correlation(
    detachments: ArrayLike,
    spreads: ArrayLike,
    curve: SurvivalCurve,
    maturity: float,
    r: CurveLike,
    recovery: float = 0.4,
    freq: int = 4,
    n_names: int | None = None,
) -> BaseCorrelationResult:
    """Bootstrap the base correlation skew from a capital structure of tranche quotes.

    The tranches are ``[0, K_1], [K_1, K_2], ...``, each quoted as a running
    par spread. The first quote's flat correlation is the first base
    correlation; each later one solves (:func:`scipy.optimize.brentq`) for
    the :math:`\\rho_k` at which :func:`base_correlation_spread` with the
    already known :math:`\\rho_{k-1}` matches the quote. Because an equity
    tranche's expected loss falls with correlation, each root is unique.

    Parameters
    ----------
    detachments : array_like, shape (k,)
        Increasing detachment points.
    spreads : array_like, shape (k,)
        Par spread of each tranche ``[K_{i-1}, K_i]``. The equity tranche is
        quoted as a running spread here, not as the market's upfront plus
        500 bp.
    curve : SurvivalCurve
        Survival curve shared by every name in the pool, e.g. from the index spread.
    maturity : float
    r : float or YieldCurve
    recovery : float
    freq : int
    n_names : int, optional
        Pool size; the large homogeneous pool if omitted.

    Returns
    -------
    BaseCorrelationResult

    Raises
    ------
    ValueError
        If no correlation in ``(0, 1)`` reprices a quote.

    Examples
    --------
    Quotes from one flat correlation give a flat skew:

    >>> from financialkit.credit import HazardCurve
    >>> curve = HazardCurve.flat(0.01)
    >>> K = [0.03, 0.07, 0.10]
    >>> s = [tranche_spread(a, d, curve, 0.25, 5.0, 0.03).par_spread for a, d in zip([0.0] + K[:-1], K)]
    >>> base_correlation(K, s, curve, 5.0, 0.03).base_correlations.round(6).tolist()
    [0.25, 0.25, 0.25]
    """
    K = np.asarray(detachments, dtype=np.float64)
    s = np.asarray(spreads, dtype=np.float64)
    if K.ndim != 1 or K.shape != s.shape:
        raise ValueError("detachments and spreads must be 1-d arrays of equal length")
    if np.any(np.diff(np.concatenate([[0.0], K])) <= 0) or K[-1] > 1.0:
        raise ValueError("detachments must increase within (0, 1]")
    t, p = _payment_grid(curve, maturity, freq)
    rhos = np.empty_like(K)
    lower, A = np.zeros_like(p), 0.0
    for i, (D, quote) in enumerate(zip(K, s, strict=True)):

        def mtm(rho, D=D, quote=quote, lower=lower, A=A):
            res = _tranche_result(A, D, t, (_equity_loss(D, p, rho, recovery, n_names) - lower) / (D - A), r)
            return res.protection_leg - quote * res.rpv01

        lo, hi = mtm(_RHO_MIN), mtm(_RHO_MAX)
        if lo * hi > 0:
            raise ValueError(f"no base correlation reprices the tranche [{A:g}, {D:g}] at spread {quote:g}")
        rhos[i] = brentq(mtm, _RHO_MIN, _RHO_MAX, xtol=1e-12)
        lower, A = _equity_loss(D, p, rhos[i], recovery, n_names), D
    return BaseCorrelationResult(detachments=K, base_correlations=rhos, spreads=s)


@renamed_parameter("frequency", "freq", "financialkit.credit.compound_correlations")
def compound_correlations(
    spread: float,
    attachment: float,
    detachment: float,
    curve: SurvivalCurve,
    maturity: float,
    r: CurveLike,
    recovery: float = 0.4,
    freq: int = 4,
    n_names: int | None = None,
    n_grid: int = 40,
) -> np.ndarray:
    """Every flat (compound) correlation at which one tranche prices at `spread`.

    A mezzanine tranche's par spread rises and then falls with
    correlation, so a quote can have zero, one or two compound
    correlations; this scans `n_grid` correlations for sign changes and
    refines each with :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    spread : float
        Quoted par spread.
    attachment, detachment : float
    curve : SurvivalCurve
    maturity : float
    r : float or YieldCurve
    recovery : float
    freq : int
    n_names : int, optional
    n_grid : int
        Number of correlations scanned in ``(0, 1)``.

    Returns
    -------
    ndarray
        The roots in increasing order; empty when none exists.

    Examples
    --------
    >>> from financialkit.credit import HazardCurve
    >>> curve = HazardCurve.flat(0.01)
    >>> s = tranche_spread(0.0, 0.03, curve, 0.2, 5.0, 0.03).par_spread
    >>> compound_correlations(s, 0.0, 0.03, curve, 5.0, 0.03).round(6).tolist()
    [0.2]
    """
    t, p = _payment_grid(curve, maturity, freq)

    def mtm(rho):
        el = np.array([expected_tranche_loss(attachment, detachment, float(pi), rho, recovery, n_names) for pi in p])
        res = _tranche_result(attachment, detachment, t, el, r)
        return res.protection_leg - spread * res.rpv01

    grid = np.linspace(_RHO_MIN, _RHO_MAX, n_grid)
    values = np.array([mtm(rho) for rho in grid])
    roots = [brentq(mtm, a, b, xtol=1e-12) for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True) if fa * fb < 0]
    roots += [float(g) for g, v in zip(grid, values, strict=True) if v == 0.0]
    return np.array(sorted(roots))
