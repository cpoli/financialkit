"""Counterparty exposure by simulation: regression mark-to-market on short-rate paths.

CVA, DVA and FVA integrate the expected positive and negative parts of a
portfolio's future value against default and funding curves, so they
need the value :math:`V(t)` on every simulated path at every exposure
date. A closed form at every node is rarely available. Instead, as in
Longstaff and Schwartz's (2001) American Monte Carlo, the realized cash
flows after :math:`t_i`, discounted to :math:`t_i` along each path, are
regressed on polynomials of the state at :math:`t_i`, and the fitted
conditional expectation is the value. One backward sweep gives the whole
exposure profile. Cesari et al. (2009) describe this as the standard
industrial exposure engine.

:func:`swap_exposure` applies it to an interest rate swap on the paths
of any one-factor short-rate model, such as
:class:`~financialkit.interest_rate_models.HullWhiteModel` fitted to a
discount curve. The state is the short rate plus the floating coupon
already fixed. Paths are discounted by the trapezoidal integral of the
short rate, so the average discount factor reproduces the model's
curve. ``method="exact"`` revalues the swap on the same paths with the
model's closed-form bond prices :math:`P(t, T; r_t)`, which isolates the
regression error.

References: F. A. Longstaff and E. S. Schwartz, "Valuing American Options
by Simulation: A Simple Least-Squares Approach," *Rev. Financial Stud.* 14
(2001) 113-147; G. Cesari, J. Aquilina, N. Charpillon, Z. Filipovic,
G. Lee and I. Manda, *Modelling, Pricing, and Hedging Counterparty Credit
Exposure* (Springer, 2009), Ch. 5; J. Gregory, *Counterparty Credit Risk*
(Wiley, 2010), Ch. 4.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import ArrayLike

from financialkit._validation import one_of, positive
from financialkit.credit.core.base import ExposureProfile
from financialkit.integrators import RNGLike
from financialkit.interest_rate_models import ShortRateModel

__all__ = ["regression_mark_to_market", "pathwise_discount", "swap_exposure"]


def _polynomial_basis(X: np.ndarray, degree: int) -> np.ndarray:
    """A constant and every monomial of total degree up to `degree` in the standardized columns of `X`."""
    std = X.std(axis=0)
    keep = std > 1e-12 * (1.0 + np.abs(X).max(axis=0))
    Z = (X[:, keep] - X[:, keep].mean(axis=0)) / std[keep]
    cols = [np.ones(X.shape[0])]
    for k in range(1, degree + 1):
        cols += [np.prod(Z[:, list(c)], axis=1) for c in combinations_with_replacement(range(Z.shape[1]), k)]
    return np.column_stack(cols)


def regression_mark_to_market(cash_flows: ArrayLike, discount: ArrayLike, states: ArrayLike, degree: int = 2) -> np.ndarray:
    """Mark-to-market paths by least-squares regression of discounted future cash flows.

    Working backward from the last date, the realized value after
    :math:`t_i` is

    .. math:: Y_i = \\frac{D(0, t_{i+1})}{D(0, t_i)} (C_{i+1} + Y_{i+1}),

    and the mark-to-market :math:`V(t_i) \\approx E[Y_i \\mid X_{t_i}]` is its
    least-squares fit on a polynomial in the state :math:`X_{t_i}`, as in
    Longstaff and Schwartz (2001). The regression sees the realized
    :math:`Y_{i+1}`, not the fitted value, so the errors of one date do
    not propagate to the next.

    Parameters
    ----------
    cash_flows : array_like, shape (n_paths, m)
        Cash flow :math:`C_i` received at :math:`t_i` on each path.
    discount : array_like, shape (n_paths, m)
        Pathwise discount factors :math:`D(0, t_i)`.
    states : array_like, shape (n_paths, m) or (n_paths, m, d)
        Regressors at each date; a constant state (as at :math:`t_0`) gives the mean.
    degree : int
        Total degree of the polynomial in the standardized states.

    Returns
    -------
    ndarray, shape (n_paths, m)
        :math:`V(t_i)` on each path, the value of the cash flows strictly after
        :math:`t_i`; the last column is zero.

    Examples
    --------
    A bond paying 1 at the last date, with a deterministic 5% rate:

    >>> t = np.linspace(0.0, 2.0, 5)
    >>> D = np.tile(np.exp(-0.05 * t), (100, 1))
    >>> C = np.zeros((100, 5)); C[:, -1] = 1.0
    >>> X = np.random.default_rng(0).normal(size=(100, 5))
    >>> V = regression_mark_to_market(C, D, X, degree=2)
    >>> bool(np.allclose(V[:, :-1], np.exp(-0.05 * (2.0 - t[:-1]))))
    True
    """
    C = np.asarray(cash_flows, dtype=np.float64)
    D = np.asarray(discount, dtype=np.float64)
    X = np.asarray(states, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, :, None]
    if C.ndim != 2 or D.shape != C.shape or X.shape[:2] != C.shape:
        raise ValueError("cash_flows, discount and states must share their (n_paths, m) shape")
    n_paths, m = C.shape
    V = np.zeros((n_paths, m))
    Y = np.zeros(n_paths)
    for i in range(m - 2, -1, -1):
        Y = D[:, i + 1] / D[:, i] * (C[:, i + 1] + Y)
        B = _polynomial_basis(X[:, i, :], int(degree))
        coef, *_ = np.linalg.lstsq(B, Y, rcond=None)
        V[:, i] = B @ coef
    return V


def pathwise_discount(times: ArrayLike, short_rates: ArrayLike) -> np.ndarray:
    """Discount factors :math:`e^{-\\int_0^t r\\,ds}` along short-rate paths, by the trapezoidal rule.

    Parameters
    ----------
    times : array_like, shape (m,)
    short_rates : array_like, shape (n_paths, m)

    Returns
    -------
    ndarray, shape (n_paths, m)

    Examples
    --------
    >>> D = pathwise_discount([0.0, 1.0, 2.0], [[0.03, 0.03, 0.03]])
    >>> bool(np.allclose(D, np.exp(-0.03 * np.array([0.0, 1.0, 2.0]))))
    True
    """
    t = np.asarray(times, dtype=np.float64)
    r = np.asarray(short_rates, dtype=np.float64)
    integral = np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * np.diff(t), axis=1)
    return np.exp(-np.concatenate([np.zeros((r.shape[0], 1)), integral], axis=1))


def swap_exposure(
    model: ShortRateModel,
    fixed_rate: float,
    maturity: float,
    freq: int = 2,
    notional: float = 1.0,
    payer: bool = True,
    steps_per_period: int = 4,
    n_paths: int = 5000,
    degree: int = 2,
    method: str = "regression",
    rng: RNGLike = None,
) -> ExposureProfile:
    """Exposure profile of a fixed-for-floating swap on a short-rate model's paths.

    The swap starts today and exchanges, at every :math:`T_j = j/\\text{freq}`,
    the floating coupon :math:`1/P(T_{j-1}, T_j) - 1` fixed at
    :math:`T_{j-1}` against :math:`K/\\text{freq}`, per unit notional. Short
    rates come from ``model.simulate`` on a grid with `steps_per_period`
    steps per period. With ``method="regression"``,
    :func:`regression_mark_to_market` values the swap from its realized
    cash flows, regressing on the short rate and the fixed coupon; with
    ``method="exact"`` the swap is revalued at every node as

    .. math:: V(t) = \\frac{P(t, T_k)}{P(T_{k-1}, T_k)} - P(t, T_n)
              - \\frac{K}{\\text{freq}} \\sum_{j \\ge k} P(t, T_j),

    with :math:`T_k` the next payment date and the model's closed-form
    :math:`P(t, T; r_t)`. Values are after the payment at :math:`t`, so at a
    payment date :math:`T_i` the discounted expected positive exposure of
    a payer swap is a payer swaption expiring at :math:`T_i`.

    Parameters
    ----------
    model : ShortRateModel
        A one-factor model with ``simulate`` and ``zero_coupon_bond(T, t, state)``,
        e.g. :class:`~financialkit.interest_rate_models.HullWhiteModel`.
    fixed_rate : float
        Fixed rate :math:`K`.
    maturity : float
        Swap maturity, a whole number of periods.
    freq : int
        Payments per year on both legs.
    notional : float
    payer : bool
        Pay fixed (receive floating) if True.
    steps_per_period : int
        Simulation and exposure dates per payment period.
    n_paths : int
    degree : int
        Total polynomial degree of the regression.
    method : {"regression", "exact"}
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ExposureProfile

    Examples
    --------
    Two years into a five-year payer swap, the discounted expected
    positive exposure is a 2y x 3y payer swaption:

    >>> from financialkit.interest_rate_models import HullWhiteModel, jamshidian_swaption
    >>> hw = HullWhiteModel(0.03, 0.1, 0.01)
    >>> prof = swap_exposure(hw, 0.03, 5.0, n_paths=4000, rng=1, method="exact")
    >>> i = 16  # t = 2.0: four steps per semiannual period
    >>> swaption = jamshidian_swaption(hw, 2.0, 3.0, 0.03, 2).price
    >>> bool(abs(prof.epe[i] - swaption) < 4 * prof.epe_std_error[i])
    True
    """
    one_of("method", method, ("regression", "exact"))
    positive(maturity=maturity, freq=freq, steps_per_period=steps_per_period, n_paths=n_paths)
    n_periods = int(round(maturity * freq))
    if n_periods < 1 or not np.isclose(n_periods, maturity * freq):
        raise ValueError("maturity must be a whole number of payment periods")
    spp = int(steps_per_period)
    times, r = model.simulate(n_periods / freq, n_periods * spp, int(n_paths), rng=rng)
    times = np.asarray(times, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    D = pathwise_discount(times, r)
    pay = times[::spp]
    tau = 1.0 / freq
    sign = 1.0 if payer else -1.0
    m = times.size
    # 1 / P(T_{j-1}, T_j) for every period j = 1..n, fixed at T_{j-1}
    resets = r[:, : n_periods * spp : spp]
    growth = 1.0 / np.column_stack([np.asarray(model.zero_coupon_bond(pay[j + 1], pay[j], resets[:, j]), dtype=np.float64) for j in range(n_periods)])
    # the coupon fixed at the last reset on or before t_i, paid at the next payment date
    period = np.minimum(np.arange(m) // spp, n_periods - 1)
    fixing = growth[:, period]
    if method == "regression":
        C = np.zeros_like(r)
        C[:, spp::spp] = growth - 1.0 - fixed_rate * tau
        # at a reset the new coupon is a function of r itself, so it is left out of the regressors there
        coupon = np.where(np.arange(m) % spp == 0, 0.0, fixing)
        V = regression_mark_to_market(C, D, np.stack([r, coupon], axis=2), degree)
    else:
        V = np.zeros_like(r)
        for i in range(m - 1):
            k = period[i] + 1
            bonds = np.column_stack([np.asarray(model.zero_coupon_bond(T, times[i], r[:, i]), dtype=np.float64) for T in pay[k:]])
            V[:, i] = bonds[:, 0] * fixing[:, i] - bonds[:, -1] - fixed_rate * tau * bonds.sum(axis=1)
    return ExposureProfile(times=times, values=sign * notional * V, discount=D, method=method)
