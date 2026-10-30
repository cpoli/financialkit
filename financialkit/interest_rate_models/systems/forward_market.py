"""Lyashenko and Mercurio's (2019) forward market model for backward-looking rates.

LIBOR was a forward-looking term rate, fixed at the start of its accrual
period. Its replacements (SOFR, SONIA, €STR) are overnight rates, and the
term rate paid at the end of a period is the overnight rate compounded in
arrears over it,

.. math:: R(T_{j-1}, T_j) = \\frac{1}{\\tau_j}\\left(e^{\\int_{T_{j-1}}^{T_j} r(s)\\,ds} - 1\\right),

known only at :math:`T_j`. Lyashenko and Mercurio extended the forward
rate to the whole of :math:`[0, T_j]`,

.. math:: R_j(t) = \\frac{1}{\\tau_j}\\left(\\frac{P(t, T_{j-1})}{P(t, T_j)} - 1\\right),

with the bond that matured at :math:`T_{j-1}` rolled at the short rate
afterwards. It is the forward-looking rate before :math:`T_{j-1}`, it
keeps moving inside the accrual period as overnight fixings arrive, and
it ends at :math:`R_j(T_j) = R(T_{j-1}, T_j)`. It is a martingale under
the :math:`T_j`-forward measure throughout, so one model prices LIBOR-style
and in-arrears products. Its volatility decays to zero over the accrual
period, :math:`dR_j = \\sigma_j g_j(t)\\,R_j\\,dW` (or without the
:math:`R_j` in the normal version), and with the linear decay

.. math:: g_j(t) = \\min\\left(1, \\frac{(T_j - t)^+}{\\tau_j}\\right)

a caplet on the compounded rate is Black's (or Bachelier's) formula with
total variance :math:`\\sigma_j^2 (T_{j-1} + \\tau_j/3)`.

* :func:`fmm_volatility_decay`: :math:`g_j(t)`.
* :func:`fmm_caplet_price`: caplets and floorlets on backward-looking
  (or forward-looking) rates, lognormal or normal.
* :func:`simulate_fmm_rate`: exact paths of :math:`R_j(t)` under the
  :math:`T_j`-forward measure, on increments from
  :func:`financialkit.integrators.brownian_increments`.

References: A. Lyashenko and F. Mercurio, "Looking Forward to
Backward-Looking Rates: A Modeling Framework for Term Rates Replacing
LIBOR," SSRN 3330240 (2019); A. Lyashenko and F. Mercurio, "Looking
Forward to Backward-Looking Rates: Completing the Generalized Forward
Market Model," SSRN 3482132 (2019).
"""

from __future__ import annotations

import numpy as np

from financialkit._validation import nonnegative
from financialkit.fixed_income.core.base import CurveLike, as_curve
from financialkit.integrators import RNGLike, brownian_increments
from financialkit.options import bachelier_price, black76_price
from financialkit.options.utils.common import as_output

__all__ = ["fmm_volatility_decay", "fmm_caplet_price", "simulate_fmm_rate"]


def _check_period(start, end):
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    if np.any(start < 0) or np.any(end <= start):
        raise ValueError("need 0 <= start < end")
    return start, end


def fmm_volatility_decay(t, start, end):
    r"""Linear volatility decay :math:`g_j(t) = \min(1, (T_j - t)^+ / \tau_j)` of the extended forward rate.

    Parameters
    ----------
    t : array_like
        Times.
    start, end : float
        Accrual period :math:`[T_{j-1}, T_j]`.

    Returns
    -------
    float or ndarray
        1 before `start`, falling linearly to 0 at `end`, and 0 after.

    Examples
    --------
    >>> fmm_volatility_decay([0.5, 1.0, 1.25, 1.5, 2.0], 1.0, 1.5).tolist()
    [1.0, 1.0, 0.5, 0.0, 0.0]
    """
    start, end = _check_period(start, end)
    t = np.asarray(t, dtype=np.float64)
    return as_output(np.clip((end - t) / (end - start), 0.0, 1.0))


def _integrated_variance(t, start, end):
    """:math:`\\int_0^t g^2(s)\\,ds` for the linear decay."""
    tau = end - start
    inside = np.clip((end - t) / tau, 0.0, 1.0)
    return np.minimum(t, start) + tau / 3.0 * (1.0 - inside**3)


def fmm_caplet_price(
    curve: CurveLike,
    start,
    end,
    strike,
    vol,
    option_type: str = "call",
    backward_looking: bool = True,
    normal: bool = False,
    notional: float = 1.0,
):
    r"""Caplet (call) or floorlet (put) on a backward-looking or forward-looking term rate.

    The payoff :math:`\tau\,(R - K)^+` is paid at :math:`T_j` = `end`.
    Under the :math:`T_j`-forward measure the extended forward rate is a
    martingale started at :math:`F = [P(0, T_{j-1})/P(0, T_j) - 1]/\tau`,
    so

    .. math:: \text{caplet} = \tau P(0, T_j)\,\text{Black}(F, K, \sigma^2 v), \qquad
              v = T_{j-1} + \tau/3

    for the rate compounded in arrears, and :math:`v = T_{j-1}` for a
    forward-looking rate fixed at the start of the period (the LIBOR
    caplet). The extra :math:`\tau/3` is the uncertainty that remains
    while the overnight fixings arrive, so even a caplet whose period has
    started has time value.

    Parameters
    ----------
    curve : YieldCurve or float
    start, end : array_like
        Accrual period, ``0 <= start < end``.
    strike : array_like
    vol : array_like
        Black (lognormal) volatility, or normal volatility if `normal`.
    option_type : {"call", "put"}
    backward_looking : bool
        Compounded in arrears (Lyashenko-Mercurio), or fixed at `start`.
    normal : bool
        Bachelier dynamics :math:`dR_j = \sigma g_j\,dW`.
    notional : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    An in-arrears caplet over [1, 1.25] is Black's formula with expiry
    1 + 0.25/3, and worth more than the forward-looking caplet:

    >>> from financialkit.options import black76_price
    >>> F = (np.exp(0.03 * 0.25) - 1) / 0.25
    >>> c = fmm_caplet_price(0.03, 1.0, 1.25, 0.03, 0.2)
    >>> bool(np.isclose(c, 0.25 * np.exp(-0.03 * 1.25) * black76_price(F, 0.03, 1.0 + 0.25 / 3, 0.0, 0.2)))
    True
    >>> bool(c > fmm_caplet_price(0.03, 1.0, 1.25, 0.03, 0.2, backward_looking=False))
    True
    """
    c = as_curve(curve)
    start, end = _check_period(start, end)
    nonnegative(vol=vol)
    tau = end - start
    P_end = c.discount(end)
    F = (c.discount(start) / P_end - 1.0) / tau
    v = start + tau / 3.0 if backward_looking else start
    if normal:
        undiscounted = bachelier_price(F, strike, v, 0.0, vol, option_type=option_type)
    else:
        undiscounted = black76_price(F, strike, v, 0.0, vol, option_type)
    return as_output(notional * tau * P_end * np.asarray(undiscounted, dtype=np.float64))


def simulate_fmm_rate(
    curve: CurveLike, start: float, end: float, vol: float, n_steps: int, n_paths: int, normal: bool = False, rng: RNGLike = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Paths of the extended forward rate :math:`R_j(t)` on ``[0, end]`` under the :math:`T_j`-forward measure.

    The rate is driftless with volatility :math:`\sigma g_j(t)`. Each step
    uses the exact integrated variance
    :math:`\sigma^2\int_{t_k}^{t_{k+1}} g_j^2\,ds`, applied to the log of
    the rate (lognormal) or to the rate itself (normal), so the terminal
    law is exact for any number of steps. The paths stop moving at
    `end`, where they equal the compounded backward-looking rate.

    Parameters
    ----------
    curve : YieldCurve or float
    start, end : float
        Accrual period.
    vol : float
    n_steps, n_paths : int
    normal : bool
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
    rates : ndarray, shape (n_paths, n_steps + 1)
    """
    c = as_curve(curve)
    start, end = (float(x) for x in _check_period(start, end))
    nonnegative(vol=vol)
    F = (c.discount(start) / c.discount(end) - 1.0) / (end - start)
    times = np.linspace(0.0, end, n_steps + 1)
    var = vol**2 * np.diff(_integrated_variance(times, start, end))
    z = brownian_increments(end, n_steps, n_paths, rng=rng) / np.sqrt(end / n_steps)
    steps = np.sqrt(var) * z
    if normal:
        increments = steps
    else:
        if F <= 0:
            raise ValueError("the lognormal forward market model needs a positive forward rate")
        increments = steps - 0.5 * var
    paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return times, F + paths if normal else F * np.exp(paths)
