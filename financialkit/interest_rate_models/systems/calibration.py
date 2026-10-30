"""Calibrating Hull-White and G2++ to Black cap and swaption volatilities.

A curve-fitted model reprices today's curve by construction; its
volatility parameters are what is left to fit, and they are fitted to
the options the model will hedge. The market quotes caps and swaptions
as Black volatilities, so each quote is turned into a price with
:func:`~financialkit.interest_rate_models.systems.bond_options.cap_floor_price`
or :func:`~financialkit.interest_rate_models.systems.bond_options.black_swaption_price`,
and the model's parameters are chosen by :func:`scipy.optimize.least_squares`
to match those prices, priced in the model by
:func:`~financialkit.interest_rate_models.systems.bond_options.model_cap_floor_price`
or :func:`~financialkit.interest_rate_models.systems.bond_options.jamshidian_swaption`
(:func:`~financialkit.interest_rate_models.systems.two_factor.g2pp_swaption`
in G2++). Each price error is divided by the instrument's Black vega, so the fit
minimises volatility errors without inverting Black's formula at every
step.

* :func:`hull_white_calibrate`: :math:`(a, \\sigma)` to caps or swaptions.
* :func:`model_implied_vols`: the Black volatilities a model implies,
  for quoting a fitted model the way the market does.
* :func:`g2pp_calibrate`: :math:`(a, \\sigma, b, \\eta, \\rho)` to caps or
  swaptions, the latter priced by Brigo and Mercurio's two-factor
  swaption integral.

References: J. C. Hull, *Options, Futures, and Other Derivatives*, 10th
ed. (Pearson, 2018), Sec. 32.8; D. Brigo and F. Mercurio, *Interest Rate
Models -- Theory and Practice*, 2nd ed. (Springer, 2006), Ch. 3.3 and 4.2 (Theorem 4.2.3 for G2++ swaptions).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares

from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CurveLike, YieldCurve, as_curve
from financialkit.interest_rate_models.core.base import GaussianShortRateModel, ShortRateCalibration
from financialkit.interest_rate_models.systems.bond_options import (
    _cap_schedule,
    black_swaption_price,
    cap_floor_price,
    forward_swap_rate,
    jamshidian_swaption,
    model_cap_floor_price,
)
from financialkit.interest_rate_models.systems.curve_fitted import HullWhiteModel
from financialkit.interest_rate_models.systems.two_factor import G2PlusPlusModel, g2pp_swaption

__all__ = ["hull_white_calibrate", "g2pp_calibrate", "model_implied_vols"]

_VOL_BRACKET = (1e-6, 5.0)


def _atm_strike(curve: YieldCurve, kind: str, expiry: float, tenor: float, freq: int) -> float:
    if kind == "swaption":
        return forward_swap_rate(curve, expiry, tenor, freq)
    resets, pays = _cap_schedule(expiry, freq)
    return float((curve.discount(resets[0]) - curve.discount(pays[-1])) / np.sum((pays - resets) * curve.discount(pays)))


def _black_price(curve: YieldCurve, kind: str, expiry: float, tenor: float, strike: float, vol: float, freq: int) -> float:
    if kind == "cap":
        return cap_floor_price(curve, strike, expiry, vol, freq).price
    return float(black_swaption_price(curve, expiry, tenor, strike, vol, freq))


def _model_price(model: GaussianShortRateModel, kind: str, expiry: float, tenor: float, strike: float, freq: int) -> float:
    if kind == "cap":
        return model_cap_floor_price(model, strike, expiry, freq).price
    if isinstance(model, G2PlusPlusModel):
        return g2pp_swaption(model, expiry, tenor, strike, freq)
    return jamshidian_swaption(model, expiry, tenor, strike, freq).price


def _implied_vol(curve: YieldCurve, kind: str, expiry: float, tenor: float, strike: float, price: float, freq: int) -> float:
    def gap(v):
        return _black_price(curve, kind, expiry, tenor, strike, v, freq) - price

    lo, hi = _VOL_BRACKET
    if gap(lo) * gap(hi) > 0:
        return np.nan
    return brentq(gap, lo, hi, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)


def _quotes(curve: YieldCurve, kind: str, expiries, tenors, strikes, freq: int | None, shape: tuple[int, ...] = ()):
    if kind not in ("cap", "swaption"):
        raise ValueError(f'kind must be "cap" or "swaption", got {kind!r}')
    if kind == "swaption" and tenors is None:
        raise ValueError("swaption quotes need tenors")
    f = int(freq if freq is not None else (4 if kind == "cap" else 2))
    tenors = 0.0 if tenors is None else tenors
    n = np.broadcast_shapes(np.shape(expiries), np.shape(tenors), shape)
    T, tau = (np.broadcast_to(np.asarray(x, dtype=np.float64), n).ravel() for x in (expiries, tenors))
    if strikes is None:
        K = np.array([_atm_strike(curve, kind, Ti, ti, f) for Ti, ti in zip(T, tau, strict=True)])
    else:
        K = np.broadcast_to(np.asarray(strikes, dtype=np.float64), T.shape).copy()
    return T, tau, K, f


def model_implied_vols(model: GaussianShortRateModel, expiries, kind: str = "cap", tenors=None, strikes=None, freq: int | None = None) -> np.ndarray:
    """Black volatilities implied by a short-rate model's cap or swaption prices.

    Each instrument is priced in the model (:func:`model_cap_floor_price`,
    or :func:`jamshidian_swaption` and, for G2++, :func:`g2pp_swaption`), and Black's formula is inverted by
    :func:`scipy.optimize.brentq` on the model's own initial curve.

    Parameters
    ----------
    model : GaussianShortRateModel
        A curve-fitted model: one-factor, or G2++ for swaptions.
    expiries : array_like, shape (n,)
        Cap maturities, or swaption expiries.
    kind : {"cap", "swaption"}
    tenors, strikes : array_like, shape (n,), optional
        As in :func:`hull_white_calibrate`.
    freq : int, optional
        As in :func:`hull_white_calibrate`.

    Returns
    -------
    ndarray, shape (n,)
        ``nan`` where Black's formula cannot match the model price.

    Examples
    --------
    >>> hw = HullWhiteModel(0.04, 0.05, 0.01)
    >>> vols = model_implied_vols(hw, [2.0, 5.0, 10.0])
    >>> bool(np.all(np.diff(vols) < 0))
    True
    """
    c = model.curve()
    T, tau, K, f = _quotes(c, kind, expiries, tenors, strikes, freq)
    return np.array([_implied_vol(c, kind, Ti, ti, Ki, _model_price(model, kind, Ti, ti, Ki, f), f) for Ti, ti, Ki in zip(T, tau, K, strict=True)])


def _calibrate(
    build: Callable[..., GaussianShortRateModel],
    names: Sequence[str],
    x0: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    curve: CurveLike,
    vols,
    expiries,
    kind: str,
    tenors,
    strikes,
    freq: int | None,
) -> ShortRateCalibration:
    c = as_curve(curve)
    v = np.asarray(vols, dtype=np.float64)
    T, tau, K, f = _quotes(c, kind, expiries, tenors, strikes, freq, v.shape)
    v = np.broadcast_to(v, T.shape).copy()
    if v.size < len(names):
        raise ValueError(f"need at least {len(names)} quotes to fit {len(names)} parameters")
    if np.any(v <= 0):
        raise ValueError("vols must be positive")
    quotes = list(zip(T, tau, K, strict=True))

    def black(vol: np.ndarray) -> np.ndarray:
        return np.array([_black_price(c, kind, Ti, ti, Ki, vi, f) for (Ti, ti, Ki), vi in zip(quotes, vol, strict=True)])

    market = black(v)
    h = 1e-4
    vega = (black(v + h) - black(v - h)) / (2 * h)

    def prices(p: np.ndarray) -> np.ndarray:
        model = build(c, *p)
        return np.array([_model_price(model, kind, Ti, ti, Ki, f) for Ti, ti, Ki in quotes])

    def resid(p: np.ndarray) -> np.ndarray:
        return (prices(p) - market) / vega

    sol = least_squares(resid, np.asarray(x0, dtype=np.float64), bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    fitted = prices(sol.x)
    model_vols = np.array([_implied_vol(c, kind, Ti, ti, Ki, Pi, f) for (Ti, ti, Ki), Pi in zip(quotes, fitted, strict=True)])
    return ShortRateCalibration(
        model=build(c, *sol.x),
        params={n: float(x) for n, x in zip(names, sol.x, strict=True)},
        kind=kind,
        expiries=T,
        tenors=tau,
        strikes=K,
        market_vols=v,
        model_vols=model_vols,
        market_prices=market,
        model_prices=fitted,
        rmse=float(np.sqrt(np.mean((model_vols - v) ** 2))),
        converged=bool(sol.success),
    )


def hull_white_calibrate(
    curve: CurveLike,
    vols,
    expiries,
    kind: str = "cap",
    tenors=None,
    strikes=None,
    freq: int | None = None,
    x0: Sequence[float] = (0.1, 0.01),
) -> ShortRateCalibration:
    r"""Fit Hull-White's :math:`(a, \sigma)` to Black cap or swaption volatilities.

    Every quote is priced with Black's formula, then in the model, and
    :func:`scipy.optimize.least_squares` minimises the vega-weighted
    price errors :math:`(V^{\text{model}}_i - V^{\text{Black}}_i)/\nu_i`,
    to first order the volatility errors. Caps are priced in the model as
    puts on zero-coupon bonds, swaptions by Jamshidian's decomposition.
    Two quotes at different expiries pin down both parameters: :math:`\sigma`
    sets the level of the volatilities and :math:`a` how they fall with
    expiry (Hull, 10th ed., Sec. 32.8).

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure, used both by Black's formula and the model.
    vols : array_like, shape (n,)
        Black volatilities: flat cap volatilities, or swaption volatilities.
    expiries : array_like, shape (n,)
        Cap maturities, or swaption expiries.
    kind : {"cap", "swaption"}
    tenors : array_like, shape (n,), optional
        Swaption tenors (required for swaptions, ignored for caps).
    strikes : array_like, shape (n,), optional
        Cap rates or fixed rates; at the money (the forward swap rate) by default.
    freq : int, optional
        Resets (caps) or fixed payments (swaptions) per year; 4 for caps and
        2 for swaptions by default.
    x0 : sequence of float
        Starting :math:`(a, \sigma)`.

    Returns
    -------
    ShortRateCalibration

    Examples
    --------
    Round trip: quote caps at the Black volatilities a known Hull-White
    model implies, and fit them back.

    >>> from financialkit.fixed_income import NelsonSiegelCurve
    >>> curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
    >>> mats = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    >>> vols = model_implied_vols(HullWhiteModel(curve, 0.08, 0.009), mats)
    >>> fit = hull_white_calibrate(curve, vols, mats)
    >>> round(fit.params["a"], 6), round(fit.params["sigma"], 6), fit.rmse < 1e-8
    (0.08, 0.009, True)
    """
    return _calibrate(
        lambda c, a, sigma: HullWhiteModel(c, a, sigma), ("a", "sigma"), x0, ([0.0, 1e-8], [5.0, 1.0]), curve, vols, expiries, kind, tenors, strikes, freq
    )


def g2pp_calibrate(
    curve: CurveLike,
    vols,
    maturities,
    strikes=None,
    freq: int | None = None,
    x0: Sequence[float] = (0.5, 0.01, 0.05, 0.008, -0.5),
    kind: str = "cap",
    tenors=None,
) -> ShortRateCalibration:
    r"""Fit G2++'s :math:`(a, \sigma, b, \eta, \rho)` to Black cap or swaption volatilities.

    The same vega-weighted least squares as :func:`hull_white_calibrate`,
    with caps priced in the model as puts on zero-coupon bonds and
    swaptions by :func:`g2pp_swaption`. A cap strip constrains the caplet
    volatility term structure only, so different parameter sets can fit it
    equally well (and :math:`(a, \sigma)` and :math:`(b, \eta)` can swap).
    Swaptions on different tenors also see the correlation of short and
    long rates, which pins down :math:`\rho`, so a swaption grid is the
    usual target. Either way, check :attr:`ShortRateCalibration.rmse`
    rather than the parameters (Brigo and Mercurio, 2nd ed., Ch. 4.2).

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure.
    vols : array_like, shape (n,)
        Flat Black cap volatilities, or swaption volatilities, ``n >= 5``.
    maturities : array_like, shape (n,)
        Cap maturities, or swaption expiries.
    strikes : array_like, shape (n,), optional
        Cap rates or fixed rates; at the money by default.
    freq : int, optional
        Resets (caps) or fixed payments (swaptions) per year; 4 for caps and
        2 for swaptions by default.
    x0 : sequence of float
        Starting :math:`(a, \sigma, b, \eta, \rho)`.
    kind : {"cap", "swaption"}
    tenors : array_like, shape (n,), optional
        Swaption tenors (required for swaptions, ignored for caps).

    Returns
    -------
    ShortRateCalibration

    Examples
    --------
    Round trip on a grid of swaptions quoted from a known model.

    >>> from financialkit.fixed_income import NelsonSiegelCurve
    >>> curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
    >>> expiries, tenors = np.meshgrid([1.0, 3.0, 5.0], [2.0, 10.0])
    >>> truth = G2PlusPlusModel(curve, 0.6, 0.012, 0.06, 0.007, -0.6)
    >>> vols = model_implied_vols(truth, expiries.ravel(), "swaption", tenors.ravel())
    >>> fit = g2pp_calibrate(curve, vols, expiries.ravel(), kind="swaption", tenors=tenors.ravel())
    >>> bool(fit.rmse < 1e-8)
    True
    """
    return _calibrate(
        lambda c, a, sigma, b, eta, rho: G2PlusPlusModel(c, a, sigma, b, eta, rho),
        ("a", "sigma", "b", "eta", "rho"),
        x0,
        ([1e-3, 1e-8, 1e-3, 1e-8, -0.999], [5.0, 1.0, 5.0, 1.0, 0.999]),
        curve,
        vols,
        maturities,
        kind,
        tenors,
        strikes,
        freq,
    )
