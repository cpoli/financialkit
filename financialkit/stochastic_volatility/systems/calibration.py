"""Calibration of characteristic-function models to a smile.

A model is calibrated by choosing its parameters so that its prices
match the quotes across strikes and maturities. Price errors are not
comparable across the surface (a deep out-of-the-money option is worth
cents), so each is divided by the option's Black-Scholes vega at the
market volatility. To first order this turns a price error into an
implied-volatility error,

.. math::

   \\min_\\theta \\sum_i \\left(\\frac{C^{\\text{model}}_i(\\theta) - C^{\\text{mkt}}_i}{\\mathcal{V}_i}\\right)^2
   \\approx \\sum_i \\left(\\sigma^{\\text{model}}_i(\\theta) - \\sigma^{\\text{mkt}}_i\\right)^2,

without inverting the model's prices at every iteration. The fit is
solved by :func:`scipy.optimize.least_squares`, and any
:class:`~financialkit.stochastic_volatility.core.base.CharacteristicFunctionModel`
(Heston, Bates, Merton, variance gamma, Kou) is priced by the COS method
or another Fourier pricer.

References: R. Cont and P. Tankov, "Non-parametric calibration of
jump-diffusion option pricing models," *J. Computational Finance* 7
(2004) 1-49; Y. Cui, S. del Baño Rollin and G. Germano, "Full and fast
calibration of the Heston stochastic volatility model," *European J.
Operational Research* 263 (2017) 625-638.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.stats import norm

from financialkit.options import black_scholes_price, implied_volatility
from financialkit.stochastic_volatility.core.base import CharacteristicFunctionModel, FourierCalibration
from financialkit.stochastic_volatility.systems.fourier import BlackScholesModel
from financialkit.stochastic_volatility.systems.heston import HestonModel
from financialkit.stochastic_volatility.systems.jumps import BatesModel, KouModel, MertonJumpModel, VarianceGammaModel

__all__ = ["fourier_calibrate", "CALIBRATION_BOUNDS"]

_HESTON = {"v0": (1e-4, 1.0), "kappa": (1e-3, 20.0), "theta": (1e-4, 1.0), "xi": (1e-3, 5.0), "rho": (-0.999, 0.999)}
_MERTON_JUMPS = {"lam": (0.0, 10.0), "mu_j": (-1.0, 1.0), "delta": (1e-3, 1.0)}

CALIBRATION_BOUNDS: dict[type, dict[str, tuple[float, float]]] = {
    BlackScholesModel: {"sigma": (1e-3, 3.0)},
    HestonModel: _HESTON,
    MertonJumpModel: {"sigma": (1e-3, 3.0), **_MERTON_JUMPS},
    BatesModel: {**_HESTON, **_MERTON_JUMPS},
    VarianceGammaModel: {"sigma": (1e-3, 3.0), "nu": (1e-3, 5.0), "theta": (-2.0, 2.0)},
    KouModel: {"sigma": (1e-3, 3.0), "lam": (0.0, 20.0), "p": (0.0, 1.0), "eta1": (1.001, 100.0), "eta2": (1e-3, 100.0)},
}
"""Default parameter bounds of :func:`fourier_calibrate` for each model class."""


def fourier_calibrate(
    model_class: type[CharacteristicFunctionModel],
    x0: Mapping[str, float],
    S: float,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    q: float = 0.0,
    prices: ArrayLike | None = None,
    vols: ArrayLike | None = None,
    option_type: str = "call",
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    method: str = "cos",
) -> FourierCalibration:
    """Fit a characteristic-function model to option quotes by vega-weighted least squares.

    The residual of each quote is the model price minus the market price,
    divided by the Black-Scholes vega at the market volatility (floored at
    a thousandth of the largest vega, so far wings cannot dominate). The
    model is priced one maturity at a time, and a parameter vector the
    model rejects (for example a variance gamma with infinite
    :math:`E[S_T]`) is given a residual of one, a hundred vol points, at
    every quote.

    Parameters
    ----------
    model_class : type
        A :class:`~financialkit.stochastic_volatility.core.base.CharacteristicFunctionModel`
        subclass, built as ``model_class(**params)``.
    x0 : mapping of str to float
        Starting values of the parameters to fit, by constructor name,
        clipped to the bounds; the model must accept them.
    S : float
        Spot.
    K, T : array_like
        Strikes and maturities of the quotes, broadcast together.
    r, q : float
        Risk-free rate and dividend yield.
    prices, vols : array_like, optional
        Market prices of ``option_type`` options, or their Black-Scholes
        implied volatilities; exactly one must be given.
    option_type : {"call", "put"}
        Type of the quoted prices.
    fixed : mapping of str to float, optional
        Constructor arguments held at the given values.
    bounds : mapping of str to (float, float), optional
        Bounds by parameter name, overriding :data:`CALIBRATION_BOUNDS`
        for ``model_class``; unbounded where neither gives one.
    method : {"cos", "lewis", "heston", "fft"}
        Fourier pricer, as in
        :meth:`~financialkit.stochastic_volatility.core.base.CharacteristicFunctionModel.price`.

    Returns
    -------
    FourierCalibration

    Examples
    --------
    Recover Heston's parameters from its own smile:

    >>> import numpy as np
    >>> K, T = np.meshgrid(np.linspace(80, 120, 9), [0.5, 1.0])
    >>> true = HestonModel(v0=0.04, kappa=1.5, theta=0.06, xi=0.6, rho=-0.7)
    >>> vols = np.array([true.implied_vol(100.0, k, t[0], 0.02) for k, t in zip(K, T)])
    >>> x0 = dict(v0=0.03, kappa=1.0, theta=0.05, xi=0.4, rho=-0.4)
    >>> fit = fourier_calibrate(HestonModel, x0, 100.0, K, T, 0.02, vols=vols)
    >>> {k: round(v, 4) for k, v in fit.params.items()}
    {'v0': 0.04, 'kappa': 1.5, 'theta': 0.06, 'xi': 0.6, 'rho': -0.7}
    >>> bool(fit.rmse < 1e-6)
    True
    """
    if (prices is None) == (vols is None):
        raise ValueError("give exactly one of prices and vols")
    fixed = dict(fixed or {})
    names = list(x0)
    signature = inspect.signature(model_class)
    unknown = [n for n in [*names, *fixed] if n not in signature.parameters]
    if unknown:
        raise ValueError(f"{model_class.__name__} has no parameters {unknown}")
    if set(names) & set(fixed):
        raise ValueError(f"parameters {sorted(set(names) & set(fixed))} are both fitted and fixed")

    quotes = np.asarray(vols if vols is not None else prices, dtype=np.float64)
    K, T, quotes = (a.ravel() for a in np.broadcast_arrays(np.asarray(K, dtype=np.float64), np.asarray(T, dtype=np.float64), quotes))
    if vols is not None:
        market_vols = quotes
        market = np.asarray(black_scholes_price(S, K, T, r, market_vols, q, option_type), dtype=np.float64)
    else:
        market = quotes
        market_vols = np.asarray(implied_volatility(market, S, K, T, r, q, option_type), dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * market_vols**2) * T) / (market_vols * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    vega = np.maximum(vega, 1e-3 * vega.max())
    groups = [(t, T == t) for t in np.unique(T)]

    table = {**CALIBRATION_BOUNDS.get(model_class, {}), **(bounds or {})}
    lo = np.array([table.get(n, (-np.inf, np.inf))[0] for n in names])
    hi = np.array([table.get(n, (-np.inf, np.inf))[1] for n in names])
    start = np.clip(np.array([x0[n] for n in names], dtype=np.float64), lo, hi)

    def build(p: np.ndarray) -> CharacteristicFunctionModel:
        return model_class(**dict(zip(names, (float(v) for v in p), strict=True)), **fixed)

    def model_prices(m: CharacteristicFunctionModel) -> np.ndarray:
        out = np.empty_like(market)
        for t, idx in groups:
            out[idx] = m.price(S, K[idx], float(t), r, q, option_type, method)
        return out

    build(start)  # an invalid starting point raises here rather than stalling on the penalty

    def resid(p: np.ndarray) -> np.ndarray:
        try:
            m = build(p)
        except ValueError:
            return np.ones_like(market)
        with np.errstate(all="ignore"):
            e = (model_prices(m) - market) / vega
        return np.where(np.isfinite(e), e, 1.0)

    sol = least_squares(resid, start, bounds=(lo, hi), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    model = build(sol.x)
    model_vols = np.empty_like(market)
    for t, idx in groups:
        model_vols[idx] = model.implied_vol(S, K[idx], float(t), r, q, method)
    params = {**dict(zip(names, (float(v) for v in sol.x), strict=True)), **{k: float(v) for k, v in fixed.items()}}
    return FourierCalibration(
        model=model,
        params=params,
        strikes=K,
        maturities=T,
        market_vols=market_vols,
        model_vols=model_vols,
        residuals=np.asarray(sol.fun),
        rmse=float(100.0 * np.sqrt(np.mean((model_vols - market_vols) ** 2))),
        converged=bool(sol.success),
        nfev=int(sol.nfev),
    )
