"""Local volatility: Cox's constant elasticity of variance and Dupire's formula.

Cox (1975) let the volatility depend on the price level,
:math:`dS = (r - q) S\\,dt + \\sigma S^{\\beta}\\,dW`. For
:math:`\\beta < 1` volatility rises as the price falls (the leverage
effect) and the implied volatility skews downward in the strike.
Schroder (1989) wrote the option prices with the noncentral chi-square
distribution (:func:`cev_price`).

Dupire (1994), and Derman and Kani in a tree, inverted the problem: given
call prices at every strike and maturity there is exactly one diffusion
:math:`dS = (r - q) S\\,dt + \\sigma(S, t) S\\,dW` reproducing them, with

.. math::

   \\sigma^2(K, T) = \\frac{\\partial_T C + (r - q) K \\partial_K C + q C}{\\tfrac12 K^2 \\partial_{KK} C}.

:func:`dupire_local_vol` applies it to a grid of prices by finite
differences, and :func:`local_vol_paths` simulates the fitted diffusion
through :func:`financialkit.integrators.euler_maruyama`.

References: J. C. Cox, "Notes on option pricing I: constant elasticity
of variance diffusions," Stanford working paper (1975), reprinted in
*J. Portfolio Management* 23 (1996) 15-17; M. Schroder, *J. Finance* 44
(1989) 211-219; B. Dupire, *Risk* 7(1) (1994) 18-20; E. Derman and
I. Kani, *Risk* 7(2) (1994) 32-39; Hull, *Options, Futures, and Other
Derivatives*, 10th ed., Sections 27.1 and 27.6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ncx2

from financialkit.integrators import RNGLike, euler_maruyama
from financialkit.options.utils.common import as_output, option_sign
from financialkit.stochastic_volatility.core.base import LocalVolSurface
from financialkit.stochastic_volatility.utils.numerics import local_vol_diffusion, local_vol_drift, local_vol_params

__all__ = ["cev_price", "cev_local_vol", "dupire_local_vol", "local_vol_paths"]


def cev_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: float,
    beta: float,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> np.ndarray | float:
    r"""European option prices under Cox's CEV model (Schroder 1989).

    For :math:`dS = (r - q) S\,dt + \sigma S^{\beta}\,dW` with
    :math:`0 < \beta < 1`, writing :math:`Q(z; k, \lambda)` for the
    noncentral chi-square CDF,

    .. math::

       C = S e^{-qT}\left[1 - Q(a; b + 2, c)\right] - K e^{-rT} Q(c; b, a),

    with :math:`a = (K e^{-(r-q)T})^{2(1-\beta)} / ((1-\beta)^2 v)`,
    :math:`b = 1/(1 - \beta)`, :math:`c = S^{2(1-\beta)} / ((1-\beta)^2 v)`
    and :math:`v = \sigma^2 [e^{2(r-q)(\beta-1)T} - 1] / (2(r-q)(\beta-1))`
    (:math:`\sigma^2 T` when :math:`r = q`). For :math:`\beta > 1` the roles
    swap: :math:`C = S e^{-qT}[1 - Q(c; -b, a)] - K e^{-rT} Q(a; 2 - b, c)`.
    Puts follow by parity.

    Parameters
    ----------
    S, K, T, r : array_like
        Spot, strike, maturity and risk-free rate.
    sigma : float
        Volatility scale; the local volatility is :math:`\sigma S^{\beta - 1}`.
    beta : float
        Elasticity exponent, positive and not 1 (that is Black-Scholes).
    q : array_like
        Dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Section 27.1, uses the same formula; with
    :math:`\beta = 0.5` and :math:`\sigma = 2` the at-the-money local
    volatility is 20%, and the price is close to Black-Scholes at 20%:

    >>> round(cev_price(100.0, 100.0, 1.0, 0.05, 2.0, 0.5), 4)
    10.4539
    """
    option_sign(option_type)
    if beta <= 0 or beta == 1:
        raise ValueError(f"beta must be positive and different from 1, got {beta}")
    S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q)))
    mu = r - q
    small = np.abs(mu) < 1e-12
    mu_safe = np.where(small, 1.0, mu)
    v = np.where(small, sigma**2 * T, sigma**2 * np.expm1(2 * mu_safe * (beta - 1) * T) / (2 * mu_safe * (beta - 1)))
    one_b = 1.0 - beta
    a = (K * np.exp(-mu * T)) ** (2 * one_b) / (one_b**2 * v)
    b = 1.0 / one_b
    c = S ** (2 * one_b) / (one_b**2 * v)
    if beta < 1:
        call = S * np.exp(-q * T) * ncx2.sf(a, b + 2, c) - K * np.exp(-r * T) * ncx2.cdf(c, b, a)
    else:
        call = S * np.exp(-q * T) * ncx2.sf(c, -b, a) - K * np.exp(-r * T) * ncx2.cdf(a, 2 - b, c)
    out = call if option_type == "call" else call - S * np.exp(-q * T) + K * np.exp(-r * T)
    return as_output(out)


def cev_local_vol(S: ArrayLike, sigma: float, beta: float) -> np.ndarray | float:
    """CEV local volatility :math:`\\sigma S^{\\beta - 1}` (proportional, as in Black-Scholes).

    Parameters
    ----------
    S : array_like
    sigma, beta : float

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(cev_local_vol(100.0, 2.0, 0.5), 6)
    0.2
    """
    return as_output(sigma * np.asarray(S, dtype=np.float64) ** (beta - 1.0))


def dupire_local_vol(
    strikes: ArrayLike,
    maturities: ArrayLike,
    call_prices: ArrayLike,
    r: float,
    q: float = 0.0,
) -> LocalVolSurface:
    r"""Dupire's (1994) local volatility from a grid of call prices.

    .. math::

       \sigma^2(K, T) = \frac{\partial_T C + (r - q) K\,\partial_K C + q C}{\tfrac12 K^2\,\partial_{KK} C}

    Derivatives are second-order finite differences (:func:`numpy.gradient`,
    non-uniform grids allowed). Nodes where the numerator or the
    butterfly :math:`\partial_{KK}C` is not positive (an arbitrage, or
    numerical noise in the wings) get ``nan``.

    Parameters
    ----------
    strikes : array_like, shape (n_K,)
        Increasing strikes.
    maturities : array_like, shape (n_T,)
        Increasing maturities (at least two).
    call_prices : array_like, shape (n_T, n_K)
        European call prices on the grid.
    r, q : float
        Risk-free rate and dividend yield.

    Returns
    -------
    LocalVolSurface

    Examples
    --------
    Black-Scholes prices give back the flat volatility:

    >>> import numpy as np
    >>> from financialkit.options import black_scholes_price
    >>> K, T = np.linspace(60, 140, 81), np.linspace(0.5, 1.5, 21)
    >>> C = black_scholes_price(100.0, K[None, :], T[:, None], 0.03, 0.25)
    >>> lv = dupire_local_vol(K, T, C, 0.03)
    >>> bool(np.nanmax(np.abs(lv.vol[5:-5, 20:-20] - 0.25)) < 2e-3)
    True
    """
    K = np.asarray(strikes, dtype=np.float64)
    T = np.asarray(maturities, dtype=np.float64)
    C = np.asarray(call_prices, dtype=np.float64)
    if C.shape != (T.size, K.size):
        raise ValueError(f"call_prices must have shape {(T.size, K.size)}, got {C.shape}")
    dC_dT = np.gradient(C, T, axis=0, edge_order=2)
    dC_dK = np.gradient(C, K, axis=1, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K, axis=1, edge_order=2)
    num = dC_dT + (r - q) * K * dC_dK + q * C
    den = 0.5 * K**2 * d2C_dK2
    ok = (num > 0) & (den > 0)
    var = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    return LocalVolSurface(strikes=K, maturities=T, vol=np.sqrt(var))


def local_vol_paths(
    surface: LocalVolSurface,
    S: float,
    T: float,
    r: float,
    q: float = 0.0,
    n_steps: int = 100,
    n_paths: int = 10_000,
    rng: RNGLike = None,
    dW: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate :math:`dS = (r - q) S\\,dt + \\sigma(S, t) S\\,dW` on a local-volatility grid.

    Euler-Maruyama on :math:`\\ln S`, through
    :func:`financialkit.integrators.euler_maruyama`; the volatility is
    interpolated bilinearly and held flat outside the grid. Nodes that are
    ``nan`` in ``surface`` are filled from their nearest finite neighbour
    in strike first.

    Parameters
    ----------
    surface : LocalVolSurface
    S, T, r, q : float
    n_steps, n_paths : int
    rng : numpy.random.Generator or int, optional
    dW : array_like, shape (n_paths, n_steps), optional
        Brownian increments to use instead of fresh ones.

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
    spot : ndarray, shape (n_paths, n_steps + 1)

    Examples
    --------
    >>> import numpy as np
    >>> flat = LocalVolSurface(np.array([50.0, 150.0]), np.array([0.0, 1.0]), np.full((2, 2), 0.2))
    >>> t, S = local_vol_paths(flat, 100.0, 1.0, 0.05, n_steps=50, n_paths=20_000, rng=0)
    >>> bool(abs(S[:, -1].mean() - 100 * np.exp(0.05)) < 0.6)
    True
    """
    vol = np.array(surface.vol, dtype=np.float64)
    for row in vol:
        good = np.isfinite(row)
        if not good.any():
            raise ValueError("a maturity slice of the surface has no finite volatility")
        idx = np.flatnonzero(good)
        row[~good] = row[idx[np.abs(np.arange(row.size)[~good, None] - idx[None, :]).argmin(axis=1)]]
    params = local_vol_params(r - q, surface.strikes, surface.maturities, vol)
    times, x = euler_maruyama(local_vol_drift, local_vol_diffusion, np.log(S), T, n_steps, n_paths, params=params, rng=rng, dW=dW)
    return times, np.exp(x)
