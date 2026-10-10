"""Simulating Heston: truncated Euler schemes and Andersen's quadratic-exponential scheme.

A plain Euler step can push the square-root variance below zero, where
:math:`\\sqrt{v}` is undefined, and the obvious repairs bias prices badly
when the Feller condition fails, as it does in most calibrations. Lord,
Koekkoek and van Dijk (2010) compared the repairs in one framework and
found that *full truncation* (let :math:`v` go negative but use
:math:`v^+` in both the drift and the diffusion) has the smallest bias
among them. The Euler schemes here run through
:func:`financialkit.integrators.euler_maruyama` with correlated
increments from :func:`financialkit.integrators.correlated_increments`.

Andersen (2008) instead matched the first two moments of the exact
noncentral chi-square transition: a scaled squared Gaussian
:math:`a(b + Z)^2` when the variance is far from zero, and a mass at zero
plus an exponential tail when it is close. The stock then takes a
step that integrates the variance with a trapezoidal rule and uses the
correlation exactly. The quadratic-exponential (QE) scheme is nearly
unbiased at step sizes where Euler is far off; its update is a
step-by-step recursion rather than an SDE scheme, so it is written out
here.

References: L. Andersen, *J. Computational Finance* 11(3) (2008) 1-42;
R. Lord, R. Koekkoek and D. van Dijk, *Quantitative Finance* 10 (2010)
177-194; G. Deelstra and F. Delbaen, *Applied Stochastic Models and Data
Analysis* 14 (1998) 77-84; D. J. Higham and X. Mao, *J. Computational
Finance* 8(3) (2005) 35-61.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from financialkit.integrators import RNGLike, correlated_increments, euler_maruyama
from financialkit.monte_carlo import MonteCarloResult
from financialkit.monte_carlo.utils.numerics import sample_mean
from financialkit.options.utils.common import option_sign
from financialkit.stochastic_volatility.core.base import HestonPaths
from financialkit.stochastic_volatility.systems.heston import HestonModel
from financialkit.stochastic_volatility.utils.numerics import HESTON_EULER_SCHEMES, heston_diffusion, heston_drift

__all__ = ["heston_paths", "heston_mc_price", "HESTON_SCHEMES"]

#: Every scheme accepted by :func:`heston_paths`.
HESTON_SCHEMES = (*HESTON_EULER_SCHEMES, "qe")


def _qe_paths(model: HestonModel, S: float, T: float, mu: float, n_steps: int, n_paths: int, rng: np.random.Generator, psi_c: float):
    dt = T / n_steps
    k, th, xi, rho = model.kappa, model.theta, model.xi, model.rho
    e = np.exp(-k * dt)
    K0 = -rho * k * th / xi * dt
    K1 = 0.5 * dt * (k * rho / xi - 0.5) - rho / xi
    K2 = 0.5 * dt * (k * rho / xi - 0.5) + rho / xi
    K3 = 0.5 * dt * (1 - rho**2)
    U = rng.random((n_steps, n_paths))
    Z = rng.standard_normal((n_steps, n_paths))
    lnS = np.empty((n_paths, n_steps + 1))
    V = np.empty((n_paths, n_steps + 1))
    lnS[:, 0], V[:, 0] = np.log(S), model.v0
    for i in range(n_steps):
        v = V[:, i]
        m = th + (v - th) * e
        s2 = v * xi**2 * e * (1 - e) / k + th * xi**2 * (1 - e) ** 2 / (2 * k)
        psi = s2 / m**2
        quad_branch = psi <= psi_c
        # Quadratic branch: a (b + Z)^2 with Z = Phi^{-1}(U).
        inv = 2.0 / np.where(quad_branch, psi, 1.0)
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        a = m / (1.0 + b2)
        zv = norm.ppf(np.clip(U[i], 1e-16, 1 - 1e-16))
        v_quad = a * (np.sqrt(b2) + zv) ** 2
        # Exponential branch: an atom at 0 of mass p, then an exponential tail.
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        v_exp = np.where(U[i] <= p, 0.0, np.log((1.0 - p) / np.maximum(1.0 - U[i], 1e-300)) / beta)
        v_new = np.where(quad_branch, v_quad, v_exp)
        lnS[:, i + 1] = lnS[:, i] + mu * dt + K0 + K1 * v + K2 * v_new + np.sqrt(K3 * (v + v_new)) * Z[i]
        V[:, i + 1] = v_new
    return np.exp(lnS), V


def heston_paths(
    model: HestonModel,
    S: float,
    T: float,
    r: float,
    q: float = 0.0,
    n_steps: int = 100,
    n_paths: int = 10_000,
    scheme: str = "qe",
    rng: RNGLike = None,
    psi_c: float = 1.5,
) -> HestonPaths:
    """Simulate Heston spot and variance paths.

    Parameters
    ----------
    model : HestonModel
    S, T, r, q : float
        Spot, horizon, risk-free rate and dividend yield.
    n_steps, n_paths : int
    scheme : {"qe", "full_truncation", "partial_truncation", "higham_mao"}
        Andersen's QE scheme, or an Euler scheme on :math:`(\\ln S, v)`
        with :math:`v` in the coefficients replaced by :math:`v^+` in both
        drift and diffusion (full truncation, Lord et al. 2010), in the
        diffusion only (partial truncation, Deelstra and Delbaen 1998), or
        by :math:`|v|` in the diffusion (Higham and Mao 2005).
    rng : numpy.random.Generator or int, optional
    psi_c : float
        QE switching threshold between the quadratic and exponential branches.

    Returns
    -------
    HestonPaths

    Examples
    --------
    >>> m = HestonModel(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    >>> res = heston_paths(m, 100.0, 1.0, 0.03, n_steps=20, n_paths=20_000, rng=0)
    >>> bool(abs(res.spot[:, -1].mean() / (100 * np.exp(0.03)) - 1) < 5e-3)
    True
    """
    if scheme not in HESTON_SCHEMES:
        raise ValueError(f"scheme must be one of {HESTON_SCHEMES}, got {scheme!r}")
    rng = np.random.default_rng(rng)
    mu = r - q
    times = np.linspace(0.0, T, n_steps + 1)
    if scheme == "qe":
        spot, var = _qe_paths(model, S, T, mu, n_steps, n_paths, rng, psi_c)
        return HestonPaths(times, spot, var, scheme)
    corr = np.array([[1.0, model.rho], [model.rho, 1.0]])
    dW = correlated_increments(corr, T, n_steps, n_paths, rng=rng)
    params = [mu, model.kappa, model.theta, model.xi, HESTON_EULER_SCHEMES[scheme]]
    times, x = euler_maruyama(heston_drift, heston_diffusion, np.array([np.log(S), model.v0]), T, n_steps, n_paths, params=params, dW=dW)
    return HestonPaths(times, np.exp(x[:, :, 0]), x[:, :, 1], scheme)


def heston_mc_price(
    model: HestonModel,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps: int = 100,
    n_paths: int = 10_000,
    scheme: str = "qe",
    rng: RNGLike = None,
) -> MonteCarloResult:
    """European option price under Heston by Monte Carlo on :func:`heston_paths`.

    The standard error measures only the statistical error; the
    discretisation bias of the scheme comes on top of it.

    Parameters
    ----------
    model : HestonModel
    S, K, T, r, q : float
    option_type : {"call", "put"}
    n_steps, n_paths : int
    scheme : str
        See :func:`heston_paths`.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    MonteCarloResult
    """
    sign = option_sign(option_type)
    paths = heston_paths(model, S, T, r, q, n_steps, n_paths, scheme, rng)
    payoff = np.exp(-r * T) * np.maximum(sign * (paths.spot[:, -1] - K), 0.0)
    return sample_mean(payoff, method=f"heston_{scheme}")
