"""Stochastic volatility: Hull-White's mixing formula and the Heston model.

Hull and White (1987) let the variance follow its own diffusion. When it
is uncorrelated with the stock, the option price is the Black-Scholes
price averaged over the distribution of the *average* variance along the
path (the mixing argument); for a lognormal variance with zero drift
they expanded that average in a series (:func:`hull_white_price`), and
it can be computed directly by simulating variance paths
(:func:`hull_white_mixing_mc`). Because the variance is uncorrelated,
the smile is symmetric in log-moneyness.

Heston (1993) chose a mean-reverting square-root variance correlated
with the stock,

.. math::

   dS = (r - q) S\\,dt + \\sqrt{v} S\\,dW_1, \\qquad
   dv = \\kappa(\\theta - v)\\,dt + \\xi\\sqrt{v}\\,dW_2, \\qquad
   dW_1\\,dW_2 = \\rho\\,dt,

which is affine, so its characteristic function is closed form and
:class:`HestonModel` prices with every Fourier method. Heston's own
inversion writes the call as :math:`S e^{-qT} P_1 - K e^{-rT} P_2` with
two exercise probabilities, each one Fourier integral
(:func:`heston_price`). A negative :math:`\\rho` gives the equity skew.

References: J. Hull and A. White, *J. Finance* 42 (1987) 281-300;
S. L. Heston, *Rev. Financial Studies* 6 (1993) 327-343; H. Albrecher,
P. Mayer, W. Schoutens and J. Tistaert, "The little Heston trap,"
*Wilmott* (Jan. 2007) 83-92; J. Gatheral, *The Volatility Surface*
(Wiley, 2006), Chapter 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad, trapezoid
from scipy.special import factorial
from scipy.stats import norm

from financialkit.integrators import RNGLike, gbm_paths
from financialkit.monte_carlo import MonteCarloResult
from financialkit.monte_carlo.utils.numerics import sample_mean
from financialkit.options import black_scholes_price
from financialkit.options.utils.common import as_output, black_d1_d2, option_sign
from financialkit.stochastic_volatility.core.base import CharacteristicFunctionModel

__all__ = ["HestonModel", "heston_price", "hull_white_price", "hull_white_mixing_mc"]


class HestonModel(CharacteristicFunctionModel):
    """Heston's (1993) square-root stochastic volatility model.

    Parameters
    ----------
    v0 : float
        Initial variance.
    kappa : float
        Speed of mean reversion of the variance.
    theta : float
        Long-run variance.
    xi : float
        Volatility of variance.
    rho : float
        Correlation between the stock and variance Brownian motions.

    Examples
    --------
    >>> m = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    >>> m.feller_satisfied
    True
    >>> complex(m.char_func(-1j, 1.0)).real  # E[S_T / F_T] = 1
    1.0
    """

    name = "Heston"

    def __init__(self, v0: float, kappa: float, theta: float, xi: float, rho: float):
        if v0 < 0 or kappa <= 0 or theta <= 0 or xi <= 0:
            raise ValueError("need v0 >= 0 and kappa, theta, xi > 0")
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        self.v0, self.kappa, self.theta, self.xi, self.rho = (float(a) for a in (v0, kappa, theta, xi, rho))

    @property
    def feller_satisfied(self) -> bool:
        """bool: Whether :math:`2\\kappa\\theta \\ge \\xi^2`, so the variance never reaches 0."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        """Characteristic function of :math:`\\ln(S_T/F_T)`, in the "little trap" form of Albrecher et al. (2007).

        That form keeps the complex logarithm on its principal branch for
        all maturities, unlike Heston's original expression.
        """
        u = np.asarray(u, dtype=np.complex128)
        k, th, xi, rho = self.kappa, self.theta, self.xi, self.rho
        beta = k - rho * xi * 1j * u
        d = np.sqrt(beta**2 + xi**2 * (1j * u + u**2))
        g = (beta - d) / (beta + d)
        e = np.exp(-d * T)
        C = k * th / xi**2 * ((beta - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        D = (beta - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
        return np.exp(C + D * self.v0)

    def variance_mean(self, t: ArrayLike) -> np.ndarray | float:
        """Expected variance :math:`E[v_t] = \\theta + (v_0 - \\theta) e^{-\\kappa t}`."""
        return as_output(self.theta + (self.v0 - self.theta) * np.exp(-self.kappa * np.asarray(t, dtype=np.float64)))


def heston_price(
    model: CharacteristicFunctionModel,
    S: float,
    K: ArrayLike,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
) -> np.ndarray | float:
    r"""European prices by Heston's (1993) two-probability formula.

    .. math::

       C = S e^{-qT} P_1 - K e^{-rT} P_2, \qquad
       P_j = \frac12 + \frac1\pi \int_0^\infty
       \operatorname{Re}\!\left[\frac{e^{i u \ln(F/K)} f_j(u)}{i u}\right] du,

    with :math:`f_2 = \phi` (the exercise probability under the
    risk-neutral measure) and :math:`f_1(u) = \phi(u - i)` (under the stock
    measure). It works for any characteristic-function model, not only
    Heston's.

    Parameters
    ----------
    model : CharacteristicFunctionModel
    S : float
    K : array_like
    T, r, q : float
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Andersen (2008), Case I (Feller condition badly violated): the
    at-the-money call over ten years is worth 13.0847.

    >>> m = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
    >>> round(heston_price(m, 100.0, 100.0, 10.0, 0.0), 4)
    13.0847
    """
    option_sign(option_type)
    K = np.asarray(K, dtype=np.float64)
    F = S * np.exp((r - q) * T)
    calls = np.empty(K.shape)
    for idx in np.ndindex(K.shape):
        kappa = np.log(F / K[idx])

        def p_integrand(u: float, shift: complex, kappa: float = kappa) -> float:
            return float(np.real(np.exp(1j * u * kappa) * model.char_func(u + shift, T) / (1j * u)))

        P1 = 0.5 + quad(p_integrand, 0.0, np.inf, args=(-1j,), limit=2000, epsabs=1e-12)[0] / np.pi
        P2 = 0.5 + quad(p_integrand, 0.0, np.inf, args=(0.0,), limit=2000, epsabs=1e-12)[0] / np.pi
        calls[idx] = np.exp(-r * T) * (F * P1 - K[idx] * P2)
    puts = calls - S * np.exp(-q * T) + K * np.exp(-r * T)
    return as_output(calls if option_type == "call" else puts)


def _average_variance_moments(V0: float, xi: float, T: float) -> tuple[float, float]:
    """:math:`E[\\bar V^2]` and :math:`E[\\bar V^3]` for :math:`dV = \\xi V dW` (Hull-White 1987, eqs. 10-11)."""
    k = xi**2 * T
    if k < 0.1:
        # The closed forms cancel catastrophically for small k (the third
        # central moment is O(k^2)), so sum their Taylor series instead;
        # twelve terms leave a truncation error below 1e-17.
        n = np.arange(12)
        m2 = float(np.sum(2.0 * k**n / factorial(n + 2)))
        m3 = float(np.sum((3.0 ** (n + 3) - 9.0) * k**n / (3.0 * factorial(n + 3))))
    else:
        m2 = 2 * (np.exp(k) - k - 1) / k**2
        m3 = (np.exp(3 * k) - 9 * np.exp(k) + 6 * k + 8) / (3 * k**3)
    return float(m2 * V0**2), float(m3 * V0**3)


def hull_white_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma0: float,
    xi: float,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> np.ndarray | float:
    r"""Hull-White (1987) series for an option under uncorrelated lognormal variance.

    With :math:`dV = \xi V\,dW` independent of the stock and
    :math:`\bar V = \frac1T\int_0^T V_t\,dt`, the price is
    :math:`E[C_{BS}(\bar V)]`; expanding about :math:`V_0 = \sigma_0^2`,

    .. math::

       f \approx C_{BS}(V_0) + \tfrac12 C''(V_0)\,\operatorname{Var}\bar V
       + \tfrac16 C'''(V_0)\,E[(\bar V - V_0)^3],

    with the derivatives in the variance in closed form. The truncated
    series is accurate while :math:`\xi^2 T` is small (below about 0.1)
    and drifts away from the exact mixing price beyond that; compare
    :func:`hull_white_mixing_mc`.

    Parameters
    ----------
    S, K, T, r : array_like
        Spot, strike, maturity and risk-free rate.
    sigma0 : float
        Initial volatility, :math:`\sqrt{V_0}`.
    xi : float
        Volatility of the variance.
    q : array_like
        Dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray

    Examples
    --------
    Without volatility of variance it is Black-Scholes:

    >>> hull_white_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0) == black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    True
    """
    option_sign(option_type)
    S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q)))
    V0 = sigma0**2
    base = np.asarray(black_scholes_price(S, K, T, r, sigma0, q, option_type), dtype=np.float64)
    if xi == 0.0:
        return as_output(base)
    F = S * np.exp((r - q) * T)
    d1, d2 = black_d1_d2(F, K, sigma0, T)
    A = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)
    dd = d1 * d2
    c2 = A * (dd - 1.0) / (4 * sigma0**3)
    c3 = A * ((dd - 1.0) * (dd - 3.0) - (d1**2 + d2**2)) / (8 * sigma0**5)
    out = np.empty(S.shape)
    for idx in np.ndindex(S.shape):
        m2, m3 = _average_variance_moments(V0, xi, float(T[idx]))
        out[idx] = base[idx] + 0.5 * c2[idx] * (m2 - V0**2) + c3[idx] / 6 * (m3 - 3 * m2 * V0 + 2 * V0**3)
    return as_output(out)


def hull_white_mixing_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma0: float,
    xi: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps: int = 100,
    n_paths: int = 20_000,
    rng: RNGLike = None,
) -> MonteCarloResult:
    """Hull-White price by mixing: average Black-Scholes prices over simulated average variances.

    Variance paths :math:`dV = \\xi V\\,dW` come from
    :func:`financialkit.integrators.gbm_paths`; each path's time-average
    variance (trapezoidal rule) sets a Black-Scholes volatility, and the
    price is the mean of those Black-Scholes prices. This conditional
    Monte Carlo has far less variance than simulating the stock.

    Parameters
    ----------
    S, K, T, r, sigma0, xi, q, option_type
        As in :func:`hull_white_price`.
    n_steps, n_paths : int
        Time steps and variance paths.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    >>> res = hull_white_mixing_mc(100.0, 100.0, 1.0, 0.0, 0.2, 0.3, n_paths=4000, rng=0)
    >>> bool(abs(res.estimate - hull_white_price(100.0, 100.0, 1.0, 0.0, 0.2, 0.3)) < 4 * res.std_error)
    True
    """
    times, V = gbm_paths(sigma0**2, 0.0, xi, T, n_steps, n_paths, rng=np.random.default_rng(rng))
    vbar = trapezoid(V, times, axis=1) / T
    prices = black_scholes_price(S, K, T, r, np.sqrt(vbar), q, option_type)
    return sample_mean(prices, method="hull_white_mixing")
