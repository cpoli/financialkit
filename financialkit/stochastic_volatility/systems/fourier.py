"""Fourier pricing: Carr-Madan's FFT, Lewis's single integral and the COS method.

Every pricer here takes a
:class:`~financialkit.stochastic_volatility.core.base.CharacteristicFunctionModel`,
so Heston, Bates, Merton, Kou and variance gamma prices all come from the
same few lines, and :class:`BlackScholesModel` checks them against the
closed form.

* Carr and Madan (1999) damp the call price by :math:`e^{\\alpha k}` in the
  log strike :math:`k`, which makes it square integrable, and invert its
  Fourier transform with one FFT, giving prices on a whole grid of
  strikes at once (:func:`carr_madan_fft`).
* Lewis (2001) moves the contour of the inversion integral to
  :math:`\\operatorname{Im} u = -\\tfrac12`, where it is symmetric between
  calls and puts, leaving one real integral per strike (:func:`lewis_price`).
* Fang and Oosterlee (2008) expand the density of the log return in a
  cosine series on a truncated interval, with coefficients read straight
  off the characteristic function; the payoff's coefficients are closed
  form, and the error decays exponentially in the number of terms for
  smooth densities (:func:`cos_price`, :func:`cos_density`).

References: P. Carr and D. B. Madan, *J. Computational Finance* 2(4)
(1999) 61-73; A. L. Lewis, *A Simple Option Formula for General
Jump-Diffusion and Other Exponential Lévy Processes* (2001), and *Option
Valuation under Stochastic Volatility* (Finance Press, 2000); F. Fang and
C. W. Oosterlee, *SIAM J. Sci. Comput.* 31 (2008) 826-848.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from financialkit.options.utils.common import as_output, option_sign
from financialkit.stochastic_volatility.core.base import CharacteristicFunctionModel, FFTResult
from financialkit.stochastic_volatility.utils.numerics import cos_chi, cos_psi, put_call_from_put, simpson_weights

__all__ = ["BlackScholesModel", "carr_madan_fft", "carr_madan_price", "lewis_price", "cos_price", "cos_density", "cos_terms", "truncation_range"]


class BlackScholesModel(CharacteristicFunctionModel):
    """Black-Scholes as a characteristic-function model, the reference for the Fourier pricers.

    :math:`X_T = -\\tfrac12\\sigma^2 T + \\sigma W_T`, so
    :math:`\\phi(u) = \\exp(-\\tfrac12\\sigma^2 T (i u + u^2))`.

    Parameters
    ----------
    sigma : float
        Volatility.

    Examples
    --------
    >>> from financialkit.options import black_scholes_price
    >>> m = BlackScholesModel(0.2)
    >>> round(m.price(100.0, 100.0, 1.0, 0.05), 6) == round(black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2), 6)
    True
    """

    name = "Black-Scholes"

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)

    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        return np.exp(-0.5 * self.sigma**2 * T * (1j * u + u**2))


def truncation_range(model: CharacteristicFunctionModel, T: float, L: float = 10.0) -> tuple[float, float]:
    """Fang-Oosterlee truncation interval :math:`c_1 \\pm L\\sqrt{c_2 + \\sqrt{c_4}}` for :math:`X_T`.

    Parameters
    ----------
    model : CharacteristicFunctionModel
    T : float
        Maturity.
    L : float
        Width in units of the (kurtosis-inflated) standard deviation.

    Returns
    -------
    a, b : float
    """
    c1, c2, c4 = model.cumulants(T)
    half = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - half, c1 + half


def cos_terms(a: float, b: float, per_unit: float = 40.0, minimum: int = 256) -> int:
    """Default number of COS terms: a power of two of at least ``per_unit`` per unit of interval width.

    A fixed count is not enough: the fourth cumulant of a long-dated
    Heston return with a large volatility of variance widens the
    truncation interval tenfold, and the series needs proportionally
    more terms to resolve the density on it.

    Parameters
    ----------
    a, b : float
        Truncation interval.
    per_unit : float
        Terms per unit of :math:`b - a`.
    minimum : int

    Returns
    -------
    int

    Examples
    --------
    >>> cos_terms(-3.0, 3.0), cos_terms(-37.0, 37.0)
    (256, 4096)
    """
    return max(minimum, 1 << int(np.ceil(np.log2(per_unit * (b - a)))))


def cos_price(
    model: CharacteristicFunctionModel,
    S: float,
    K: ArrayLike,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    N: int | None = None,
    L: float = 10.0,
) -> np.ndarray | float:
    r"""European option prices by the Fang-Oosterlee (2008) COS method.

    With :math:`y = \ln(S_T/K)` truncated to :math:`[a_y, b_y]` and
    :math:`u_k = k\pi/(b_y - a_y)`,

    .. math::

       V = e^{-rT} \sum_{k=0}^{N-1}{}' \operatorname{Re}\!\left[\phi(u_k) e^{-i u_k a}\right] V_k,

    where :math:`a = a_y - \ln(F/K)` is the interval's left end for
    :math:`X_T`, the prime halves the first term, and :math:`V_k` are the
    cosine coefficients of the payoff, in closed form. Puts are priced
    by the series (their payoff is bounded) and calls by parity, as Fang
    and Oosterlee recommend.

    Parameters
    ----------
    model : CharacteristicFunctionModel
    S : float
        Spot.
    K : array_like
        Strikes.
    T, r, q : float
        Maturity, risk-free rate and dividend yield.
    option_type : {"call", "put"}
    N : int, optional
        Number of cosine terms; by default :func:`cos_terms` of the interval.
    L : float
        Truncation width, see :func:`truncation_range`.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Fang and Oosterlee (2008), Section 5.2: Heston with S = K = 100,
    T = 1, r = 0 has the reference call price 5.785155450.

    >>> from financialkit.stochastic_volatility import HestonModel
    >>> m = HestonModel(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
    >>> round(cos_price(m, 100.0, 100.0, 1.0, 0.0), 6)
    5.785155
    """
    option_sign(option_type)
    K = np.asarray(K, dtype=np.float64)
    a, b = truncation_range(model, T, L)
    u = np.arange(cos_terms(a, b) if N is None else N) * np.pi / (b - a)
    A = np.real(model.char_func(u, T) * np.exp(-1j * u * a))
    A[0] *= 0.5
    F = S * np.exp((r - q) * T)
    x = np.log(F / np.atleast_1d(K))[:, None]
    lo = x + a
    hi = np.minimum(x + b, 0.0)
    Vk = np.where(lo < 0.0, 2.0 / (b - a) * (cos_psi(u, lo, hi, lo) - cos_chi(u, lo, hi, lo)), 0.0)
    puts = np.atleast_1d(K) * np.exp(-r * T) * (Vk @ A)
    out = put_call_from_put(puts, S, np.atleast_1d(K), T, r, q, option_type)
    return as_output(out.reshape(K.shape))


def cos_density(model: CharacteristicFunctionModel, x: ArrayLike, T: float, N: int | None = None, L: float = 10.0) -> np.ndarray:
    r"""Density of :math:`X_T = \ln(S_T / F_T)` by the cosine expansion.

    .. math::

       f(x) \approx \frac{2}{b - a} \sum_{k=0}^{N-1}{}' \operatorname{Re}\!\left[\phi(u_k) e^{-i u_k a}\right] \cos(u_k (x - a))

    Parameters
    ----------
    model : CharacteristicFunctionModel
    x : array_like
        Points inside the truncation interval (the density is 0 outside).
    T : float
    N : int, optional
        Number of cosine terms; by default :func:`cos_terms` of the interval.
    L : float

    Returns
    -------
    ndarray

    Examples
    --------
    >>> from scipy.stats import norm
    >>> f = cos_density(BlackScholesModel(0.2), [0.0], 1.0)
    >>> bool(abs(f[0] - norm.pdf(0.0, -0.02, 0.2)) < 1e-10)
    True
    """
    x = np.asarray(x, dtype=np.float64)
    a, b = truncation_range(model, T, L)
    u = np.arange(cos_terms(a, b) if N is None else N) * np.pi / (b - a)
    A = np.real(model.char_func(u, T) * np.exp(-1j * u * a))
    A[0] *= 0.5
    f = 2.0 / (b - a) * (np.cos(np.multiply.outer(x - a, u)) @ A)
    return np.where((x >= a) & (x <= b), f, 0.0)


def lewis_price(
    model: CharacteristicFunctionModel,
    S: float,
    K: ArrayLike,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
) -> np.ndarray | float:
    r"""European option prices by Lewis's (2001) single-integral formula.

    .. math::

       C = S e^{-qT} - \frac{\sqrt{F K}\, e^{-rT}}{\pi} \int_0^\infty
       \operatorname{Re}\!\left[e^{i u \ln(F/K)} \phi\!\left(u - \tfrac{i}{2}\right)\right] \frac{du}{u^2 + \tfrac14}

    The integrand decays like :math:`u^{-2}` even when :math:`\phi` does not
    decay, so the formula is robust for pure-jump models. Puts by parity.

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
    >>> from financialkit.options import black_scholes_price
    >>> m = BlackScholesModel(0.25)
    >>> abs(lewis_price(m, 100.0, 110.0, 0.5, 0.03) - black_scholes_price(100.0, 110.0, 0.5, 0.03, 0.25)) < 1e-8
    True
    """
    option_sign(option_type)
    K = np.asarray(K, dtype=np.float64)
    F = S * np.exp((r - q) * T)
    calls = np.empty(K.shape)
    for idx in np.ndindex(K.shape):
        kappa = np.log(F / K[idx])

        def integrand(u: float, kappa: float = kappa) -> float:
            return float(np.real(np.exp(1j * u * kappa) * model.char_func(u - 0.5j, T))) / (u * u + 0.25)

        integral = quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-12, epsrel=1e-10)[0]
        calls[idx] = S * np.exp(-q * T) - np.sqrt(F * K[idx]) * np.exp(-r * T) / np.pi * integral
    puts = calls - S * np.exp(-q * T) + K * np.exp(-r * T)
    return as_output(calls if option_type == "call" else puts)


def carr_madan_fft(
    model: CharacteristicFunctionModel,
    S: float,
    T: float,
    r: float,
    q: float = 0.0,
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> FFTResult:
    r"""Call prices on a grid of log strikes by the Carr-Madan (1999) FFT.

    The damped call :math:`c(k) = e^{\alpha k} C(k)` has Fourier transform

    .. math::

       \psi(v) = \frac{e^{-rT} \phi_{\ln S_T}(v - (\alpha + 1) i)}{\alpha^2 + \alpha - v^2 + i (2\alpha + 1) v},

    and the inverse transform at :math:`N` log strikes spaced
    :math:`\lambda = 2\pi/(N\eta)` around :math:`\ln F` is one FFT of
    :math:`\psi` sampled at :math:`v_j = j\eta` with Simpson weights.

    Parameters
    ----------
    model : CharacteristicFunctionModel
    S, T, r, q : float
    alpha : float
        Damping exponent; :math:`E[S_T^{\alpha + 1}]` must be finite.
    N : int
        Number of grid points (a power of two is fastest).
    eta : float
        Spacing of the integration grid in :math:`v`.

    Returns
    -------
    FFTResult

    Examples
    --------
    >>> from financialkit.options import black_scholes_price
    >>> res = carr_madan_fft(BlackScholesModel(0.2), 100.0, 1.0, 0.05)
    >>> i = abs(res.strikes - 100.0).argmin()
    >>> bool(abs(res.calls[i] - black_scholes_price(100.0, res.strikes[i], 1.0, 0.05, 0.2)) < 1e-6)
    True
    """
    F = S * np.exp((r - q) * T)
    lam = 2 * np.pi / (N * eta)
    k0 = np.log(F) - 0.5 * N * lam
    v = np.arange(N) * eta
    u = v - (alpha + 1) * 1j
    phi_lnS = np.exp(1j * u * np.log(F)) * model.char_func(u, T)
    psi = np.exp(-r * T) * phi_lnS / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    y = fft(np.exp(-1j * v * k0) * psi * eta * simpson_weights(N))
    k = k0 + lam * np.arange(N)
    calls = np.exp(-alpha * k) / np.pi * np.real(y)
    return FFTResult(strikes=np.exp(k), calls=calls, alpha=float(alpha), eta=float(eta))


def carr_madan_price(
    model: CharacteristicFunctionModel,
    S: float,
    K: ArrayLike,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    alpha: float = 1.5,
    N: int = 4096,
    eta: float = 0.25,
) -> np.ndarray | float:
    """European option prices at arbitrary strikes, by cubic interpolation of :func:`carr_madan_fft`.

    Parameters
    ----------
    model : CharacteristicFunctionModel
    S : float
    K : array_like
    T, r, q : float
    option_type : {"call", "put"}
    alpha, N, eta
        As in :func:`carr_madan_fft`.

    Returns
    -------
    float or ndarray
    """
    option_sign(option_type)
    K = np.asarray(K, dtype=np.float64)
    res = carr_madan_fft(model, S, T, r, q, alpha, N, eta)
    calls = CubicSpline(np.log(res.strikes), res.calls)(np.log(K))
    puts = calls - S * np.exp(-q * T) + K * np.exp(-r * T)
    return as_output(calls if option_type == "call" else puts)
