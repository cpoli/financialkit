"""Rough volatility: fractional kernels, rough Bergomi and the roughness of volatility.

Bayer, Friz and Gatheral (2016) drove the variance by a Riemann-Liouville
(Volterra) process

.. math::

   \\tilde W_t = \\sqrt{2H} \\int_0^t (t - s)^{H - 1/2}\\, dW_s, \\qquad
   \\operatorname{Var}\\tilde W_t = t^{2H},

and wrote the rough Bergomi model (:class:`RoughBergomiModel`)

.. math::

   v_t = \\xi_0 \\exp\\left(\\eta \\tilde W_t - \\tfrac12 \\eta^2 t^{2H}\\right), \\qquad
   \\frac{dS_t}{S_t} = (r - q)\\,dt + \\sqrt{v_t}\\left(\\rho\\, dW_t + \\sqrt{1 - \\rho^2}\\, dW^\\perp_t\\right).

The kernel is singular at :math:`s = t` for :math:`H < 1/2`, so the
process is not Markov and the SDE schemes of
:mod:`financialkit.integrators` do not apply; the kernels are written here
(:func:`volterra_paths`). Bayer, Friz and Gatheral simulated the Gaussian
vector :math:`(\\tilde W_{t_i}, W_{t_i})` exactly by a Cholesky
factorisation of its covariance (``scheme="cholesky"``), which costs
:math:`O(n^3)` once and :math:`O(n^2)` per path. Bennedsen, Lunde and
Pakkanen's (2017) hybrid scheme (``scheme="hybrid"``) integrates the kernel
exactly over the last step, where it is singular, and approximates it
elsewhere by its value at optimally chosen points, a discrete convolution
done by FFT in :math:`O(n \\log n)` per path. The plain left-point Riemann
sum (``scheme="riemann"``) is kept for comparison: it badly understates the
variance when :math:`H` is small.

Gatheral, Jaisson and Rosenbaum (2018) measured the scaling of
log-volatility increments, :math:`E|\\log\\sigma_{t+\\Delta} - \\log\\sigma_t|^q
\\propto \\Delta^{qH}`, found :math:`H \\approx 0.1` in every market
(:func:`volatility_roughness`), and modelled log-volatility as a fractional
Brownian motion (:func:`fbm_paths`, by Davies and Harte's circulant
embedding).

References: C. Bayer, P. Friz and J. Gatheral, "Pricing under rough
volatility," *Quantitative Finance* 16 (2016) 887-904; M. Bennedsen,
A. Lunde and M. S. Pakkanen, "Hybrid scheme for Brownian semistationary
processes," *Finance and Stochastics* 21 (2017) 931-965; J. Gatheral,
T. Jaisson and M. Rosenbaum, "Volatility is rough," *Quantitative Finance*
18 (2018) 933-949; R. B. Davies and D. S. Harte, *Biometrika* 74 (1987)
95-101; R. McCrickerd and M. S. Pakkanen, *Quantitative Finance* 18
(2018) 1877-1886.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, irfft, rfft
from scipy.special import hyp2f1

from financialkit.integrators import RNGLike
from financialkit.monte_carlo import MonteCarloResult
from financialkit.monte_carlo.utils.numerics import sample_mean
from financialkit.options.utils.common import option_sign
from financialkit.stochastic_volatility.core.base import RoughBergomiPaths, RoughnessEstimate

__all__ = [
    "VOLTERRA_SCHEMES",
    "RoughBergomiModel",
    "volterra_covariance",
    "volterra_paths",
    "hybrid_scheme_variance",
    "rough_bergomi_paths",
    "rough_bergomi_mc_price",
    "fbm_paths",
    "volatility_roughness",
]

#: Every scheme accepted by :func:`volterra_paths`.
VOLTERRA_SCHEMES = ("hybrid", "cholesky", "riemann")


def _check_hurst(H: float) -> None:
    if not 0.0 < H < 1.0:
        raise ValueError(f"H must lie in (0, 1), got {H}")


def _check_grid(T: float, n_steps: int, n_paths: int) -> None:
    if not T > 0:
        raise ValueError(f"T must be positive, got {T}")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")


@dataclass
class RoughBergomiModel:
    """Bayer, Friz and Gatheral's (2016) rough Bergomi model with a flat forward variance.

    Examples
    --------
    >>> m = RoughBergomiModel(H=0.07, eta=1.9, rho=-0.9, xi0=0.235**2)
    >>> m.alpha
    -0.43
    """

    H: float
    """Hurst exponent of the Volterra driver, in (0, 1); rough when below 1/2."""

    eta: float
    """Volatility of log-variance, non-negative."""

    rho: float
    """Correlation between the stock and the variance driver, in [-1, 1]."""

    xi0: float
    """Flat forward variance :math:`\\xi_0(t) = E[v_t]`, positive."""

    def __post_init__(self) -> None:
        _check_hurst(self.H)
        if self.eta < 0 or self.xi0 <= 0:
            raise ValueError("need eta >= 0 and xi0 > 0")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got {self.rho}")

    @property
    def alpha(self) -> float:
        """float: Kernel exponent :math:`\\alpha = H - 1/2`."""
        return round(self.H - 0.5, 15)


def volterra_covariance(H: float, times: ArrayLike) -> np.ndarray:
    """Joint covariance of :math:`(\\tilde W_{t_1..t_n}, W_{t_1..t_n})` for the Volterra process.

    With :math:`\\gamma = 1/2 - H` and :math:`u \\le v` (Bayer, Friz and
    Gatheral 2016, Section 4),

    .. math::

       \\operatorname{Cov}(\\tilde W_u, \\tilde W_v) = u^{2H} G(v/u), \\quad
       G(x) = 2H\\left[\\frac{x^{-\\gamma}}{1 - \\gamma}
       + \\frac{\\gamma\\, x^{-1-\\gamma}}{(1 - \\gamma)(2 - \\gamma)}\\,
       {}_2F_1\\!\\left(1, 1 + \\gamma; 3 - \\gamma; \\tfrac1x\\right)\\right],

    :math:`\\operatorname{Cov}(\\tilde W_v, W_u) = \\frac{\\sqrt{2H}}{H + 1/2}
    \\left(v^{H+1/2} - (v - \\min(u, v))^{H+1/2}\\right)` and
    :math:`\\operatorname{Cov}(W_u, W_v) = \\min(u, v)`.

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1).
    times : array_like, shape (n,)
        Positive times.

    Returns
    -------
    ndarray, shape (2n, 2n)
        Volterra block first, then the Brownian block.

    Examples
    --------
    >>> C = volterra_covariance(0.1, [0.5, 1.0])
    >>> np.round(np.diag(C), 12)
    array([0.87055056, 1.        , 0.5       , 1.        ])
    """
    _check_hurst(H)
    t = np.asarray(times, dtype=np.float64).ravel()
    if np.any(t <= 0):
        raise ValueError("times must be positive")
    g = 0.5 - H
    u = np.minimum.outer(t, t)
    v = np.maximum.outer(t, t)
    x = v / u
    G = 2 * H * (x**-g / (1 - g) + g * x ** (-1 - g) / ((1 - g) * (2 - g)) * hyp2f1(1.0, 1.0 + g, 3.0 - g, 1.0 / x))
    cvv = u ** (2 * H) * G
    cvw = np.sqrt(2 * H) / (H + 0.5) * (t[:, None] ** (H + 0.5) - (t[:, None] - np.minimum(t[:, None], t[None, :])) ** (H + 0.5))
    return np.block([[cvv, cvw], [cvw.T, u]])


def _hybrid_points(alpha: float, n: int) -> np.ndarray:
    """Bennedsen et al.'s optimal evaluation points :math:`b_k`, k = 1..n (Proposition 2.8)."""
    k = np.arange(1, n + 1, dtype=np.float64)
    if abs(alpha) < 1e-12:
        return k - 0.5  # g is constant, so any point in [k - 1, k] is exact
    return ((k ** (alpha + 1) - (k - 1) ** (alpha + 1)) / (alpha + 1)) ** (1.0 / alpha)


def hybrid_scheme_variance(H: float, T: float, n_steps: int, scheme: str = "hybrid") -> np.ndarray:
    """Variance of the discretised Volterra process at each grid time, from the scheme's weights.

    The exact variance is :math:`t^{2H}`; comparing with it shows the
    discretisation error without any sampling noise.

    Parameters
    ----------
    H, T : float
    n_steps : int
    scheme : {"hybrid", "riemann"}

    Returns
    -------
    ndarray, shape (n_steps + 1,)

    Examples
    --------
    >>> v = hybrid_scheme_variance(0.1, 1.0, 100)
    >>> bool(abs(v[-1] - 1.0) < 0.01)
    True
    """
    _check_hurst(H)
    if scheme not in ("hybrid", "riemann"):
        raise ValueError(f"scheme must be 'hybrid' or 'riemann', got {scheme!r}")
    a, dt = H - 0.5, T / n_steps
    points = _hybrid_points(a, n_steps) if scheme == "hybrid" else np.arange(1.0, n_steps + 1)
    w = (points * dt) ** (2 * a) * dt
    if scheme == "hybrid":
        w[0] = dt ** (2 * a + 1) / (2 * a + 1)
    return 2 * H * np.concatenate([[0.0], np.cumsum(w)])


def volterra_paths(
    H: float,
    T: float,
    n_steps: int,
    n_paths: int,
    scheme: str = "hybrid",
    rng: RNGLike = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the Volterra process :math:`\\tilde W` together with its driving Brownian motion.

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1).
    T : float
        Horizon.
    n_steps, n_paths : int
    scheme : {"hybrid", "cholesky", "riemann"}
        Bennedsen-Lunde-Pakkanen's hybrid scheme with :math:`\\kappa = 1`
        (exact Wiener integral over the last step, optimal points before
        it), Bayer-Friz-Gatheral's exact Cholesky sampling on the grid, or
        the left-point Riemann sum. At :math:`H = 1/2` the Volterra process
        is the Brownian motion itself, so the Cholesky covariance is
        singular; use the hybrid scheme there.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
    X : ndarray, shape (n_paths, n_steps + 1)
        :math:`\\tilde W_{t_i}`, with :math:`\\tilde W_0 = 0`.
    dW : ndarray, shape (n_paths, n_steps)
        Increments of the Brownian motion that drives :math:`\\tilde W`.

    Examples
    --------
    >>> t, X, dW = volterra_paths(0.1, 1.0, 50, 20_000, rng=0)
    >>> bool(abs(X[:, -1].var() - 1.0) < 0.05)
    True
    """
    _check_hurst(H)
    _check_grid(T, n_steps, n_paths)
    if scheme not in VOLTERRA_SCHEMES:
        raise ValueError(f"scheme must be one of {VOLTERRA_SCHEMES}, got {scheme!r}")
    rng = np.random.default_rng(rng)
    times = np.linspace(0.0, T, n_steps + 1)
    n, dt, a = n_steps, T / n_steps, H - 0.5
    X = np.zeros((n_paths, n + 1))
    if scheme == "cholesky":
        L = np.linalg.cholesky(volterra_covariance(H, times[1:]))
        Y = rng.standard_normal((n_paths, 2 * n)) @ L.T
        X[:, 1:] = Y[:, :n]
        dW = np.diff(np.concatenate([np.zeros((n_paths, 1)), Y[:, n:]], axis=1), axis=1)
        return times, X, dW
    if scheme == "hybrid":
        # (dW_i, int_{t_i}^{t_{i+1}} (t_{i+1} - s)^alpha dW_s) is Gaussian with this covariance.
        # Its Cholesky factor is written out, since it is singular at H = 1/2.
        c = dt ** (a + 1) / (a + 1)
        resid = np.sqrt(max(dt ** (2 * a + 1) / (2 * a + 1) - c * c / dt, 0.0))
        Z = rng.standard_normal((n_paths, n, 2))
        dW = np.sqrt(dt) * Z[..., 0]
        near = c / np.sqrt(dt) * Z[..., 0] + resid * Z[..., 1]
        gamma = np.zeros(n + 1)
        gamma[2:] = (_hybrid_points(a, n)[1:] * dt) ** a
    else:
        dW = rng.standard_normal((n_paths, n)) * np.sqrt(dt)
        near = np.zeros((n_paths, n))
        gamma = np.zeros(n + 1)
        gamma[1:] = (np.arange(1, n + 1) * dt) ** a
    # far[i] = sum_k gamma_k dW_{i-k}: a causal convolution, done by FFT.
    m = 2 * n + 1
    far = irfft(rfft(dW, m, axis=1) * rfft(gamma, m), m, axis=1)[:, 1 : n + 1]
    X[:, 1:] = np.sqrt(2 * H) * (near + far)
    return times, X, dW


def rough_bergomi_paths(
    model: RoughBergomiModel,
    S: float,
    T: float,
    r: float,
    q: float = 0.0,
    n_steps: int = 100,
    n_paths: int = 10_000,
    scheme: str = "hybrid",
    rng: RNGLike = None,
) -> RoughBergomiPaths:
    """Simulate spot and variance paths under rough Bergomi.

    The Volterra driver comes from :func:`volterra_paths`; the log spot
    takes forward Euler steps with the variance at the left end of each
    step, as in Bayer, Friz and Gatheral (2016).

    Parameters
    ----------
    model : RoughBergomiModel
    S, T, r, q : float
        Spot, horizon, risk-free rate and dividend yield.
    n_steps, n_paths : int
    scheme : {"hybrid", "cholesky", "riemann"}
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    RoughBergomiPaths

    Examples
    --------
    >>> m = RoughBergomiModel(0.1, 1.5, -0.7, 0.04)
    >>> p = rough_bergomi_paths(m, 100.0, 1.0, 0.02, n_steps=50, n_paths=20_000, rng=1)
    >>> bool(abs(p.spot[:, -1].mean() / (100 * np.exp(0.02)) - 1) < 5e-3)
    True
    """
    rng = np.random.default_rng(rng)
    times, X, dW = volterra_paths(model.H, T, n_steps, n_paths, scheme, rng)
    dt = T / n_steps
    v = model.xi0 * np.exp(model.eta * X - 0.5 * model.eta**2 * times ** (2 * model.H))
    dZ = model.rho * dW + np.sqrt(1.0 - model.rho**2) * rng.standard_normal(dW.shape) * np.sqrt(dt)
    vl = v[:, :-1]
    steps = (r - q - 0.5 * vl) * dt + np.sqrt(vl) * dZ
    lnS = np.log(S) + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)
    return RoughBergomiPaths(times, np.exp(lnS), v, scheme)


def rough_bergomi_mc_price(
    model: RoughBergomiModel,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps: int = 100,
    n_paths: int = 10_000,
    scheme: str = "hybrid",
    rng: RNGLike = None,
) -> MonteCarloResult:
    """European option price under rough Bergomi by Monte Carlo on :func:`rough_bergomi_paths`.

    Parameters
    ----------
    model : RoughBergomiModel
    S, K, T, r, q : float
    option_type : {"call", "put"}
    n_steps, n_paths : int
    scheme : str
        See :func:`volterra_paths`.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    MonteCarloResult
    """
    sign = option_sign(option_type)
    paths = rough_bergomi_paths(model, S, T, r, q, n_steps, n_paths, scheme, rng)
    payoff = np.exp(-r * T) * np.maximum(sign * (paths.spot[:, -1] - K), 0.0)
    return sample_mean(payoff, method=f"rough_bergomi_{scheme}")


def fbm_paths(H: float, T: float, n_steps: int, n_paths: int, rng: RNGLike = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact fractional Brownian motion on a uniform grid by Davies and Harte's (1987) circulant embedding.

    The increments (fractional Gaussian noise) have autocovariance
    :math:`\\gamma(k) = \\tfrac12 \\Delta^{2H}(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})`.
    Embedded in a circulant matrix of size :math:`2n`, whose eigenvalues
    are the FFT of its first row and are non-negative, they are sampled by
    one FFT per path.

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1); 1/2 is Brownian motion.
    T : float
    n_steps, n_paths : int
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
    B : ndarray, shape (n_paths, n_steps + 1)
        :math:`B^H_{t_i}`, with :math:`B^H_0 = 0` and :math:`\\operatorname{Var} B^H_t = t^{2H}`.

    Examples
    --------
    >>> t, B = fbm_paths(0.2, 1.0, 64, 20_000, rng=0)
    >>> bool(abs(B[:, -1].var() - 1.0) < 0.05)
    True
    """
    _check_hurst(H)
    _check_grid(T, n_steps, n_paths)
    rng = np.random.default_rng(rng)
    n, dt = n_steps, T / n_steps
    k = np.arange(n + 1, dtype=np.float64)
    acov = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))
    lam = np.maximum(np.real(fft(np.concatenate([acov, acov[-2:0:-1]]))), 0.0)
    Z = rng.standard_normal((n_paths, 2 * n)) + 1j * rng.standard_normal((n_paths, 2 * n))
    fgn = np.real(fft(np.sqrt(lam / (2 * n)) * Z, axis=1))[:, :n] * dt**H
    times = np.linspace(0.0, T, n + 1)
    return times, np.concatenate([np.zeros((n_paths, 1)), np.cumsum(fgn, axis=1)], axis=1)


def volatility_roughness(log_vol: ArrayLike, lags: ArrayLike | None = None, q: ArrayLike = (0.5, 1.0, 1.5, 2.0, 3.0)) -> RoughnessEstimate:
    """Gatheral, Jaisson and Rosenbaum's (2018) scaling estimate of the roughness of volatility.

    For each moment :math:`q` and lag :math:`\\Delta` the empirical
    :math:`m(q, \\Delta) = \\overline{|x_{t+\\Delta} - x_t|^q}` of the
    log-volatility series :math:`x` is computed; a straight line in
    :math:`\\log\\Delta` gives :math:`\\log m(q, \\Delta) \\approx \\zeta_q \\log\\Delta + c_q`,
    and :math:`\\zeta_q = qH` for a process with fractional-Brownian scaling.
    :math:`H` is the least-squares slope of :math:`\\zeta_q` on :math:`q`
    through the origin.

    Parameters
    ----------
    log_vol : array_like, shape (n,)
        Log-volatility observed at equally spaced times.
    lags : array_like of int, optional
        Lags :math:`\\Delta` in observations; 1 to 50 by default.
    q : array_like
        Moments.

    Returns
    -------
    RoughnessEstimate

    Examples
    --------
    >>> t, B = fbm_paths(0.15, 1.0, 4000, 1, rng=2)
    >>> bool(abs(volatility_roughness(B[0]).H - 0.15) < 0.03)
    True
    """
    x = np.asarray(log_vol, dtype=np.float64).ravel()
    lag = np.arange(1, 51) if lags is None else np.asarray(lags, dtype=np.int64).ravel()
    qs = np.asarray(q, dtype=np.float64).ravel()
    if np.any(lag < 1) or lag.max() >= x.size - 1:
        raise ValueError("lags must be at least 1 and shorter than the series")
    if lag.size < 2 or np.any(qs <= 0):
        raise ValueError("need at least two lags and positive moments")
    m = np.array([[np.mean(np.abs(x[d:] - x[:-d]) ** qq) for d in lag] for qq in qs])
    zeta = np.polyfit(np.log(lag), np.log(m).T, 1)[0]
    H = float(qs @ zeta / (qs @ qs))
    return RoughnessEstimate(lag.astype(np.float64), qs, m, zeta, H)
