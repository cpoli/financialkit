"""Linear time-series models: Yule's autoregression, Box-Jenkins ARMA and order selection.

Yule (1927) modelled sunspot numbers as a disturbed second-order
autoregression whose complex roots produce a pseudo-cycle, and estimated it
from the sample autocorrelations (the Yule-Walker equations, solved here by
:func:`scipy.linalg.solve_toeplitz`). Box and Jenkins (1970) turned
autoregressive moving-average models into a method: identify the orders
from the autocorrelation and partial autocorrelation functions, estimate,
check the residuals, forecast. Akaike (1974) replaced the identification
by eye with an information criterion. ARMA(p, q) models are estimated by
conditional maximum likelihood, which for Gaussian innovations is
conditional least squares (:func:`scipy.optimize.least_squares`; pure
autoregressions are solved exactly by :func:`numpy.linalg.lstsq`).
References: G. U. Yule, Philosophical Transactions of the Royal Society A
226 (1927), 267-298; G. E. P. Box and G. M. Jenkins, *Time Series
Analysis: Forecasting and Control* (Holden-Day, 1970), Chs. 3, 6-7; H.
Akaike, IEEE Transactions on Automatic Control 19(6) (1974), 716-723; G.
Schwarz, Annals of Statistics 6(2) (1978), 461-464; J. F. Monahan,
Biometrika 71(2) (1984), 403-404; Hamilton, *Time Series
Analysis* (Princeton, 1994), Chs. 3-5.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.optimize import least_squares
from scipy.signal import lfilter

from financialkit.time_series.core.base import ARMAResult, OrderSelectionResult
from financialkit.time_series.systems.efficiency import autocorrelation, autocovariance

__all__ = [
    "yule_walker",
    "ar_cycle_period",
    "partial_autocorrelation",
    "arma_psi_weights",
    "arma_acf",
    "simulate_arma",
    "fit_arma",
    "select_arma_order",
]


def _as_coeffs(c) -> np.ndarray:
    return np.atleast_1d(np.asarray([] if c is None else c, dtype=np.float64))


def _conditional_result(x: np.ndarray, ar, ma, mean: float, start: int, include_mean: bool, sigma2: Optional[float] = None) -> ARMAResult:
    ar, ma = _as_coeffs(ar), _as_coeffs(ma)
    e = _css_residuals(x, ar, ma, mean, start)
    n = e.size
    s2 = float(e @ e / n) if sigma2 is None else float(sigma2)
    ll = -0.5 * n * (math.log(2 * math.pi * s2)) - 0.5 * float(e @ e) / s2
    k = ar.size + ma.size + int(include_mean) + 1
    return ARMAResult(ar=ar, ma=ma, mean=mean, sigma2=s2, loglik=ll, nobs=n, n_params=k, residuals=e, data=x)


def _css_residuals(x: np.ndarray, ar: np.ndarray, ma: np.ndarray, mean: float, start: int) -> np.ndarray:
    """Innovations for ``t >= start``, with pre-sample innovations set to zero."""
    xc = x - mean
    w = lfilter(np.r_[1.0, -ar], [1.0], xc)[start:]
    return lfilter([1.0], np.r_[1.0, ma], w)


def yule_walker(x, order: int, demean: bool = True) -> ARMAResult:
    """Fit an AR(`order`) model by solving the Yule-Walker equations.

    The equations :math:`\\rho_k = \\sum_{i=1}^p \\phi_i \\rho_{k-i}`,
    ``k = 1..p``, form a Toeplitz system in the sample autocorrelations, and
    :math:`\\sigma^2 = \\hat\\gamma_0 (1 - \\sum_i \\phi_i \\hat\\rho_i)`.
    Yule's (1927) sunspot fit was :math:`\\phi_1 \\approx 1.34`,
    :math:`\\phi_2 \\approx -0.65`.

    Parameters
    ----------
    x : array_like
    order : int
    demean : bool
        Subtract the sample mean first.

    Returns
    -------
    ARMAResult
        With ``ma`` empty; the log-likelihood is conditional on the first `order` values.

    Examples
    --------
    >>> import numpy as np
    >>> x = simulate_arma(ar=[1.34, -0.65], n=50_000, rng=0)
    >>> yule_walker(x, 2).ar.round(2).tolist()
    [1.34, -0.65]
    """
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean()) if demean else 0.0
    g = autocovariance(x, order) if demean else _raw_autocovariance(x, order)
    phi = solve_toeplitz(g[:order], g[1 : order + 1]) if order > 0 else np.empty(0)
    sigma2 = float(g[0] - phi @ g[1 : order + 1])
    return _conditional_result(x, phi, None, mean, order, demean, sigma2)


def _raw_autocovariance(x: np.ndarray, nlags: int) -> np.ndarray:
    n = x.size
    return np.array([x[k:] @ x[: n - k] / n for k in range(nlags + 1)])


def ar_cycle_period(ar) -> float:
    """Pseudo-period of the dominant complex pair of AR roots, in periods (``inf`` if all roots are real).

    For AR(2) with :math:`\\phi_2 < 0` and :math:`\\phi_1^2 + 4\\phi_2 < 0`,
    the period is :math:`2\\pi / \\arccos\\bigl(\\phi_1 / (2\\sqrt{-\\phi_2})\\bigr)`.

    Examples
    --------
    >>> round(ar_cycle_period([1.34, -0.65]), 2)
    10.65
    """
    ar = _as_coeffs(ar)
    z = np.roots(np.r_[1.0, -ar])  # inverse roots of phi(B)
    z = z[np.abs(z.imag) > 1e-12]
    if z.size == 0:
        return math.inf
    dom = z[np.argmax(np.abs(z))]
    return float(2 * math.pi / abs(np.angle(dom)))


def partial_autocorrelation(x, nlags: int = 20) -> np.ndarray:
    """Sample partial autocorrelations by the Durbin-Levinson recursion, ``k = 0..nlags``.

    The lag-`k` value is the last coefficient of the AR(`k`) Yule-Walker
    fit; it cuts off after lag `p` for an AR(`p`) process, which is how Box
    and Jenkins identify the autoregressive order.

    Examples
    --------
    >>> import numpy as np
    >>> pacf = partial_autocorrelation(simulate_arma(ar=[0.6], n=20_000, rng=1), 3)
    >>> round(float(pacf[1]), 1), bool(abs(pacf[2]) < 0.03)
    (0.6, True)
    """
    rho = autocorrelation(x, nlags)
    out = np.empty(nlags + 1)
    out[0] = 1.0
    phi = np.empty(0)
    v = 1.0
    for k in range(1, nlags + 1):
        a = (rho[k] - phi @ rho[1:k][::-1]) / v
        phi = np.r_[phi - a * phi[::-1], a]
        v *= 1 - a * a
        out[k] = a
    return out


def arma_psi_weights(ar, ma, n: int) -> np.ndarray:
    """Weights :math:`\\psi_0 = 1, \\psi_1, \\dots, \\psi_{n-1}` of :math:`x_t = \\sum_j \\psi_j \\varepsilon_{t-j}`.

    Examples
    --------
    >>> arma_psi_weights([0.5], [0.4], 4).round(4).tolist()
    [1.0, 0.9, 0.45, 0.225]
    """
    impulse = np.zeros(n)
    impulse[0] = 1.0
    return lfilter(np.r_[1.0, _as_coeffs(ma)], np.r_[1.0, -_as_coeffs(ar)], impulse)


def arma_acf(ar=None, ma=None, nlags: int = 20, n_terms: int = 2000) -> np.ndarray:
    """Theoretical autocorrelations ``k = 0..nlags`` of a stationary ARMA process.

    Uses :math:`\\gamma_k = \\sigma^2 \\sum_j \\psi_j \\psi_{j+k}` with the
    MA(:math:`\\infty`) weights truncated at `n_terms`.

    Examples
    --------
    >>> arma_acf(ar=[0.5], nlags=3).round(6).tolist()
    [1.0, 0.5, 0.25, 0.125]
    """
    psi = arma_psi_weights(ar, ma, n_terms + nlags)
    g = np.array([psi[:n_terms] @ psi[k : k + n_terms] for k in range(nlags + 1)])
    return g / g[0]


def simulate_arma(
    ar=None, ma=None, n: int = 1000, sigma: float = 1.0, mean: float = 0.0, burn: int = 500, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Simulate :math:`\\phi(B)(x_t - \\mu) = \\theta(B)\\varepsilon_t` with Gaussian innovations.

    Filters white noise through :func:`scipy.signal.lfilter` and discards
    `burn` start-up values.

    Parameters
    ----------
    ar, ma : array_like, optional
        :math:`\\phi_1..\\phi_p` and :math:`\\theta_1..\\theta_q`.
    n : int
    sigma : float
        Innovation standard deviation.
    mean : float
    burn : int
    rng : numpy.random.Generator, int or None

    Returns
    -------
    ndarray, shape (n,)
    """
    rng = np.random.default_rng(rng)
    e = sigma * rng.standard_normal(n + burn)
    return mean + lfilter(np.r_[1.0, _as_coeffs(ma)], np.r_[1.0, -_as_coeffs(ar)], e)[burn:]


def fit_arma(x, p: int, q: int = 0, include_mean: bool = True, start: Optional[int] = None) -> ARMAResult:
    """Fit ARMA(`p`, `q`) by conditional maximum likelihood (Box-Jenkins estimation).

    Conditioning on the first `start` observations and on zero pre-sample
    innovations, the Gaussian log-likelihood is maximized by minimizing the
    sum of squared innovations, which :func:`scipy.optimize.least_squares`
    does over a parameterization that keeps the AR part stationary and the
    MA part invertible (Monahan's transform of the partial
    autocorrelations); :math:`\\hat\\sigma^2` is the mean squared innovation. A pure AR
    model is linear in its parameters and is solved exactly by
    :func:`numpy.linalg.lstsq`.

    Parameters
    ----------
    x : array_like
    p, q : int
        Autoregressive and moving-average orders.
    include_mean : bool
        Estimate the mean :math:`\\mu` (otherwise :math:`\\mu = 0`).
    start : int, optional
        Observations conditioned on (default `p`). Use a common value to
        compare models of different orders on the same sample.

    Returns
    -------
    ARMAResult

    Examples
    --------
    >>> x = simulate_arma(ar=[0.7], ma=[0.4], n=5000, mean=1.0, rng=2)
    >>> res = fit_arma(x, 1, 1)
    >>> res.ar.round(1).tolist(), res.ma.round(1).tolist(), round(res.mean, 1)
    ([0.7], [0.4], 1.0)
    """
    x = np.asarray(x, dtype=np.float64)
    start = p if start is None else start
    if start < p:
        raise ValueError("start must be at least p")
    if x.size - start <= p + q + 1:
        raise ValueError("series too short for this order")
    if q == 0:
        cols = [x[start - i : x.size - i] for i in range(1, p + 1)]
        X = np.column_stack(([np.ones(x.size - start)] if include_mean else []) + cols) if (cols or include_mean) else np.empty((x.size - start, 0))
        y = x[start:]
        coef = np.linalg.lstsq(X, y, rcond=None)[0] if X.shape[1] else np.empty(0)
        phi = coef[int(include_mean) :]
        c = coef[0] if include_mean else 0.0
        mean = float(c / (1 - phi.sum())) if include_mean and abs(1 - phi.sum()) > 1e-12 else 0.0
        return _conditional_result(x, phi, None, mean, start, include_mean)

    # Stationarity and invertibility are imposed by optimizing over
    # unconstrained transforms of the partial autocorrelations (Monahan 1984).
    phi0 = yule_walker(x, p).ar if p else np.empty(0)
    u0 = np.arctanh(np.clip(_coeffs_to_pacf(phi0), -0.99, 0.99))
    m0 = [float(x.mean())] if include_mean else []

    def unpack(v):
        mean = v[0] if include_mean else 0.0
        o = int(include_mean)
        return mean, _pacf_to_coeffs(np.tanh(v[o : o + p])), -_pacf_to_coeffs(np.tanh(v[o + p :]))

    def resid(v):
        mean, ar, ma = unpack(v)
        e = _css_residuals(x, ar, ma, mean, start)
        return np.where(np.isfinite(e), e, 1e6)

    v0 = np.r_[m0, u0, np.zeros(q)]
    sol = least_squares(resid, v0, method="lm" if v0.size <= x.size - start else "trf")
    mean, ar, ma = unpack(sol.x)
    return _conditional_result(x, ar, ma, float(mean), start, include_mean)


def _pacf_to_coeffs(r: np.ndarray) -> np.ndarray:
    """Coefficients of a stationary :math:`1 - \\sum_j \\phi_j B^j` from partial autocorrelations in (-1, 1)."""
    phi = np.empty(0)
    for rk in r:
        phi = np.r_[phi - rk * phi[::-1], rk]
    return phi


def _coeffs_to_pacf(phi: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_pacf_to_coeffs` (the backward Levinson recursion)."""
    phi = np.asarray(phi, dtype=np.float64)
    r = np.empty(phi.size)
    for k in range(phi.size, 0, -1):
        rk = phi[-1]
        r[k - 1] = rk
        phi = (phi[:-1] + rk * phi[:-1][::-1]) / (1 - rk * rk)
    return r


def select_arma_order(x, max_p: int = 4, max_q: int = 2, include_mean: bool = True) -> OrderSelectionResult:
    """AIC and BIC of every ARMA(p, q) with ``p <= max_p`` and ``q <= max_q``.

    All models are fitted on the same sample (conditioning on the first
    `max_p` observations), so their likelihoods are comparable. Akaike's
    criterion :math:`-2\\ell + 2k` estimates the expected Kullback-Leibler
    divergence and tends to overfit in large samples; Schwarz's
    :math:`-2\\ell + k\\ln n` is consistent.

    Returns
    -------
    OrderSelectionResult

    Examples
    --------
    >>> x = simulate_arma(ar=[0.5, 0.3], n=3000, rng=4)
    >>> select_arma_order(x, max_p=4, max_q=0).best("bic")
    (2, 0)
    """
    x = np.asarray(x, dtype=np.float64)
    aic = np.full((max_p + 1, max_q + 1), np.nan)
    bic = np.full_like(aic, np.nan)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                res = fit_arma(x, p, q, include_mean=include_mean, start=max_p)
            except (ValueError, np.linalg.LinAlgError):
                continue
            aic[p, q], bic[p, q] = res.aic, res.bic
    return OrderSelectionResult(aic=aic, bic=bic)
