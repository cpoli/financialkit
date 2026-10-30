"""Delta-gamma VaR: the exact distribution of a quadratic profit and loss.

A book of options is approximated to second order in its risk-factor
moves, :math:`\\Delta V \\approx \\theta h + \\delta^\\top \\Delta X +
\\tfrac12 \\Delta X^\\top \\Gamma \\Delta X` with
:math:`\\Delta X \\sim N(0, \\Sigma h)`. Writing
:math:`\\Delta X = C z` with :math:`C C^\\top = \\Sigma h` and
diagonalising :math:`C^\\top \\Gamma C = P \\Lambda P^\\top` turns it
into a sum of independent noncentral chi-squared terms,

.. math::

   \\Delta V = \\theta h + \\sum_k \\big(b_k y_k + \\tfrac12 \\lambda_k y_k^2\\big),
   \\qquad b = P^\\top C^\\top \\delta, \\quad y \\sim N(0, I),

whose characteristic function is known in closed form. Rouvinez (1997)
inverted it numerically to get the exact distribution, and with it the
VaR, instead of matching moments.

References: C. Rouvinez, "Going Greek with VaR," *Risk* 10(2) (1997)
57-65; J. P. Imhof, "Computing the Distribution of Quadratic Forms in
Normal Variables," *Biometrika* 48 (1961) 419-426; J. Gil-Pelaez, "Note
on the Inversion Theorem," *Biometrika* 38 (1951) 481-482.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, optimize

from financialkit.risk.systems.var import cornish_fisher_var
from financialkit.risk.utils.tail import check_alpha

__all__ = ["delta_gamma_cumulants", "delta_gamma_cdf", "delta_gamma_var"]


def _terms(delta: ArrayLike, gamma: ArrayLike, cov: ArrayLike, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.atleast_1d(np.asarray(delta, dtype=np.float64))
    G = np.atleast_2d(np.asarray(gamma, dtype=np.float64))
    S = np.atleast_2d(np.asarray(cov, dtype=np.float64)) * horizon
    if G.shape != (d.size, d.size) or S.shape != G.shape:
        raise ValueError(f"delta (d,), gamma (d, d) and cov (d, d) must agree, got {d.shape}, {G.shape}, {S.shape}")
    # C = V sqrt(D) also handles a singular covariance, unlike Cholesky.
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    C = V * np.sqrt(np.clip(w, 0.0, None))
    lam, P = np.linalg.eigh(C.T @ (0.5 * (G + G.T)) @ C)
    return P.T @ (C.T @ d), lam


def delta_gamma_cumulants(delta: ArrayLike, gamma: ArrayLike, cov: ArrayLike, horizon: float = 1.0, theta: float = 0.0) -> np.ndarray:
    """First four cumulants of the delta-gamma profit and loss.

    From the cumulant generating function of the diagonal form,
    :math:`\\kappa_1 = \\theta h + \\tfrac12 \\sum \\lambda_k`,
    :math:`\\kappa_2 = \\sum (b_k^2 + \\tfrac12 \\lambda_k^2)`,
    :math:`\\kappa_3 = \\sum (3 b_k^2 \\lambda_k + \\lambda_k^3)` and
    :math:`\\kappa_4 = \\sum (12 b_k^2 \\lambda_k^2 + 3 \\lambda_k^4)`.

    Parameters
    ----------
    delta : array_like, shape (d,)
        First derivatives of the book's value in the risk factors.
    gamma : array_like, shape (d, d)
        Second derivatives.
    cov : array_like, shape (d, d)
        Covariance of the one-period factor moves.
    horizon : float
        Number of periods.
    theta : float
        Value drift per period (the book's theta).

    Returns
    -------
    ndarray, shape (4,)

    Examples
    --------
    A short gamma position in one factor is a scaled chi-squared:
    mean :math:`-1/2`, variance :math:`1/2`.

    >>> delta_gamma_cumulants([0.0], [[-1.0]], [[1.0]]).tolist()
    [-0.5, 0.5, -1.0, 3.0]
    """
    b, lam = _terms(delta, gamma, cov, horizon)
    b2 = b**2
    return np.array(
        [
            theta * horizon + 0.5 * lam.sum(),
            np.sum(b2 + 0.5 * lam**2),
            np.sum(3.0 * b2 * lam + lam**3),
            np.sum(12.0 * b2 * lam**2 + 3.0 * lam**4),
        ]
    )


def _make_cdf(b: np.ndarray, lam: np.ndarray, shift: float):
    """Gil-Pelaez cdf of a quadratic form already scaled to unit variance."""
    # Factors with negligible curvature are linear: their b^2 t^2 term damps.
    quad = np.abs(lam) > 1e-8
    lin_b2 = float(np.sum(b[~quad] ** 2))
    bq, lq = b[quad], lam[quad]
    # The b-terms of curved factors rotate the phase by -b^2 / (2 lam) per
    # unit t for large t; removing that slope c leaves a smooth envelope
    # psi, so the tail integral is a Fourier integral (QUADPACK's QAWF).
    c = shift - 0.5 * float(np.sum(bq**2 / lq))

    def psi(t: float) -> complex:
        z = 1.0 - 1j * lq * t
        log_phi = 1j * shift * t - 0.5 * lin_b2 * t**2 + np.sum(-0.5 * np.log(z) - 0.5 * bq**2 * t**2 / z)
        return complex(np.exp(log_phi - 1j * c * t))

    def cdf(x: float) -> float:
        w = x - c

        def head(t: float) -> float:
            return (psi(t) * np.exp(-1j * w * t)).imag / t

        # The head covers a full period of e^{-iwt}, so QAWF's cycles start
        # where psi / t varies slowly across each of them.
        if abs(w) < 1e-9:
            w = 0.0
        t0 = max(1.0, 2.0 * np.pi / abs(w)) if w else 1.0
        # Decade breakpoints keep each piece smooth when t0 is large.
        edges = np.r_[0.0, np.geomspace(1.0, t0, int(np.ceil(np.log10(t0))) + 1)] if t0 > 1.0 else np.array([0.0, 1.0])
        val = sum(integrate.quad(head, lo, hi, limit=200)[0] for lo, hi in zip(edges[:-1], edges[1:], strict=True))
        if w == 0.0:
            val += integrate.quad(lambda t: psi(t).imag / t, t0, np.inf, limit=200)[0]
        else:
            # Im[psi e^{-iwt}] = Im psi cos(wt) - Re psi sin(wt); shift t -> t0 + s.
            a = abs(w)
            sgn = np.sign(w)
            cos0, sin0 = np.cos(w * t0), np.sin(w * t0)
            f_im = lambda s: psi(t0 + s).imag / (t0 + s)
            f_re = lambda s: psi(t0 + s).real / (t0 + s)
            # cos(w(t0 + s)) = cos0 cos(as) - sin0 sgn sin(as); sin(w(t0 + s)) = sin0 cos(as) + cos0 sgn sin(as).
            ic_im = integrate.quad(f_im, 0.0, np.inf, weight="cos", wvar=a)[0]
            is_im = integrate.quad(f_im, 0.0, np.inf, weight="sin", wvar=a)[0]
            ic_re = integrate.quad(f_re, 0.0, np.inf, weight="cos", wvar=a)[0]
            is_re = integrate.quad(f_re, 0.0, np.inf, weight="sin", wvar=a)[0]
            val += (cos0 * ic_im - sin0 * sgn * is_im) - (sin0 * ic_re + cos0 * sgn * is_re)
        return float(np.clip(0.5 - val / np.pi, 0.0, 1.0))

    return cdf


def delta_gamma_cdf(x: ArrayLike, delta: ArrayLike, gamma: ArrayLike, cov: ArrayLike, horizon: float = 1.0, theta: float = 0.0) -> np.ndarray | float:
    """Exact distribution function of the delta-gamma profit and loss.

    Gil-Pelaez inversion of the characteristic function

    .. math::

       \\varphi(t) = e^{i\\theta h t} \\prod_k (1 - i\\lambda_k t)^{-1/2}
       \\exp\\left(-\\frac{b_k^2 t^2}{2(1 - i\\lambda_k t)}\\right),
       \\qquad F(x) = \\frac12 - \\frac1\\pi \\int_0^\\infty \\frac{\\operatorname{Im}\\big[e^{-itx}\\varphi(t)\\big]}{t}\\, dt,

    with :func:`scipy.integrate.quad`; the slowly decaying tail of the
    integral is done as a Fourier integral (QUADPACK's QAWF).

    Parameters
    ----------
    x : array_like
        Profit-and-loss levels.
    delta, gamma, cov, horizon, theta
        As in :func:`delta_gamma_cumulants`.

    Returns
    -------
    float or ndarray
        :math:`P(\\Delta V \\le x)`, shaped like ``x``.

    Examples
    --------
    With no gamma the profit and loss is normal.

    >>> round(delta_gamma_cdf(-2.0, [1.0], [[0.0]], [[1.0]]), 6)
    0.02275
    """
    b, lam = _terms(delta, gamma, cov, horizon)
    k = delta_gamma_cumulants(delta, gamma, cov, horizon, theta)
    s = float(np.sqrt(k[1]))
    xs = np.asarray(x, dtype=np.float64)
    if s == 0.0:
        out = (xs >= k[0]).astype(np.float64)
    else:
        cdf = _make_cdf(b / s, lam / s, theta * horizon / s)
        out = np.vectorize(cdf, otypes=[np.float64])(xs / s)
    return float(out) if out.ndim == 0 else out


def delta_gamma_var(
    delta: ArrayLike,
    gamma: ArrayLike,
    cov: ArrayLike,
    alpha: float = 0.99,
    horizon: float = 1.0,
    theta: float = 0.0,
    method: str = "exact",
) -> float:
    """Delta-gamma VaR (Rouvinez 1997), exact or by Cornish-Fisher.

    ``method="exact"`` solves :math:`F(-\\mathrm{VaR}) = 1 - \\alpha` with
    :func:`scipy.optimize.brentq` on :func:`delta_gamma_cdf`;
    ``method="cornish_fisher"`` feeds the four cumulants of
    :func:`delta_gamma_cumulants` to
    :func:`~financialkit.risk.systems.var.cornish_fisher_var`, the moment
    approximation the exact inversion replaced. With ``gamma = 0`` both are
    the delta-normal VaR.

    Parameters
    ----------
    delta : array_like, shape (d,)
        First derivatives of the book's value in the risk factors.
    gamma : array_like, shape (d, d)
        Second derivatives.
    cov : array_like, shape (d, d)
        Covariance of the one-period factor moves.
    alpha : float
        Confidence level.
    horizon : float
        Number of periods.
    theta : float
        Value drift per period.
    method : {"exact", "cornish_fisher"}

    Returns
    -------
    float
        A positive loss.

    Examples
    --------
    One factor, short gamma: the loss is :math:`\\tfrac12 \\chi^2_1`, so
    the 99% VaR is half the chi-squared quantile 6.6349.

    >>> round(delta_gamma_var([0.0], [[-1.0]], [[1.0]], 0.99), 4)
    3.3174
    """
    alpha = check_alpha(alpha)
    k = delta_gamma_cumulants(delta, gamma, cov, horizon, theta)
    s = float(np.sqrt(k[1]))
    if method == "cornish_fisher":
        skew = k[2] / s**3 if s else 0.0
        kurt = k[3] / s**4 if s else 0.0
        return float(cornish_fisher_var(k[0], s, skew, kurt, alpha))
    if method != "exact":
        raise ValueError(f"method must be 'exact' or 'cornish_fisher', got {method!r}")
    if s == 0.0:
        return float(-k[0])
    b, lam = _terms(delta, gamma, cov, horizon)
    cdf = _make_cdf(b / s, lam / s, theta * horizon / s)
    p, m = 1.0 - alpha, k[0] / s
    # Cantelli's inequality brackets the p-quantile of a unit-variance law.
    lo, hi = m - np.sqrt(1.0 / p) - 1.0, m + np.sqrt(1.0 / (1.0 - p)) + 1.0
    x = optimize.brentq(lambda y: cdf(y) - p, lo, hi, xtol=1e-12)
    return float(-x * s)
