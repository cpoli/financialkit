"""Jäckel's "Let's be rational" implied volatility.

Peter Jäckel (2015) reduced every Black price to the normalised call
:math:`b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2)` with
:math:`x = \\ln(F/K) \\le 0` (an out-of-the-money call, reached by put-call
symmetry and by subtracting the intrinsic value) and total volatility
:math:`s = \\sigma\\sqrt{T}`. Three ideas give the inverse to machine
precision in two iterations:

* :math:`b` is evaluated without cancellation in every regime: an
  asymptotic expansion for very negative :math:`h = x/s`, a Taylor series
  in :math:`t = s/2` for small total volatility, the scaled complementary
  error function :func:`scipy.special.erfcx` in between, and the normal
  CDF where both terms are large;
* the initial guess interpolates the inverse with Delbourgo-Gregory
  rational cubics on four branches split at :math:`s_c = \\sqrt{2|x|}`, the
  inflection point of :math:`b`, and at the tangents through it, after
  mapping the two outer branches onto nearly linear functions;
* Householder's third-order method then runs on an objective chosen per
  branch: :math:`1/\\ln b` on the lowest, :math:`b` in the middle and
  :math:`\\ln(b_{\\max} - b)` on the highest.

Here the whole array is solved at once: every element computes all four
branches and keeps its own.

References: P. Jäckel, "Let's Be Rational," *Wilmott* 2015(75) 40-53;
P. Jäckel, "By Implication," *Wilmott* 2006(26) 60-66; R. Delbourgo and
J. A. Gregory, "Shape Preserving Piecewise Rational Interpolation,"
*SIAM J. Sci. Stat. Comput.* 6 (1985) 967-976.
"""

from __future__ import annotations

import math
from math import comb

import numpy as np
from scipy.special import erfcx, erfinv, ndtr, ndtri

__all__ = ["normalised_black_call", "normalised_implied_volatility"]

_EPS = np.finfo(np.float64).eps
_SQRT_2PI = np.sqrt(2 * np.pi)
_SMALL_T = 2 * _EPS**0.0625  # Taylor series below this t = s/2
_ASYMPTOTIC_H = -10.0  # asymptotic expansion below this h = x/s
_CDF_SWITCH = 0.85  # normal CDF once h + t exceeds this
_R_MIN = -(1 - np.sqrt(_EPS))
_R_MAX = 2 / (_EPS * _EPS)

# Asymptotic expansion in q = (h / (h^2 - t^2))^2: the k-th term is
# (-1)^k (2k - 1)!! q^k sum_j 2 C(2k + 1, 2j + 1) e^j with e = (t / h)^2.
_N_ASYMPTOTIC = 12
_ASYMPTOTIC_COEFFS = [[(-1) ** k * float(np.prod(np.arange(1, 2 * k, 2))) * 2 * comb(2 * k + 1, 2 * j + 1) for j in range(k + 1)] for k in range(_N_ASYMPTOTIC)]


# Taylor series in w = t^2: the k-th term is w^k / (2k + 1)! times sum_j h^{2j} (c_kj + d_kj a),
# with a = 1 + h N(h) / n(h); pairs (c_kj, d_kj) listed by increasing j.
_SMALL_T_COEFFS = [
    [(0, 1)],
    [(-1, 3), (0, 1)],
    [(-7, 15), (-1, 10), (0, 1)],
    [(-57, 105), (-18, 105), (-1, 21), (0, 1)],
    [(-561, 945), (-285, 1260), (-33, 378), (-1, 36), (0, 1)],
    [(-6555, 10395), (-4680, 17325), (-840, 6930), (-52, 990), (-1, 55), (0, 1)],
    [(-89055, 135135), (-82845, 270270), (-20370, 135135), (-1926, 25740), (-75, 2145), (-1, 78), (0, 1)],
]


def _asymptotic(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """:math:`b` for :math:`h \\ll -1`, where the two erfcx terms nearly cancel."""
    e, r = (t / h) ** 2, (h + t) * (h - t)
    q = (h / r) ** 2
    total = np.zeros_like(h)
    for coeffs in reversed(_ASYMPTOTIC_COEFFS):
        total = total * q + np.polynomial.polynomial.polyval(e, coeffs)
    return np.abs(np.exp(-0.5 * (h * h + t * t)) * (t / r) * total / _SQRT_2PI)


def _small_t(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """:math:`b` as a Taylor series in :math:`t`, where the two CDF terms nearly cancel."""
    a = 1 + h * (0.5 * _SQRT_2PI) * erfcx(-h / np.sqrt(2))
    w, h2 = t * t, h * h
    series = np.zeros_like(h)
    for k in reversed(range(len(_SMALL_T_COEFFS))):
        p_k = np.zeros_like(h)
        for c, d in reversed(_SMALL_T_COEFFS[k]):
            p_k = p_k * h2 + (c + d * a)
        series = series * w + p_k / math.factorial(2 * k + 1)
    series = 2 * t * series
    return np.abs(np.maximum(np.exp(-0.5 * (h2 + w)) * series / _SQRT_2PI, 0.0))


def _erfcx_form(h: np.ndarray, t: np.ndarray) -> np.ndarray:
    """:math:`b = \\tfrac12 e^{-(h^2 + t^2)/2}[\\mathrm{erfcx}(-(h + t)/\\sqrt2) - \\mathrm{erfcx}(-(h - t)/\\sqrt2)]`."""
    return np.abs(np.maximum(0.5 * np.exp(-0.5 * (h * h + t * t)) * (erfcx(-(h + t) / np.sqrt(2)) - erfcx(-(h - t) / np.sqrt(2))), 0.0))


def _cdf_form(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.abs(np.maximum(np.exp(0.5 * x) * ndtr(x / s + 0.5 * s) - np.exp(-0.5 * x) * ndtr(x / s - 0.5 * s), 0.0))


def normalised_black_call(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Normalised Black call :math:`b(x, s) = c / \\sqrt{FK}`, to full relative precision.

    Parameters
    ----------
    x : array_like
        Log-moneyness :math:`\\ln(F/K)`.
    s : array_like
        Total volatility :math:`\\sigma\\sqrt{T} \\ge 0`.

    Returns
    -------
    ndarray

    Examples
    --------
    At the money :math:`b = 2N(s/2) - 1`:

    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> bool(np.isclose(normalised_black_call(0.0, 0.4), 2 * norm.cdf(0.2) - 1))
    True
    """
    x, s = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(s, dtype=np.float64))
    intrinsic = np.where(x > 0, 2 * np.sinh(0.5 * x), 0.0)  # e^{x/2} - e^{-x/2}, the in-the-money part
    xm = -np.abs(x)
    out = np.zeros(x.shape)
    live = s > 0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        h, t = xm / s, 0.5 * s
        asym = live & (h < _ASYMPTOTIC_H) & (h + t < _ASYMPTOTIC_H + _SMALL_T)
        small = live & ~asym & (t < _SMALL_T)
        cdf = live & ~asym & ~small & (h + t > _CDF_SWITCH)
        rest = live & ~asym & ~small & ~cdf
        out[asym] = _asymptotic(h[asym], t[asym])
        out[small] = _small_t(h[small], t[small])
        out[cdf] = _cdf_form(xm[cdf], s[cdf])
        out[rest] = _erfcx_form(h[rest], t[rest])
    return out + intrinsic


def _vega(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """:math:`\\partial b / \\partial s = e^{-(x^2/s^2 + s^2/4)/2} / \\sqrt{2\\pi}`."""
    return np.exp(-0.5 * ((x / s) ** 2 + 0.25 * s * s)) / _SQRT_2PI


def _rational_cubic(x, x_l, x_r, y_l, y_r, d_l, d_r, r):
    """Delbourgo-Gregory rational cubic through (x_l, y_l), (x_r, y_r) with end slopes d_l, d_r."""
    h = x_r - x_l
    u = (x - x_l) / h
    v = 1 - u
    linear = y_r * u + y_l * v
    cubic = (y_r * u**3 + (r * y_r - h * d_r) * u * u * v + (r * y_l + h * d_l) * u * v * v + y_l * v**3) / (1 + (r - 3) * u * v)
    return np.where(np.abs(h) > 0, np.where(r >= _R_MAX, linear, cubic), y_l)


def _minimum_control(d_l, d_r, slope):
    """Smallest control parameter that keeps the interpolant monotone and convex (or concave) where the data are."""
    monotonic = (d_l * slope >= 0) & (d_r * slope >= 0)
    convex = (d_l <= slope) & (slope <= d_r)
    concave = (d_l >= slope) & (slope >= d_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(monotonic, np.where(slope != 0, (d_r + d_l) / slope, _R_MAX), _R_MIN)
        shape_r = np.maximum(np.abs((d_r - d_l) / (d_r - slope)), np.abs((d_r - d_l) / (slope - d_l)))
        degenerate = (slope - d_l == 0) | (d_r - slope == 0)
        r2 = np.where(convex | concave, np.where(degenerate, _R_MAX, shape_r), np.where(monotonic, _R_MAX, _R_MIN))
    r = np.maximum(_R_MIN, np.maximum(np.nan_to_num(r1, nan=_R_MIN), np.nan_to_num(r2, nan=_R_MIN)))
    return np.where(monotonic | convex | concave, r, _R_MIN)


def _control_fitting(x_l, x_r, y_l, y_r, d_l, d_r, second, left: bool):
    """Control parameter matching a second derivative at one end, raised to keep the shape."""
    h = x_r - x_l
    slope = (y_r - y_l) / h
    numerator = 0.5 * h * second + (d_r - d_l)
    denominator = (slope - d_l) if left else (d_r - slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(numerator == 0, 0.0, np.where(denominator == 0, np.where(numerator > 0, _R_MAX, _R_MIN), numerator / denominator))
    return np.maximum(r, _minimum_control(d_l, d_r, slope))


def _householder_factor(newton, halley, hh3):
    return (1 + 0.5 * halley * newton) / (1 + newton * (halley + hh3 * newton / 6))


def normalised_implied_volatility(beta: np.ndarray, x: np.ndarray, n_iter: int = 2) -> np.ndarray:
    """Invert the normalised Black call: the total volatility :math:`s` with :math:`b(x, s) = \\beta`.

    Parameters
    ----------
    beta : array_like
        Normalised call price :math:`c / \\sqrt{FK}`, strictly between the
        normalised intrinsic value and :math:`e^{x/2}`.
    x : array_like
        Log-moneyness :math:`\\ln(F/K)`.
    n_iter : int
        Householder iterations after the rational initial guess; Jäckel
        shows that two reach machine precision.

    Returns
    -------
    ndarray
        Total volatility :math:`\\sigma\\sqrt{T}`.

    Examples
    --------
    >>> import numpy as np
    >>> s = normalised_implied_volatility(normalised_black_call(-0.5, 0.3), -0.5)
    >>> bool(abs(s - 0.3) < 1e-15)
    True
    """
    beta, x = np.broadcast_arrays(np.asarray(beta, dtype=np.float64), np.asarray(x, dtype=np.float64))
    shape = beta.shape
    beta, x = beta.ravel().copy(), x.ravel().copy()
    # reduce to an out-of-the-money call: subtract the intrinsic value and use b(x) - b(-x) = intrinsic
    itm = x > 0
    beta[itm] = beta[itm] - 2 * np.sinh(0.5 * x[itm])
    x = -np.abs(x)
    s = np.empty_like(beta)
    # no time value left (or none possible): Jäckel returns 0 below and infinity above
    s[beta <= 0] = 0.0
    s[beta >= np.exp(0.5 * x)] = np.inf
    atm = (x == 0) & (beta > 0) & (beta < 1)
    s[atm] = 2 * np.sqrt(2) * erfinv(beta[atm])  # b(0, s) = 2 N(s/2) - 1 = erf(s / 2 sqrt 2)
    solve = (x < 0) & (beta > 0) & (beta < np.exp(0.5 * x))
    if not np.any(solve):
        return s.reshape(shape)
    b, xo = beta[solve], x[solve]
    b_max = np.exp(0.5 * xo)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        s_c = np.sqrt(2 * np.abs(xo))  # inflection point of b in s
        b_c, v_c = normalised_black_call(xo, s_c), _vega(xo, s_c)
        s_l = s_c - b_c / v_c  # tangent at s_c hits b = 0 ...
        s_u = s_c + (b_max - b_c) / v_c  # ... and b = b_max
        s_l = np.maximum(s_l, 0.0)
        b_l, b_u = normalised_black_call(xo, s_l), normalised_black_call(xo, s_u)
        v_l, v_u = _vega(xo, s_l), _vega(xo, s_u)

        # lowest branch: f = 2 pi |x| N(-|x| / (sqrt 3 s))^3 / sqrt 27 is nearly linear in b near 0
        ax = np.abs(xo)
        c3 = ax / np.sqrt(3)
        z = c3 / s_l
        Nz, nz = ndtr(-z), np.exp(-0.5 * z * z) / _SQRT_2PI
        A = 2 * np.pi * ax / np.sqrt(27)
        f_l = A * Nz**3
        fs = 3 * A * Nz**2 * nz * z * z / c3
        fss = (3 * A / c3) * Nz * nz * z * (-2 * nz * z - Nz * z * z + 2 * Nz) * (-z * z / c3)
        bss_over_bs = (xo / s_l) ** 2 / s_l - 0.25 * s_l
        fp_l = fs / v_l
        fpp_l = (fss - fs * bss_over_bs) / v_l**2
        r_ll = _control_fitting(0.0, b_l, 0.0, f_l, 1.0, fp_l, fpp_l, left=False)
        f = _rational_cubic(b, 0.0, b_l, 0.0, f_l, 1.0, fp_l, r_ll)
        y = np.cbrt(f / A)
        guess_low = np.where(f > 0, -c3 / ndtri(y), 0.0)

        # two central branches: interpolate s(b) directly, with s'' = 0 at the inflection point
        r_lm = _control_fitting(b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, 0.0, left=False)
        guess_lm = _rational_cubic(b, b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, r_lm)
        r_hm = _control_fitting(b_c, b_u, s_c, s_u, 1 / v_c, 1 / v_u, 0.0, left=True)
        guess_hm = _rational_cubic(b, b_c, b_u, s_c, s_u, 1 / v_c, 1 / v_u, r_hm)

        # highest branch: f = N(-s/2) runs linearly into 0 at b_max
        f_u = ndtr(-0.5 * s_u)
        fs_u = -0.5 * np.exp(-0.125 * s_u * s_u) / _SQRT_2PI
        fss_u = 0.125 * s_u * np.exp(-0.125 * s_u * s_u) / _SQRT_2PI
        bss_over_bs_u = (xo / s_u) ** 2 / s_u - 0.25 * s_u
        fp_u = fs_u / v_u
        fpp_u = (fss_u - fs_u * bss_over_bs_u) / v_u**2
        r_hh = _control_fitting(b_u, b_max, f_u, 0.0, fp_u, -0.5, fpp_u, left=True)
        f_hi = _rational_cubic(b, b_u, b_max, f_u, 0.0, fp_u, -0.5, r_hh)
        guess_high = -2 * ndtri(np.clip(f_hi, 0.0, 0.5))

        low, high = b < b_l, b > b_u
        sigma = np.select([low, b <= b_c, ~high], [guess_low, guess_lm, guess_hm], guess_high)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, s_c)
        ln_beta = np.log(b)

        for _ in range(n_iter):
            bb, vv = normalised_black_call(xo, sigma), _vega(xo, sigma)
            hh = xo / sigma
            halley_b = hh * hh / sigma - 0.25 * sigma  # b'' / b'
            hh3_b = halley_b**2 - 3 * (hh / sigma) ** 2 - 0.25  # b''' / b'
            # lowest branch: objective 1 / ln b
            ln_b = np.log(bb)
            bpob = vv / bb
            newton_lo = (ln_beta - ln_b) * ln_b / ln_beta / bpob
            halley_lo = halley_b - bpob * (1 + 2 / ln_b)
            hh3_lo = hh3_b + 2 * bpob**2 * (1 + 3 / ln_b * (1 + 1 / ln_b)) - 3 * halley_b * bpob * (1 + 2 / ln_b)
            # middle: objective b
            newton_mid = (b - bb) / vv
            # highest branch: objective ln(b_max - b)
            gp = vv / (b_max - bb)
            newton_hi = -np.log((b_max - b) / (b_max - bb)) / gp
            halley_hi = halley_b + gp
            hh3_hi = hh3_b + gp * (2 * gp + 3 * halley_b)
            step = np.select(
                [low & (bb > 0), high & (bb < b_max)],
                [newton_lo * _householder_factor(newton_lo, halley_lo, hh3_lo), newton_hi * _householder_factor(newton_hi, halley_hi, hh3_hi)],
                newton_mid * _householder_factor(newton_mid, halley_b, hh3_b),
            )
            step = np.where(np.isfinite(step), step, 0.0)
            sigma = np.maximum(sigma + step, 0.5 * sigma)  # never jump below half the current value
    s[solve] = sigma
    return s.reshape(shape)
