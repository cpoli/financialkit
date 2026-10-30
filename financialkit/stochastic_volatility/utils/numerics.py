"""Supporting numerics for the stochastic-volatility models: the COS payoff
coefficients, Simpson weights for the FFT, and SDE callbacks for
:mod:`financialkit.integrators`.

The drift and diffusion callbacks follow the integrators' ``f(x, t,
params)`` convention; every model parameter travels in ``params``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "cos_chi",
    "cos_psi",
    "simpson_weights",
    "put_call_from_put",
    "heston_drift",
    "heston_diffusion",
    "local_vol_drift",
    "local_vol_diffusion",
    "local_vol_params",
]

#: Variance treatments of the Euler schemes for Heston, as coded in ``params``
#: (Lord, Koekkoek and van Dijk 2010, Table 1).
HESTON_EULER_SCHEMES = {"full_truncation": 0.0, "partial_truncation": 1.0, "higham_mao": 2.0}


def cos_chi(u: ArrayLike, c: ArrayLike, d: ArrayLike, a: ArrayLike) -> np.ndarray:
    r"""Cosine coefficients of :math:`e^y` on :math:`[c, d]` for an interval starting at ``a``.

    .. math::

       \chi_k(c, d) = \int_c^d e^y \cos(u_k (y - a))\,dy

    Parameters
    ----------
    u : array_like, shape (N,)
        Frequencies :math:`u_k = k\pi/(b - a)`.
    c, d, a : array_like
        Integration limits and the interval's left end, broadcast against ``u``.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> round(float(cos_chi([0.0], 0.0, 1.0, 0.0)[0]), 6)  # e - 1
    1.718282
    """
    u, c, d, a = (np.asarray(v, dtype=np.float64) for v in (u, c, d, a))
    ed, ec = np.exp(d), np.exp(c)
    return (np.cos(u * (d - a)) * ed - np.cos(u * (c - a)) * ec + u * (np.sin(u * (d - a)) * ed - np.sin(u * (c - a)) * ec)) / (1.0 + u * u)


def cos_psi(u: ArrayLike, c: ArrayLike, d: ArrayLike, a: ArrayLike) -> np.ndarray:
    r"""Cosine coefficients of 1 on :math:`[c, d]`: :math:`\psi_k(c, d) = \int_c^d \cos(u_k (y - a))\,dy`.

    Parameters
    ----------
    u, c, d, a : array_like
        As in :func:`cos_chi`.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> cos_psi([0.0], 0.0, 2.0, 0.0)
    array([2.])
    """
    u, c, d, a = (np.asarray(v, dtype=np.float64) for v in (u, c, d, a))
    safe = np.where(u == 0.0, 1.0, u)
    return np.where(u == 0.0, d - c, (np.sin(u * (d - a)) - np.sin(u * (c - a))) / safe)


def simpson_weights(N: int) -> np.ndarray:
    """Simpson's rule weights ``[1, 4, 2, 4, ..., 2, 4] / 3`` used by Carr and Madan (1999).

    Parameters
    ----------
    N : int

    Returns
    -------
    ndarray, shape (N,)

    Examples
    --------
    >>> simpson_weights(5) * 3
    array([1., 4., 2., 4., 2.])
    """
    w = 3.0 + (-1.0) ** (np.arange(N) + 1)
    w[0] = 1.0
    return w / 3.0


def put_call_from_put(puts: ArrayLike, S: float, K: ArrayLike, T: float, r: float, q: float, option_type: str) -> np.ndarray:
    """Return ``puts`` for a put, or the parity call :math:`P + S e^{-qT} - K e^{-rT}`.

    Parameters
    ----------
    puts : array_like
    S : float
    K : array_like
    T, r, q : float
    option_type : {"call", "put"}

    Returns
    -------
    ndarray
    """
    puts = np.asarray(puts, dtype=np.float64)
    if option_type == "put":
        return puts
    return puts + S * np.exp(-q * T) - np.asarray(K, dtype=np.float64) * np.exp(-r * T)


def _heston_parts(x, p):
    v = x[:, 1]
    vp = np.maximum(v, 0.0)
    if p[4] == 2.0:
        vd = np.abs(v)
        vs = vd
    elif p[4] == 1.0:
        vd = v
        vs = vp
    else:
        vd = vp
        vs = vp
    return vd, vs


def heston_drift(x, t, p):
    """Drift of ``(ln S, v)`` under Heston; ``p = [r - q, kappa, theta, xi, scheme]``.

    ``scheme`` 0 is full truncation (the drift and diffusion both use
    :math:`v^+`), 1 partial truncation (only the diffusion does) and 2
    Higham-Mao (the diffusion uses :math:`|v|`).
    """
    vd, vs = _heston_parts(x, p)
    out = np.empty_like(x)
    out[:, 0] = p[0] - 0.5 * vs
    out[:, 1] = p[1] * (p[2] - vd)
    return out


def heston_diffusion(x, t, p):
    """Diffusion :math:`(\\sqrt{v}, \\xi\\sqrt{v})` of ``(ln S, v)``, with ``v`` treated as in :func:`heston_drift`."""
    vd, vs = _heston_parts(x, p)
    out = np.empty_like(x)
    out[:, 0] = np.sqrt(vs)
    out[:, 1] = p[3] * np.sqrt(vs)
    return out


def local_vol_params(mu: float, strikes: ArrayLike, maturities: ArrayLike, vol: ArrayLike) -> np.ndarray:
    """Pack a local-volatility grid into the ``params`` vector of :func:`local_vol_drift`.

    Parameters
    ----------
    mu : float
        :math:`r - q`.
    strikes : array_like, shape (n_K,)
    maturities : array_like, shape (n_T,)
    vol : array_like, shape (n_T, n_K)

    Returns
    -------
    ndarray
        ``[mu, n_K, n_T, strikes..., maturities..., vol (row-major)...]``.
    """
    K = np.asarray(strikes, dtype=np.float64)
    T = np.asarray(maturities, dtype=np.float64)
    V = np.asarray(vol, dtype=np.float64)
    return np.concatenate([[mu, K.size, T.size], K, T, V.ravel()])


def _local_vol(x, t, p):
    nK = int(p[1])
    nT = int(p[2])
    Ks = p[3 : 3 + nK]
    Ts = p[3 + nK : 3 + nK + nT]
    V = np.ascontiguousarray(p[3 + nK + nT : 3 + nK + nT + nK * nT]).reshape((nT, nK))
    S = np.exp(x[:, 0])
    if t <= Ts[0]:
        return np.interp(S, Ks, V[0])
    if t >= Ts[nT - 1]:
        return np.interp(S, Ks, V[nT - 1])
    j = np.searchsorted(Ts, t)
    w = (t - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
    return (1.0 - w) * np.interp(S, Ks, V[j - 1]) + w * np.interp(S, Ks, V[j])


def local_vol_drift(x, t, p):
    """Drift :math:`r - q - \\tfrac12\\sigma(S, t)^2` of :math:`\\ln S`; ``p`` from :func:`local_vol_params`."""
    s = _local_vol(x, t, p)
    out = np.empty_like(x)
    out[:, 0] = p[0] - 0.5 * s * s
    return out


def local_vol_diffusion(x, t, p):
    """Diffusion :math:`\\sigma(S, t)` of :math:`\\ln S`, bilinear in the grid and flat outside it."""
    out = np.empty_like(x)
    out[:, 0] = _local_vol(x, t, p)
    return out
