"""Adjoint Monte Carlo Greeks in the LIBOR market model (Giles-Glasserman 2006).

Pathwise derivatives propagated forward through a simulation cost one
tangent per input: the deltas to :math:`N` forward rates need :math:`N`
tangents, each updated at every time step. Giles and Glasserman
("smoking adjoints") ran the chain rule backwards instead. One adjoint
vector, swept from the payoff to time zero, gives the derivative of the
payoff to every input at once, for a cost that does not depend on how
many inputs there are.

The test case is theirs: one-factor LIBOR market model, forward rates
:math:`L_i` for the accrual periods :math:`[T_i, T_{i+1}]`,
:math:`T_i = i\\delta`, Euler steps of size :math:`h = \\delta` in
:math:`\\log L` under the spot measure,

.. math::

    L_i(n+1) = L_i(n) \\exp\\Bigl[\\bigl(\\mu_i(n) - \\tfrac12\\sigma_i^2\\bigr)h
    + \\sigma_i\\sqrt h\\,Z_{n+1}\\Bigr], \\qquad
    \\mu_i(n) = \\sigma_i \\sum_{j=n+1}^{i} \\frac{\\sigma_j\\delta L_j(n)}{1 + \\delta L_j(n)},

for :math:`i \\ge n + 1`, with the rates that have fixed held constant. A
portfolio of payer swaptions expiring at :math:`T_m` pays
:math:`\\sum_s \\max(0, \\sum_{j=m}^{m+k_s-1} \\delta (L_j - K_s) P_j)` with
:math:`P_j = \\prod_{l=m}^{j} (1 + \\delta L_l)^{-1}`, divided by the
spot numeraire :math:`B(T_m) = \\prod_{l<m} (1 + \\delta L_l(T_l))`.

The adjoint of one Euler step needs two reverse cumulative sums, so it
costs :math:`O(N)` per path; the forward step's tangents cost
:math:`O(N)` each, :math:`O(N^2)` for all of them.

References: M. Giles and P. Glasserman, "Smoking Adjoints: Fast Monte
Carlo Greeks," *Risk* 19 (January 2006) 88-92; P. Glasserman, *Monte
Carlo Methods in Financial Engineering* (Springer, 2004), Sections 3.7
and 7.2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from financialkit._validation import nonnegative, one_of, positive
from financialkit.integrators import RNGLike
from financialkit.monte_carlo.core.base import LMMGreeks

__all__ = ["lmm_swaption_greeks"]

# Floats held per chunk of paths: the adjoint stores the rates at every
# step, the forward mode a 2N-column tangent per rate.
_CHUNK_FLOATS = 2**22


def _step(L: np.ndarray, sigma: np.ndarray, delta: float, n: int, z: np.ndarray) -> np.ndarray:
    """One Euler step in log L from T_n to T_{n+1} for the rates i >= n + 1."""
    s = sigma[n + 1 :]
    lo = L[:, n + 1 :]
    a = delta * lo / (1.0 + delta * lo)
    mu = s * np.cumsum(s * a, axis=1)
    out = L.copy()
    out[:, n + 1 :] = lo * np.exp((mu - 0.5 * s**2) * delta + s * np.sqrt(delta) * z[:, None])
    return out


def _payoff(L: np.ndarray, delta: float, m: int, swaptions: Sequence[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Discounted swaption-portfolio payoff and its gradient in the rates at T_m."""
    B = np.prod(1.0 + delta * L[:, :m], axis=1)
    disc = np.cumprod(1.0 / (1.0 + delta * L[:, m:]), axis=1)
    payoff = np.zeros(L.shape[0])
    grad = np.zeros_like(L)
    for k, K in swaptions:
        terms = delta * (L[:, m : m + k] - K) * disc[:, :k]
        itm = terms.sum(axis=1) > 0.0
        payoff += np.where(itm, terms.sum(axis=1), 0.0)
        tail = np.cumsum(terms[:, ::-1], axis=1)[:, ::-1]
        dS = delta * disc[:, :k] - delta / (1.0 + delta * L[:, m : m + k]) * tail
        grad[:, m : m + k] += itm[:, None] * dS
    V = payoff / B
    grad[:, m:] /= B[:, None]
    grad[:, :m] = -V[:, None] * delta / (1.0 + delta * L[:, :m])
    return V, grad


def _adjoint(L0, sigma, delta, m, swaptions, z):
    P, N = z.shape[0], L0.size
    hist = np.empty((m + 1, P, N))
    hist[0] = L0
    for n in range(m):
        hist[n + 1] = _step(hist[n], sigma, delta, n, z[:, n])
    V, Lbar = _payoff(hist[m], delta, m, swaptions)
    sbar = np.zeros((P, N))
    for n in range(m - 1, -1, -1):
        s = sigma[n + 1 :]
        lo, ln = hist[n][:, n + 1 :], hist[n + 1][:, n + 1 :]
        a = delta * lo / (1.0 + delta * lo)
        c = np.cumsum(s * a, axis=1)
        w = Lbar[:, n + 1 :] * ln  # adjoint of log L_i(n+1)
        tail = np.cumsum((w * s)[:, ::-1], axis=1)[:, ::-1]  # sum over i >= j of w_i sigma_i
        sbar[:, n + 1 :] += w * (np.sqrt(delta) * z[:, n : n + 1] - s * delta + delta * c) + delta * a * tail
        Lbar[:, n + 1 :] = Lbar[:, n + 1 :] * ln / lo + delta * s * delta / (1.0 + delta * lo) ** 2 * tail
    return V, Lbar, sbar


def _forward(L0, sigma, delta, m, swaptions, z):
    P, N = z.shape[0], L0.size
    L = np.tile(L0, (P, 1))
    D = np.zeros((P, N, 2 * N))  # dL_i / d(L0, sigma)
    D[:, np.arange(N), np.arange(N)] = 1.0
    for n in range(m):
        idx = np.arange(n + 1, N)
        s = sigma[n + 1 :]
        lo = L[:, n + 1 :]
        a = delta * lo / (1.0 + delta * lo)
        c = np.cumsum(s * a, axis=1)
        u = (s * delta / (1.0 + delta * lo) ** 2)[:, :, None] * D[:, n + 1 :, :]
        u[:, idx - n - 1, N + idx] += a
        dmu = s[None, :, None] * np.cumsum(u, axis=1)
        dmu[:, idx - n - 1, N + idx] += c
        dlog = D[:, n + 1 :, :] / lo[:, :, None] + delta * dmu
        dlog[:, idx - n - 1, N + idx] += np.sqrt(delta) * z[:, n : n + 1] - s * delta
        L = _step(L, sigma, delta, n, z[:, n])
        D[:, n + 1 :, :] = L[:, n + 1 :, None] * dlog
    V, grad = _payoff(L, delta, m, swaptions)
    g = np.einsum("pi,pij->pj", grad, D)
    return V, g[:, :N], g[:, N:]


def lmm_swaption_greeks(
    L0: ArrayLike,
    sigma: ArrayLike,
    delta: float,
    expiry: int,
    swaptions: Sequence[tuple[int, float]],
    n_paths: int = 10_000,
    rng: RNGLike = None,
    z: ArrayLike | None = None,
    method: str = "adjoint",
) -> LMMGreeks:
    """Price, deltas and vegas of a swaption portfolio in the LIBOR market model, by adjoints (Giles-Glasserman 2006).

    Simulates the one-factor model of the module docstring to the expiry
    and returns the pathwise derivative of the price to every initial
    forward rate and every volatility. ``method="adjoint"`` sweeps one
    adjoint vector back through the stored path; ``method="forward"``
    carries :math:`2N` tangents forward. Both give the same numbers to
    rounding; only the cost differs.

    Parameters
    ----------
    L0 : array_like, shape (N,)
        Initial forward rates :math:`L_i(0)` for the periods
        :math:`[i\\delta, (i + 1)\\delta]`.
    sigma : float or array_like, shape (N,)
        Lognormal volatility of each forward rate.
    delta : float
        Accrual fraction, which is also the time step.
    expiry : int
        Index :math:`m` of the common exercise date :math:`T_m = m\\delta`,
        ``1 <= m < N``.
    swaptions : sequence of (int, float)
        ``(k, K)`` for each payer swaption of unit notional on the swap
        from :math:`T_m` to :math:`T_{m+k}` at fixed rate :math:`K`;
        ``m + k <= N``.
    n_paths : int
        Number of paths; ignored if `z` is given.
    rng : numpy.random.Generator or int, optional
        Generator or seed.
    z : array_like, shape (n_paths, expiry), optional
        Standard normals driving the paths, e.g. common random numbers
        for a bump-and-reprice check.
    method : {"adjoint", "forward"}

    Returns
    -------
    LMMGreeks

    Examples
    --------
    The adjoint and the forward mode agree path by path:

    >>> import numpy as np
    >>> book = [(4, 0.05), (8, 0.055)]
    >>> adj = lmm_swaption_greeks(np.full(12, 0.05), 0.2, 0.25, 4, book, n_paths=500, rng=0)
    >>> fwd = lmm_swaption_greeks(np.full(12, 0.05), 0.2, 0.25, 4, book, n_paths=500, rng=0, method="forward")
    >>> bool(np.allclose(adj.delta, fwd.delta) and np.allclose(adj.vega, fwd.vega))
    True
    """
    one_of("method", method, ("adjoint", "forward"))
    L0 = np.asarray(L0, dtype=np.float64).ravel()
    N = L0.size
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (N,)).copy()
    positive(L0=L0, delta=delta)
    nonnegative(sigma=sig)
    m = int(expiry)
    if not 1 <= m < N:
        raise ValueError(f"expiry must satisfy 1 <= expiry < len(L0) = {N}, got {expiry}")
    book = [(int(k), float(K)) for k, K in swaptions]
    if not book:
        raise ValueError("swaptions must not be empty")
    for k, _ in book:
        if not 1 <= k <= N - m:
            raise ValueError(f"swap length must satisfy 1 <= k <= len(L0) - expiry = {N - m}, got {k}")
    if z is None:
        z = np.random.default_rng(rng).standard_normal((int(n_paths), m))
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != m:
        raise ValueError(f"z must have shape (n_paths, {m}), got {z.shape}")

    if method == "adjoint":
        sweep, chunk = _adjoint, max(1, _CHUNK_FLOATS // ((m + 1) * N))
    else:
        sweep, chunk = _forward, max(1, _CHUNK_FLOATS // (2 * N * N))
    parts = [sweep(L0, sig, delta, m, book, z[i : i + chunk]) for i in range(0, z.shape[0], chunk)]
    V, dL, dsig = (np.concatenate(x) for x in zip(*parts, strict=True))
    root_n = np.sqrt(V.size)
    return LMMGreeks(
        price=float(V.mean()),
        price_se=float(V.std(ddof=1) / root_n),
        delta=dL.mean(axis=0),
        delta_se=dL.std(axis=0, ddof=1) / root_n,
        vega=dsig.mean(axis=0),
        vega_se=dsig.std(axis=0, ddof=1) / root_n,
        n_paths=V.size,
        method=method,
    )
