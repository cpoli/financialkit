"""Markov regime switching: Hamilton's filter, Kim's smoother and EM estimation.

Hamilton (1989) let the mean growth rate of U.S. GNP switch between an
expansion and a recession state that follows a hidden Markov chain, and
gave the nonlinear filter that turns each observation into updated
probabilities of the two states, together with the likelihood. Kim (1994)
gave the backward recursion for the smoothed probabilities given the
whole sample, and Hamilton (1990) the EM algorithm whose M-step is a
probability-weighted mean, variance and transition count. In finance the
same model with switching means *and* variances separates calm bull
markets from turbulent bear markets. References: J. D. Hamilton, "A New
Approach to the Economic Analysis of Nonstationary Time Series and the
Business Cycle," Econometrica 57(2) (1989), 357-384; J. D. Hamilton,
"Analysis of Time Series Subject to Changes in Regime," Journal of
Econometrics 45(1-2) (1990), 39-70; C.-J. Kim, "Dynamic Linear Models with
Markov-Switching," Journal of Econometrics 60(1-2) (1994), 1-22; Hamilton,
*Time Series Analysis* (Princeton, 1994), Ch. 22.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm

from financialkit.time_series.core.base import RegimeSwitchingResult

__all__ = ["hamilton_filter", "kim_smoother", "simulate_markov_switching", "fit_markov_switching"]


def _densities(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return norm.pdf(x[:, None], mu[None, :], sigma[None, :])


def hamilton_filter(x, mu, sigma, transition, p0=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Hamilton's filter for :math:`x_t \\mid s_t = j \\sim N(\\mu_j, \\sigma_j^2)`.

    With :math:`\\xi_{t|t}` the vector of :math:`P(s_t = j \\mid x_1..x_t)`
    and :math:`\\eta_t` the vector of regime densities,

    .. math:: \\xi_{t|t-1} = P'\\xi_{t-1|t-1}, \\qquad
       \\xi_{t|t} = \\frac{\\xi_{t|t-1} \\odot \\eta_t}{\\mathbf 1'(\\xi_{t|t-1} \\odot \\eta_t)},

    and the denominators are the conditional densities of :math:`x_t`
    whose logs sum to the log-likelihood.

    Parameters
    ----------
    x : array_like, shape (n,)
    mu, sigma : array_like, shape (k,)
    transition : array_like, shape (k, k)
        ``transition[i, j]`` is :math:`P(s_t = j \\mid s_{t-1} = i)`; rows sum to one.
    p0 : array_like, shape (k,), optional
        Regime probabilities before the first observation (the chain's
        ergodic distribution by default).

    Returns
    -------
    filtered : ndarray, shape (n, k)
    predicted : ndarray, shape (n, k)
    loglik : float

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0.95, 0.05], [0.10, 0.90]])
    >>> filt, _, _ = hamilton_filter([0.0, 0.1, 5.0], [0.0, 0.0], [1.0, 3.0], P)
    >>> filt.argmax(axis=1).tolist()
    [0, 0, 1]
    """
    x = np.asarray(x, dtype=np.float64)
    mu, sigma = np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    P = np.asarray(transition, dtype=np.float64)
    if p0 is None:
        w, v = np.linalg.eig(P.T)
        p0 = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    xi = np.asarray(p0, dtype=np.float64) / np.sum(p0)
    dens = _densities(x, mu, sigma)
    n, k = dens.shape
    filt, pred = np.empty((n, k)), np.empty((n, k))
    loglik = 0.0
    for t in range(n):
        xi = xi @ P
        pred[t] = xi
        f = xi * dens[t]
        c = f.sum()
        if c <= 0.0:
            return filt, pred, -math.inf
        xi = f / c
        filt[t] = xi
        loglik += math.log(c)
    return filt, pred, loglik


def kim_smoother(filtered, predicted, transition) -> np.ndarray:
    """Kim's (1994) smoothed probabilities :math:`P(s_t = j \\mid x_1..x_n)`.

    .. math:: \\xi_{t|n} = \\xi_{t|t} \\odot \\bigl[P\\,(\\xi_{t+1|n} \\oslash \\xi_{t+1|t})\\bigr]

    Returns
    -------
    ndarray, shape (n, k)
    """
    filt = np.asarray(filtered, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    P = np.asarray(transition, dtype=np.float64)
    sm = filt.copy()
    for t in range(filt.shape[0] - 2, -1, -1):
        sm[t] = filt[t] * (P @ (sm[t + 1] / pred[t + 1]))
    return sm


def simulate_markov_switching(n: int, mu, sigma, transition, s0: int = 0, rng: np.random.Generator | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Gaussian Markov-switching series and its hidden regimes.

    Returns
    -------
    x : ndarray, shape (n,)
    regimes : ndarray of int, shape (n,)
    """
    rng = np.random.default_rng(rng)
    mu, sigma = np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    cum = np.cumsum(np.asarray(transition, dtype=np.float64), axis=1)
    u = rng.random(n)
    s = np.empty(n, dtype=np.int64)
    state = s0
    for t in range(n):
        state = min(int(np.searchsorted(cum[state], u[t], side="right")), mu.size - 1)
        s[t] = state
    return mu[s] + sigma[s] * rng.standard_normal(n), s


def fit_markov_switching(x, k: int = 2, max_iter: int = 500, tol: float = 1e-8, init: Optional[dict] = None) -> RegimeSwitchingResult:
    """Fit a `k`-regime Gaussian Markov-switching model by the EM algorithm (Hamilton 1990).

    The E-step runs :func:`hamilton_filter` and :func:`kim_smoother`; the
    M-step sets each regime's mean and variance to the smoothed-probability
    weighted moments and each transition probability to the expected
    number of :math:`i \\to j` switches over the expected time spent in
    :math:`i`. Every iteration raises the likelihood. Regimes are returned
    in increasing order of volatility.

    Parameters
    ----------
    x : array_like, shape (n,)
    k : int
        Number of regimes.
    max_iter : int
    tol : float
        Stop when the log-likelihood rises by less than ``tol * n``.
    init : dict, optional
        Starting ``mu``, ``sigma`` and ``transition``; by default the
        regimes start at the sample mean with standard deviations spread
        around the sample one, and stay put with probability 0.9.

    Returns
    -------
    RegimeSwitchingResult

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[0.98, 0.02], [0.05, 0.95]])
    >>> x, s = simulate_markov_switching(3000, [0.05, -0.1], [0.5, 2.0], P, rng=0)
    >>> res = fit_markov_switching(x)
    >>> res.sigma.round(1).tolist(), res.transition.diagonal().round(2).tolist()
    ([0.5, 2.0], [0.98, 0.95])
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if init is None:
        sd = float(np.std(x))
        mu = np.full(k, float(np.mean(x)))
        sigma = sd * np.linspace(0.5, 1.5, k) if k > 1 else np.array([sd])
        P = np.full((k, k), 0.1 / max(k - 1, 1))
        np.fill_diagonal(P, 0.9 if k > 1 else 1.0)
    else:
        mu = np.asarray(init["mu"], dtype=np.float64).copy()
        sigma = np.asarray(init["sigma"], dtype=np.float64).copy()
        P = np.asarray(init["transition"], dtype=np.float64).copy()
    p0 = np.full(k, 1.0 / k)
    ll_old, converged, n_iter = -math.inf, False, 0
    while n_iter < max_iter:
        n_iter += 1
        filt, pred, ll = hamilton_filter(x, mu, sigma, P, p0)
        sm = kim_smoother(filt, pred, P)
        # Expected transitions: P(s_{t-1}=i, s_t=j | all) = filt_{t-1,i} P_ij sm_{t,j} / pred_{t,j},
        # with the pre-sample probabilities p0 in place of filt_{-1}.
        before = np.vstack([p0, filt[:-1]])
        joint = before[:, :, None] * P[None] * (sm / pred)[:, None, :]
        p0 = joint[0].sum(axis=1)
        joint = joint.sum(axis=0)
        P = joint / joint.sum(axis=1, keepdims=True)
        w = sm.sum(axis=0)
        mu = sm.T @ x / w
        sigma = np.sqrt(np.maximum(np.einsum("tj,tj->j", sm, (x[:, None] - mu) ** 2) / w, 1e-12 * float(np.var(x))))
        if ll - ll_old < tol * n:
            converged = True
            break
        ll_old = ll
    filt, pred, ll = hamilton_filter(x, mu, sigma, P, p0)
    sm = kim_smoother(filt, pred, P)
    order = np.argsort(sigma)
    return RegimeSwitchingResult(
        mu=mu[order],
        sigma=sigma[order],
        transition=P[np.ix_(order, order)],
        filtered_probs=filt[:, order],
        smoothed_probs=sm[:, order],
        loglik=float(ll),
        n_obs=n,
        n_iter=n_iter,
        converged=converged,
    )
