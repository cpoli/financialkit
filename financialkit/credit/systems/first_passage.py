"""Black and Cox's first-passage model: default the first time assets touch a barrier.

In Merton's model a firm can only default at maturity, however low its
assets fall in between. Black and Cox (1976) let bond covenants force
default the first time the asset value hits a barrier
:math:`K e^{-\\gamma (T - t)}`. Writing :math:`X_t = \\ln V_t - \\gamma t`,
a Brownian motion with drift :math:`m = \\mu - \\sigma^2/2 - \\gamma`, the
barrier becomes flat at :math:`b = \\ln K - \\gamma T`, and the reflection
principle gives the survival probability to :math:`t` with
:math:`h = \\ln V_0 - b`:

.. math::

   Q(t) = N\\left(\\frac{h + m t}{\\sigma\\sqrt t}\\right)
          - e^{-2 m h / \\sigma^2} N\\left(\\frac{-h + m t}{\\sigma\\sqrt t}\\right).

References: F. Black and J. C. Cox, "Valuing Corporate Securities: Some
Effects of Bond Indenture Provisions," *J. Finance* 31 (1976) 351-367;
D. Lando, *Credit Risk Modeling* (Princeton, 2004), Section 2.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from financialkit.options.utils.common import as_output

__all__ = ["black_cox_survival", "black_cox_default_probability", "first_passage_crossing_probability"]


def black_cox_survival(
    V: ArrayLike, K: ArrayLike, t: ArrayLike, mu: ArrayLike, sigma: ArrayLike, gamma: float = 0.0, T: ArrayLike | None = None
) -> np.ndarray | float:
    """Probability that assets stay above the Black-Cox barrier up to time ``t``.

    Parameters
    ----------
    V : array_like
        Current asset value (above the barrier).
    K : array_like
        Barrier level at ``T``.
    t : array_like
        Horizon.
    mu : array_like
        Asset drift (``r - payout`` under the pricing measure).
    sigma : array_like
        Asset volatility.
    gamma : float
        Growth rate of the barrier; ``0`` gives a flat barrier ``K``.
    T : array_like, optional
        Date at which the barrier equals ``K``; defaults to ``t``.

    Returns
    -------
    float or ndarray

    Examples
    --------
    With zero log-drift, the reflection principle doubles the probability
    of ending below the barrier:

    >>> q = black_cox_survival(100.0, 70.0, 1.0, 0.02, 0.2)
    >>> p_end = float(stats.norm.cdf(np.log(70 / 100) / 0.2))
    >>> round(1 - q, 10) == round(2 * p_end, 10)
    True
    """
    V, K, t, mu, sigma = (np.asarray(a, dtype=np.float64) for a in (V, K, t, mu, sigma))
    T = t if T is None else np.asarray(T, dtype=np.float64)
    m = mu - 0.5 * sigma**2 - gamma
    h = np.log(V / K) + gamma * T
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = sigma * np.sqrt(t)
        q = stats.norm.cdf((h + m * t) / sd) - np.exp(-2.0 * m * h / sigma**2) * stats.norm.cdf((-h + m * t) / sd)
    q = np.where(t <= 0, 1.0, q)
    return as_output(np.where(h <= 0, 0.0, np.clip(q, 0.0, 1.0)))


def black_cox_default_probability(
    V: ArrayLike, K: ArrayLike, t: ArrayLike, mu: ArrayLike, sigma: ArrayLike, gamma: float = 0.0, T: ArrayLike | None = None
) -> np.ndarray | float:
    """First-passage default probability, ``1 - black_cox_survival(...)``.

    Parameters
    ----------
    V, K, t, mu, sigma, gamma, T
        As in :func:`black_cox_survival`.

    Returns
    -------
    float or ndarray
    """
    return as_output(1.0 - np.asarray(black_cox_survival(V, K, t, mu, sigma, gamma, T)))


def first_passage_crossing_probability(paths: ArrayLike, barrier: ArrayLike, sigma: float, dt: float) -> np.ndarray:
    """Brownian-bridge probability that each discretely sampled path crossed the barrier.

    Between two log-asset values above a log barrier :math:`b` a
    Brownian bridge crosses with probability
    :math:`\\exp(-2 (x_i - b)(x_{i+1} - b) / (\\sigma^2 \\Delta t))`, which
    removes the bias of only checking the barrier on the grid.

    Parameters
    ----------
    paths : array_like, shape (n_paths, n_steps + 1)
        Asset paths.
    barrier : array_like
        Barrier on the same grid (broadcast to ``paths``).
    sigma : float
        Asset volatility.
    dt : float
        Time step.

    Returns
    -------
    ndarray, shape (n_paths,)
        Probability that each path hit the barrier.
    """
    x = np.log(np.asarray(paths, dtype=np.float64))
    b = np.log(np.broadcast_to(np.asarray(barrier, dtype=np.float64), x.shape))
    d = x - b
    if np.any(d[:, 0] <= 0):
        raise ValueError("paths must start above the barrier")
    p_cross = np.exp(-2.0 * np.maximum(d[:, :-1], 0.0) * np.maximum(d[:, 1:], 0.0) / (sigma**2 * dt))
    p_cross = np.where((d[:, 1:] <= 0) | (d[:, :-1] <= 0), 1.0, p_cross)
    return 1.0 - np.prod(1.0 - p_cross, axis=1)
