"""Synthetic price data for tests and gallery examples (no market data is bundled)."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from financialkit._deprecation import renamed_parameter
from financialkit.constants import TRADING_DAYS_PER_YEAR
from financialkit.integrators import gbm_paths
from financialkit.time_series.core.base import OHLC

__all__ = ["simulate_ohlc", "simulate_intraday_returns", "simulate_cointegrated_pair", "simulate_dickey_fuller"]


def simulate_ohlc(
    n_days: int,
    sigma: float,
    steps_per_day: int = 390,
    s0: float = 100.0,
    mu: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> OHLC:
    """Daily open, high, low and close of geometric Brownian motion.

    Each day is sampled on `steps_per_day` points of an exact GBM path
    (:func:`financialkit.integrators.gbm_paths`); the next day opens at the
    previous close. The sampled high and low miss the true extremes between
    grid points, so range-based estimators are biased slightly downward on
    coarse grids.

    Parameters
    ----------
    n_days : int
    sigma : float
        Annual volatility.
    steps_per_day : int
        Intraday grid (390 one-minute bars by default).
    s0, mu : float
        Initial price and annual drift.
    rng : numpy.random.Generator, int or None

    Returns
    -------
    OHLC
    """
    dt_day = 1.0 / TRADING_DAYS_PER_YEAR
    _, S = gbm_paths(s0, mu, sigma, T=n_days * dt_day, n_steps=n_days * steps_per_day, n_paths=1, rng=rng)
    S = S[0]
    days = np.lib.stride_tricks.sliding_window_view(S, steps_per_day + 1)[::steps_per_day]
    return OHLC(open=days[:, 0].copy(), high=days.max(axis=1), low=days.min(axis=1), close=days[:, -1].copy())


def simulate_intraday_returns(daily_variance, m: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Intraday returns whose day-`t` variance sums to ``daily_variance[t]``.

    Each of the `m` intraday returns of day `t` is
    :math:`\\sigma_t z_{t,i} / \\sqrt m`, so the daily return
    :math:`\\sum_i r_{t,i}` has exactly variance :math:`\\sigma_t^2`, as in
    the continuous-time GARCH diffusion behind Andersen and Bollerslev
    (1998).

    Returns
    -------
    ndarray, shape (n_days, m)
    """
    rng = np.random.default_rng(rng)
    v = np.asarray(daily_variance, dtype=np.float64)
    return np.sqrt(v / m)[:, None] * rng.standard_normal((v.size, m))


def simulate_cointegrated_pair(
    n: int,
    beta: float = 1.0,
    intercept: float = 0.0,
    phi: float = 0.9,
    sigma_x: float = 1.0,
    sigma_u: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A random walk `x` and ``y = intercept + beta x + u`` with a stationary AR(1) error `u`.

    Returns
    -------
    y, x : ndarray, shape (n,)
    """
    rng = np.random.default_rng(rng)
    x = np.cumsum(sigma_x * rng.standard_normal(n))
    u = lfilter([1.0], [1.0, -phi], sigma_u * rng.standard_normal(n))
    return intercept + beta * x + u, x


@renamed_parameter("n_sims", "n_paths", "financialkit.time_series.simulate_dickey_fuller")
def simulate_dickey_fuller(n_obs: int, n_paths: int, regression: str = "c", rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Dickey-Fuller t-statistics of `n_paths` driftless Gaussian random walks.

    Regresses :math:`\\Delta y_t` on :math:`y_{t-1}` (plus a constant for
    ``"c"``, and a trend for ``"ct"``) for all walks at once, which traces
    out the nonstandard null distribution that Dickey and Fuller tabulated.

    Returns
    -------
    ndarray, shape (n_paths,)
    """
    rng = np.random.default_rng(rng)
    y = np.cumsum(rng.standard_normal((n_paths, n_obs + 1)), axis=1)
    dy = np.diff(y, axis=1)
    ylag = y[:, :-1]
    t = np.arange(n_obs, dtype=np.float64)
    det = {"n": np.empty((n_obs, 0)), "c": np.ones((n_obs, 1)), "ct": np.column_stack([np.ones(n_obs), t])}[regression]
    if det.shape[1]:
        # Frisch-Waugh: partial the deterministic terms out of both sides.
        Q, _ = np.linalg.qr(det)
        dy = dy - (dy @ Q) @ Q.T
        ylag = ylag - (ylag @ Q) @ Q.T
    sxx = np.sum(ylag**2, axis=1)
    gamma = np.sum(ylag * dy, axis=1) / sxx
    e = dy - gamma[:, None] * ylag
    s2 = np.sum(e**2, axis=1) / (n_obs - 1 - det.shape[1])
    return gamma / np.sqrt(s2 / sxx)
