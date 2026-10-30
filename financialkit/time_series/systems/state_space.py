"""State-space models: Kalman's filter and the local-level model.

Kalman (1960) wrote a linear dynamic system as a hidden state that evolves
by :math:`x_t = F x_{t-1} + w_t` and is observed with noise as
:math:`y_t = H x_t + v_t`, and gave the recursion that updates the
Gaussian conditional mean and covariance of the state with each new
observation. The one-step prediction errors are independent, so the same
pass also gives the exact Gaussian likelihood (the prediction-error
decomposition of Schweppe 1965), which is how state-space models are
estimated. Rauch, Tung and Striebel (1965) added the backward pass that
conditions every state on the whole sample. The local-level model of Muth
(1960), a random walk observed with noise, is the simplest case and the
statistical form of an efficient price seen through microstructure noise.
References: R. E. Kalman, "A New Approach to Linear Filtering and
Prediction Problems," Journal of Basic Engineering 82(1) (1960), 35-45;
F. C. Schweppe, IEEE Transactions on Information Theory 11(1) (1965),
61-70; H. E. Rauch, F. Tung and C. T. Striebel, AIAA Journal 3(8) (1965),
1445-1450; J. F. Muth, Journal of the American Statistical Association
55(290) (1960), 299-306; J. Durbin and S. J. Koopman, *Time Series
Analysis by State Space Methods*, 2nd ed. (Oxford, 2012), Chs. 2 and 4;
Hamilton, *Time Series Analysis* (Princeton, 1994), Ch. 13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize

from financialkit.time_series.core.base import KalmanResult

__all__ = ["kalman_filter", "LocalLevelFit", "local_level_steady_state_gain", "fit_local_level"]

_DIFFUSE = 1e7


def _as_matrix(a, name: str) -> np.ndarray:
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    if a.ndim != 2:
        raise ValueError(f"{name} must be a scalar or a matrix")
    return a


def kalman_filter(y, F, H, Q, R, x0=None, P0=None) -> KalmanResult:
    """Kalman's filter for :math:`x_t = F x_{t-1} + w_t`, :math:`y_t = H x_t + v_t`.

    With :math:`w_t \\sim N(0, Q)`, :math:`v_t \\sim N(0, R)`, each step
    predicts

    .. math:: \\hat x_{t|t-1} = F\\hat x_{t-1|t-1}, \\qquad P_{t|t-1} = F P_{t-1|t-1} F' + Q

    and updates with the innovation :math:`e_t = y_t - H\\hat x_{t|t-1}`
    and gain :math:`K_t = P_{t|t-1}H'S_t^{-1}`, :math:`S_t = H P_{t|t-1}H' + R`:

    .. math:: \\hat x_{t|t} = \\hat x_{t|t-1} + K_t e_t, \\qquad P_{t|t} = (I - K_t H) P_{t|t-1}

    The log-likelihood is
    :math:`-\\tfrac12\\sum_t[m\\ln 2\\pi + \\ln|S_t| + e_t'S_t^{-1}e_t]`.
    Missing observations (NaN rows of `y`) skip the update.

    Parameters
    ----------
    y : array_like, shape (n,) or (n, m)
        Observations.
    F, H, Q, R : array_like
        Transition (s, s), observation (m, s), state noise (s, s) and
        observation noise (m, m) matrices; scalars for a univariate model.
    x0, P0 : array_like, optional
        Mean and covariance of the state before the first observation.
        Default to zero and the stationary covariance (the solution of
        :math:`P = FPF' + Q` by :func:`scipy.linalg.solve_discrete_lyapunov`)
        when `F` is stable, or a diffuse :math:`10^7 I` otherwise.

    Returns
    -------
    KalmanResult

    Examples
    --------
    A random walk observed with noise: the gain settles at its steady state.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(200)) + rng.standard_normal(200)
    >>> res = kalman_filter(y, F=1.0, H=1.0, Q=1.0, R=1.0, x0=0.0, P0=1.0)
    >>> round(float(res.gain[-1, 0, 0]), 4), round(local_level_steady_state_gain(1.0), 4)
    (0.618, 0.618)
    """
    Y = np.asarray(y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    F, H, Q, R = (_as_matrix(a, nm) for a, nm in ((F, "F"), (H, "H"), (Q, "Q"), (R, "R")))
    n, m = Y.shape
    s = F.shape[0]
    if F.shape != (s, s) or H.shape != (m, s) or Q.shape != (s, s) or R.shape != (m, m):
        raise ValueError("inconsistent shapes of y, F, H, Q and R")
    x = np.zeros(s) if x0 is None else np.atleast_1d(np.asarray(x0, dtype=np.float64)).copy()
    if P0 is None:
        stable = np.max(np.abs(np.linalg.eigvals(F))) < 1.0
        P = solve_discrete_lyapunov(F, Q) if stable else _DIFFUSE * np.eye(s)
    else:
        P = _as_matrix(P0, "P0").copy()

    xp, Pp = np.empty((n, s)), np.empty((n, s, s))
    xf, Pf = np.empty((n, s)), np.empty((n, s, s))
    e, S_all, K_all = np.full((n, m), np.nan), np.empty((n, m, m)), np.zeros((n, s, m))
    loglik, n_obs = 0.0, 0
    I = np.eye(s)
    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        xp[t], Pp[t] = x, P
        S = H @ P @ H.T + R
        S_all[t] = S
        if not np.any(np.isnan(Y[t])):
            v = Y[t] - H @ x
            K = np.linalg.solve(S, H @ P).T  # P H' S^-1, S and P symmetric
            x = x + K @ v
            P = (I - K @ H) @ P
            P = 0.5 * (P + P.T)
            e[t], K_all[t] = v, K
            _, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (m * math.log(2 * math.pi) + logdet + float(v @ np.linalg.solve(S, v)))
            n_obs += 1
        xf[t], Pf[t] = x, P
    return KalmanResult(
        predicted_state=xp,
        predicted_cov=Pp,
        filtered_state=xf,
        filtered_cov=Pf,
        innovations=e,
        innovation_cov=S_all,
        gain=K_all,
        loglik=float(loglik),
        n_obs=n_obs,
        F=F,
        Q=Q,
    )


def local_level_steady_state_gain(q: float) -> float:
    """Steady-state Kalman gain of the local-level model with signal-to-noise ratio :math:`q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon`.

    The predicted variance solves the Riccati equation
    :math:`P = P\\sigma^2_\\varepsilon/(P + \\sigma^2_\\varepsilon) + \\sigma^2_\\eta`,
    so :math:`P/\\sigma^2_\\varepsilon = (q + \\sqrt{q^2 + 4q})/2` and
    :math:`K = P/(P + \\sigma^2_\\varepsilon)`. The filtered level is then an
    exponentially weighted moving average with weight :math:`K` on the
    newest observation (Muth 1960).

    Examples
    --------
    >>> round(local_level_steady_state_gain(1.0), 6)  # 1 / golden ratio
    0.618034
    """
    p = 0.5 * (q + math.sqrt(q * q + 4.0 * q))
    return p / (p + 1.0)


@dataclass
class LocalLevelFit:
    """A local-level model :math:`y_t = \\mu_t + \\varepsilon_t`, :math:`\\mu_t = \\mu_{t-1} + \\eta_t`, fitted by maximum likelihood."""

    sigma2_level: float
    """float: :math:`\\sigma^2_\\eta`, the variance of the level's innovations."""

    sigma2_noise: float
    """float: :math:`\\sigma^2_\\varepsilon`, the variance of the observation noise."""

    loglik: float
    """float: Log-likelihood conditional on the first observation."""

    filter: KalmanResult
    """KalmanResult: The filter run at the estimate; index 0 is the first observation."""

    converged: bool = True

    @property
    def signal_to_noise(self) -> float:
        """:math:`q = \\sigma^2_\\eta / \\sigma^2_\\varepsilon`."""
        return self.sigma2_level / self.sigma2_noise

    @property
    def steady_state_gain(self) -> float:
        return local_level_steady_state_gain(self.signal_to_noise)


def fit_local_level(y, start: Optional[tuple[float, float]] = None) -> LocalLevelFit:
    """Maximum-likelihood estimate of the local-level (random walk plus noise) model.

    The likelihood comes from :func:`kalman_filter` and is conditional on
    the first observation, which is equivalent to a diffuse prior on the
    initial level. :func:`scipy.optimize.minimize` searches over the log
    variances.

    Parameters
    ----------
    y : array_like, shape (n,)
    start : (float, float), optional
        Starting :math:`(\\sigma^2_\\eta, \\sigma^2_\\varepsilon)`; defaults to
        half the variance of the first differences each.

    Returns
    -------
    LocalLevelFit

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> y = np.cumsum(0.5 * rng.standard_normal(2000)) + rng.standard_normal(2000)
    >>> fit = fit_local_level(y)
    >>> round(fit.sigma2_level, 1), round(fit.sigma2_noise, 1)
    (0.2, 1.0)
    """
    x = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("y must be a series of at least 3 observations")
    v = float(np.var(np.diff(x)))
    if v == 0:
        raise ValueError("y has constant differences")
    s0 = (0.5 * v, 0.5 * v) if start is None else start

    # Conditioning on x[0] under a diffuse prior leaves the level N(x[0], sigma2_noise) at t = 0.
    def conditional(s_eta, s_eps):
        return kalman_filter(x[1:], F=1.0, H=1.0, Q=s_eta, R=s_eps, x0=[x[0]], P0=[[s_eps]])

    sol = minimize(
        lambda th: -conditional(*np.exp(th)).loglik / x.size, np.log(np.asarray(s0, dtype=np.float64)), method="L-BFGS-B", bounds=[(-30.0, 30.0)] * 2
    )
    s_eta, s_eps = (float(a) for a in np.exp(sol.x))
    res = conditional(s_eta, s_eps)
    full = kalman_filter(x, F=1.0, H=1.0, Q=s_eta, R=s_eps, x0=[x[0]], P0=[[_DIFFUSE]])
    return LocalLevelFit(sigma2_level=s_eta, sigma2_noise=s_eps, loglik=res.loglik, filter=full, converged=bool(sol.success))
