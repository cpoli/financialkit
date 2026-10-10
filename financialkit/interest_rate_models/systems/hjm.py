"""The Heath-Jarrow-Morton framework: forward-curve dynamics and the drift condition.

Heath, Jarrow and Morton (1992) modelled the whole instantaneous forward
curve, :math:`df(t, T) = \\alpha(t, T)\\,dt + \\sigma(t, T)\\,dW`, starting
from today's curve. They showed that absence of arbitrage leaves no
freedom in the risk-neutral drift: discounted bond prices are
martingales if and only if

.. math:: \\alpha(t, T) = \\sigma(t, T) \\int_t^T \\sigma(t, u)\\,du.

The volatility is the whole model. A constant :math:`\\sigma` gives
Ho-Lee and :math:`\\sigma e^{-a(T - t)}` gives Hull-White, each fitted to
the initial curve by construction.

* :func:`hjm_drift`: the drift for any volatility function, by
  :func:`scipy.integrate.quad`.
* :func:`simulate_hjm`: one-factor forward curves on a grid by
  :func:`financialkit.integrators.euler_maruyama` (one shared Brownian
  driver across maturities), with or without the drift condition, so
  the arbitrage that appears without it can be measured.

References: D. Heath, R. Jarrow and A. Morton, "Bond Pricing and the
Term Structure of Interest Rates: A New Methodology for Contingent Claims
Valuation," Econometrica 60(1) (1992) 77-105; P. Glasserman, *Monte Carlo
Methods in Financial Engineering* (Springer, 2003), Sec. 3.6.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from financialkit.fixed_income.core.base import CurveLike, as_curve
from financialkit.integrators import RNGLike, brownian_increments, euler_maruyama
from financialkit.interest_rate_models.core.base import HJMResult
from financialkit.interest_rate_models.utils.numerics import hjm_diffusion_callback, hjm_drift_callback

__all__ = ["hjm_drift", "simulate_hjm"]


def hjm_drift(sigma: Callable[[float, float], float], t: float, T) -> np.ndarray | float:
    r"""HJM no-arbitrage drift :math:`\alpha(t, T) = \sigma(t, T)\int_t^T \sigma(t, u)\,du`.

    Parameters
    ----------
    sigma : callable
        ``sigma(t, T) -> float``, the forward-rate volatility.
    t : float
        Current time.
    T : array_like
        Maturities, ``T >= t``.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Constant volatility (Ho-Lee) gives :math:`\sigma^2 (T - t)`:

    >>> round(hjm_drift(lambda t, T: 0.01, 0.0, 5.0), 10)
    0.0005
    """
    T = np.asarray(T, dtype=np.float64)
    out = np.array([sigma(t, Tj) * quad(lambda u: sigma(t, u), t, Tj)[0] for Tj in T.ravel()]).reshape(T.shape)
    return float(out) if out.ndim == 0 else out


def simulate_hjm(
    curve: CurveLike,
    sigma: float,
    a: float = 0.0,
    horizon: float = 5.0,
    n_steps: int = 50,
    n_paths: int = 2000,
    drift_condition: bool = True,
    rng: RNGLike = None,
) -> HJMResult:
    r"""Simulate one-factor HJM forward curves with :math:`\sigma(t, T) = \sigma e^{-a(T - t)}`.

    The maturity grid is the simulation grid, so the short rate is
    :math:`r(t_i) = f(t_i, t_i)`. Every maturity shares one Brownian
    driver, and the drift is the HJM drift (or zero when
    ``drift_condition=False``).

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure; its instantaneous forwards start the simulation.
    sigma : float
        Forward-rate volatility at zero time to maturity.
    a : float
        Exponential decay of volatility with maturity; 0 is Ho-Lee.
    horizon : float
        Last date and maturity.
    n_steps : int
    n_paths : int
    drift_condition : bool
        Apply the no-arbitrage drift.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    HJMResult

    Examples
    --------
    >>> res = simulate_hjm(0.03, 0.01, horizon=2.0, n_steps=20, n_paths=500, rng=0)
    >>> res.forwards.shape
    (500, 21, 21)
    """
    c = as_curve(curve)
    times = np.linspace(0.0, horizon, n_steps + 1)
    f0 = np.asarray(c.instantaneous_forward(times), dtype=np.float64)
    m = times.size
    dW1 = brownian_increments(horizon, n_steps, n_paths, rng=rng)
    dW = np.repeat(dW1[:, :, None], m, axis=2)
    params = np.concatenate([[sigma, a, 1.0 if drift_condition else 0.0], times])
    _, forwards = euler_maruyama(hjm_drift_callback, hjm_diffusion_callback, f0, horizon, n_steps, n_paths, params=params, dW=dW)
    return HJMResult(times=times, forwards=forwards, initial_curve=f0, drift_condition=bool(drift_condition))
