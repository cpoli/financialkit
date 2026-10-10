"""Brigo and Mercurio's G2++ two-factor Gaussian short-rate model.

.. math:: r(t) = x(t) + y(t) + \\varphi(t), \\quad dx = -a x\\,dt + \\sigma\\,dW_1, \\quad
          dy = -b y\\,dt + \\eta\\,dW_2, \\quad dW_1\\,dW_2 = \\rho\\,dt,

with :math:`\\varphi` fitted so that today's curve is repriced exactly.
A one-factor model moves every yield with one shock, so all yields are
perfectly correlated; a second factor with :math:`\\rho < 0` decorrelates
short and long rates and can produce a humped volatility term structure.
The model is the two-factor Hull-White model in a parameterization that
keeps everything Gaussian and in closed form.

References: D. Brigo and F. Mercurio, *Interest Rate Models -- Theory and
Practice*, 2nd ed. (Springer, 2006), Ch. 4.2 (first edition 2001); J.
Hull and A. White, Journal of Derivatives 2(2) (1994) 37-48.
"""

from __future__ import annotations

import numpy as np

from financialkit.fixed_income.core.base import CurveLike, YieldCurve, as_curve
from financialkit.integrators import RNGLike, correlated_increments, euler_maruyama
from financialkit.interest_rate_models.core.base import GaussianShortRateModel
from financialkit.interest_rate_models.utils.numerics import g2_diffusion, g2_drift

__all__ = ["G2PlusPlusModel"]


def _B(k: float, tau):
    return -np.expm1(-k * tau) / k


class G2PlusPlusModel(GaussianShortRateModel):
    r"""The G2++ model, fitted to an initial curve.

    Bond prices given the factors :math:`(x, y)` at time :math:`t` are

    .. math:: P(t, T) = \frac{P(0, T)}{P(0, t)}
              \exp\left\{\tfrac12[V(t, T) - V(0, T) + V(0, t)] - B_a(\tau) x - B_b(\tau) y\right\},

    with :math:`B_k(\tau) = (1 - e^{-k\tau})/k` and :math:`V(t, T)` the
    variance of :math:`\int_t^T (x + y)\,du` (Brigo-Mercurio eq. 4.10).

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure.
    a, sigma : float
        Mean reversion and volatility of the first factor.
    b, eta : float
        Mean reversion and volatility of the second factor.
    rho : float
        Correlation of the two Brownian drivers, in ``(-1, 1)``.

    Examples
    --------
    >>> m = G2PlusPlusModel(0.03, 0.5, 0.01, 0.05, 0.008, -0.7)
    >>> round(float(m.zero_coupon_bond(10.0)), 6) == round(float(np.exp(-0.3)), 6)
    True
    """

    n_factors = 2

    def __init__(self, curve: CurveLike, a: float, sigma: float, b: float, eta: float, rho: float):
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive")
        if sigma < 0 or eta < 0:
            raise ValueError("sigma and eta must be non-negative")
        if not -1 < rho < 1:
            raise ValueError("rho must lie strictly between -1 and 1")
        self.curve_ = as_curve(curve)
        self.a, self.sigma, self.b, self.eta, self.rho = float(a), float(sigma), float(b), float(eta), float(rho)
        self.r0 = float(self.curve_.instantaneous_forward(0.0))

    def curve(self) -> YieldCurve:
        """The initial curve the model was fitted to."""
        return self.curve_

    def V(self, t, T):
        """Variance of :math:`\\int_t^T [x(u) + y(u)]\\,du` given the factors at ``t``."""
        tau = np.asarray(T, dtype=np.float64) - np.asarray(t, dtype=np.float64)
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho

        def single(k, vol):
            return vol**2 / k**2 * (tau + 2 / k * np.exp(-k * tau) - 1 / (2 * k) * np.exp(-2 * k * tau) - 3 / (2 * k))

        cross = 2 * rho * s * e / (a * b) * (tau + np.expm1(-a * tau) / a + np.expm1(-b * tau) / b - np.expm1(-(a + b) * tau) / (a + b))
        return single(a, s) + single(b, e) + cross

    def shift(self, t):
        r""":math:`\varphi(t) = f(0, t) + \tfrac{\sigma^2}{2}B_a(t)^2 + \tfrac{\eta^2}{2}B_b(t)^2 + \rho\sigma\eta B_a(t) B_b(t)`."""
        t = np.asarray(t, dtype=np.float64)
        Ba, Bb = _B(self.a, t), _B(self.b, t)
        return self.curve_.instantaneous_forward(t) + 0.5 * self.sigma**2 * Ba**2 + 0.5 * self.eta**2 * Bb**2 + self.rho * self.sigma * self.eta * Ba * Bb

    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        T = np.asarray(T, dtype=np.float64)
        P0T = self.curve_.discount(T)
        if t == 0.0 and state is None:
            return P0T
        xy = np.zeros(2) if state is None else np.asarray(state, dtype=np.float64)
        x, y = xy[..., 0], xy[..., 1]
        tau = T - t
        conv = 0.5 * (self.V(t, T) - self.V(0.0, T) + self.V(0.0, t))
        return P0T / self.curve_.discount(t) * np.exp(conv - _B(self.a, tau) * x - _B(self.b, tau) * y)

    def bond_option_volatility(self, T, S):
        T, S = np.asarray(T, dtype=np.float64), np.asarray(S, dtype=np.float64)
        a, b, s, e, rho = self.a, self.b, self.sigma, self.eta, self.rho
        Ba, Bb = _B(a, S - T), _B(b, S - T)
        var = (
            s**2 / (2 * a) * Ba**2 * -np.expm1(-2 * a * T)
            + e**2 / (2 * b) * Bb**2 * -np.expm1(-2 * b * T)
            + 2 * rho * s * e / (a + b) * Ba * Bb * -np.expm1(-(a + b) * T)
        )
        return np.sqrt(var)

    def yield_correlation(self, T1, T2):
        r"""Instantaneous correlation of the zero yields of maturities ``T1`` and ``T2``.

        A yield :math:`R(t, t + \tau)` moves by
        :math:`[B_a(\tau)\sigma\,dW_1 + B_b(\tau)\eta\,dW_2]/\tau`, so two
        yields are perfectly correlated in any one-factor model but not here.

        Parameters
        ----------
        T1, T2 : array_like
            Times to maturity.

        Returns
        -------
        float or ndarray
        """
        T1, T2 = np.asarray(T1, dtype=np.float64), np.asarray(T2, dtype=np.float64)
        u1, v1 = self.sigma * _B(self.a, T1) / T1, self.eta * _B(self.b, T1) / T1
        u2, v2 = self.sigma * _B(self.a, T2) / T2, self.eta * _B(self.b, T2) / T2
        cov = u1 * u2 + v1 * v2 + self.rho * (u1 * v2 + u2 * v1)
        var1 = u1**2 + v1**2 + 2 * self.rho * u1 * v1
        var2 = u2**2 + v2**2 + 2 * self.rho * u2 * v2
        return cov / np.sqrt(var1 * var2)

    def simulate_factors(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None) -> tuple[np.ndarray, np.ndarray]:
        """Factor paths by Euler-Maruyama on correlated increments.

        Returns
        -------
        times : ndarray, shape (n_steps + 1,)
        xy : ndarray, shape (n_paths, n_steps + 1, 2)
        """
        corr = np.array([[1.0, self.rho], [self.rho, 1.0]])
        dW = correlated_increments(corr, T, n_steps, n_paths, rng=rng)
        return euler_maruyama(g2_drift, g2_diffusion, np.zeros(2), T, n_steps, n_paths, params=[self.a, self.b, self.sigma, self.eta], dW=dW)

    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None):
        times, xy = self.simulate_factors(T, n_steps, n_paths, rng)
        return times, xy.sum(axis=2) + self.shift(times)

    def __repr__(self) -> str:
        return f"G2PlusPlusModel(a={self.a!r}, sigma={self.sigma!r}, b={self.b!r}, eta={self.eta!r}, rho={self.rho!r})"
