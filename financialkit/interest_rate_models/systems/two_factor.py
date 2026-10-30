"""Brigo and Mercurio's G2++ two-factor Gaussian short-rate model.

.. math:: r(t) = x(t) + y(t) + \\varphi(t), \\quad dx = -a x\\,dt + \\sigma\\,dW_1, \\quad
          dy = -b y\\,dt + \\eta\\,dW_2, \\quad dW_1\\,dW_2 = \\rho\\,dt,

with :math:`\\varphi` fitted so that today's curve is repriced exactly.
A one-factor model moves every yield with one shock, so all yields are
perfectly correlated; a second factor with :math:`\\rho < 0` decorrelates
short and long rates and can produce a humped volatility term structure.
The model is the two-factor Hull-White model in a parameterization that
keeps everything Gaussian and in closed form.

Swaptions have no Jamshidian decomposition in two factors;
:func:`g2pp_swaption` prices them by Brigo and Mercurio's one-dimensional
integral.

References: D. Brigo and F. Mercurio, *Interest Rate Models -- Theory and
Practice*, 2nd ed. (Springer, 2006), Ch. 4.2 (first edition 2001); J.
Hull and A. White, Journal of Derivatives 2(2) (1994) 37-48.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from financialkit._validation import in_interval, nonnegative, positive
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CurveLike, YieldCurve, as_curve
from financialkit.integrators import RNGLike, correlated_increments, euler_maruyama
from financialkit.interest_rate_models.core.base import GaussianShortRateModel
from financialkit.interest_rate_models.systems.bond_options import swaption_schedule
from financialkit.interest_rate_models.utils.numerics import g2_diffusion, g2_drift

__all__ = ["G2PlusPlusModel", "g2pp_swaption"]


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
        positive(a=a, b=b)
        nonnegative(sigma=sigma, eta=eta)
        in_interval(-1.0, 1.0, rho=rho)
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


def g2pp_swaption(
    model: G2PlusPlusModel, expiry: float, tenor: float, strike: float, freq: int = 2, payer: bool = True, notional: float = 1.0, n_nodes: int = 64
) -> float:
    r"""European swaption in G2++, by Brigo and Mercurio's one-dimensional integral.

    Under the :math:`T`-forward measure the factors at expiry are jointly
    Gaussian with means :math:`(\mu_x, \mu_y)`, standard deviations
    :math:`(\sigma_x, \sigma_y)` and correlation :math:`\rho_{xy}`. Given
    :math:`x`, the swap's fixed leg :math:`\sum_i c_i P(T, t_i)` is
    decreasing in :math:`y` and equals par at a single :math:`\bar y(x)`,
    so the conditional payoff is a closed form in normal CDFs and only the
    integral over :math:`x` is numerical:

    .. math:: \mathrm{PS} = \omega P(0, T) \int \frac{e^{-\frac12(\frac{x - \mu_x}{\sigma_x})^2}}{\sigma_x\sqrt{2\pi}}
              \Big[\Phi(-\omega h_1(x)) - \sum_i \lambda_i(x) e^{\kappa_i(x)} \Phi(-\omega h_2^{(i)}(x))\Big]\,dx,

    with :math:`\omega = 1` for a payer and :math:`-1` for a receiver
    (Brigo and Mercurio, 2nd ed., Theorem 4.2.3). The integral is
    evaluated by Gauss-Hermite quadrature, and :math:`\bar y(x)` by
    Newton's method on the log of the fixed leg, which is convex in :math:`y`.
    The factors' roles are symmetric, and the integral runs over the one
    with the smaller terminal standard deviation, which keeps the
    integrand smooth when the other factor is nearly deterministic.

    Parameters
    ----------
    model : G2PlusPlusModel
        Needs :math:`\sigma, \eta > 0`.
    expiry, tenor : float
        Option expiry and length of the underlying swap.
    strike : float
        Fixed rate, positive.
    freq : int
        Fixed-leg payments per year.
    payer : bool
        Payer (right to pay fixed) or receiver swaption.
    notional : float
    n_nodes : int
        Gauss-Hermite nodes in :math:`x`.

    Returns
    -------
    float
        Scaled by `notional`.

    Examples
    --------
    Payer minus receiver is the forward swap, :math:`P(0, T) - \sum_i c_i P(0, t_i)`.

    >>> m = G2PlusPlusModel(0.04, 0.5, 0.01, 0.05, 0.008, -0.7)
    >>> parity = g2pp_swaption(m, 2.0, 5.0, 0.05) - g2pp_swaption(m, 2.0, 5.0, 0.05, payer=False)
    >>> t = np.arange(1, 11) / 2 + 2.0
    >>> forward = np.exp(-0.08) - 0.025 * np.exp(-0.04 * t).sum() - np.exp(-0.28)
    >>> bool(abs(parity - forward) < 1e-12)
    True
    """
    positive(expiry=expiry, tenor=tenor, strike=strike, sigma=model.sigma, eta=model.eta)
    T = float(expiry)
    times = swaption_schedule(T, tenor, freq)
    c = strike * np.diff(np.concatenate([[T], times]))
    c[-1] += 1.0
    a, b, s, e, rho = model.a, model.b, model.sigma, model.eta, model.rho
    sx = s * np.sqrt(-np.expm1(-2 * a * T) / (2 * a))
    sy = e * np.sqrt(-np.expm1(-2 * b * T) / (2 * b))
    # The model is symmetric in its factors; integrating over the narrower one
    # keeps the integrand smooth when the other's conditional payoff is near a step.
    if sx > sy:
        a, b, s, e, sx, sy = b, a, e, s, sy, sx
    gab = -np.expm1(-(a + b) * T)
    mx = -(s**2 / a**2 + rho * s * e / (a * b)) * -np.expm1(-a * T) + s**2 / (2 * a**2) * -np.expm1(-2 * a * T) + rho * s * e / (b * (a + b)) * gab
    my = -(e**2 / b**2 + rho * s * e / (a * b)) * -np.expm1(-b * T) + e**2 / (2 * b**2) * -np.expm1(-2 * b * T) + rho * s * e / (a * (a + b)) * gab
    rxy = rho * s * e / ((a + b) * sx * sy) * gab

    Ba, Bb = _B(a, times - T), _B(b, times - T)
    logcA = np.log(c * model.zero_coupon_bond(times, T, np.zeros(2)))
    z, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    x = mx + sx * z

    # ybar(x): log sum_i c_i A_i exp(-Ba_i x - Bb_i y) = 0, convex and decreasing in y.
    base = logcA - np.outer(x, Ba)
    y = np.zeros_like(x)
    for _ in range(DEFAULT_MAX_ITER):
        L = base - np.outer(y, Bb)
        top = L.max(axis=1, keepdims=True)
        p = np.exp(L - top)
        g = top[:, 0] + np.log(p.sum(axis=1))
        step = g / -((p * Bb).sum(axis=1) / p.sum(axis=1))
        y -= step
        if np.max(np.abs(step)) < DEFAULT_ATOL:
            break

    root = np.sqrt(1 - rxy**2)
    h1 = (y - my) / (sy * root) - rxy * (x - mx) / (sx * root)
    h2 = h1[:, None] + Bb * sy * root
    kappa = -Bb * (my - 0.5 * root**2 * sy**2 * Bb + rxy * sy * ((x - mx) / sx)[:, None])
    lam = np.exp(logcA - np.outer(x, Ba) + kappa)
    om = 1.0 if payer else -1.0
    inner = norm.cdf(-om * h1) - (lam * norm.cdf(-om * h2)).sum(axis=1)
    return float(notional * om * model.zero_coupon_bond(T) * (w @ inner) / np.sqrt(2 * np.pi))
