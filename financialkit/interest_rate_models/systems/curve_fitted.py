"""No-arbitrage short-rate models fitted to today's yield curve: Ho-Lee,
Hull-White and Black-Derman-Toy, and the futures convexity adjustment.

An equilibrium model (Vasicek, CIR) produces its own curve, which never
matches the market's. Ho and Lee (1986) instead took the initial curve
as an input and let a time-dependent drift absorb it; Hull and White
(1990) added mean reversion to the same construction; Black, Derman and
Toy (1990) built a lognormal binomial tree calibrated level by level.
Each model takes a :class:`~financialkit.fixed_income.core.base.YieldCurve`
(or a flat rate) and reprices every zero-coupon bond on it exactly.

* :class:`HoLeeModel`: :math:`dr = \\theta(t)\\,dt + \\sigma\\,dW`.
* :class:`HullWhiteModel`: :math:`dr = (\\theta(t) - a r)\\,dt + \\sigma\\,dW`,
  simulated as an exact Ornstein-Uhlenbeck deviation
  (:func:`financialkit.integrators.ou_paths`) around a deterministic shift.
* :func:`black_derman_toy_tree`: a recombining binomial tree of lognormal
  rates, each level's median rate solved by :func:`scipy.optimize.brentq`
  against Arrow-Debreu prices (forward induction).
* :func:`hull_white_tree`: Hull and White's (1994) trinomial tree for the
  same model, with its branching turned inward at :math:`|j| = j_{max}`
  and each level's shift fitted to the curve by forward induction.
* :func:`futures_convexity_adjustment`: the gap between a futures rate
  and the forward rate under Ho-Lee and Hull-White.

References: T. S. Y. Ho and S.-B. Lee, Journal of Finance 41(5) (1986)
1011-1029; J. Hull and A. White, Review of Financial Studies 3(4) (1990)
573-592; J. Hull and A. White, Journal of Derivatives 2(1) (1994) 7-16;
F. Black, E. Derman and W. Toy, Financial Analysts Journal
46(1) (1990) 33-39; F. Jamshidian, Journal of Fixed Income 1(1) (1991)
62-74 (forward induction); J. C. Hull, *Options, Futures, and Other
Derivatives*, 10th ed. (Pearson, 2018), Sec. 6.3 and Ch. 32.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from financialkit._validation import nonnegative, positive
from financialkit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from financialkit.fixed_income.core.base import CurveLike, YieldCurve, as_curve
from financialkit.integrators import RNGLike, ou_paths
from financialkit.interest_rate_models.core.base import GaussianShortRateModel, RateTree, TrinomialRateTree

__all__ = ["HoLeeModel", "HullWhiteModel", "black_derman_toy_tree", "hull_white_tree", "futures_convexity_adjustment"]


class _CurveFittedGaussian(GaussianShortRateModel):
    """Shared machinery: :math:`r(t) = x(t) + \\varphi(t)`, with :math:`x` an OU process from 0."""

    a: float = 0.0
    sigma: float = 0.0
    curve_: YieldCurve

    def _B(self, tau):
        return tau if self.a == 0 else -np.expm1(-self.a * tau) / self.a

    def _var_x(self, t):
        """Variance of the deviation at ``t``, :math:`\\sigma^2(1 - e^{-2at})/(2a)`."""
        t = np.asarray(t, dtype=np.float64)
        return self.sigma**2 * (t if self.a == 0 else -np.expm1(-2 * self.a * t) / (2 * self.a))

    def shift(self, t):
        r""":math:`\varphi(t) = E[r(t)] = f(0, t) + \tfrac12 \sigma^2 B(0, t)^2`, the deterministic part of the rate."""
        t = np.asarray(t, dtype=np.float64)
        return self.curve_.instantaneous_forward(t) + 0.5 * self.sigma**2 * self._B(t) ** 2

    def theta(self, t, h: float = 1e-4):
        r"""Drift function :math:`\theta(t) = \varphi'(t) + a\varphi(t)`, by central differences."""
        t = np.asarray(t, dtype=np.float64)
        lo = np.maximum(t - h, 0.0)
        return (self.shift(t + h) - self.shift(lo)) / (t + h - lo) + self.a * self.shift(t)

    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        T = np.asarray(T, dtype=np.float64)
        P0T = self.curve_.discount(T)
        if t == 0.0 and state is None:
            return P0T
        tau = T - t
        B = self._B(tau)
        f0t = self.curve_.instantaneous_forward(t)
        r = np.asarray(self.r0 if state is None else state, dtype=np.float64)
        lnA = np.log(P0T / self.curve_.discount(t)) + B * f0t - 0.5 * B**2 * self._var_x(t)
        return np.exp(lnA - B * r)

    def bond_option_volatility(self, T, S):
        T, S = np.asarray(T, dtype=np.float64), np.asarray(S, dtype=np.float64)
        return self._B(S - T) * np.sqrt(self._var_x(T))

    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None):
        times, x = ou_paths(0.0, self.a, 0.0, self.sigma, T, n_steps, n_paths, rng=rng)
        return times, x + self.shift(times)


class HoLeeModel(_CurveFittedGaussian):
    r"""Ho and Lee's (1986) arbitrage-free model, in its continuous-time limit.

    .. math:: dr = \theta(t)\,dt + \sigma\,dW, \qquad \theta(t) = f_T(0, t) + \sigma^2 t,

    where :math:`f(0, t)` is today's instantaneous forward curve. The drift
    is whatever it takes to reprice every zero-coupon bond, so the model
    fits any initial curve exactly. Bonds at later dates are

    .. math:: P(t, T) = \frac{P(0, T)}{P(0, t)}
              \exp\left(\tau f(0, t) - \tfrac12\sigma^2 t \tau^2 - \tau r(t)\right).

    Ho and Lee built it as a binomial tree of whole discount curves; it is
    Merton's Gaussian rate with a curve-fitting drift.

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure (a float is a flat rate).
    sigma : float
        Absolute volatility of the short rate.

    Examples
    --------
    >>> from financialkit.fixed_income import NelsonSiegelCurve
    >>> curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
    >>> bool(np.isclose(HoLeeModel(curve, 0.01).zero_coupon_bond(7.0), curve.discount(7.0)))
    True
    """

    def __init__(self, curve: CurveLike, sigma: float):
        nonnegative(sigma=sigma)
        self.curve_ = as_curve(curve)
        self.a = 0.0
        self.sigma = float(sigma)
        self.r0 = float(self.curve_.instantaneous_forward(0.0))

    def curve(self) -> YieldCurve:
        """The initial curve the model was fitted to."""
        return self.curve_

    def __repr__(self) -> str:
        return f"HoLeeModel(curve={self.curve_!r}, sigma={self.sigma!r})"


class HullWhiteModel(_CurveFittedGaussian):
    r"""Hull and White's (1990) extended Vasicek model.

    .. math:: dr = (\theta(t) - a r)\,dt + \sigma\,dW, \qquad
              \theta(t) = f_T(0, t) + a f(0, t) + \frac{\sigma^2}{2a}(1 - e^{-2at}).

    Vasicek's mean reversion with Ho-Lee's curve-fitting drift: the
    initial curve is matched exactly and volatility of long rates decays
    like :math:`B(\tau)/\tau`. The short rate is
    :math:`r(t) = x(t) + \varphi(t)` with :math:`x` an Ornstein-Uhlenbeck
    process started at 0 and
    :math:`\varphi(t) = f(0, t) + \tfrac{\sigma^2}{2a^2}(1 - e^{-at})^2`.

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure (a float is a flat rate).
    a : float
        Mean-reversion speed; ``a = 0`` is Ho-Lee.
    sigma : float
        Absolute volatility of the short rate.

    Examples
    --------
    >>> hw = HullWhiteModel(0.04, 0.1, 0.01)
    >>> round(float(hw.zero_coupon_bond(5.0)), 6)
    0.818731
    """

    def __init__(self, curve: CurveLike, a: float, sigma: float):
        nonnegative(a=a, sigma=sigma)
        self.curve_ = as_curve(curve)
        self.a = float(a)
        self.sigma = float(sigma)
        self.r0 = float(self.curve_.instantaneous_forward(0.0))

    def curve(self) -> YieldCurve:
        """The initial curve the model was fitted to."""
        return self.curve_

    def __repr__(self) -> str:
        return f"HullWhiteModel(curve={self.curve_!r}, a={self.a!r}, sigma={self.sigma!r})"


def black_derman_toy_tree(curve: CurveLike, sigma, maturity: float, n_steps: int) -> RateTree:
    r"""Calibrate a Black-Derman-Toy lognormal binomial short-rate tree to a zero curve.

    At step :math:`i` the rates are :math:`r_{i,j} = U_i e^{\sigma_i\sqrt{\Delta t}\,(2j - i)}`,
    :math:`j = 0, \dots, i`, with probability 1/2 on each branch, so the
    log short rate has volatility :math:`\sigma_i` and rates stay
    positive. Each median level :math:`U_i` is solved by
    :func:`scipy.optimize.brentq` so that the tree reprices the bond
    maturing at :math:`t_{i+1}`,

    .. math:: \sum_j Q_{i,j} e^{-r_{i,j}\Delta t} = P(0, t_{i+1}),

    where the Arrow-Debreu prices :math:`Q` are carried forward one level
    at a time (Jamshidian's 1991 forward induction). This is the
    short-rate-volatility form of the model; Black, Derman and Toy's
    original also matched a term structure of yield volatilities.

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure.
    sigma : float or array_like, shape (n_steps,)
        Volatility of the log short rate at each step.
    maturity : float
        Tree horizon in years.
    n_steps : int
        Number of steps.

    Returns
    -------
    RateTree

    Examples
    --------
    >>> from financialkit.fixed_income import FlatCurve
    >>> tree = black_derman_toy_tree(FlatCurve(0.05), 0.2, 5.0, 50)
    >>> round(tree.zero_coupon_bond(5.0), 8) == round(float(np.exp(-0.25)), 8)
    True
    """
    curve = as_curve(curve)
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    n = int(n_steps)
    dt = maturity / n
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n,)).copy()
    times = np.linspace(0.0, maturity, n + 1)
    P = curve.discount(times)
    rates = np.full((n, n), np.nan)
    Q = np.zeros((n + 1, n + 1))
    Q[0, 0] = 1.0
    for i in range(n):
        spread = np.exp(sig[i] * np.sqrt(dt) * (2 * np.arange(i + 1) - i))
        q = Q[i, : i + 1]

        def mismatch(logU, q=q, spread=spread, target=P[i + 1]):
            return float(np.sum(q * np.exp(-np.exp(logU) * spread * dt))) - target

        forward = -np.log(P[i + 1] / P[i]) / dt
        if forward <= 0:
            raise ValueError("the lognormal Black-Derman-Toy tree needs positive forward rates")
        # The Arrow-Debreu prices q sum to P[i], so every node rate at or above
        # (below) the forward makes the mismatch non-positive (non-negative);
        # the spreads lie within exp(+-sigma sqrt(dt) i), which brackets log U.
        width = abs(sig[i]) * np.sqrt(dt) * i + 1.0
        logU = brentq(mismatch, np.log(forward) - width, np.log(forward) + width, xtol=DEFAULT_ATOL, maxiter=DEFAULT_MAX_ITER)
        rates[i, : i + 1] = np.exp(logU) * spread
        d = q * np.exp(-rates[i, : i + 1] * dt)
        Q[i + 1, : i + 1] += 0.5 * d
        Q[i + 1, 1 : i + 2] += 0.5 * d
    return RateTree(times=times, rates=rates, arrow_debreu=Q, dt=dt, sigma=sig)


def hull_white_tree(curve: CurveLike, a: float, sigma: float, maturity: float, n_steps: int) -> TrinomialRateTree:
    r"""Build Hull and White's (1994) trinomial tree for :math:`dr = (\theta(t) - a r)\,dt + \sigma\,dW`.

    The rate over one step is :math:`R = x + \alpha_i`. The deviation
    :math:`x` lives on the grid :math:`j\Delta x` with
    :math:`\Delta x = \sigma\sqrt{3\Delta t}`, and its branch
    probabilities match the mean :math:`jM\Delta x`, :math:`M = -a\Delta t`,
    and variance :math:`\sigma^2\Delta t` of one step:

    .. math:: p_u = \tfrac16 + \tfrac12(j^2M^2 + jM), \quad
              p_m = \tfrac23 - j^2M^2, \quad
              p_d = \tfrac16 + \tfrac12(j^2M^2 - jM).

    At :math:`|j| = j_{max}`, the smallest integer above
    :math:`0.184/(a\Delta t)`, the branching turns inward (to
    :math:`j, j \mp 1, j \mp 2`) so the probabilities stay positive and
    the tree stops widening. Each shift :math:`\alpha_i` is then fitted to
    the bond maturing at :math:`t_{i+1}` through the Arrow-Debreu prices
    :math:`Q_{i,j}`,

    .. math:: \alpha_i = \frac{1}{\Delta t}\left[\ln \sum_j Q_{i,j} e^{-j\Delta x\Delta t} - \ln P(0, t_{i+1})\right],

    so the tree reprices the curve exactly.

    Parameters
    ----------
    curve : YieldCurve or float
        Initial term structure.
    a : float
        Mean-reversion speed; ``a = 0`` gives an untruncated Ho-Lee tree
        (``j_max = n_steps + 1``, never reached).
    sigma : float
        Absolute volatility of the short rate.
    maturity : float
        Tree horizon in years.
    n_steps : int
        Number of steps.

    Returns
    -------
    TrinomialRateTree

    Examples
    --------
    Hull, 10th ed., Sec. 32.7: three yearly steps with ``a = 0.1`` and
    ``sigma = 0.01``. The branching turns inward at ``j_max = 2``, and the
    first two shifts are 3.824% and 5.205%:

    >>> from financialkit.fixed_income import ZeroCurve
    >>> curve = ZeroCurve([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.0343, 0.03824, 0.04183, 0.04512, 0.04812, 0.05086])
    >>> tree = hull_white_tree(curve, 0.1, 0.01, 3.0, 3)
    >>> tree.j_max, [round(float(x), 5) for x in tree.alpha[:2]]
    (2, [0.03824, 0.05205])
    """
    curve = as_curve(curve)
    nonnegative(a=a)
    positive(sigma=sigma, maturity=maturity)
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    n = int(n_steps)
    dt = maturity / n
    dx = sigma * np.sqrt(3 * dt)
    M = -a * dt
    j_max = n + 1 if a == 0 else int(np.floor(0.184 / (a * dt))) + 1
    width = min(j_max, n)
    j = np.arange(-width, width + 1)
    jM = j * M
    up = 1 / 6 + 0.5 * (jM**2 + jM)
    mid = 2 / 3 - jM**2
    down = 1 / 6 + 0.5 * (jM**2 - jM)
    centers = j.copy()
    top, bottom = j == j_max, j == -j_max
    # Turned inward at the edges: from j_max to j, j - 1, j - 2, and mirrored at -j_max.
    up[top], mid[top], down[top] = 7 / 6 + 0.5 * (jM[top] ** 2 + 3 * jM[top]), -1 / 3 - jM[top] ** 2 - 2 * jM[top], 1 / 6 + 0.5 * (jM[top] ** 2 + jM[top])
    up[bottom], mid[bottom], down[bottom] = (
        1 / 6 + 0.5 * (jM[bottom] ** 2 - jM[bottom]),
        -1 / 3 - jM[bottom] ** 2 + 2 * jM[bottom],
        7 / 6 + 0.5 * (jM[bottom] ** 2 - 3 * jM[bottom]),
    )
    centers[top] -= 1
    centers[bottom] += 1
    probabilities = np.column_stack([down, mid, up])
    if np.any(probabilities < 0):
        raise ValueError("negative branch probabilities: a * dt is too large, use more steps")
    # Edge nodes of an untruncated tree are never reached as parents; clipping keeps their indices in range.
    centers = np.clip(centers, -width + 1, width - 1)
    times = np.linspace(0.0, maturity, n + 1)
    P = curve.discount(times)
    alpha = np.empty(n)
    Q = np.zeros((n + 1, j.size))
    Q[0, width] = 1.0
    for i in range(n):
        alpha[i] = (np.log(np.sum(Q[i] * np.exp(-j * dx * dt))) - np.log(P[i + 1])) / dt
        d = Q[i] * np.exp(-(alpha[i] + j * dx) * dt)
        for k in range(3):
            np.add.at(Q[i + 1], centers + width + k - 1, d * probabilities[:, k])
    return TrinomialRateTree(times=times, alpha=alpha, dx=float(dx), dt=float(dt), j_max=j_max, probabilities=probabilities, centers=centers, arrow_debreu=Q)


def futures_convexity_adjustment(t1, t2, sigma, a: float = 0.0):
    r"""Futures rate minus forward rate for a futures contract on the rate over ``[t1, t2]``.

    Futures are settled daily, so their rate is the risk-neutral
    expectation of the future rate rather than the forward rate, and is
    higher by a convexity adjustment. Under Ho-Lee (``a = 0``),

    .. math:: \text{futures} - \text{forward} = \tfrac12 \sigma^2 t_1 t_2,

    and under Hull-White

    .. math:: \frac{B(t_1, t_2)}{t_2 - t_1}\left[B(t_1, t_2)(1 - e^{-2at_1}) + 2aB(0, t_1)^2\right]\frac{\sigma^2}{4a},

    with :math:`B(s, t) = (1 - e^{-a(t-s)})/a`. Both rates are
    continuously compounded.

    Parameters
    ----------
    t1, t2 : array_like
        Start and end of the underlying rate's period, ``t2 > t1 >= 0``.
    sigma : array_like
        Short-rate volatility.
    a : float
        Hull-White mean reversion; 0 gives the Ho-Lee formula.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Hull, 10th ed., Example 6.4: sigma = 1.2%, an 8-year futures contract
    on 3-month rates.

    >>> round(100 * futures_convexity_adjustment(8.0, 8.25, 0.012), 3)
    0.475
    """
    t1, t2, sigma = (np.asarray(x, dtype=np.float64) for x in (t1, t2, sigma))
    if np.any(t2 <= t1) or np.any(t1 < 0):
        raise ValueError("need t2 > t1 >= 0")
    if a == 0:
        out = 0.5 * sigma**2 * t1 * t2
    else:
        B12 = -np.expm1(-a * (t2 - t1)) / a
        B01 = -np.expm1(-a * t1) / a
        out = B12 / (t2 - t1) * (B12 * -np.expm1(-2 * a * t1) + 2 * a * B01**2) * sigma**2 / (4 * a)
    return float(out) if out.ndim == 0 else out
