"""Equilibrium short-rate models: Merton, Vasicek and Cox-Ingersoll-Ross,
with the Feller condition and Feynman-Kac bond prices by Monte Carlo.

* :func:`monte_carlo_bond_prices`: the Feynman-Kac representation
  :math:`P(0, T) = E^Q[e^{-\\int_0^T r\\,dt}]` evaluated on paths from any
  :class:`~financialkit.interest_rate_models.core.base.ShortRateModel`,
  the check on every closed form in this domain.
* :class:`MertonShortRateModel`: Merton's (1973) arithmetic Brownian
  short rate, :math:`dr = \\mu\\,dt + \\sigma\\,dW`.
* :class:`VasicekModel`: Vasicek's (1977) mean-reverting Gaussian rate,
  simulated exactly by :func:`financialkit.integrators.ou_paths`.
* :class:`CIRModel`: Cox, Ingersoll and Ross's (1985) square-root rate,
  sampled exactly from its noncentral chi-square transition
  (:data:`scipy.stats.ncx2`) or by full-truncation Euler through
  :func:`financialkit.integrators.euler_maruyama`.
* :func:`feller_condition`: Feller's (1951) test :math:`2\\kappa\\theta \\ge
  \\sigma^2` for the square-root process to stay away from zero.

References: M. Kac, Trans. AMS 65 (1949) 1-13, after R. P. Feynman, Rev.
Mod. Phys. 20 (1948) 367-387; W. Feller, Annals of Mathematics 54(1)
(1951) 173-182; R. C. Merton, Bell J. Econ. Manag. Sci. 4 (1973) 141-183;
O. Vasicek, J. Financial Economics 5 (1977) 177-188; J. C. Cox, J. E.
Ingersoll and S. A. Ross, Econometrica 53(2) (1985) 385-407; J. C. Hull,
*Options, Futures, and Other Derivatives*, 10th ed. (Pearson, 2018), Ch. 31;
D. Brigo and F. Mercurio, *Interest Rate Models -- Theory and Practice*,
2nd ed. (Springer, 2006), Ch. 3.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ncx2

from financialkit.integrators import RNGLike, euler_maruyama, ou_paths
from financialkit.interest_rate_models.core.base import BondMonteCarloResult, GaussianShortRateModel, ShortRateModel
from financialkit.interest_rate_models.utils.numerics import cir_diffusion, cir_drift, path_discount_factors

__all__ = ["monte_carlo_bond_prices", "feller_condition", "MertonShortRateModel", "VasicekModel", "CIRModel"]


def _state(model: ShortRateModel, state) -> np.ndarray:
    return np.asarray(model.r0 if state is None else state, dtype=np.float64)


def _vasicek_B(kappa: float, tau):
    return (1.0 - np.exp(-kappa * tau)) / kappa


def monte_carlo_bond_prices(model: ShortRateModel, maturities, n_steps: int = 200, n_paths: int = 10_000, rng: RNGLike = None) -> BondMonteCarloResult:
    r"""Zero-coupon bond prices by simulating the short rate (Feynman-Kac).

    Kac (1949) showed that the solution of the bond-pricing PDE
    :math:`\partial_t P + \mu \partial_r P + \tfrac12 \sigma^2 \partial_{rr} P - rP = 0`
    is the expectation

    .. math:: P(0, T) = E^Q\left[\exp\left(-\int_0^T r(s)\,ds\right)\right],

    so a bond is priced by averaging discount factors over simulated
    paths. The integral is the trapezoidal rule on the simulation grid, an
    :math:`O(\Delta t^2)` bias well below the sampling error.

    Parameters
    ----------
    model : ShortRateModel
    maturities : array_like, shape (m,)
        Positive bond maturities; the paths run to the longest.
    n_steps : int
        Time steps to the longest maturity.
    n_paths : int
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    BondMonteCarloResult
        With the model's closed-form prices in ``exact``.

    Examples
    --------
    >>> res = monte_carlo_bond_prices(VasicekModel(0.03, 0.5, 0.05, 0.01), [1.0, 5.0], n_paths=4000, rng=0)
    >>> bool(abs(res.z_scores).max() < 4)
    True
    """
    maturities = np.atleast_1d(np.asarray(maturities, dtype=np.float64))
    if np.any(maturities <= 0):
        raise ValueError("maturities must be positive")
    T = float(maturities.max())
    times, rates = model.simulate(T, n_steps, n_paths, rng=rng)
    disc = path_discount_factors(times, rates)
    pos = maturities / (T / n_steps)
    lo = np.minimum(np.floor(pos).astype(int), n_steps - 1)
    w = pos - lo
    log_d = -np.log(disc)
    at_T = np.exp(-((1 - w) * log_d[:, lo] + w * log_d[:, lo + 1]))
    return BondMonteCarloResult(
        maturities=maturities,
        prices=at_T.mean(axis=0),
        std_errors=at_T.std(axis=0, ddof=1) / np.sqrt(n_paths),
        exact=np.asarray(model.zero_coupon_bond(maturities), dtype=np.float64),
        n_paths=int(n_paths),
    )


def feller_condition(kappa, theta, sigma):
    r"""Feller's ratio :math:`2\kappa\theta / \sigma^2` for the square-root process.

    Feller (1951) classified the boundary at zero of
    :math:`dr = \kappa(\theta - r)\,dt + \sigma\sqrt{r}\,dW`: the process
    never reaches zero when the ratio is at least 1, and touches it (and
    is reflected) when it is below 1. The ratio is also the shape
    parameter of the stationary gamma distribution, whose density
    diverges at zero when it is below 1.

    Parameters
    ----------
    kappa, theta, sigma : array_like
        Mean-reversion speed, long-run level and volatility.

    Returns
    -------
    float or ndarray
        :math:`2\kappa\theta/\sigma^2`; the condition holds where it is ``>= 1``.

    Examples
    --------
    >>> round(feller_condition(0.5, 0.05, 0.2), 10)
    1.25
    """
    ratio = 2.0 * np.asarray(kappa, dtype=np.float64) * np.asarray(theta, dtype=np.float64) / np.asarray(sigma, dtype=np.float64) ** 2
    return float(ratio) if ratio.ndim == 0 else ratio


class MertonShortRateModel(GaussianShortRateModel):
    r"""Merton's (1973) Gaussian short rate, :math:`dr = \mu\,dt + \sigma\,dW`.

    The first stochastic interest rate in continuous time, from Section 6
    of Merton's "Theory of Rational Option Pricing". Bond prices are

    .. math:: P(t, T) = \exp\left(-r\tau - \tfrac12 \mu\tau^2 + \tfrac16 \sigma^2\tau^3\right),
              \quad \tau = T - t,

    so the cubic convexity term eventually makes long yields fall without
    bound: rates without mean reversion cannot carry a long curve.

    Parameters
    ----------
    r0 : float
        Initial short rate.
    mu : float
        Constant drift of the short rate, per year.
    sigma : float
        Absolute volatility of the short rate.

    Examples
    --------
    >>> m = MertonShortRateModel(0.05, 0.0, 0.0)
    >>> round(float(m.zero_coupon_bond(2.0)), 6)
    0.904837
    """

    def __init__(self, r0: float, mu: float, sigma: float):
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.r0, self.mu, self.sigma = float(r0), float(mu), float(sigma)

    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        tau = np.asarray(T, dtype=np.float64) - t
        r = _state(self, state)
        return np.exp(-r * tau - 0.5 * self.mu * tau**2 + self.sigma**2 * tau**3 / 6.0)

    def bond_option_volatility(self, T, S):
        T, S = np.asarray(T, dtype=np.float64), np.asarray(S, dtype=np.float64)
        return self.sigma * (S - T) * np.sqrt(T)

    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None):
        times, w = ou_paths(0.0, 0.0, 0.0, self.sigma, T, n_steps, n_paths, rng=rng)
        return times, self.r0 + self.mu * times + w

    def __repr__(self) -> str:
        return f"MertonShortRateModel(r0={self.r0!r}, mu={self.mu!r}, sigma={self.sigma!r})"


class VasicekModel(GaussianShortRateModel):
    r"""Vasicek's (1977) mean-reverting short rate, :math:`dr = \kappa(\theta - r)\,dt + \sigma\,dW`.

    The rate is an Ornstein-Uhlenbeck process, so bond prices are
    exponential-affine,

    .. math:: P(t, T) = A(\tau) e^{-B(\tau) r}, \quad B = \frac{1 - e^{-\kappa\tau}}{\kappa}, \quad
              \ln A = \left(\theta - \frac{\sigma^2}{2\kappa^2}\right)(B - \tau) - \frac{\sigma^2 B^2}{4\kappa},

    and long yields tend to :math:`\theta - \sigma^2/(2\kappa^2)`.
    Depending on :math:`r_0` against that level, the curve is upward
    sloping, inverted or humped.

    Parameters
    ----------
    r0, kappa, theta, sigma : float
        Initial rate, mean-reversion speed, long-run mean and volatility.

    Examples
    --------
    >>> m = VasicekModel(0.06, 0.1, 0.1, 0.02)
    >>> round(float(m.zero_coupon_bond(10.0)), 4)
    0.4899
    >>> round(m.long_yield, 4)
    0.08
    """

    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.r0, self.kappa, self.theta, self.sigma = float(r0), float(kappa), float(theta), float(sigma)

    @property
    def long_yield(self) -> float:
        r"""float: :math:`\lim_{T\to\infty} R(0, T) = \theta - \sigma^2/(2\kappa^2)`."""
        return self.theta - self.sigma**2 / (2.0 * self.kappa**2)

    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        tau = np.asarray(T, dtype=np.float64) - t
        k, s = self.kappa, self.sigma
        B = _vasicek_B(k, tau)
        lnA = (self.theta - s**2 / (2 * k**2)) * (B - tau) - s**2 * B**2 / (4 * k)
        return np.exp(lnA - B * _state(self, state))

    def bond_option_volatility(self, T, S):
        T, S = np.asarray(T, dtype=np.float64), np.asarray(S, dtype=np.float64)
        k = self.kappa
        return self.sigma * _vasicek_B(k, S - T) * np.sqrt((1.0 - np.exp(-2 * k * T)) / (2 * k))

    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None):
        return ou_paths(self.r0, self.kappa, self.theta, self.sigma, T, n_steps, n_paths, rng=rng)

    def __repr__(self) -> str:
        return f"VasicekModel(r0={self.r0!r}, kappa={self.kappa!r}, theta={self.theta!r}, sigma={self.sigma!r})"


class CIRModel(ShortRateModel):
    r"""Cox, Ingersoll and Ross's (1985) square-root short rate.

    .. math:: dr = \kappa(\theta - r)\,dt + \sigma\sqrt{r}\,dW

    Volatility falls with the rate, so (under the Feller condition) rates
    stay positive. Bond prices are exponential-affine,

    .. math:: B = \frac{2(e^{\gamma\tau} - 1)}{(\gamma + \kappa)(e^{\gamma\tau} - 1) + 2\gamma}, \quad
              A = \left[\frac{2\gamma e^{(\kappa + \gamma)\tau/2}}{(\gamma + \kappa)(e^{\gamma\tau} - 1) + 2\gamma}\right]^{2\kappa\theta/\sigma^2},

    with :math:`\gamma = \sqrt{\kappa^2 + 2\sigma^2}` and
    :math:`P = A e^{-Br}`. The transition is a scaled noncentral
    chi-square, and the stationary law a gamma distribution with shape
    :math:`2\kappa\theta/\sigma^2` and scale :math:`\sigma^2/(2\kappa)`.

    Parameters
    ----------
    r0, kappa, theta, sigma : float
        Initial rate (non-negative), mean-reversion speed, long-run mean
        and volatility.

    Examples
    --------
    >>> m = CIRModel(0.06, 0.1, 0.1, 0.11)
    >>> round(float(m.zero_coupon_bond(10.0)), 4)
    0.5039
    >>> round(m.feller_ratio, 3)
    1.653
    """

    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            raise ValueError("kappa, theta and sigma must be positive")
        if r0 < 0:
            raise ValueError("r0 must be non-negative")
        self.r0, self.kappa, self.theta, self.sigma = float(r0), float(kappa), float(theta), float(sigma)

    @property
    def feller_ratio(self) -> float:
        r"""float: :math:`2\kappa\theta/\sigma^2` (see :func:`feller_condition`)."""
        return float(feller_condition(self.kappa, self.theta, self.sigma))

    @property
    def gamma(self) -> float:
        r"""float: :math:`\sqrt{\kappa^2 + 2\sigma^2}`."""
        return float(np.sqrt(self.kappa**2 + 2 * self.sigma**2))

    def _AB(self, tau):
        k, g = self.kappa, self.gamma
        e = np.expm1(g * tau)
        den = (g + k) * e + 2 * g
        B = 2 * e / den
        lnA = self.feller_ratio * (np.log(2 * g) + 0.5 * (k + g) * tau - np.log(den))
        return lnA, B

    def zero_coupon_bond(self, T, t: float = 0.0, state=None):
        lnA, B = self._AB(np.asarray(T, dtype=np.float64) - t)
        return np.exp(lnA - B * _state(self, state))

    def stationary_shape_scale(self) -> tuple[float, float]:
        """Shape and scale of the stationary gamma distribution of the rate."""
        return self.feller_ratio, self.sigma**2 / (2 * self.kappa)

    def simulate(self, T: float, n_steps: int, n_paths: int, rng: RNGLike = None, method: str = "exact"):
        """Short-rate paths; ``method="exact"`` samples the noncentral chi-square
        transition, ``"euler"`` runs full-truncation Euler-Maruyama.

        Parameters
        ----------
        T : float
        n_steps, n_paths : int
        rng : numpy.random.Generator or int, optional
        method : {"exact", "euler"}

        Returns
        -------
        times : ndarray, shape (n_steps + 1,)
        rates : ndarray, shape (n_paths, n_steps + 1)
        """
        rng = np.random.default_rng(rng)
        if method == "euler":
            times, x = euler_maruyama(cir_drift, cir_diffusion, self.r0, T, n_steps, n_paths, params=[self.kappa, self.theta, self.sigma], rng=rng)
            return times, np.maximum(x, 0.0)
        if method != "exact":
            raise ValueError(f'method must be "exact" or "euler", got {method!r}')
        times = np.linspace(0.0, T, n_steps + 1)
        dt = T / n_steps
        c = self.sigma**2 * -np.expm1(-self.kappa * dt) / (4 * self.kappa)
        df = 4 * self.kappa * self.theta / self.sigma**2
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = self.r0
        for i in range(n_steps):
            out[:, i + 1] = c * ncx2.rvs(df, out[:, i] * np.exp(-self.kappa * dt) / c, random_state=rng)
        return times, out

    def __repr__(self) -> str:
        return f"CIRModel(r0={self.r0!r}, kappa={self.kappa!r}, theta={self.theta!r}, sigma={self.sigma!r})"
