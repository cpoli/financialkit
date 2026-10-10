"""Jumps: Merton's jump diffusion, variance gamma, Kou's double-exponential jumps and Bates.

Merton (1976) added Poisson jumps with lognormal sizes to geometric
Brownian motion. When the jump risk is diversifiable the call price is
a Poisson mixture of Black-Scholes prices (:func:`merton_jump_price`),
and short-dated smiles become steep, which diffusions cannot do.

Madan and Seneta (1990) replaced calendar time by a gamma-distributed
business clock: Brownian motion run on gamma time is the variance gamma
process, a pure-jump Lévy process with infinitely many small jumps and
fat, semi-heavy tails. Madan, Carr and Chang (1998) added a drift in
business time, :math:`\\theta`, which skews the distribution and the
smile (:class:`VarianceGammaModel`).

Kou (2002) took double-exponential jump sizes: asymmetric,
leptokurtic, and memoryless, which keeps barrier and lookback options
analytically tractable (:class:`KouModel`). Bates (1996) put Merton's
jumps on top of Heston's stochastic variance, the jumps shaping the
short end of the smile and the variance its long end
(:class:`BatesModel`).

Every model's characteristic function is normalised by the forward,
with the martingale correction folded into the drift, so they all price
with the Fourier methods.

References: R. C. Merton, *J. Financial Economics* 3 (1976) 125-144;
D. B. Madan and E. Seneta, *J. Business* 63 (1990) 511-524; D. B. Madan,
P. P. Carr and E. C. Chang, *European Finance Review* 2 (1998) 79-105;
S. G. Kou, *Management Science* 48 (2002) 1086-1101; D. S. Bates,
*Rev. Financial Studies* 9 (1996) 69-107.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln, kv

from financialkit.integrators import RNGLike
from financialkit.options import black_scholes_price
from financialkit.options.utils.common import as_output, option_sign
from financialkit.stochastic_volatility.core.base import CharacteristicFunctionModel
from financialkit.stochastic_volatility.systems.heston import HestonModel

__all__ = ["MertonJumpModel", "merton_jump_price", "VarianceGammaModel", "KouModel", "BatesModel"]


class MertonJumpModel(CharacteristicFunctionModel):
    """Merton's (1976) jump diffusion: GBM plus compound Poisson lognormal jumps.

    :math:`dS/S_- = (r - q - \\lambda\\bar k)\\,dt + \\sigma\\,dW + (J - 1)\\,dN`,
    with :math:`\\ln J \\sim N(\\mu_J, \\delta^2)` and
    :math:`\\bar k = E[J] - 1 = e^{\\mu_J + \\delta^2/2} - 1`.

    Parameters
    ----------
    sigma : float
        Diffusive volatility.
    lam : float
        Jump intensity (jumps per year).
    mu_j : float
        Mean log jump size.
    delta : float
        Standard deviation of the log jump size.

    Examples
    --------
    >>> m = MertonJumpModel(0.2, 1.0, -0.1, 0.15)
    >>> round(m.mean_jump, 6)
    -0.084926
    """

    name = "Merton"

    def __init__(self, sigma: float, lam: float, mu_j: float, delta: float):
        if sigma < 0 or lam < 0 or delta < 0:
            raise ValueError("sigma, lam and delta must be non-negative")
        self.sigma, self.lam, self.mu_j, self.delta = float(sigma), float(lam), float(mu_j), float(delta)

    @property
    def mean_jump(self) -> float:
        """float: :math:`\\bar k = e^{\\mu_J + \\delta^2/2} - 1`, the mean relative jump."""
        return float(np.expm1(self.mu_j + 0.5 * self.delta**2))

    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        jump = np.exp(1j * u * self.mu_j - 0.5 * self.delta**2 * u**2) - 1.0
        return np.exp(T * (-0.5 * self.sigma**2 * (1j * u + u**2) - 1j * u * self.lam * self.mean_jump + self.lam * jump))

    def sample_terminal(self, T: float, n: int, rng: RNGLike = None) -> np.ndarray:
        """Draws of :math:`X_T = \\ln(S_T/F_T)`: Gaussian diffusion plus a Poisson number of Gaussian jumps.

        Parameters
        ----------
        T : float
        n : int
            Number of draws.
        rng : numpy.random.Generator or int, optional

        Returns
        -------
        ndarray, shape (n,)
        """
        rng = np.random.default_rng(rng)
        N = rng.poisson(self.lam * T, n)
        z = rng.standard_normal((2, n))
        drift = -(0.5 * self.sigma**2 + self.lam * self.mean_jump) * T
        return drift + self.sigma * np.sqrt(T) * z[0] + N * self.mu_j + np.sqrt(N) * self.delta * z[1]


def merton_jump_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    model: MertonJumpModel,
    q: ArrayLike = 0.0,
    option_type: str = "call",
    n_terms: int = 80,
) -> np.ndarray | float:
    r"""Merton's (1976) series: a Poisson mixture of Black-Scholes prices.

    .. math::

       C = \sum_{n \ge 0} \frac{e^{-\lambda' T} (\lambda' T)^n}{n!}\,
       C_{BS}(S, K, T, r_n, \sigma_n, q),

    with :math:`\lambda' = \lambda(1 + \bar k)`,
    :math:`\sigma_n^2 = \sigma^2 + n\delta^2/T` and
    :math:`r_n = r - \lambda\bar k + n \ln(1 + \bar k)/T`.

    Parameters
    ----------
    S, K, T, r : array_like
    model : MertonJumpModel
    q : array_like
    option_type : {"call", "put"}
    n_terms : int
        Number of Poisson terms summed.

    Returns
    -------
    float or ndarray

    Examples
    --------
    Without jumps the series is Black-Scholes:

    >>> m = MertonJumpModel(0.2, 0.0, -0.1, 0.1)
    >>> merton_jump_price(100.0, 100.0, 1.0, 0.05, m) == black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    True
    """
    option_sign(option_type)
    S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, T, r, q)))
    kbar = model.mean_jump
    lam_p = model.lam * (1.0 + kbar)
    if lam_p == 0.0:
        return as_output(np.asarray(black_scholes_price(S, K, T, r, model.sigma, q, option_type)))
    total = np.zeros(S.shape)
    for n in range(n_terms):
        weight = np.exp(-lam_p * T + n * np.log(lam_p * T) - gammaln(n + 1))
        sigma_n = np.sqrt(model.sigma**2 + n * model.delta**2 / T)
        r_n = r - model.lam * kbar + n * np.log1p(kbar) / T
        # Discounting at r_n rather than r is absorbed by the weights using lambda' instead of lambda.
        total += weight * np.asarray(black_scholes_price(S, K, T, r_n, sigma_n, q, option_type))
    return as_output(total)


class VarianceGammaModel(CharacteristicFunctionModel):
    """Variance gamma: Brownian motion with drift run on a gamma clock (Madan-Seneta 1990, Madan-Carr-Chang 1998).

    :math:`X_T = \\omega T + \\theta G_T + \\sigma W_{G_T}`, with
    :math:`G_T \\sim \\Gamma(T/\\nu, \\nu)` and
    :math:`\\omega = \\ln(1 - \\theta\\nu - \\tfrac12\\sigma^2\\nu)/\\nu` so that
    :math:`E[e^{X_T}] = 1`. ``theta = 0`` is the symmetric Madan-Seneta
    process; ``theta < 0`` skews it to the left.

    Parameters
    ----------
    sigma : float
        Volatility of the Brownian motion in business time.
    nu : float
        Variance rate of the gamma clock (excess kurtosis).
    theta : float
        Drift in business time (skewness).

    Examples
    --------
    >>> m = VarianceGammaModel(sigma=0.12, nu=0.2, theta=-0.14)
    >>> round(m.price(100.0, 90.0, 1.0, 0.1), 6)  # Fang-Oosterlee (2008), Table 7
    19.099355
    """

    name = "variance gamma"

    def __init__(self, sigma: float, nu: float, theta: float = 0.0):
        if sigma <= 0 or nu <= 0:
            raise ValueError("sigma and nu must be positive")
        if 1.0 - theta * nu - 0.5 * sigma**2 * nu <= 0:
            raise ValueError("E[S_T] is infinite: need 1 - theta nu - sigma^2 nu / 2 > 0")
        self.sigma, self.nu, self.theta = float(sigma), float(nu), float(theta)

    @property
    def omega(self) -> float:
        """float: Martingale correction :math:`\\omega` per unit time."""
        return float(np.log(1.0 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu) / self.nu)

    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        base = 1.0 - 1j * u * self.theta * self.nu + 0.5 * self.sigma**2 * self.nu * u**2
        return np.exp(1j * u * self.omega * T - T / self.nu * np.log(base))

    def pdf(self, x: ArrayLike, T: float) -> np.ndarray:
        r"""Closed-form density of :math:`X_T` (Madan, Carr and Chang 1998, eq. 23).

        .. math::

           f(z) = \frac{2 e^{\theta z/\sigma^2}}{\nu^{T/\nu}\sqrt{2\pi}\,\sigma\,\Gamma(T/\nu)}
           \left(\frac{z^2}{2\sigma^2/\nu + \theta^2}\right)^{\frac{T}{2\nu} - \frac14}
           K_{\frac{T}{\nu} - \frac12}\!\left(\frac{\sqrt{z^2 (2\sigma^2/\nu + \theta^2)}}{\sigma^2}\right),

        with :math:`z = x - \omega T` and :math:`K` the modified Bessel
        function of the second kind.

        Parameters
        ----------
        x : array_like
        T : float

        Returns
        -------
        ndarray
        """
        z = np.asarray(x, dtype=np.float64) - self.omega * T
        s2, nu, th = self.sigma**2, self.nu, self.theta
        c = 2 * s2 / nu + th**2
        a = T / nu
        az = np.maximum(np.abs(z), 1e-300)
        log_norm = np.log(2.0) + th * z / s2 - a * np.log(nu) - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2) - gammaln(a)
        return np.exp(log_norm + (0.5 * a - 0.25) * np.log(az**2 / c)) * kv(a - 0.5, az * np.sqrt(c) / s2)

    def critical_moments(self) -> tuple[float, float]:
        """Lee's critical moments: the largest :math:`p` with :math:`E[S_T^{1+p}] < \\infty` and :math:`q` with :math:`E[S_T^{-q}] < \\infty`.

        The moment generating function :math:`E[e^{sX}]` is finite exactly
        while :math:`1 - \\theta\\nu s - \\tfrac12\\sigma^2\\nu s^2 > 0`, for
        every maturity.

        Returns
        -------
        p_right, q_left : float
        """
        a, b = 0.5 * self.sigma**2 * self.nu, self.theta * self.nu
        disc = np.sqrt(b * b + 4 * a)
        s_plus, s_minus = (-b + disc) / (2 * a), (-b - disc) / (2 * a)
        return float(s_plus - 1.0), float(-s_minus)

    def sample_terminal(self, T: float, n: int, rng: RNGLike = None) -> np.ndarray:
        """Draws of :math:`X_T` by subordination: a gamma time, then a normal given it.

        Parameters
        ----------
        T : float
        n : int
        rng : numpy.random.Generator or int, optional

        Returns
        -------
        ndarray, shape (n,)
        """
        rng = np.random.default_rng(rng)
        G = rng.gamma(T / self.nu, self.nu, n)
        return self.omega * T + self.theta * G + self.sigma * np.sqrt(G) * rng.standard_normal(n)


class KouModel(CharacteristicFunctionModel):
    """Kou's (2002) double-exponential jump diffusion.

    Log jump sizes are :math:`\\mathrm{Exp}(\\eta_1)` upward with probability
    :math:`p` and :math:`-\\mathrm{Exp}(\\eta_2)` downward otherwise.

    Parameters
    ----------
    sigma : float
        Diffusive volatility.
    lam : float
        Jump intensity.
    p : float
        Probability that a jump is upward.
    eta1 : float
        Rate of upward jumps; must exceed 1 so that :math:`E[J] < \\infty`.
    eta2 : float
        Rate of downward jumps.

    Examples
    --------
    >>> m = KouModel(0.16, 1.0, 0.4, 10.0, 5.0)
    >>> m.critical_moments()
    (9.0, 5.0)
    """

    name = "Kou"

    def __init__(self, sigma: float, lam: float, p: float, eta1: float, eta2: float):
        if sigma < 0 or lam < 0 or not 0 <= p <= 1:
            raise ValueError("need sigma, lam >= 0 and 0 <= p <= 1")
        if eta1 <= 1 or eta2 <= 0:
            raise ValueError("need eta1 > 1 and eta2 > 0")
        self.sigma, self.lam, self.p, self.eta1, self.eta2 = (float(a) for a in (sigma, lam, p, eta1, eta2))

    def _jump_mgf(self, s):
        return self.p * self.eta1 / (self.eta1 - s) + (1 - self.p) * self.eta2 / (self.eta2 + s)

    @property
    def mean_jump(self) -> float:
        """float: :math:`\\zeta = E[e^Y] - 1`."""
        return float(self._jump_mgf(1.0) - 1.0)

    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        return np.exp(T * (-0.5 * self.sigma**2 * (1j * u + u**2) - 1j * u * self.lam * self.mean_jump + self.lam * (self._jump_mgf(1j * u) - 1.0)))

    def critical_moments(self) -> tuple[float, float]:
        """Lee's critical moments :math:`(\\eta_1 - 1, \\eta_2)`: the exponential tails cap the moments.

        Returns
        -------
        p_right, q_left : float
        """
        return self.eta1 - 1.0, self.eta2

    def sample_terminal(self, T: float, n: int, rng: RNGLike = None) -> np.ndarray:
        """Draws of :math:`X_T`; the up and down jump totals are gamma given their counts.

        Parameters
        ----------
        T : float
        n : int
        rng : numpy.random.Generator or int, optional

        Returns
        -------
        ndarray, shape (n,)
        """
        rng = np.random.default_rng(rng)
        N = rng.poisson(self.lam * T, n)
        n_up = rng.binomial(N, self.p)
        jumps = rng.gamma(n_up, 1 / self.eta1) - rng.gamma(N - n_up, 1 / self.eta2)
        drift = -(0.5 * self.sigma**2 + self.lam * self.mean_jump) * T
        return drift + self.sigma * np.sqrt(T) * rng.standard_normal(n) + jumps


class BatesModel(CharacteristicFunctionModel):
    """Bates's (1996) model: Heston's stochastic variance plus Merton's lognormal jumps.

    Parameters
    ----------
    v0, kappa, theta, xi, rho : float
        Heston parameters, see :class:`~financialkit.stochastic_volatility.systems.heston.HestonModel`.
    lam, mu_j, delta : float
        Merton jump parameters, see :class:`MertonJumpModel`.

    Examples
    --------
    With no jumps it is Heston:

    >>> b = BatesModel(0.04, 2.0, 0.04, 0.3, -0.7, lam=0.0, mu_j=-0.1, delta=0.1)
    >>> h = HestonModel(0.04, 2.0, 0.04, 0.3, -0.7)
    >>> bool(abs(b.price(100.0, 95.0, 1.0, 0.03) - h.price(100.0, 95.0, 1.0, 0.03)) < 1e-12)
    True
    """

    name = "Bates"

    def __init__(self, v0: float, kappa: float, theta: float, xi: float, rho: float, lam: float, mu_j: float, delta: float):
        self.heston = HestonModel(v0, kappa, theta, xi, rho)
        self.jumps = MertonJumpModel(0.0, lam, mu_j, delta)

    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        return self.heston.char_func(u, T) * self.jumps.char_func(u, T)
