"""Abstract base classes and result containers for financialkit.stochastic_volatility.

Every model beyond Black-Scholes in this domain that is priced by
transform methods implements :class:`CharacteristicFunctionModel`: it
supplies the characteristic function of the normalised log return
:math:`X_T = \\ln(S_T / F_T)`, where :math:`F_T = S e^{(r - q) T}` is the
forward, and inherits pricing by any of the Fourier methods (Heston's
two probabilities, Carr-Madan's FFT, Lewis's single integral, the COS
expansion) and Black-Scholes implied volatilities. Because :math:`X_T`
is normalised by the forward, :math:`E[e^{X_T}] = 1` and the
characteristic function does not depend on rates or dividends, so one
model instance serves every spot, rate and strike.

Everything with more structure returns a small dataclass that the
visualizers plot: a Carr-Madan strike grid, a Dupire local-volatility
surface, simulated Heston and rough Bergomi paths, SABR and SVI fits, a
characteristic-function model calibrated to a smile, and a roughness
estimate of log-volatility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from financialkit._deprecation import deprecated_field

__all__ = [
    "CharacteristicFunctionModel",
    "FFTResult",
    "LocalVolSurface",
    "HestonPaths",
    "SABRFit",
    "SVIFit",
    "FourierCalibration",
    "RoughBergomiPaths",
    "RoughnessEstimate",
]


class CharacteristicFunctionModel(ABC):
    """A model for :math:`X_T = \\ln(S_T / F_T)` known through its characteristic function.

    Subclasses implement :meth:`char_func`; pricing, implied volatilities
    and cumulants follow from it.
    """

    name: str = ""

    @abstractmethod
    def char_func(self, u: ArrayLike, T: float) -> np.ndarray:
        """Characteristic function :math:`\\phi(u) = E[e^{i u X_T}]`.

        Parameters
        ----------
        u : array_like of complex
            Arguments; complex values are allowed wherever the moment
            generating function exists, which the Fourier pricers need.
        T : float
            Maturity in years.

        Returns
        -------
        ndarray of complex
            With :math:`\\phi(0) = 1` and :math:`\\phi(-i) = E[e^{X_T}] = 1`.
        """

    def cumulants(self, T: float, h: float = 0.02) -> tuple[float, float, float]:
        """First, second and fourth cumulants of :math:`X_T`.

        By central differences of the cumulant generating function
        :math:`\\kappa(s) = \\ln \\phi(-i s)` at :math:`s = 0`. The COS method
        uses them to size its truncation interval.

        Parameters
        ----------
        T : float
            Maturity in years.
        h : float
            Step in :math:`s`; the moment generating function must exist on
            :math:`[-2h, 2h]`.

        Returns
        -------
        c1, c2, c4 : float
        """
        s = h * np.arange(-2, 3)
        k = np.log(np.real(self.char_func(-1j * s, T)))
        c1 = (k[3] - k[1]) / (2 * h)
        c2 = (k[3] - 2 * k[2] + k[1]) / h**2
        c4 = (k[4] - 4 * k[3] + 6 * k[2] - 4 * k[1] + k[0]) / h**4
        return float(c1), float(c2), float(c4)

    def price(
        self,
        S: float,
        K: ArrayLike,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
        method: str = "cos",
    ) -> np.ndarray | float:
        """European option prices under the model.

        Parameters
        ----------
        S : float
            Spot.
        K : array_like
            Strikes.
        T, r, q : float
            Maturity, risk-free rate and dividend yield.
        option_type : {"call", "put"}
        method : {"cos", "lewis", "heston", "fft"}
            :func:`~financialkit.stochastic_volatility.systems.fourier.cos_price`,
            :func:`~financialkit.stochastic_volatility.systems.fourier.lewis_price`,
            :func:`~financialkit.stochastic_volatility.systems.heston.heston_price` or
            :func:`~financialkit.stochastic_volatility.systems.fourier.carr_madan_price`.

        Returns
        -------
        float or ndarray
        """
        from financialkit.stochastic_volatility.systems.fourier import carr_madan_price, cos_price, lewis_price
        from financialkit.stochastic_volatility.systems.heston import heston_price

        pricers = {"cos": cos_price, "lewis": lewis_price, "heston": heston_price, "fft": carr_madan_price}
        if method not in pricers:
            raise ValueError(f"method must be one of {sorted(pricers)}, got {method!r}")
        return pricers[method](self, S, K, T, r, q, option_type)

    def implied_vol(self, S: float, K: ArrayLike, T: float, r: float, q: float = 0.0, method: str = "cos") -> np.ndarray | float:
        """Black-Scholes implied volatilities of the model's prices.

        Out-of-the-money options are inverted (puts below the forward,
        calls above), where the time value is the whole price.

        Parameters
        ----------
        S : float
        K : array_like
        T, r, q : float
        method : str
            As in :meth:`price`.

        Returns
        -------
        float or ndarray
        """
        from financialkit.options import implied_volatility

        K = np.asarray(K, dtype=np.float64)
        k = np.atleast_1d(K)
        calls = np.atleast_1d(np.asarray(self.price(S, k, T, r, q, "call", method), dtype=np.float64))
        puts = calls - S * np.exp(-q * T) + k * np.exp(-r * T)
        otm_call = k >= S * np.exp((r - q) * T)
        out = np.empty(k.shape)
        if np.any(otm_call):
            out[otm_call] = implied_volatility(calls[otm_call], S, k[otm_call], T, r, q, "call")
        if np.any(~otm_call):
            out[~otm_call] = implied_volatility(puts[~otm_call], S, k[~otm_call], T, r, q, "put")
        return float(out[0]) if K.ndim == 0 else out.reshape(K.shape)


@dataclass
class FFTResult:
    """Call prices on the Carr-Madan log-strike grid."""

    strikes: np.ndarray
    """ndarray, shape (N,): Strikes :math:`e^{k_j}` on the FFT grid."""

    calls: np.ndarray
    """ndarray, shape (N,): European call prices at those strikes."""

    alpha: float
    """Damping exponent of the modified call price :math:`e^{\\alpha k} C(k)`."""

    eta: float
    """Spacing of the integration grid; the log-strike spacing is :math:`2\\pi / (N \\eta)`."""


@dataclass
class LocalVolSurface:
    """A local volatility :math:`\\sigma(K, T)` on a strike-maturity grid."""

    strikes: np.ndarray
    """ndarray, shape (n_K,): Increasing strikes."""

    maturities: np.ndarray
    """ndarray, shape (n_T,): Increasing maturities."""

    vol: np.ndarray
    """ndarray, shape (n_T, n_K): Local volatility at each node."""

    def __call__(self, S: ArrayLike, t: ArrayLike) -> np.ndarray:
        """Bilinear interpolation, held flat outside the grid.

        Parameters
        ----------
        S, t : array_like
            Spot levels and times, broadcast together.

        Returns
        -------
        ndarray
        """
        from scipy.interpolate import RegularGridInterpolator

        S, t = np.broadcast_arrays(np.asarray(S, dtype=np.float64), np.asarray(t, dtype=np.float64))
        interp = RegularGridInterpolator((self.maturities, self.strikes), self.vol)
        pts = np.stack([np.clip(t, self.maturities[0], self.maturities[-1]), np.clip(S, self.strikes[0], self.strikes[-1])], axis=-1)
        return interp(pts)


@dataclass
class HestonPaths:
    """Simulated spot and variance paths under the Heston model."""

    times: np.ndarray
    """ndarray, shape (n_steps + 1,): The time grid."""

    spot: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Spot paths."""

    variance: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Variance paths (negative values
    can appear under the truncation schemes, which only use their positive part)."""

    scheme: str
    """The discretisation: ``"full_truncation"``, ``"partial_truncation"``,
    ``"higham_mao"`` or ``"qe"``."""


@dataclass
class SABRFit:
    """SABR parameters fitted to a smile at one expiry."""

    alpha: float
    """Initial volatility level :math:`\\alpha`."""

    beta: float
    """CEV exponent :math:`\\beta` (held fixed in the fit)."""

    rho: float
    """Correlation between the forward and the volatility."""

    nu: float
    """Volatility of volatility :math:`\\nu`."""

    F: float
    """Forward."""

    T: float
    """Expiry in years."""

    rmse: float
    """Root-mean-square implied volatility error of the fit."""

    def implied_vol(self, K: ArrayLike) -> np.ndarray | float:
        """Hagan's implied volatility at strikes ``K`` with the fitted parameters."""
        from financialkit.stochastic_volatility.systems.smile import sabr_implied_vol

        return sabr_implied_vol(self.F, K, self.T, self.alpha, self.beta, self.rho, self.nu)


@dataclass
class SVIFit:
    """Raw SVI parameters fitted to total implied variance at one expiry."""

    a: float
    """Vertical level of the total variance."""

    b: float
    """Slope of the wings."""

    rho: float
    """Asymmetry, in :math:`(-1, 1)`."""

    m: float
    """Horizontal shift."""

    sigma: float
    """Smoothness at the vertex."""

    T: float
    """Expiry in years."""

    rmse: float
    """Root-mean-square total variance error of the fit."""

    def total_variance(self, k: ArrayLike) -> np.ndarray | float:
        """Total implied variance :math:`w(k)` at log-moneyness ``k``."""
        from financialkit.stochastic_volatility.systems.smile import svi_total_variance

        return svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)

    def implied_vol(self, k: ArrayLike) -> np.ndarray | float:
        """Implied volatility :math:`\\sqrt{w(k) / T}` at log-moneyness ``k``."""
        w = np.asarray(self.total_variance(k))
        out = np.sqrt(w / self.T)
        return float(out) if out.ndim == 0 else out


@dataclass
class FourierCalibration:
    """A characteristic-function model fitted to option quotes across strikes and maturities."""

    model: CharacteristicFunctionModel
    """The fitted model, ready to price."""

    params: dict[str, float]
    """The fitted parameters by constructor name (fixed ones included)."""

    strikes: np.ndarray
    """ndarray, shape (n,): Strikes of the quotes."""

    maturities: np.ndarray
    """ndarray, shape (n,): Maturity of each quote."""

    market_vols: np.ndarray
    """ndarray, shape (n,): Black-Scholes implied volatilities of the quotes."""

    model_vols: np.ndarray
    """ndarray, shape (n,): Black-Scholes implied volatilities of the fitted model."""

    residuals: np.ndarray
    """ndarray, shape (n,): Vega-weighted price errors (model minus market over
    vega) at the solution, approximately implied-volatility errors in decimals."""

    rmse: float
    """Root-mean-square implied volatility error in vol points (1 point = 0.01)."""

    converged: bool
    """Whether :func:`scipy.optimize.least_squares` reported convergence."""
    success = deprecated_field("converged")

    nfev: int
    """Number of residual evaluations."""


@dataclass
class RoughBergomiPaths:
    """Simulated spot and variance paths under the rough Bergomi model."""

    times: np.ndarray
    """ndarray, shape (n_steps + 1,): The time grid."""

    spot: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Spot paths."""

    variance: np.ndarray
    """ndarray, shape (n_paths, n_steps + 1): Instantaneous variance paths."""

    scheme: str
    """The Volterra scheme: ``"hybrid"``, ``"cholesky"`` or ``"riemann"``."""


@dataclass
class RoughnessEstimate:
    """Scaling of log-volatility increments, :math:`m(q, \\Delta) \\propto \\Delta^{\\zeta_q}`."""

    lags: np.ndarray
    """ndarray, shape (n_lags,): Lags :math:`\\Delta`, in observations."""

    q: np.ndarray
    """ndarray, shape (n_q,): Moments."""

    moments: np.ndarray
    """ndarray, shape (n_q, n_lags): Empirical :math:`m(q, \\Delta) = \\overline{|x_{t+\\Delta} - x_t|^q}`."""

    zeta: np.ndarray
    """ndarray, shape (n_q,): Fitted log-log slopes :math:`\\zeta_q`."""

    H: float
    """Hurst exponent: the slope of :math:`\\zeta_q` against :math:`q` through the origin."""
