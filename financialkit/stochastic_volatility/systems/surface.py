"""An implied-volatility surface from SVI slices, checked for arbitrage.

A surface quoted at a few expiries is fitted one expiry at a time by
Gatheral's raw SVI (:func:`~financialkit.stochastic_volatility.systems.smile.svi_fit`)
in log-forward-moneyness :math:`k = \\ln(K/F_T)`, and the slices are joined
linearly in total implied variance :math:`w(k, T) = \\sigma_{BS}^2 T` at
fixed :math:`k` (:class:`SVISurface`, :func:`svi_surface`).

Gatheral and Jacquier (2014) split static arbitrage in two. A slice is
free of butterfly arbitrage when its implied density is non-negative,
that is when Durrleman's function
:func:`~financialkit.stochastic_volatility.systems.smile.durrleman_g` is,
and its wings respect Lee's bound :math:`b(1 + |\\rho|) \\le 2`. The
surface is free of calendar-spread arbitrage when :math:`w(k, T)` does not
decrease in :math:`T` at any :math:`k`. Linear interpolation in :math:`w`
keeps the second property between slices that have it.

Prices on a strike-maturity grid then feed Dupire's formula
(:func:`~financialkit.stochastic_volatility.systems.local_vol.dupire_local_vol`),
so a local volatility can be built from implied volatilities
(:meth:`SVISurface.local_vol`).

Gatheral and Jacquier's surface SVI (SSVI) parametrises the whole surface
at once by the at-the-money total variance :math:`\\theta_T` and a
curvature function :math:`\\varphi(\\theta)`,

.. math::

   w(k, \\theta_T) = \\frac{\\theta_T}{2}\\left[1 + \\rho\\varphi(\\theta_T) k
   + \\sqrt{(\\varphi(\\theta_T) k + \\rho)^2 + 1 - \\rho^2}\\right],

and gives conditions on :math:`\\varphi` under which it is free of static
arbitrage (Theorems 4.1 and 4.2, :meth:`SSVISurface.is_arbitrage_free`).
With the power law :math:`\\varphi = \\eta\\,\\theta^{-\\gamma}(1 + \\theta)^{\\gamma - 1}`,
:math:`\\gamma \\le 1/2` and :math:`\\eta(1 + |\\rho|) \\le 2` are enough, which
:func:`ssvi_surface` builds into its parameters, so its fits are
arbitrage-free by construction. Each SSVI slice is a raw SVI slice
(:meth:`SSVISurface.to_svi`).

References: J. Gatheral, *The Volatility Surface* (Wiley, 2006),
Chapters 1 and 3; J. Gatheral and A. Jacquier, "Arbitrage-free SVI
volatility surfaces," *Quantitative Finance* 14 (2014) 59-71, Sections 4
and 5; R. W. Lee,
*Mathematical Finance* 14 (2004) 469-480; B. Dupire, *Risk* 7(1) (1994)
18-20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from financialkit.options import black_scholes_price
from financialkit.options.utils.common import as_output
from financialkit.stochastic_volatility.core.base import LocalVolSurface, SVIFit
from financialkit.stochastic_volatility.systems.local_vol import dupire_local_vol
from financialkit.stochastic_volatility.systems.smile import durrleman_g, svi_fit

__all__ = ["SVISurface", "svi_surface", "SSVISurface", "ssvi_surface", "SSVI_PHIS"]

_DEFAULT_K = np.linspace(-1.5, 1.5, 301)


class _ImpliedVarianceSurface:
    """Pricing shared by surfaces given by their total implied variance :math:`w(k, T)`.

    Subclasses hold ``S``, ``r`` and ``q`` and implement ``total_variance``.
    """

    S: float
    r: float
    q: float

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> np.ndarray | float:
        raise NotImplementedError

    def forward(self, T: ArrayLike) -> np.ndarray | float:
        """Forward price :math:`S e^{(r - q)T}`."""
        return as_output(self.S * np.exp((self.r - self.q) * np.asarray(T, dtype=np.float64)))

    def implied_vol(self, K: ArrayLike, T: ArrayLike) -> np.ndarray | float:
        """Black-Scholes implied volatility at strike ``K`` and expiry ``T`` (broadcast)."""
        K = np.asarray(K, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        k = np.log(K / np.asarray(self.forward(T)))
        return as_output(np.sqrt(np.asarray(self.total_variance(k, T)) / T))

    def price(self, K: ArrayLike, T: ArrayLike, option_type: str = "call") -> np.ndarray | float:
        """Black-Scholes price at the surface's implied volatility."""
        return black_scholes_price(self.S, K, T, self.r, self.implied_vol(K, T), self.q, option_type)

    def local_vol(self, strikes: ArrayLike, maturities: ArrayLike) -> LocalVolSurface:
        """Dupire local volatility from the surface's call prices on a grid.

        The calls are priced at the surface's implied volatilities and
        passed to
        :func:`~financialkit.stochastic_volatility.systems.local_vol.dupire_local_vol`.

        Parameters
        ----------
        strikes : array_like, shape (n_K,)
            Increasing strikes.
        maturities : array_like, shape (n_T,)
            Increasing positive maturities (at least two).

        Returns
        -------
        LocalVolSurface
        """
        K = np.asarray(strikes, dtype=np.float64)
        T = np.asarray(maturities, dtype=np.float64)
        return dupire_local_vol(K, T, np.asarray(self.price(K[None, :], T[:, None])), self.r, self.q)


@dataclass
class SVISurface(_ImpliedVarianceSurface):
    """An implied-volatility surface made of raw SVI slices.

    Each slice is in log-forward-moneyness :math:`k = \\ln(K/F_T)` with
    :math:`F_T = S e^{(r - q)T}`. Between expiries the total variance is
    linear in :math:`T` at fixed :math:`k`; before the first expiry and
    after the last the implied volatility of the nearest slice is held
    flat, so :math:`w` is proportional to :math:`T` there.

    Examples
    --------
    >>> s = SVISurface([SVIFit(0.04, 0.0, 0.0, 0.0, 0.1, 1.0, 0.0)], S=100.0, r=0.0)
    >>> round(s.implied_vol(120.0, 0.5), 12)
    0.2
    """

    slices: list[SVIFit]
    """Fitted slices, one per expiry, sorted by :attr:`SVIFit.T`."""

    S: float
    """Spot price."""

    r: float
    """Risk-free rate."""

    q: float = 0.0
    """Dividend yield."""

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("an SVI surface needs at least one slice")
        self.slices = sorted(self.slices, key=lambda s: s.T)
        T = self.maturities
        if T[0] <= 0 or np.any(np.diff(T) <= 0):
            raise ValueError("slice expiries must be positive and distinct")

    @property
    def maturities(self) -> np.ndarray:
        """ndarray, shape (n_slices,): The slice expiries."""
        return np.array([s.T for s in self.slices])

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> np.ndarray | float:
        """Total implied variance :math:`w(k, T)`, ``k`` and ``T`` broadcast together.

        Parameters
        ----------
        k : array_like
            Log-forward-moneyness :math:`\\ln(K/F_T)`.
        T : array_like
            Expiries, positive.

        Returns
        -------
        float or ndarray
        """
        k, T = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64))
        Tn = self.maturities
        W = np.stack([np.broadcast_to(s.total_variance(k), k.shape) for s in self.slices])
        if Tn.size == 1:
            return as_output(W[0] * T / Tn[0])
        i = np.clip(np.searchsorted(Tn, T), 1, Tn.size - 1)
        lam = (T - Tn[i - 1]) / (Tn[i] - Tn[i - 1])
        w0 = np.take_along_axis(W, (i - 1)[None], axis=0)[0]
        w1 = np.take_along_axis(W, i[None], axis=0)[0]
        w = (1.0 - lam) * w0 + lam * w1
        w = np.where(T < Tn[0], W[0] * T / Tn[0], w)
        w = np.where(T > Tn[-1], W[-1] * T / Tn[-1], w)
        return as_output(w)

    def butterfly_arbitrage(self, k: ArrayLike | None = None) -> np.ndarray:
        """Durrleman's :math:`g(k)` for every slice.

        A slice is free of butterfly arbitrage when :math:`g \\ge 0` and its
        wings satisfy Lee's bound (:meth:`is_arbitrage_free` checks both).
        Nodes with :math:`w \\le 0` are set to ``-inf``.

        Parameters
        ----------
        k : array_like, shape (n_k,), optional
            Log-forward-moneyness grid; default 301 points on ``[-1.5, 1.5]``.

        Returns
        -------
        ndarray, shape (n_slices, n_k)
        """
        k = _DEFAULT_K if k is None else np.asarray(k, dtype=np.float64)
        g = np.stack([np.atleast_1d(durrleman_g(k, s.a, s.b, s.rho, s.m, s.sigma)) for s in self.slices])
        w = np.stack([np.atleast_1d(s.total_variance(k)) for s in self.slices])
        return np.where(w > 0, g, -np.inf)

    def calendar_arbitrage(self, k: ArrayLike | None = None) -> np.ndarray:
        """Increments :math:`w(k, T_{i+1}) - w(k, T_i)` between consecutive slices.

        The surface is free of calendar-spread arbitrage when every
        increment is non-negative (Gatheral and Jacquier 2014, Lemma 2.1).

        Parameters
        ----------
        k : array_like, shape (n_k,), optional
            Log-forward-moneyness grid; default 301 points on ``[-1.5, 1.5]``.

        Returns
        -------
        ndarray, shape (n_slices - 1, n_k)
        """
        k = _DEFAULT_K if k is None else np.asarray(k, dtype=np.float64)
        w = np.stack([np.atleast_1d(s.total_variance(k)) for s in self.slices])
        return np.diff(w, axis=0)

    def is_arbitrage_free(self, k: ArrayLike | None = None, tol: float = 0.0) -> bool:
        """Whether every slice passes the butterfly test and the slices the calendar test.

        Parameters
        ----------
        k : array_like, optional
            Log-forward-moneyness grid for :meth:`butterfly_arbitrage` and
            :meth:`calendar_arbitrage`.
        tol : float
            Violations no larger than ``tol`` are accepted.

        Returns
        -------
        bool
        """
        lee = all(s.b * (1.0 + abs(s.rho)) <= 2.0 + tol for s in self.slices)
        return bool(lee and np.all(self.butterfly_arbitrage(k) >= -tol) and np.all(self.calendar_arbitrage(k) >= -tol))


def svi_surface(strikes: ArrayLike, maturities: ArrayLike, vols: ArrayLike, S: float, r: float, q: float = 0.0) -> SVISurface:
    """Fit an :class:`SVISurface` to implied-volatility quotes, one SVI slice per expiry.

    Parameters
    ----------
    strikes, maturities, vols : array_like, shape (n,)
        One quote per entry: strike, expiry and Black-Scholes implied
        volatility. Every expiry needs at least five quotes.
    S, r, q : float
        Spot, risk-free rate and dividend yield, which set the forwards.

    Returns
    -------
    SVISurface

    Examples
    --------
    >>> import numpy as np
    >>> K, T = np.meshgrid(np.linspace(70, 140, 15), [0.5, 1.0])
    >>> vol = 0.2 - 0.1 * np.log(K / 100.0)
    >>> s = svi_surface(K.ravel(), T.ravel(), vol.ravel(), 100.0, 0.0)
    >>> bool(np.max(np.abs(s.implied_vol(K, T) - vol)) < 1e-3)
    True
    """
    K = np.asarray(strikes, dtype=np.float64).ravel()
    T = np.asarray(maturities, dtype=np.float64).ravel()
    v = np.asarray(vols, dtype=np.float64).ravel()
    if not K.size == T.size == v.size:
        raise ValueError("strikes, maturities and vols must have the same length")
    slices = []
    for t in np.unique(T):
        sel = T == t
        if sel.sum() < 5:
            raise ValueError(f"expiry {t} has {sel.sum()} quotes; raw SVI needs at least five")
        k = np.log(K[sel] / (S * np.exp((r - q) * t)))
        slices.append(svi_fit(k, v[sel] ** 2 * t, float(t)))
    return SVISurface(slices, float(S), float(r), float(q))


#: Curvature functions accepted by :class:`SSVISurface`.
SSVI_PHIS = ("power_law", "heston")


@dataclass
class SSVISurface(_ImpliedVarianceSurface):
    """Gatheral and Jacquier's (2014) surface SVI.

    The at-the-money total variance :math:`\\theta_T` is linear in
    :math:`T` between the given expiries, starts at 0 and is extended past
    the last expiry at constant implied volatility. The curvature is the
    power law :math:`\\varphi(\\theta) = \\eta\\,\\theta^{-\\gamma}(1 + \\theta)^{\\gamma - 1}`
    or the Heston-like :math:`\\varphi(\\theta) = \\frac{1}{\\lambda\\theta}
    \\left(1 - \\frac{1 - e^{-\\lambda\\theta}}{\\lambda\\theta}\\right)`.

    Examples
    --------
    >>> s = SSVISurface(-0.4, [1.0], [0.04], S=100.0, r=0.0, eta=1.0, gamma=0.5)
    >>> round(s.implied_vol(100.0, 1.0), 12)
    0.2
    >>> s.is_arbitrage_free()
    True
    """

    rho: float
    """Correlation-like skew parameter, in (-1, 1)."""

    expiries: np.ndarray
    """ndarray, shape (n,): Increasing positive expiries of :attr:`theta`."""

    theta: np.ndarray
    """ndarray, shape (n,): At-the-money total variance :math:`\\theta_T = \\sigma_{ATM}^2 T` at :attr:`expiries`."""

    S: float
    """Spot price."""

    r: float
    """Risk-free rate."""

    q: float = 0.0
    """Dividend yield."""

    phi: str = "power_law"
    """Curvature function, ``"power_law"`` or ``"heston"``."""

    eta: float = 1.0
    """Power-law level :math:`\\eta > 0`."""

    gamma: float = 0.5
    """Power-law exponent :math:`\\gamma \\in (0, 1)`."""

    lam: float = 1.0
    """Heston-like :math:`\\lambda > 0`."""

    rmse: float = float("nan")
    """Root-mean-square implied volatility error of the fit (``nan`` if not fitted)."""

    def __post_init__(self) -> None:
        self.expiries = np.atleast_1d(np.asarray(self.expiries, dtype=np.float64))
        self.theta = np.atleast_1d(np.asarray(self.theta, dtype=np.float64))
        if self.expiries.shape != self.theta.shape or self.expiries.ndim != 1:
            raise ValueError("expiries and theta must be 1-D and the same length")
        if self.expiries[0] <= 0 or np.any(np.diff(self.expiries) <= 0):
            raise ValueError("expiries must be positive and increasing")
        if np.any(self.theta <= 0):
            raise ValueError("theta must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho}")
        if self.phi not in SSVI_PHIS:
            raise ValueError(f"phi must be one of {SSVI_PHIS}, got {self.phi!r}")
        if self.eta <= 0 or not 0.0 < self.gamma < 1.0 or self.lam <= 0:
            raise ValueError("need eta > 0, 0 < gamma < 1 and lam > 0")

    def atm_variance(self, T: ArrayLike) -> np.ndarray | float:
        """At-the-money total variance :math:`\\theta_T`."""
        T = np.asarray(T, dtype=np.float64)
        th = np.interp(T, np.concatenate([[0.0], self.expiries]), np.concatenate([[0.0], self.theta]))
        return as_output(np.where(T > self.expiries[-1], self.theta[-1] * T / self.expiries[-1], th))

    def curvature(self, theta: ArrayLike) -> np.ndarray | float:
        """The curvature function :math:`\\varphi(\\theta)`."""
        th = np.asarray(theta, dtype=np.float64)
        if self.phi == "power_law":
            return as_output(self.eta * th**-self.gamma * (1.0 + th) ** (self.gamma - 1.0))
        x = self.lam * th
        return as_output((1.0 - (1.0 - np.exp(-x)) / x) / x)

    def _theta_phi_derivative(self, th: np.ndarray) -> np.ndarray:
        """:math:`\\partial_\\theta(\\theta\\varphi(\\theta))`, in closed form."""
        if self.phi == "power_law":
            return self.eta * (1.0 - self.gamma) * th**-self.gamma * (1.0 + th) ** (self.gamma - 2.0)
        x = self.lam * th
        return (1.0 - np.exp(-x) - x * np.exp(-x)) / x**2

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> np.ndarray | float:
        """Total implied variance :math:`w(k, \\theta_T)`, ``k`` and ``T`` broadcast together.

        Parameters
        ----------
        k : array_like
            Log-forward-moneyness :math:`\\ln(K/F_T)`.
        T : array_like
            Expiries, positive.

        Returns
        -------
        float or ndarray
        """
        k, T = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64))
        th = np.asarray(self.atm_variance(T))
        p = np.asarray(self.curvature(th))
        rho = self.rho
        return as_output(0.5 * th * (1.0 + rho * p * k + np.sqrt((p * k + rho) ** 2 + 1.0 - rho**2)))

    def to_svi(self, T: float) -> SVIFit:
        """The raw SVI slice at expiry ``T`` (Gatheral and Jacquier 2014, Section 3).

        :math:`a = \\tfrac12\\theta(1 - \\rho^2)`, :math:`b = \\tfrac12\\theta\\varphi`,
        :math:`m = -\\rho/\\varphi` and :math:`\\sigma = \\sqrt{1 - \\rho^2}/\\varphi`.

        Parameters
        ----------
        T : float

        Returns
        -------
        SVIFit
            With ``rmse`` set to 0.
        """
        th = float(self.atm_variance(T))
        p = float(self.curvature(th))
        rho = self.rho
        return SVIFit(0.5 * th * (1 - rho**2), 0.5 * th * p, rho, -rho / p, np.sqrt(1 - rho**2) / p, float(T), 0.0)

    def to_svi_surface(self) -> SVISurface:
        """An :class:`SVISurface` made of the raw SVI slices at :attr:`expiries`."""
        return SVISurface([self.to_svi(t) for t in self.expiries], self.S, self.r, self.q)

    def arbitrage_conditions(self, theta: ArrayLike | None = None) -> dict[str, np.ndarray]:
        """Gatheral and Jacquier's sufficient conditions, as margins that are non-negative when met.

        * ``"calendar"`` (Theorem 4.1): :math:`0 \\le \\partial_\\theta(\\theta\\varphi)
          \\le \\frac{\\varphi}{\\rho^2}\\left(1 + \\sqrt{1 - \\rho^2}\\right)` (the smaller
          of the two margins), together with :math:`\\theta_T` non-decreasing;
        * ``"butterfly"`` (Theorem 4.2): :math:`\\theta\\varphi(1 + |\\rho|) < 4` and
          :math:`\\theta\\varphi^2(1 + |\\rho|) \\le 4` (the smaller margin).

        Parameters
        ----------
        theta : array_like, optional
            :math:`\\theta` values to check; by default 200 points up to
            :math:`\\max\\theta_T` and the given ones.

        Returns
        -------
        dict
            ``"theta"``, ``"calendar"`` and ``"butterfly"``, each of shape
            (n,), and ``"monotone"``, the increments of :attr:`theta`.
        """
        th = np.sort(np.concatenate([np.geomspace(1e-4, self.theta.max(), 200), self.theta])) if theta is None else np.asarray(theta, dtype=np.float64)
        p = np.asarray(self.curvature(th))
        d = self._theta_phi_derivative(th)
        rho2 = self.rho**2
        upper = np.inf if rho2 == 0 else p / rho2 * (1.0 + np.sqrt(1.0 - rho2)) - d
        cal = np.minimum(d, upper)
        a = 1.0 + abs(self.rho)
        bfly = np.minimum(4.0 - th * p * a, 4.0 - th * p**2 * a)
        return {"theta": th, "calendar": cal, "butterfly": bfly, "monotone": np.diff(np.concatenate([[0.0], self.theta]))}

    def is_arbitrage_free(self, tol: float = 0.0) -> bool:
        """Whether :meth:`arbitrage_conditions` all hold, to within ``tol``."""
        c = self.arbitrage_conditions()
        return bool(np.all(c["calendar"] >= -tol) and np.all(c["butterfly"] >= -tol) and np.all(c["monotone"] >= -tol))


def ssvi_surface(
    strikes: ArrayLike,
    maturities: ArrayLike,
    vols: ArrayLike,
    S: float,
    r: float,
    q: float = 0.0,
    phi: str = "power_law",
) -> SSVISurface:
    """Fit an :class:`SSVISurface` to implied-volatility quotes by :func:`scipy.optimize.least_squares`.

    The parameters are one :math:`\\theta_T` per expiry, :math:`\\rho` and the
    curvature. They are mapped so that the fit is free of static arbitrage
    by construction (Gatheral and Jacquier 2014, Section 4): :math:`\\theta_T` rises through non-negative increments,
    :math:`\\gamma \\in (0, 1/2]` and :math:`\\eta = 2c/(1 + |\\rho|)` with
    :math:`c \\in (0, 1]` for the power law, and
    :math:`\\lambda \\ge (1 + |\\rho|)/4` for the Heston-like curvature.

    Parameters
    ----------
    strikes, maturities, vols : array_like, shape (n,)
        One quote per entry: strike, expiry and Black-Scholes implied
        volatility. Every expiry needs at least two quotes.
    S, r, q : float
        Spot, risk-free rate and dividend yield, which set the forwards.
    phi : {"power_law", "heston"}

    Returns
    -------
    SSVISurface

    Examples
    --------
    >>> import numpy as np
    >>> true = SSVISurface(-0.6, [0.5, 1.0], [0.02, 0.045], 100.0, 0.0, eta=1.2, gamma=0.4)
    >>> K, T = np.meshgrid(np.linspace(70, 140, 15), [0.5, 1.0])
    >>> fit = ssvi_surface(K.ravel(), T.ravel(), true.implied_vol(K, T).ravel(), 100.0, 0.0)
    >>> round(fit.rho, 4), round(fit.eta, 4), round(fit.gamma, 4)
    (-0.6, 1.2, 0.4)
    """
    K = np.asarray(strikes, dtype=np.float64).ravel()
    T = np.asarray(maturities, dtype=np.float64).ravel()
    v = np.asarray(vols, dtype=np.float64).ravel()
    if not K.size == T.size == v.size:
        raise ValueError("strikes, maturities and vols must have the same length")
    if phi not in SSVI_PHIS:
        raise ValueError(f"phi must be one of {SSVI_PHIS}, got {phi!r}")
    expiries = np.unique(T)
    k = np.log(K / (S * np.exp((r - q) * T)))
    atm = []
    for t in expiries:
        sel = T == t
        if sel.sum() < 2:
            raise ValueError(f"expiry {t} has {sel.sum()} quotes; SSVI needs at least two per expiry")
        o = np.argsort(k[sel])
        atm.append(float(np.interp(0.0, k[sel][o], (v[sel] ** 2 * t)[o])))
    th0 = np.maximum.accumulate(np.maximum(atm, 1e-6))
    n = expiries.size

    def build(p: np.ndarray) -> SSVISurface:
        theta = np.cumsum(p[:n])
        rho = float(p[n])
        a = 1.0 + abs(rho)
        if phi == "power_law":
            return SSVISurface(rho, expiries, theta, S, r, q, phi, eta=float(2.0 * p[n + 1] / a), gamma=float(p[n + 2]))
        return SSVISurface(rho, expiries, theta, S, r, q, phi, lam=a / 4.0 + float(p[n + 1]))

    def resid(p: np.ndarray) -> np.ndarray:
        return np.asarray(build(p).implied_vol(K, T)) - v

    inc0 = np.diff(np.concatenate([[0.0], th0]))
    lo_th, hi_th = [1e-8] + [0.0] * (n - 1), [np.inf] * n
    if phi == "power_law":
        x0, lo, hi = [*inc0, -0.3, 0.5, 0.3], [*lo_th, -0.999, 1e-6, 1e-3], [*hi_th, 0.999, 1.0, 0.5]
    else:
        x0, lo, hi = [*inc0, -0.3, 1.0], [*lo_th, -0.999, 0.0], [*hi_th, 0.999, np.inf]
    x0 = np.clip(x0, lo, hi)
    sol = least_squares(resid, x0, bounds=(lo, hi), xtol=1e-15, ftol=1e-15, gtol=1e-15)
    fit = build(sol.x)
    fit.rmse = float(np.sqrt(np.mean(sol.fun**2)))
    return fit
