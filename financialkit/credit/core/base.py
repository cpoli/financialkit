"""Abstract base classes and result containers for financialkit.credit.

Conventions, package-wide in this domain: time is in years, the
risk-free rate ``r`` is a flat continuously compounded rate, ``recovery``
is the fraction of face value recovered on default (so the loss given
default is ``1 - recovery``), and default probabilities, hazard rates and
spreads are decimals (``0.01`` is 100 basis points).

:class:`SurvivalCurve` is the interface every reduced-form model and the
CDS pricers work with: a concrete curve supplies the cumulative hazard
:math:`\\Lambda(t) = \\int_0^t \\lambda(s)\\,ds`, and survival, default
probabilities and default-time sampling follow from it. Models with
several outputs return a small dataclass, never a bare tuple, so the
visualizers have a stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from financialkit.integrators import RNGLike
from financialkit.options.utils.common import as_output

__all__ = [
    "SurvivalCurve",
    "MertonResult",
    "KMVResult",
    "LelandResult",
    "CDSResult",
    "CreditMetricsResult",
    "LossDistribution",
    "TrancheResult",
]


class SurvivalCurve(ABC):
    """A default-time distribution given by its cumulative hazard :math:`\\Lambda(t)`.

    Survival is :math:`Q(t) = e^{-\\Lambda(t)}`; Lando (1998) calls this the
    Cox-process view of default, with the intensity deterministic here.
    """

    @abstractmethod
    def cumulative_hazard(self, t: ArrayLike) -> np.ndarray | float:
        """Integrated hazard :math:`\\Lambda(t)`, non-decreasing with :math:`\\Lambda(0) = 0`."""

    @abstractmethod
    def hazard(self, t: ArrayLike) -> np.ndarray | float:
        """Instantaneous default intensity :math:`\\lambda(t)`."""

    @abstractmethod
    def inverse_cumulative_hazard(self, x: ArrayLike) -> np.ndarray | float:
        """The time at which :math:`\\Lambda` reaches ``x``."""

    def survival(self, t: ArrayLike) -> np.ndarray | float:
        """Survival probability :math:`Q(t) = P(\\tau > t)`."""
        return as_output(np.exp(-np.asarray(self.cumulative_hazard(t), dtype=np.float64)))

    def default_probability(self, t: ArrayLike, s: ArrayLike | None = None) -> np.ndarray | float:
        """:math:`P(\\tau \\le t)`, or :math:`P(t < \\tau \\le s)` when ``s`` is given."""
        if s is None:
            return as_output(-np.expm1(-np.asarray(self.cumulative_hazard(t), dtype=np.float64)))
        return as_output(np.asarray(self.survival(t)) - np.asarray(self.survival(s)))

    def default_density(self, t: ArrayLike) -> np.ndarray | float:
        """Density of the default time, :math:`\\lambda(t) Q(t)`."""
        return as_output(np.asarray(self.hazard(t)) * np.asarray(self.survival(t)))

    def sample_default_times(self, n: int, rng: RNGLike = None, uniforms: ArrayLike | None = None) -> np.ndarray:
        """Default times by inversion: :math:`\\tau = \\Lambda^{-1}(-\\ln U)`.

        Parameters
        ----------
        n : int
            Number of draws (ignored when ``uniforms`` is given).
        rng : numpy.random.Generator, int or None
        uniforms : array_like, optional
            Uniforms to invert, e.g. a copula sample; ``inf`` marks no default.

        Returns
        -------
        ndarray
        """
        u = np.random.default_rng(rng).random(n) if uniforms is None else np.asarray(uniforms, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.asarray(self.inverse_cumulative_hazard(-np.log(u)), dtype=np.float64)


@dataclass
class MertonResult:
    """Merton's (1974) firm: equity as a call on the assets, debt as the rest."""

    equity: float
    """Market value of equity, a Black-Scholes call on the assets struck at the face value."""

    debt: float
    """Market value of the zero-coupon debt, ``V - equity``."""

    asset_value: float
    """Firm asset value ``V``."""

    asset_volatility: float
    """Asset volatility ``sigma_V``."""

    equity_volatility: float
    """Instantaneous equity volatility, :math:`N(d_1)\\sigma_V V / E`."""

    default_probability: float
    """Risk-neutral probability of default at maturity, :math:`N(-d_2)`."""

    distance_to_default: float
    """:math:`d_2`, the number of standard deviations between ``V`` and the default point."""

    credit_spread: float
    """Yield on the debt minus ``r``, continuously compounded."""

    recovery_rate: float
    """Risk-neutral expected recovery as a fraction of face value, given default."""


@dataclass
class KMVResult:
    """Crosbie and Bohn's (2003) iterative estimate of the asset path and distance to default."""

    asset_values: np.ndarray
    """ndarray, shape (n,): Implied asset values, one per equity observation."""

    asset_volatility: float
    """Annualised volatility of the implied asset log returns."""

    drift: float
    """Annualised drift ``mu`` of the asset values."""

    distance_to_default: float
    """Physical distance to default at the last observation."""

    expected_default_frequency: float
    """:math:`N(-DD)`, the normal-model EDF (Moody's KMV maps DD to EDF empirically)."""

    n_iter: int
    """Iterations until the asset volatility converged."""

    volatility_history: np.ndarray
    """ndarray: Asset volatility after each iteration, starting from the initial guess."""


@dataclass
class LelandResult:
    """Leland's (1994) perpetual-debt firm with endogenous default."""

    default_barrier: float
    """Asset value ``V_B`` at which shareholders choose to default."""

    debt: float
    """Market value of the perpetual debt."""

    equity: float
    """Market value of equity."""

    firm_value: float
    """Unlevered assets plus tax shields minus bankruptcy costs."""

    leverage: float
    """``debt / firm_value``."""

    credit_spread: float
    """``coupon / debt - r``."""

    default_probability_weight: float
    """:math:`p_B = (V/V_B)^{-2r/\\sigma^2}`, the price of a claim paying 1 at default."""


@dataclass
class CDSResult:
    """Legs of a credit default swap, per unit notional."""

    par_spread: float
    """Spread that sets the premium and protection legs equal."""

    protection_leg: float
    """Present value of the default payment ``(1 - R)``."""

    rpv01: float
    """Risky annuity: premium-leg value per unit of spread, accrued premium included."""

    accrued_on_default: float
    """Part of ``rpv01`` paid as premium accrued at default."""

    payment_times: np.ndarray
    """ndarray: Premium payment dates."""

    def mtm(self, spread: float) -> float:
        """Value to the protection buyer of a contract paying ``spread``."""
        return self.protection_leg - spread * self.rpv01


@dataclass
class CreditMetricsResult:
    """A CreditMetrics value distribution over rating states."""

    values: np.ndarray
    """ndarray: Portfolio values, one per scenario or rating state."""

    probabilities: np.ndarray
    """ndarray: Probability of each value."""

    @property
    def mean(self) -> float:
        """float: Expected value."""
        return float(self.probabilities @ self.values)

    @property
    def std(self) -> float:
        """float: Standard deviation of the value."""
        return float(np.sqrt(self.probabilities @ (self.values - self.mean) ** 2))

    def percentile(self, p: float) -> float:
        """Smallest value whose cumulative probability reaches ``p`` (e.g. ``0.01``)."""
        order = np.argsort(self.values)
        cum = np.cumsum(self.probabilities[order])
        i = min(int(np.searchsorted(cum, p - 1e-12)), cum.size - 1)
        return float(self.values[order][i])

    def credit_var(self, alpha: float = 0.99) -> float:
        """Mean value minus the ``1 - alpha`` percentile."""
        return self.mean - self.percentile(1.0 - alpha)


@dataclass
class LossDistribution:
    """A portfolio credit loss distribution on a grid of loss levels."""

    losses: np.ndarray
    """ndarray, shape (m,): Loss levels, increasing."""

    pmf: np.ndarray
    """ndarray, shape (m,): Probability of each loss level."""

    method: str
    """How the distribution was obtained, e.g. ``"creditrisk+"`` or ``"monte_carlo"``."""

    @property
    def mean(self) -> float:
        """float: Expected loss."""
        return float(self.pmf @ self.losses)

    @property
    def std(self) -> float:
        """float: Standard deviation of the loss."""
        return float(np.sqrt(self.pmf @ (self.losses - self.mean) ** 2))

    def quantile(self, alpha: float) -> float:
        """Loss quantile (credit VaR before subtracting the mean)."""
        return float(self.losses[min(int(np.searchsorted(np.cumsum(self.pmf), alpha - 1e-12)), self.losses.size - 1)])

    def unexpected_loss(self, alpha: float = 0.999) -> float:
        """Quantile minus expected loss, the economic-capital figure."""
        return self.quantile(alpha) - self.mean


@dataclass
class TrancheResult:
    """A synthetic CDO tranche priced in the one-factor Gaussian copula."""

    attachment: float
    """Lower loss point, a fraction of the portfolio notional."""

    detachment: float
    """Upper loss point."""

    par_spread: float
    """Running spread that prices the tranche at par."""

    expected_loss: np.ndarray
    """ndarray: Expected tranche loss (fraction of tranche notional) at each payment time."""

    times: np.ndarray
    """ndarray: Payment times."""

    protection_leg: float
    """Present value of tranche losses, per unit tranche notional."""

    rpv01: float
    """Premium-leg value per unit spread, per unit tranche notional."""
