"""Abstract base classes and result containers for financialkit.fixed_income.

Every term structure in this domain is a :class:`YieldCurve`: a function
from maturity (years) to a continuously compounded zero rate, from which
discount factors, Lutz forward rates and par yields follow by the same
formulas whatever the curve's construction. Concrete curves are an
interpolated :class:`ZeroCurve` (bootstrapping), a :class:`FlatCurve`, and
the parametric and spline curves in
:mod:`financialkit.fixed_income.systems.curves`. Bonds, swaps and
liabilities are :class:`CashFlows`, and every risk measure returns a small
dataclass rather than a bare tuple, so the visualizers have a stable
interface.

Functions that take a ``curve`` also accept a plain float, read as a flat
continuously compounded rate (see :func:`as_curve`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "YieldCurve",
    "FlatCurve",
    "ZeroCurve",
    "ShiftedCurve",
    "CurveLike",
    "as_curve",
    "coupon_times",
    "CashFlows",
    "BondRisk",
    "CurveFitResult",
    "CurvePCAResult",
    "KeyRateDurationResult",
    "ImmunizationResult",
    "SwapValuation",
]


class YieldCurve(ABC):
    """A term structure of continuously compounded zero rates.

    Subclasses implement :meth:`zero_rate`; discount factors, forward rates
    and par yields are derived here, so every curve answers them the same way.
    """

    @abstractmethod
    def zero_rate(self, t):
        """Continuously compounded zero rate for maturity `t` (years); broadcasts."""

    def discount(self, t):
        """Discount factor :math:`P(0, t) = e^{-r(t) t}`; broadcasts."""
        t = np.asarray(t, dtype=np.float64)
        return np.exp(-self.zero_rate(t) * t)

    def forward_rate(self, t1, t2):
        r"""Continuously compounded forward rate between `t1` and `t2` (Lutz, 1940).

        .. math:: f(t_1, t_2) = \frac{r(t_2) t_2 - r(t_1) t_1}{t_2 - t_1}
        """
        t1 = np.asarray(t1, dtype=np.float64)
        t2 = np.asarray(t2, dtype=np.float64)
        if np.any(t2 <= t1):
            raise ValueError("t2 must exceed t1")
        return (self.zero_rate(t2) * t2 - self.zero_rate(t1) * t1) / (t2 - t1)

    def instantaneous_forward(self, t, h: float = 1e-5):
        r"""Instantaneous forward rate :math:`f(t) = \partial_t [r(t) t]`, by central differences."""
        t = np.asarray(t, dtype=np.float64)
        lo = np.maximum(t - h, 0.0)
        hi = t + h
        return (self.zero_rate(hi) * hi - self.zero_rate(lo) * lo) / (hi - lo)

    def par_yield(self, maturity, freq: int = 2):
        r"""Coupon rate that prices a bond (or the fixed leg of a swap) at par.

        .. math:: c = m \, \frac{1 - P(0, T)}{\sum_i P(0, t_i)}

        with coupons at ``T, T - 1/m, ...`` down to the first date after 0.
        """
        maturity = np.atleast_1d(np.asarray(maturity, dtype=np.float64))
        out = np.empty_like(maturity)
        for k, T in enumerate(maturity):
            times = coupon_times(T, freq)
            out[k] = freq * (1.0 - self.discount(T)) / np.sum(self.discount(times))
        return out if out.size > 1 else float(out[0])

    def shifted(self, delta) -> ShiftedCurve:
        """This curve with `delta` added to every zero rate (a float, or a callable of `t`)."""
        return ShiftedCurve(self, delta)


def coupon_times(maturity: float, freq: int) -> np.ndarray:
    """Coupon dates ``maturity - k / freq > 0``, ascending; the first period may be short.

    Examples
    --------
    >>> coupon_times(1.25, 2).tolist()
    [0.25, 0.75, 1.25]
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if freq <= 0:
        raise ValueError("freq must be a positive integer")
    n = int(np.ceil(maturity * freq - 1e-9))
    times = maturity - np.arange(n)[::-1] / freq
    return times[times > 1e-12]


class FlatCurve(YieldCurve):
    """The same continuously compounded rate at every maturity."""

    def __init__(self, rate: float):
        self.rate = float(rate)

    def zero_rate(self, t):
        return np.full_like(np.asarray(t, dtype=np.float64), self.rate)

    def __repr__(self) -> str:
        return f"FlatCurve(rate={self.rate!r})"


class ZeroCurve(YieldCurve):
    """Zero rates at node maturities, linearly interpolated and flat beyond the ends.

    Linear interpolation in the zero rate is the convention of Hull's
    bootstrapping examples (*Options, Futures, and Other Derivatives*, 9th
    ed., Sec. 4.5), and is what
    :func:`~financialkit.fixed_income.systems.curves.bootstrap_zero_curve`
    solves under.

    Parameters
    ----------
    tenors : array_like, shape (n,)
        Strictly increasing positive maturities in years.
    rates : array_like, shape (n,)
        Continuously compounded zero rates at `tenors`.

    Examples
    --------
    >>> curve = ZeroCurve([1.0, 2.0], [0.03, 0.04])
    >>> round(float(curve.zero_rate(1.5)), 6)
    0.035
    >>> round(float(curve.forward_rate(1.0, 2.0)), 6)
    0.05
    """

    def __init__(self, tenors, rates):
        self.tenors = np.asarray(tenors, dtype=np.float64)
        self.rates = np.asarray(rates, dtype=np.float64)
        if self.tenors.ndim != 1 or self.tenors.shape != self.rates.shape or self.tenors.size == 0:
            raise ValueError("tenors and rates must be 1-D arrays of the same nonzero length")
        if np.any(self.tenors <= 0) or np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be positive and strictly increasing")

    def zero_rate(self, t):
        return np.interp(np.asarray(t, dtype=np.float64), self.tenors, self.rates)

    def __repr__(self) -> str:
        return f"ZeroCurve(tenors={self.tenors!r}, rates={self.rates!r})"


class ShiftedCurve(YieldCurve):
    """A base curve plus a spread on its zero rates: a parallel shift (float) or a shaped shock (callable)."""

    def __init__(self, base: YieldCurve, delta: float | Callable[[np.ndarray], np.ndarray]):
        self.base = base
        self.delta = delta

    def zero_rate(self, t):
        t = np.asarray(t, dtype=np.float64)
        d = self.delta(t) if callable(self.delta) else self.delta
        return self.base.zero_rate(t) + d


CurveLike = YieldCurve | float


def as_curve(curve: CurveLike) -> YieldCurve:
    """Return `curve` itself, or a :class:`FlatCurve` if it is a number."""
    if isinstance(curve, YieldCurve):
        return curve
    return FlatCurve(float(curve))


@dataclass
class CashFlows:
    """A stream of fixed amounts paid at times (years from today)."""

    times: np.ndarray
    """ndarray, shape (n,): Payment times, ascending."""

    amounts: np.ndarray
    """ndarray, shape (n,): Payment amounts."""

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.amounts = np.asarray(self.amounts, dtype=np.float64)
        if self.times.shape != self.amounts.shape:
            raise ValueError("times and amounts must have the same shape")

    def present_value(self, curve: CurveLike) -> float:
        """Sum of the amounts discounted on `curve`."""
        return float(np.sum(self.amounts * as_curve(curve).discount(self.times)))

    def __add__(self, other: CashFlows) -> CashFlows:
        times = np.concatenate([self.times, other.times])
        amounts = np.concatenate([self.amounts, other.amounts])
        order = np.argsort(times, kind="stable")
        return CashFlows(times[order], amounts[order])

    def scaled(self, factor: float) -> CashFlows:
        """The same dates with every amount multiplied by `factor`."""
        return CashFlows(self.times.copy(), self.amounts * factor)


@dataclass
class BondRisk:
    """Price and first/second-order yield sensitivities of a cash-flow stream."""

    price: float
    """float: Present value at the given yield."""

    macaulay_duration: float
    """float: Present-value-weighted average time to payment."""

    modified_duration: float
    """float: :math:`-\\frac{1}{P}\\frac{dP}{dy}`; equals Macaulay under continuous compounding."""

    convexity: float
    """float: :math:`\\frac{1}{P}\\frac{d^2P}{dy^2}`."""

    yield_: float = 0.0
    """float: The yield the measures were computed at."""

    compounding: object = None
    """int or None: Compounding frequency of the yield (``None`` = continuous)."""

    @property
    def dollar_duration(self) -> float:
        """float: :math:`-dP/dy = D_{mod} P`."""
        return self.modified_duration * self.price

    @property
    def dv01(self) -> float:
        """float: Price change for a one-basis-point fall in yield."""
        return self.dollar_duration * 1e-4

    def price_change(self, dy, order: int = 2):
        """Duration (``order=1``) or duration-plus-convexity (``order=2``) estimate of the price change for a yield move `dy`."""
        dy = np.asarray(dy, dtype=np.float64)
        rel = -self.modified_duration * dy
        if order >= 2:
            rel = rel + 0.5 * self.convexity * dy**2
        return self.price * rel


@dataclass
class CurveFitResult:
    """A fitted term structure and its goodness of fit."""

    curve: YieldCurve
    """YieldCurve: The fitted curve."""

    params: np.ndarray
    """ndarray: Fitted parameters, in the model's documented order."""

    residuals: np.ndarray
    """ndarray: Fitted minus observed values (yields or prices)."""

    model: str = ""
    """str: ``"nelson_siegel"``, ``"svensson"`` or ``"mcculloch_spline"``."""

    @property
    def rmse(self) -> float:
        """float: Root-mean-square residual."""
        return float(np.sqrt(np.mean(self.residuals**2)))


@dataclass
class CurvePCAResult:
    """Principal components of yield-curve changes (Litterman-Scheinkman)."""

    tenors: np.ndarray
    """ndarray, shape (n,): Maturities of the yield columns."""

    eigenvalues: np.ndarray
    """ndarray, shape (n,): Component variances, descending."""

    loadings: np.ndarray
    """ndarray, shape (n, n): Unit eigenvectors as columns, ``loadings[:, k]`` pairs with ``eigenvalues[k]``.
    Signs are fixed so the first component loads positively on average and the second rises with maturity."""

    scores: np.ndarray
    """ndarray, shape (n_obs, n): The centred changes projected on the components."""

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """ndarray: Share of total variance carried by each component."""
        return self.eigenvalues / np.sum(self.eigenvalues)


@dataclass
class KeyRateDurationResult:
    """Ho's key rate durations of a cash-flow stream."""

    key_tenors: np.ndarray
    """ndarray, shape (k,): Key maturities."""

    durations: np.ndarray
    """ndarray, shape (k,): Duration with respect to each key rate."""

    parallel_duration: float
    """float: Effective duration under a parallel shift; equals ``durations.sum()`` to first order."""

    price: float
    """float: Present value on the unshocked curve."""


@dataclass
class ImmunizationResult:
    """Asset holdings that immunize a liability stream against parallel shifts (Redington)."""

    holdings: np.ndarray
    """ndarray, shape (n_assets,): Units of each asset held."""

    assets: list
    """list of CashFlows: Each asset's cash flows per unit."""

    liabilities: CashFlows
    """CashFlows: The liability stream."""

    curve: YieldCurve
    """YieldCurve: The curve the match was made on."""

    pv_assets: float = 0.0
    """float: Present value of the asset portfolio."""

    pv_liabilities: float = 0.0
    """float: Present value of the liabilities."""

    duration_assets: float = 0.0
    """float: Fisher-Weil duration of the asset portfolio."""

    duration_liabilities: float = 0.0
    """float: Fisher-Weil duration of the liabilities."""

    convexity_assets: float = 0.0
    """float: Convexity of the asset portfolio."""

    convexity_liabilities: float = 0.0
    """float: Convexity of the liabilities."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics."""

    @property
    def portfolio(self) -> CashFlows:
        """CashFlows: The combined asset portfolio."""
        out = CashFlows(np.empty(0), np.empty(0))
        for h, a in zip(self.holdings, self.assets, strict=True):
            out = out + a.scaled(float(h))
        return out

    @property
    def redington_satisfied(self) -> bool:
        """bool: Asset convexity exceeds liability convexity, so the surplus has a local minimum at zero shift."""
        return self.convexity_assets > self.convexity_liabilities

    def surplus(self, shifts) -> np.ndarray:
        """Asset minus liability value after parallel shifts `shifts` of the curve."""
        shifts = np.atleast_1d(np.asarray(shifts, dtype=np.float64))
        port = self.portfolio
        return np.array([port.present_value(self.curve.shifted(s)) - self.liabilities.present_value(self.curve.shifted(s)) for s in shifts])


@dataclass
class SwapValuation:
    """Value of a swap as the difference of two bonds: the leg received less the leg paid."""

    value: float
    """float: ``receive_leg - pay_leg``, in the valuation currency."""

    receive_leg: float
    """float: Present value of the leg received."""

    pay_leg: float
    """float: Present value of the leg paid."""

    kind: str = ""
    """str: ``"pay_fixed"``, ``"receive_fixed"``, ``"receive_foreign"`` or ``"pay_foreign"``."""
