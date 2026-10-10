"""Abstract base classes and result containers for financialkit.time_series.

Every statistical test returns a small dataclass (a :class:`TestResult`, or
the richer :class:`VarianceRatioResult`, :class:`UnitRootResult` and
:class:`CointegrationResult`), every fitted model a fit record
(:class:`ARMAResult`, :class:`VolatilityFit`), and every estimator of a
scaling exponent its own record (:class:`TailIndexResult`,
:class:`HurstResult`), never a bare tuple, so the visualizers have a stable
interface.

Conditional-volatility models share the :class:`VolatilityModel` base:
a model is a frozen set of parameters that knows its one-step variance
update (the *news impact curve*), its Gaussian log-likelihood and how to
simulate itself, and :meth:`VolatilityModel.fit` estimates any subclass by
maximum likelihood with :func:`scipy.optimize.minimize`.

Returns are plain decimals (``0.01`` is one percent), one per period, and
variances are per period; annualize with
:data:`financialkit.constants.TRADING_DAYS_PER_YEAR`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "ReturnStatistics",
    "TailIndexResult",
    "TestResult",
    "VarianceRatioResult",
    "ARMAResult",
    "OrderSelectionResult",
    "UnitRootResult",
    "CointegrationResult",
    "HurstResult",
    "OHLC",
    "VolatilityPath",
    "VolatilityFit",
    "VolatilityModel",
]


@dataclass
class ReturnStatistics:
    """Sample moments of a return series and the Jarque-Bera normality test."""

    nobs: int
    """int: Number of returns."""

    mean: float
    """float: Per-period sample mean."""

    std: float
    """float: Per-period sample standard deviation (``ddof=1``)."""

    skewness: float
    """float: Sample skewness; zero for a normal distribution."""

    excess_kurtosis: float
    """float: Sample excess kurtosis; zero for a normal distribution."""

    jarque_bera: float
    """float: Jarque-Bera statistic (:func:`scipy.stats.jarque_bera`)."""

    jarque_bera_pvalue: float
    """float: Jarque-Bera p-value."""


@dataclass
class TailIndexResult:
    """Hill estimates of a tail index for a range of order statistics."""

    k: np.ndarray
    """ndarray of int: Number of upper order statistics used."""

    alpha: np.ndarray
    """ndarray: Tail index estimate :math:`\\hat\\alpha_k` for each `k`."""

    std_error: np.ndarray
    """ndarray: Asymptotic standard error :math:`\\hat\\alpha_k / \\sqrt{k}`."""

    threshold: np.ndarray
    """ndarray: The order statistic :math:`X_{(k+1)}` each estimate conditions on."""


@dataclass
class TestResult:
    """A test statistic with its p-value."""

    __test__ = False  # keep pytest from collecting this class

    name: str
    """str: Name of the test."""

    statistic: float
    """float: The test statistic."""

    pvalue: float
    """float: p-value of the statistic."""

    df: Optional[int] = None
    """int or None: Degrees of freedom of a chi-squared reference distribution."""

    nobs: int = 0
    """int: Observations used."""

    def rejects(self, level: float = 0.05) -> bool:
        """Whether the null hypothesis is rejected at significance `level`."""
        return bool(self.pvalue < level)


@dataclass
class VarianceRatioResult:
    """Lo-MacKinlay variance ratios for one or more horizons."""

    q: np.ndarray
    """ndarray of int: Aggregation horizons."""

    ratio: np.ndarray
    """ndarray: Bias-corrected variance ratios :math:`VR(q)`; 1 under a random walk."""

    z: np.ndarray
    """ndarray: Standard normal statistics under homoskedastic increments."""

    z_robust: np.ndarray
    """ndarray: Heteroskedasticity-consistent statistics :math:`z^*(q)`."""

    nobs: int
    """int: Number of one-period returns :math:`T`."""

    @property
    def pvalue(self) -> np.ndarray:
        """Two-sided p-values of :attr:`z`."""
        from scipy.stats import norm

        return 2.0 * norm.sf(np.abs(self.z))

    @property
    def pvalue_robust(self) -> np.ndarray:
        """Two-sided p-values of :attr:`z_robust`."""
        from scipy.stats import norm

        return 2.0 * norm.sf(np.abs(self.z_robust))


@dataclass
class ARMAResult:
    """An ARMA(p, q) model fitted by conditional maximum likelihood.

    The model is :math:`\\phi(B)(x_t - \\mu) = \\theta(B)\\varepsilon_t`
    with :math:`\\phi(B) = 1 - \\phi_1 B - \\dots - \\phi_p B^p` and
    :math:`\\theta(B) = 1 + \\theta_1 B + \\dots + \\theta_q B^q`.
    """

    ar: np.ndarray
    """ndarray: Autoregressive coefficients :math:`\\phi_1..\\phi_p`."""

    ma: np.ndarray
    """ndarray: Moving-average coefficients :math:`\\theta_1..\\theta_q`."""

    mean: float
    """float: :math:`\\mu` (0 when the mean was not estimated)."""

    sigma2: float
    """float: Innovation variance."""

    loglik: float
    """float: Conditional Gaussian log-likelihood."""

    nobs: int
    """int: Observations in the likelihood (after conditioning)."""

    n_params: int
    """int: Estimated parameters, including :math:`\\sigma^2`."""

    residuals: np.ndarray
    """ndarray: Conditional innovations :math:`\\hat\\varepsilon_t` after the conditioning sample."""

    data: np.ndarray
    """ndarray: The series the model was fitted to."""

    @property
    def aic(self) -> float:
        """Akaike's information criterion :math:`-2\\ell + 2k`."""
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        """Schwarz's Bayesian information criterion :math:`-2\\ell + k \\ln n`."""
        return -2.0 * self.loglik + self.n_params * math.log(self.nobs)

    def psi_weights(self, n: int) -> np.ndarray:
        """First `n` weights of the MA(:math:`\\infty`) representation, :math:`\\psi_0 = 1`."""
        from financialkit.time_series.systems.linear_models import arma_psi_weights

        return arma_psi_weights(self.ar, self.ma, n)

    def forecast(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Minimum mean-square-error forecasts 1..`h` steps ahead and their standard errors.

        Box and Jenkins's recursion: future innovations are replaced by 0 and
        future values by their forecasts; the h-step error variance is
        :math:`\\sigma^2 \\sum_{j<h} \\psi_j^2`.

        Returns
        -------
        mean : ndarray, shape (h,)
        std_error : ndarray, shape (h,)
        """
        p, q = self.ar.size, self.ma.size
        x = list(self.data - self.mean)
        e = list(self.residuals)
        out = np.empty(h)
        for k in range(h):
            ar_part = sum(self.ar[i] * x[-1 - i] for i in range(p))
            ma_part = sum(self.ma[j] * e[-1 - j] for j in range(q) if j < len(e))
            x.append(ar_part + ma_part)
            e.append(0.0)
            out[k] = x[-1]
        se = np.sqrt(self.sigma2 * np.cumsum(self.psi_weights(h) ** 2))
        return out + self.mean, se


@dataclass
class OrderSelectionResult:
    """Information criteria over a grid of ARMA orders."""

    aic: np.ndarray
    """ndarray, shape (max_p + 1, max_q + 1): Akaike criterion of ARMA(p, q) at ``[p, q]`` (NaN where a fit failed)."""

    bic: np.ndarray
    """ndarray, shape (max_p + 1, max_q + 1): Schwarz criterion of ARMA(p, q) at ``[p, q]`` (NaN where a fit failed)."""

    def best(self, criterion: str = "aic") -> tuple[int, int]:
        """The ``(p, q)`` minimizing `criterion` (``"aic"`` or ``"bic"``)."""
        grid = {"aic": self.aic, "bic": self.bic}[criterion]
        p, q = np.unravel_index(np.nanargmin(grid), grid.shape)
        return int(p), int(q)


@dataclass
class UnitRootResult:
    """An augmented Dickey-Fuller test."""

    statistic: float
    """float: The t-ratio of :math:`\\gamma` in :math:`\\Delta y_t = \\text{det}_t + \\gamma y_{t-1} + \\sum_i \\delta_i \\Delta y_{t-i} + e_t`."""

    critical_values: dict
    """dict: MacKinnon (2010) critical values keyed by ``"1%"``, ``"5%"``, ``"10%"``."""

    lags: int
    """int: Lagged differences included."""

    nobs: int
    """int: Observations in the regression."""

    regression: str
    """str: Deterministic terms: ``"n"``, ``"c"`` or ``"ct"``."""

    gamma: float
    """float: Estimated :math:`\\gamma = \\rho - 1`."""

    def rejects(self, level: str = "5%") -> bool:
        """Whether the unit root is rejected at `level` (``"1%"``, ``"5%"`` or ``"10%"``)."""
        return bool(self.statistic < self.critical_values[level])


@dataclass
class CointegrationResult:
    """An Engle-Granger two-step cointegration test."""

    statistic: float
    """float: ADF t-ratio of the first-step residuals."""

    critical_values: dict
    """dict: MacKinnon (2010) critical values for the number of variables."""

    intercept: float
    """float: Intercept :math:`a` of the first-step regression."""

    beta: np.ndarray
    """ndarray: Cointegrating coefficients on the regressors."""

    residuals: np.ndarray
    """ndarray: The estimated equilibrium error :math:`y_t - a - b' x_t`."""

    adf: UnitRootResult
    """UnitRootResult: The second-step test on the residuals."""

    r_squared: float
    """float: :math:`R^2` of the first-step regression."""

    def rejects(self, level: str = "5%") -> bool:
        """Whether "no cointegration" is rejected at `level`."""
        return bool(self.statistic < self.critical_values[level])


@dataclass
class HurstResult:
    """Rescaled-range (R/S) analysis."""

    window: np.ndarray
    """ndarray of int: Window lengths :math:`n`."""

    rs: np.ndarray
    """ndarray: Average rescaled range :math:`(R/S)_n` over non-overlapping windows."""

    expected_rs: np.ndarray
    """ndarray: Anis-Lloyd expectation of :math:`(R/S)_n` for independent normals."""

    hurst: float
    """float: Hurst's exponent: the slope of :math:`\\log (R/S)_n` on :math:`\\log n`."""

    hurst_corrected: float
    """float: :math:`0.5` plus the slope of :math:`\\log (R/S)_n - \\log E[(R/S)_n]`, which removes the small-sample bias of the raw slope."""


@dataclass
class OHLC:
    """Open, high, low and close prices per period."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@dataclass
class VolatilityPath:
    """A simulated return series and its conditional variance."""

    returns: np.ndarray
    """ndarray, shape (n,): Simulated returns."""

    variance: np.ndarray
    """ndarray, shape (n,): :math:`\\sigma_t^2`, the variance of ``returns[t]`` given the past."""

    @property
    def volatility(self) -> np.ndarray:
        """Conditional standard deviation :math:`\\sigma_t`."""
        return np.sqrt(self.variance)


@dataclass
class VolatilityFit:
    """A volatility model estimated by Gaussian maximum likelihood."""

    model: VolatilityModel
    """VolatilityModel: The fitted model, in the units of the input returns."""

    loglik: float
    """float: Maximized Gaussian log-likelihood."""

    nobs: int
    """int: Number of returns."""

    variance: np.ndarray
    """ndarray: In-sample conditional variances (the one-step-ahead forecasts)."""

    std_errors: dict = field(default_factory=dict)
    """dict: Standard errors from the inverse numerical Hessian, keyed by parameter."""

    converged: bool = True
    """bool: Whether the optimizer reported success."""

    @property
    def n_params(self) -> int:
        return len(self.model.param_names)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + self.n_params * math.log(self.nobs)

    @property
    def params(self) -> dict:
        """Fitted parameters keyed by name."""
        return self.model.params


class VolatilityModel(ABC):
    """A conditional-variance model :math:`r_t = \\mu + \\sigma_t z_t`, :math:`z_t \\sim N(0, 1)`.

    Subclasses define their parameters (:attr:`param_names`), the one-step
    update :meth:`news_impact`, the bounds and constraints of estimation, and
    how the parameters change when returns are rescaled. Variance filtering,
    the log-likelihood, simulation and maximum-likelihood fitting are
    shared. The pre-sample variance is the sample variance of the demeaned
    returns ("backcasting").
    """

    param_names: tuple[str, ...] = ()

    @property
    def params(self) -> dict:
        return {k: float(getattr(self, k)) for k in self.param_names}

    def _vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self.param_names], dtype=np.float64)

    @classmethod
    def _from_vector(cls, v) -> VolatilityModel:
        return cls(**dict(zip(cls.param_names, (float(x) for x in v), strict=True)))

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={getattr(self, k):.6g}" for k in self.param_names)
        return f"{type(self).__name__}({inner})"

    # --- model definition --------------------------------------------------

    @abstractmethod
    def news_impact(self, eps, var_prev):
        """Next period's variance given today's shock `eps` and variance `var_prev`; broadcasts."""

    @property
    @abstractmethod
    def unconditional_variance(self) -> float:
        """Long-run variance (``inf`` for a nonstationary model)."""

    @classmethod
    @abstractmethod
    def _start(cls, r: np.ndarray) -> np.ndarray:
        """Starting values for unit-variance returns `r`."""

    @classmethod
    @abstractmethod
    def _bounds(cls) -> list:
        """Bounds of the estimated parameters, for unit-variance returns."""

    @classmethod
    def _constraints(cls) -> list:
        """:func:`scipy.optimize.minimize` inequality constraints (stationarity)."""
        return []

    @abstractmethod
    def _rescaled(self, s: float) -> VolatilityModel:
        """The same model for returns multiplied by `s`."""

    # --- shared machinery --------------------------------------------------

    @property
    def mean(self) -> float:
        return float(getattr(self, "mu", 0.0))

    def variance(self, returns, v0: Optional[float] = None) -> np.ndarray:
        """Conditional variances :math:`\\sigma_t^2` of `returns` (one-step-ahead forecasts).

        Parameters
        ----------
        returns : array_like, shape (n,)
        v0 : float, optional
            Variance of the first observation; defaults to the sample
            variance of the demeaned returns.

        Returns
        -------
        ndarray, shape (n,)
        """
        eps = np.asarray(returns, dtype=np.float64) - self.mean
        out = np.empty_like(eps)
        out[0] = float(np.mean(eps**2)) if v0 is None else v0
        for t in range(1, eps.size):
            out[t] = self.news_impact(eps[t - 1], out[t - 1])
        return out

    def loglikelihood(self, returns, v0: Optional[float] = None) -> float:
        """Gaussian log-likelihood :math:`-\\tfrac12\\sum_t [\\ln 2\\pi + \\ln\\sigma_t^2 + \\varepsilon_t^2/\\sigma_t^2]`."""
        r = np.asarray(returns, dtype=np.float64)
        var = self.variance(r, v0)
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            return -np.inf
        eps = r - self.mean
        return float(-0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + eps**2 / var))

    def standardized_residuals(self, returns) -> np.ndarray:
        """:math:`z_t = (r_t - \\mu) / \\sigma_t`."""
        r = np.asarray(returns, dtype=np.float64)
        return (r - self.mean) / np.sqrt(self.variance(r))

    def simulate(self, n: int, burn: int = 500, rng: np.random.Generator | int | None = None) -> VolatilityPath:
        """Simulate `n` returns with Gaussian innovations, after `burn` discarded periods.

        Returns
        -------
        VolatilityPath
        """
        rng = np.random.default_rng(rng)
        z = rng.standard_normal(n + burn)
        var = np.empty(n + burn)
        eps = np.empty(n + burn)
        v = self.unconditional_variance
        var[0] = v if np.isfinite(v) else 1.0
        for t in range(n + burn):
            if t > 0:
                var[t] = self.news_impact(eps[t - 1], var[t - 1])
            eps[t] = np.sqrt(var[t]) * z[t]
        return VolatilityPath(returns=self.mean + eps[burn:], variance=var[burn:])

    @classmethod
    def fit(cls, returns) -> VolatilityFit:
        """Maximum-likelihood estimate on `returns` (Gaussian quasi-likelihood).

        The returns are scaled to unit variance before optimizing (so every
        parameter is of order one) and the estimate is mapped back, which the
        Gaussian likelihood allows because each model family is closed under
        rescaling. :func:`scipy.optimize.minimize` (SLSQP) handles the
        bounds and stationarity constraints.

        Returns
        -------
        VolatilityFit
        """
        r = np.asarray(returns, dtype=np.float64)
        s = float(np.std(r))
        if s == 0:
            raise ValueError("returns have zero variance")
        z = r / s

        def objective(v):
            ll = cls._from_vector(v).loglikelihood(z)
            return -ll / z.size if np.isfinite(ll) else 1e10

        res = minimize(objective, cls._start(z), method="SLSQP", bounds=cls._bounds(), constraints=cls._constraints(), options={"maxiter": 500, "ftol": 1e-10})
        unit = cls._from_vector(res.x)
        from financialkit.time_series.utils.numerics import numerical_hessian

        H = numerical_hessian(lambda v: cls._from_vector(v).loglikelihood(z), res.x)
        try:
            cov = np.linalg.inv(-H)
            se_unit = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except np.linalg.LinAlgError:
            se_unit = np.full(res.x.size, np.nan)
        model = unit._rescaled(s)
        # Map the standard errors through the (diagonal) rescaling.
        scale = np.array([unit._param_scale(k, s) for k in cls.param_names])
        std_errors = dict(zip(cls.param_names, se_unit * scale, strict=True))
        return VolatilityFit(
            model=model, loglik=model.loglikelihood(r), nobs=r.size, variance=model.variance(r), std_errors=std_errors, converged=bool(res.success)
        )

    def _param_scale(self, name: str, s: float) -> float:
        """Factor by which parameter `name`'s standard error scales when returns are multiplied by `s`."""
        return {"mu": s, "omega": s**2}.get(name, 1.0)
