"""Abstract base classes and result containers for financialkit.time_series.

Every statistical test returns a small dataclass (a :class:`TestResult`, or
the richer :class:`VarianceRatioResult`, :class:`UnitRootResult`,
:class:`StationarityResult`, :class:`CointegrationResult` and
:class:`JohansenResult`), every fitted model a fit record
(:class:`ARMAResult`, :class:`VectorAutoregressionResult`, :class:`VolatilityFit`,
:class:`DCCFit`, :class:`HARResult`, :class:`RegimeSwitchingResult`), the
Kalman filter a :class:`KalmanResult`, and every estimator of a
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

from financialkit._deprecation import deprecated_field

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
    "KalmanResult",
    "VectorAutoregressionResult",
    "JohansenResult",
    "StationarityResult",
    "RegimeSwitchingResult",
    "DCCFit",
    "HARResult",
]


@dataclass
class ReturnStatistics:
    """Sample moments of a return series and the Jarque-Bera normality test."""

    n_obs: int
    """int: Number of returns."""
    nobs = deprecated_field("n_obs")

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

    jarque_bera_p_value: float
    """float: Jarque-Bera p-value."""
    jarque_bera_pvalue = deprecated_field("jarque_bera_p_value")


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

    p_value: float
    """float: p-value of the statistic."""
    pvalue = deprecated_field("p_value")

    df: Optional[int] = None
    """int or None: Degrees of freedom of a chi-squared reference distribution."""

    n_obs: int = 0
    """int: Observations used."""
    nobs = deprecated_field("n_obs")

    def rejects(self, level: float = 0.05) -> bool:
        """Whether the null hypothesis is rejected at significance `level`."""
        return bool(self.p_value < level)


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

    n_obs: int
    """int: Number of one-period returns :math:`T`."""
    nobs = deprecated_field("n_obs")

    @property
    def p_value(self) -> np.ndarray:
        """Two-sided p-values of :attr:`z`."""
        from scipy.stats import norm

        return 2.0 * norm.sf(np.abs(self.z))

    pvalue = deprecated_field("p_value")

    @property
    def p_value_robust(self) -> np.ndarray:
        """Two-sided p-values of :attr:`z_robust`."""
        from scipy.stats import norm

        return 2.0 * norm.sf(np.abs(self.z_robust))

    pvalue_robust = deprecated_field("p_value_robust")


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

    n_obs: int
    """int: Observations in the likelihood (after conditioning)."""
    nobs = deprecated_field("n_obs")

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
        return -2.0 * self.loglik + self.n_params * math.log(self.n_obs)

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

    n_obs: int
    """int: Observations in the regression."""
    nobs = deprecated_field("n_obs")

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

    n_obs: int
    """int: Number of returns."""
    nobs = deprecated_field("n_obs")

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
        return -2.0 * self.loglik + self.n_params * math.log(self.n_obs)

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
            model=model, loglik=model.loglikelihood(r), n_obs=r.size, variance=model.variance(r), std_errors=std_errors, converged=bool(res.success)
        )

    def _param_scale(self, name: str, s: float) -> float:
        """Factor by which parameter `name`'s standard error scales when returns are multiplied by `s`."""
        return {"mu": s, "omega": s**2}.get(name, 1.0)


@dataclass
class KalmanResult:
    """Kalman filter output for :math:`x_t = F x_{t-1} + w_t`, :math:`y_t = H x_t + v_t`.

    :math:`w_t \\sim N(0, Q)` and :math:`v_t \\sim N(0, R)`. Arrays are
    indexed by observation `t`; "predicted" means given
    :math:`y_1..y_{t-1}` and "filtered" given :math:`y_1..y_t`.
    """

    predicted_state: np.ndarray
    """ndarray, shape (n, s): :math:`\\hat x_{t|t-1}`."""

    predicted_cov: np.ndarray
    """ndarray, shape (n, s, s): :math:`P_{t|t-1}`."""

    filtered_state: np.ndarray
    """ndarray, shape (n, s): :math:`\\hat x_{t|t}`."""

    filtered_cov: np.ndarray
    """ndarray, shape (n, s, s): :math:`P_{t|t}`."""

    innovations: np.ndarray
    """ndarray, shape (n, m): One-step prediction errors :math:`y_t - H\\hat x_{t|t-1}` (NaN where `y` is missing)."""

    innovation_cov: np.ndarray
    """ndarray, shape (n, m, m): :math:`S_t = H P_{t|t-1} H' + R`."""

    gain: np.ndarray
    """ndarray, shape (n, s, m): Kalman gain :math:`K_t = P_{t|t-1} H' S_t^{-1}` (zero where `y` is missing)."""

    loglik: float
    """float: Gaussian log-likelihood by the prediction-error decomposition."""

    n_obs: int
    """int: Non-missing observations in the likelihood."""
    nobs = deprecated_field("n_obs")

    F: np.ndarray
    """ndarray, shape (s, s): Transition matrix."""

    Q: np.ndarray
    """ndarray, shape (s, s): State noise covariance."""

    def smooth(self) -> tuple[np.ndarray, np.ndarray]:
        """Rauch-Tung-Striebel smoothed states :math:`\\hat x_{t|n}` and covariances :math:`P_{t|n}`.

        .. math:: \\hat x_{t|n} = \\hat x_{t|t} + J_t(\\hat x_{t+1|n} - \\hat x_{t+1|t}), \\qquad J_t = P_{t|t} F' P_{t+1|t}^{-1}

        Returns
        -------
        state : ndarray, shape (n, s)
        cov : ndarray, shape (n, s, s)
        """
        xs = self.filtered_state.copy()
        Ps = self.filtered_cov.copy()
        for t in range(xs.shape[0] - 2, -1, -1):
            J = np.linalg.solve(self.predicted_cov[t + 1].T, (self.filtered_cov[t] @ self.F.T).T).T
            xs[t] = self.filtered_state[t] + J @ (xs[t + 1] - self.predicted_state[t + 1])
            Ps[t] = self.filtered_cov[t] + J @ (Ps[t + 1] - self.predicted_cov[t + 1]) @ J.T
        return xs, Ps


@dataclass
class VectorAutoregressionResult:
    """A vector autoregression :math:`y_t = c + A_1 y_{t-1} + \\dots + A_p y_{t-p} + u_t` fitted by least squares."""

    coefs: np.ndarray
    """ndarray, shape (p, k, k): :math:`A_1..A_p`; ``coefs[i, a, b]`` is the effect of variable `b` at lag ``i + 1`` on variable `a`."""

    intercept: np.ndarray
    """ndarray, shape (k,): :math:`c` (zeros when no constant was estimated)."""

    sigma: np.ndarray
    """ndarray, shape (k, k): Maximum-likelihood residual covariance :math:`\\hat u'\\hat u / T`."""

    residuals: np.ndarray
    """ndarray, shape (T, k): :math:`\\hat u_t` after conditioning on the first values."""

    loglik: float
    """float: Gaussian log-likelihood at the estimate."""

    n_obs: int
    """int: Observations :math:`T` in the regression."""
    nobs = deprecated_field("n_obs")

    n_params: int
    """int: Estimated coefficients (without :math:`\\Sigma`)."""

    data: np.ndarray
    """ndarray, shape (n, k): The series the model was fitted to."""

    @property
    def order(self) -> int:
        return int(self.coefs.shape[0])

    @property
    def aic(self) -> float:
        """:math:`-2\\ell + 2k` with `k` the number of coefficients."""
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        """:math:`-2\\ell + k \\ln T`."""
        return -2.0 * self.loglik + self.n_params * math.log(self.n_obs)

    def companion(self) -> np.ndarray:
        """The ``(kp, kp)`` companion matrix of the VAR(1) form."""
        p, k, _ = self.coefs.shape
        C = np.zeros((k * p, k * p))
        C[:k] = np.hstack(list(self.coefs))
        C[k:, :-k] = np.eye(k * (p - 1))
        return C

    def is_stable(self) -> bool:
        """Whether every eigenvalue of the companion matrix lies inside the unit circle."""
        return bool(np.max(np.abs(np.linalg.eigvals(self.companion()))) < 1.0)

    def ma_weights(self, h: int) -> np.ndarray:
        """Moving-average matrices :math:`\\Phi_0 = I, \\Phi_1, \\dots, \\Phi_h` of :math:`y_t = \\mu + \\sum_s \\Phi_s u_{t-s}`.

        Returns
        -------
        ndarray, shape (h + 1, k, k)
        """
        p, k, _ = self.coefs.shape
        Phi = np.zeros((h + 1, k, k))
        Phi[0] = np.eye(k)
        for s in range(1, h + 1):
            for i in range(min(p, s)):
                Phi[s] += Phi[s - 1 - i] @ self.coefs[i]
        return Phi

    def impulse_response(self, h: int, orthogonalized: bool = True) -> np.ndarray:
        """Responses ``[s, a, b]`` of variable `a`, `s` periods after a shock to `b`, for ``s = 0..h``.

        Orthogonalized responses (Sims 1980) are :math:`\\Phi_s P` with
        :math:`P` the lower Cholesky factor of :math:`\\Sigma`: a one
        standard deviation structural shock, with the variables' ordering
        as the causal ordering of the contemporaneous effects. Otherwise
        the responses are to a unit reduced-form shock :math:`u_b`.

        Returns
        -------
        ndarray, shape (h + 1, k, k)
        """
        Phi = self.ma_weights(h)
        return Phi @ np.linalg.cholesky(self.sigma) if orthogonalized else Phi

    def fevd(self, h: int) -> np.ndarray:
        """Forecast-error variance decomposition: ``[s, a, b]`` is the share of the ``s + 1``-step error variance of `a` due to orthogonalized shock `b`.

        Each ``[s, a, :]`` sums to one.

        Returns
        -------
        ndarray, shape (h, k, k)
        """
        theta2 = np.cumsum(self.impulse_response(h - 1) ** 2, axis=0)
        return theta2 / theta2.sum(axis=2, keepdims=True)

    def forecast(self, h: int) -> np.ndarray:
        """Forecasts 1..`h` steps after the end of the data.

        Returns
        -------
        ndarray, shape (h, k)
        """
        p = self.order
        hist = list(self.data[-p:])
        out = np.empty((h, self.data.shape[1]))
        for s in range(h):
            out[s] = self.intercept + sum(self.coefs[i] @ hist[-1 - i] for i in range(p))
            hist.append(out[s])
        return out


@dataclass
class JohansenResult:
    """Johansen's trace and maximum-eigenvalue tests of the cointegration rank.

    Entry `r` of each statistic tests :math:`H_0`: rank :math:`\\le r`
    (trace) or rank :math:`= r` against :math:`r + 1` (maximum eigenvalue).
    """

    eigenvalues: np.ndarray
    """ndarray, shape (k,): Eigenvalues in decreasing order, the squared canonical correlations of :math:`\\Delta y_t` and :math:`y_{t-1}`."""

    trace: np.ndarray
    """ndarray, shape (k,): :math:`-T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)` for ``r = 0..k-1``."""

    max_eigen: np.ndarray
    """ndarray, shape (k,): :math:`-T\\ln(1 - \\hat\\lambda_{r+1})` for ``r = 0..k-1``."""

    trace_critical_values: dict
    """dict: Critical values of :attr:`trace` keyed by ``"10%"``, ``"5%"``, ``"1%"`` (arrays of shape (k,))."""

    max_eigen_critical_values: dict
    """dict: Critical values of :attr:`max_eigen`, keyed like :attr:`trace_critical_values`."""

    beta: np.ndarray
    """ndarray, shape (k, k): Eigenvectors in columns, normalized so :math:`\\beta' S_{11}\\beta = I`; the first `r` span the cointegrating space."""

    alpha: np.ndarray
    """ndarray, shape (k, k): Adjustment coefficients :math:`S_{01}\\beta` of the error-correction form."""

    lags: int
    """int: Lagged differences in the error-correction model."""

    n_obs: int
    """int: Observations :math:`T` in the reduced-rank regression."""
    nobs = deprecated_field("n_obs")

    regression: str
    """str: Deterministic terms: ``"n"`` (none) or ``"c"`` (unrestricted constant)."""

    def rank(self, level: str = "5%", test: str = "trace") -> int:
        """The cointegration rank chosen by testing ``r = 0, 1, ...`` in turn and stopping at the first non-rejection."""
        stat, cv = (self.trace, self.trace_critical_values) if test == "trace" else (self.max_eigen, self.max_eigen_critical_values)
        for r in range(stat.size):
            if stat[r] < cv[level][r]:
                return r
        return int(stat.size)

    def cointegrating_vectors(self, r: int) -> np.ndarray:
        """The first `r` cointegrating vectors, each scaled so its first element is one.

        Returns
        -------
        ndarray, shape (k, r)
        """
        return self.beta[:, :r] / self.beta[0, :r]


@dataclass
class StationarityResult:
    """The KPSS test of (level or trend) stationarity."""

    statistic: float
    """float: :math:`\\eta = T^{-2}\\sum_t S_t^2 / \\hat\\sigma^2_\\infty`, with :math:`S_t` the partial sums of the residuals."""

    critical_values: dict
    """dict: KPSS (1992, Table 1) asymptotic critical values keyed by ``"10%"``, ``"5%"``, ``"2.5%"``, ``"1%"``."""

    lags: int
    """int: Bartlett-window lags in the long-run variance."""

    n_obs: int
    """int: Observations."""
    nobs = deprecated_field("n_obs")

    regression: str
    """str: ``"c"`` (level stationarity) or ``"ct"`` (trend stationarity)."""

    @property
    def p_value(self) -> float:
        """p-value interpolated in the table, clipped to [0.01, 0.10]."""
        levels = np.array([0.10, 0.05, 0.025, 0.01])
        cvs = np.array(list(self.critical_values.values()))
        return float(np.interp(self.statistic, cvs, levels))

    pvalue = deprecated_field("p_value")

    def rejects(self, level: str = "5%") -> bool:
        """Whether stationarity is rejected at `level`; large values reject."""
        return bool(self.statistic > self.critical_values[level])


@dataclass
class RegimeSwitchingResult:
    """A Markov-switching model :math:`x_t \\mid s_t = j \\sim N(\\mu_j, \\sigma_j^2)` estimated by EM.

    Regimes are ordered by increasing :math:`\\sigma_j`.
    """

    mu: np.ndarray
    """ndarray, shape (k,): Regime means."""

    sigma: np.ndarray
    """ndarray, shape (k,): Regime standard deviations."""

    transition: np.ndarray
    """ndarray, shape (k, k): ``transition[i, j]`` is :math:`P(s_t = j \\mid s_{t-1} = i)`."""

    filtered_probs: np.ndarray
    """ndarray, shape (n, k): :math:`P(s_t = j \\mid x_1..x_t)`."""

    smoothed_probs: np.ndarray
    """ndarray, shape (n, k): :math:`P(s_t = j \\mid x_1..x_n)` (Kim's smoother)."""

    loglik: float
    """float: Log-likelihood from Hamilton's filter."""

    n_obs: int
    """int: Observations."""
    nobs = deprecated_field("n_obs")

    n_iter: int
    """int: EM iterations."""

    converged: bool
    """bool: Whether the log-likelihood change fell below the tolerance."""

    @property
    def n_regimes(self) -> int:
        return int(self.mu.size)

    @property
    def n_params(self) -> int:
        k = self.n_regimes
        return 2 * k + k * (k - 1)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + self.n_params * math.log(self.n_obs)

    @property
    def expected_duration(self) -> np.ndarray:
        """Mean stay in each regime, :math:`1 / (1 - p_{jj})` periods."""
        return 1.0 / (1.0 - np.diag(self.transition))

    @property
    def ergodic_probs(self) -> np.ndarray:
        """Long-run regime probabilities: the left eigenvector of :attr:`transition` for eigenvalue 1."""
        w, v = np.linalg.eig(self.transition.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        return pi / pi.sum()

    def most_likely_regime(self) -> np.ndarray:
        """The regime with the highest smoothed probability at each date."""
        return np.argmax(self.smoothed_probs, axis=1)


@dataclass
class DCCFit:
    """Engle's DCC-GARCH(1,1): univariate GARCH variances and dynamic conditional correlations.

    .. math::

       Q_t = (1 - a - b)\\bar Q + a\\,z_{t-1}z_{t-1}' + b\\,Q_{t-1}, \\qquad
       R_t = \\operatorname{diag}(Q_t)^{-1/2} Q_t \\operatorname{diag}(Q_t)^{-1/2}
    """

    a: float
    """float: News coefficient of the correlation dynamics."""

    b: float
    """float: Decay coefficient of the correlation dynamics."""

    Qbar: np.ndarray
    """ndarray, shape (k, k): Unconditional covariance of the standardized residuals (correlation targeting)."""

    univariate: list
    """list of VolatilityFit: The first-step GARCH fit of each series."""

    correlation: np.ndarray
    """ndarray, shape (n, k, k): Conditional correlations :math:`R_t` given data to ``t - 1``."""

    standardized_residuals: np.ndarray
    """ndarray, shape (n, k): :math:`z_{t,i} = (r_{t,i} - \\mu_i)/\\sigma_{t,i}`."""

    loglik: float
    """float: Total Gaussian log-likelihood (volatility plus correlation parts)."""

    correlation_loglik: float
    """float: The correlation part :math:`-\\tfrac12\\sum_t(\\ln|R_t| + z_t'R_t^{-1}z_t - z_t'z_t)` maximized in step 2."""

    n_obs: int
    """int: Observations."""
    nobs = deprecated_field("n_obs")

    converged: bool = True
    """bool: Whether the step-2 optimizer reported success."""

    @property
    def persistence(self) -> float:
        return self.a + self.b

    @property
    def volatility(self) -> np.ndarray:
        """ndarray, shape (n, k): Conditional standard deviations from the univariate fits."""
        return np.column_stack([np.sqrt(f.variance) for f in self.univariate])

    @property
    def covariance(self) -> np.ndarray:
        """ndarray, shape (n, k, k): :math:`H_t = D_t R_t D_t`."""
        d = self.volatility
        return d[:, :, None] * self.correlation * d[:, None, :]


@dataclass
class HARResult:
    """Corsi's heterogeneous autoregression of realized variance, fitted by least squares.

    .. math:: RV_{t+1} = \\beta_0 + \\beta_d RV_t + \\beta_w RV^{(5)}_t + \\beta_m RV^{(22)}_t + e_{t+1}

    with :math:`RV^{(h)}_t` the mean of :math:`RV_{t-h+1}..RV_t`.
    """

    coef: np.ndarray
    """ndarray: :math:`\\beta_0` followed by one coefficient per horizon."""

    horizons: tuple
    """tuple of int: Averaging windows (``(1, 5, 22)`` by default: a day, a week, a month)."""

    ols: object
    """OLSResult: The underlying regression, with standard errors and :math:`R^2`."""

    data: np.ndarray
    """ndarray, shape (n,): The realized-variance series the model was fitted to."""

    def predict(self, rv) -> np.ndarray:
        """One-step forecasts of ``rv[t]`` from ``rv[:t]``, NaN for ``t < max(horizons)``.

        Applies the fitted coefficients to any series, so passing a hold-out
        sample gives out-of-sample forecasts.

        Returns
        -------
        ndarray, shape (n,)
        """
        from financialkit.time_series.systems.realized import har_regressors

        x = np.asarray(rv, dtype=np.float64)
        X = har_regressors(x, self.horizons)
        out = np.full(x.size, np.nan)
        out[max(self.horizons) :] = self.coef[0] + X[:-1] @ self.coef[1:]
        return out

    def forecast(self) -> float:
        """Forecast of the value after the end of :attr:`data`."""
        from financialkit.time_series.systems.realized import har_regressors

        return float(self.coef[0] + har_regressors(self.data, self.horizons)[-1] @ self.coef[1:])
