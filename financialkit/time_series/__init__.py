"""financialkit.time_series: returns, efficiency tests, ARMA, unit roots, long memory and volatility.

Returns and their stylized facts (fat tails by the Hill estimator, the
Jarque-Bera test via :func:`scipy.stats.jarque_bera`); tests of the random
walk (serial correlation by :mod:`scipy.fft`, Fama's runs test, Box-Pierce
and Ljung-Box, Lo-MacKinlay variance ratios); Yule-Walker autoregressions
(:func:`scipy.linalg.solve_toeplitz`), partial autocorrelations, ARMA
simulation (:func:`scipy.signal.lfilter`) and conditional maximum
likelihood (:func:`scipy.optimize.least_squares`) with AIC/BIC order
selection; the augmented Dickey-Fuller and Engle-Granger tests with
MacKinnon's critical values; Hurst's rescaled range and fractional Gaussian
noise; and ARCH, GARCH, EGARCH, GJR-GARCH and RiskMetrics EWMA volatility
models fitted by maximum likelihood (:func:`scipy.optimize.minimize`),
Engle's ARCH-LM test, Parkinson's range estimator and realized variance.

Returns are decimals per period and variances are per period; annualize
with :data:`financialkit.constants.TRADING_DAYS_PER_YEAR`.
"""

from financialkit import __version__
from financialkit.time_series.core.base import (
    OHLC,
    ARMAResult,
    CointegrationResult,
    HurstResult,
    OrderSelectionResult,
    ReturnStatistics,
    TailIndexResult,
    TestResult,
    UnitRootResult,
    VarianceRatioResult,
    VolatilityFit,
    VolatilityModel,
    VolatilityPath,
)
from financialkit.time_series.systems.efficiency import (
    autocorrelation,
    autocovariance,
    bartlett_band,
    box_pierce,
    ljung_box,
    runs_test,
    variance_ratio,
)
from financialkit.time_series.systems.garch import (
    ARCH,
    EGARCH,
    EWMA,
    GARCH,
    GJRGARCH,
    arch_lm_test,
)
from financialkit.time_series.systems.linear_models import (
    ar_cycle_period,
    arma_acf,
    arma_psi_weights,
    fit_arma,
    partial_autocorrelation,
    select_arma_order,
    simulate_arma,
    yule_walker,
)
from financialkit.time_series.systems.long_memory import (
    anis_lloyd_expected_rs,
    fgn_autocovariance,
    hurst_exponent,
    rescaled_range,
    simulate_fgn,
)
from financialkit.time_series.systems.realized import (
    close_to_close_variance,
    mincer_zarnowitz,
    parkinson_variance,
    realized_variance,
)
from financialkit.time_series.systems.stylized_facts import (
    hill_estimator,
    log_returns,
    return_statistics,
    simple_returns,
    tail_survival,
)
from financialkit.time_series.systems.unit_roots import (
    adf_test,
    engle_granger,
    mackinnon_critical_values,
)
from financialkit.time_series.utils.numerics import (
    OLSResult,
    lag_matrix,
    numerical_hessian,
    ols,
)
from financialkit.time_series.utils.synthetic import (
    simulate_cointegrated_pair,
    simulate_dickey_fuller,
    simulate_intraday_returns,
    simulate_ohlc,
)
from financialkit.time_series.visualizers.plots import (
    plot_acf,
    plot_conditional_volatility,
    plot_hill,
    plot_news_impact,
    plot_rescaled_range,
    plot_tail,
    plot_variance_ratio,
)

__all__ = [
    "__version__",
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
    "log_returns",
    "simple_returns",
    "return_statistics",
    "tail_survival",
    "hill_estimator",
    "autocovariance",
    "autocorrelation",
    "bartlett_band",
    "runs_test",
    "box_pierce",
    "ljung_box",
    "variance_ratio",
    "yule_walker",
    "ar_cycle_period",
    "partial_autocorrelation",
    "arma_psi_weights",
    "arma_acf",
    "simulate_arma",
    "fit_arma",
    "select_arma_order",
    "mackinnon_critical_values",
    "adf_test",
    "engle_granger",
    "rescaled_range",
    "anis_lloyd_expected_rs",
    "hurst_exponent",
    "fgn_autocovariance",
    "simulate_fgn",
    "ARCH",
    "GARCH",
    "EGARCH",
    "GJRGARCH",
    "EWMA",
    "arch_lm_test",
    "close_to_close_variance",
    "parkinson_variance",
    "realized_variance",
    "mincer_zarnowitz",
    "OLSResult",
    "ols",
    "lag_matrix",
    "numerical_hessian",
    "simulate_ohlc",
    "simulate_intraday_returns",
    "simulate_cointegrated_pair",
    "simulate_dickey_fuller",
    "plot_acf",
    "plot_tail",
    "plot_hill",
    "plot_rescaled_range",
    "plot_variance_ratio",
    "plot_conditional_volatility",
    "plot_news_impact",
]
