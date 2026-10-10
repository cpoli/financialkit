"""Concrete time-series models: stylized facts, efficiency tests, ARMA, unit roots, long memory and volatility."""

from financialkit.time_series.systems.efficiency import autocorrelation, autocovariance, bartlett_band, box_pierce, ljung_box, runs_test, variance_ratio
from financialkit.time_series.systems.garch import ARCH, EGARCH, EWMA, GARCH, GJRGARCH, arch_lm_test
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
from financialkit.time_series.systems.long_memory import anis_lloyd_expected_rs, fgn_autocovariance, hurst_exponent, rescaled_range, simulate_fgn
from financialkit.time_series.systems.realized import close_to_close_variance, mincer_zarnowitz, parkinson_variance, realized_variance
from financialkit.time_series.systems.stylized_facts import hill_estimator, log_returns, return_statistics, simple_returns, tail_survival
from financialkit.time_series.systems.unit_roots import adf_test, engle_granger, mackinnon_critical_values

__all__ = [
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
]
