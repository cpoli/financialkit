"""Renamed public functions keep their old names as warning aliases (ROADMAP A4)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import financialkit as fk  # noqa: E402


def test_fixed_income_accrued_interest_alias():
    with pytest.warns(DeprecationWarning, match="bond_accrued_interest"):
        old = fk.fixed_income.accrued_interest(100.0, 0.08, 1.375, freq=2)
    assert old == fk.fixed_income.bond_accrued_interest(100.0, 0.08, 1.375, freq=2)


def test_exotic_options_realized_variance_alias():
    paths = 100.0 * np.exp([[0.0, 0.1, 0.0], [0.0, -0.2, 0.1]])
    with pytest.warns(DeprecationWarning, match="annualized_realized_variance"):
        old = fk.exotic_options.realized_variance(paths, 0.5)
    np.testing.assert_array_equal(old, fk.exotic_options.annualized_realized_variance(paths, 0.5))


def test_fixed_income_plot_cash_flows_alias():
    cf = fk.fixed_income.bond_cash_flows(100.0, 0.06, 3.0, freq=2)
    with pytest.warns(DeprecationWarning, match="plot_discounted_cash_flows"):
        ax = fk.fixed_income.plot_cash_flows(cf, 0.05)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_each_name_has_one_meaning():
    """The new names are distinct, and the one shared name is the alias itself."""
    assert fk.time_value.accrued_interest is not fk.fixed_income.bond_accrued_interest
    assert fk.time_series.realized_variance is not fk.exotic_options.annualized_realized_variance
    assert fk.fixed_income.plot_cash_flows.__doc__.startswith("Deprecated alias")


# ROADMAP C10: one spelling per parameter and result field.

RENAMED_PARAMETERS = [
    (fk.credit.simulate_migrations, "n_sims", "n_paths"),
    (fk.credit.creditmetrics_portfolio, "n_sims", "n_paths"),
    (fk.credit.simulate_one_factor_defaults, "n_sims", "n_paths"),
    (fk.time_series.simulate_dickey_fuller, "n_sims", "n_paths"),
    (fk.time_series.mackinnon_critical_values, "nobs", "n_obs"),
    (fk.time_series.bartlett_band, "nobs", "n_obs"),
    (fk.exotic_options.longstaff_schwartz, "n_exercise", "n_steps"),
    (fk.credit.kmv_iterate, "rtol", "tol"),
    (fk.risk.monte_carlo_var, "ci_level", "confidence"),
    (fk.credit.cds_legs, "frequency", "freq"),
    (fk.credit.cds_par_spread, "frequency", "freq"),
    (fk.credit.cds_upfront, "frequency", "freq"),
    (fk.credit.upfront_to_spread, "frequency", "freq"),
    (fk.credit.bootstrap_hazard_curve, "frequency", "freq"),
    (fk.credit.tranche_spread, "frequency", "freq"),
    (fk.credit.base_correlation_spread, "frequency", "freq"),
    (fk.credit.base_correlation, "frequency", "freq"),
    (fk.credit.compound_correlations, "frequency", "freq"),
]


@pytest.mark.parametrize(("func", "old", "new"), RENAMED_PARAMETERS, ids=lambda x: getattr(x, "__name__", x))
def test_renamed_parameter_is_wired(func, old, new):
    """The signature shows the new name, and the old keyword is still recognised."""
    import inspect

    params = inspect.signature(func).parameters
    assert new in params and old not in params
    with pytest.raises(TypeError, match="pass only"):
        func(**{old: 1, new: 1})


def test_renamed_parameters_warn_and_forward():
    with pytest.warns(DeprecationWarning, match=r"n_obs="):
        assert fk.time_series.bartlett_band(nobs=100) == fk.time_series.bartlett_band(100)
    with pytest.warns(DeprecationWarning, match=r"n_paths="):
        old = fk.time_series.simulate_dickey_fuller(50, n_sims=10, rng=0)
    np.testing.assert_array_equal(old, fk.time_series.simulate_dickey_fuller(50, n_paths=10, rng=0))
    curve = fk.credit.HazardCurve.flat(0.02)
    with pytest.warns(DeprecationWarning, match=r"freq="):
        old = fk.credit.cds_par_spread(curve, 5.0, 0.03, frequency=4)
    assert old == fk.credit.cds_par_spread(curve, 5.0, 0.03, freq=4)
    with pytest.warns(DeprecationWarning, match=r"n_steps="):
        old = fk.exotic_options.longstaff_schwartz(36.0, 40.0, 1.0, 0.06, 0.2, n_exercise=10, n_paths=500, rng=1)
    new = fk.exotic_options.longstaff_schwartz(36.0, 40.0, 1.0, 0.06, 0.2, n_steps=10, n_paths=500, rng=1)
    assert old.estimate == new.estimate


def test_renamed_result_fields_keep_their_old_names():
    rng = np.random.default_rng(0)
    test = fk.time_series.ljung_box(rng.standard_normal(200), 5)
    stats = fk.time_series.return_statistics(rng.standard_normal(200))
    vr = fk.time_series.variance_ratio(np.cumsum(rng.standard_normal(200)), [2, 4])
    for res, old, new in [
        (test, "pvalue", "p_value"),
        (test, "nobs", "n_obs"),
        (stats, "nobs", "n_obs"),
        (stats, "jarque_bera_pvalue", "jarque_bera_p_value"),
        (vr, "pvalue", "p_value"),
        (vr, "pvalue_robust", "p_value_robust"),
    ]:
        with pytest.warns(DeprecationWarning, match=f"{type(res).__name__}.{new}"):
            np.testing.assert_array_equal(getattr(res, old), getattr(res, new))
    lcp = fk.pde_methods.brennan_schwartz(np.array([0.0, -1.0]), np.array([2.0, 2.0]), np.array([-1.0, 0.0]), np.array([1.0, 1.0]), np.zeros(2))
    with pytest.warns(DeprecationWarning, match="LCPSolution.n_iter"):
        assert lcp.iterations == lcp.n_iter == 1


@pytest.mark.parametrize("cls", [fk.stochastic_volatility.FourierCalibration, fk.interest_rate_models.ShortRateCalibration])
def test_calibration_success_is_converged(cls):
    assert "converged" in cls.__dataclass_fields__ and "success" not in cls.__dataclass_fields__
    assert cls.success.new == "converged"


def test_every_domain_exports_its_visualizers():
    import importlib

    for domain in fk.__all__:
        mod = getattr(fk, domain)
        if not hasattr(mod, "visualizers"):
            continue
        plots = importlib.import_module(f"financialkit.{domain}.visualizers")
        for name in getattr(plots, "__all__", [n for n in dir(plots) if n.startswith("plot_")]):
            assert name in mod.__all__ and getattr(mod, name) is getattr(plots, name), f"{domain}.{name}"
