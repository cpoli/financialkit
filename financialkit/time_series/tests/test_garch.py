"""Tests for ARCH, GARCH, GJR-GARCH, EGARCH, EWMA and the ARCH-LM test."""

import math

import numpy as np
import pytest

from financialkit.time_series import ARCH, EGARCH, EWMA, GARCH, GJRGARCH, VolatilityModel, arch_lm_test

MODELS = [
    ARCH(omega=0.5, alpha=0.4, mu=0.05),
    GARCH(omega=0.05, alpha=0.1, beta=0.85, mu=0.02),
    GJRGARCH(omega=0.05, alpha=0.03, gamma=0.12, beta=0.85, mu=0.02),
    EGARCH(omega=0.0, alpha=0.2, gamma=-0.1, beta=0.95, mu=0.02),
    EWMA(0.94),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_fast_variance_filters_match_generic_recursion(model):
    r = model.simulate(500, rng=0).returns
    np.testing.assert_allclose(model.variance(r), VolatilityModel.variance(model, r), rtol=1e-12)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_rescaling_is_exact(model):
    """Scaling returns by s shifts the Gaussian log-likelihood by -n ln s under the rescaled model."""
    r = model.simulate(300, rng=1).returns
    s = 0.01
    assert model._rescaled(s).loglikelihood(s * r) == pytest.approx(model.loglikelihood(r) - r.size * math.log(s), rel=1e-10)


def test_nested_models_coincide():
    r = GARCH(0.05, 0.1, 0.85).simulate(400, rng=2).returns
    g = GARCH(omega=0.05, alpha=0.1, beta=0.85, mu=0.01)
    np.testing.assert_allclose(GJRGARCH(0.05, 0.1, 0.0, 0.85, mu=0.01).variance(r), g.variance(r), rtol=1e-13)
    np.testing.assert_allclose(ARCH(0.05, 0.1, mu=0.01).variance(r), GARCH(0.05, 0.1, 0.0, mu=0.01).variance(r), rtol=1e-13)
    np.testing.assert_allclose(EWMA(0.9).variance(r), GARCH(0.0, 0.1, 0.9).variance(r), rtol=1e-13)


def test_garch_simulated_variance_matches_unconditional():
    """Sample variance of a long GARCH path versus omega / (1 - alpha - beta); the standard
    error comes from batch means, because squared returns are autocorrelated."""
    m = GARCH(omega=0.1, alpha=0.08, beta=0.82)
    r2 = m.simulate(400_000, rng=3).returns ** 2
    batches = r2.reshape(400, -1).mean(axis=1)
    assert abs(r2.mean() - m.unconditional_variance) < 4 * batches.std(ddof=1) / np.sqrt(batches.size)
    assert GARCH(0.1, 0.5, 0.5).unconditional_variance == math.inf


def test_egarch_unconditional_variance_closed_form_and_simulation():
    assert EGARCH(omega=-0.1, alpha=0.0, gamma=0.0, beta=0.9).unconditional_variance == pytest.approx(math.exp(-1.0))
    m = EGARCH(omega=-0.05, alpha=0.2, gamma=-0.1, beta=0.9)
    var = m.simulate(400_000, rng=4).variance
    batches = var.reshape(400, -1).mean(axis=1)
    assert abs(var.mean() - m.unconditional_variance) < 4 * batches.std(ddof=1) / np.sqrt(batches.size)


def test_garch_forecast_matches_monte_carlo():
    m = GARCH(omega=0.05, alpha=0.12, beta=0.83)
    r = m.simulate(1000, rng=5).returns
    f = m.forecast(r, 10)
    v_next = m.news_impact(r[-1], m.variance(r)[-1])
    assert f[0] == pytest.approx(v_next)
    # Simulate 20,000 continuations of ten periods from the same state.
    rng = np.random.default_rng(6)
    v = np.full(20_000, v_next)
    for _ in range(9):
        v = m.news_impact(np.sqrt(v) * rng.standard_normal(v.size), v)
    assert abs(v.mean() - f[-1]) < 4 * v.std() / np.sqrt(v.size)
    assert m.forecast(r, 2000)[-1] == pytest.approx(m.unconditional_variance, rel=1e-8)


def test_gjr_and_ewma_forecasts():
    m = GJRGARCH(omega=0.05, alpha=0.03, gamma=0.12, beta=0.85)
    r = m.simulate(500, rng=7).returns
    assert m.forecast(r, 3000)[-1] == pytest.approx(m.unconditional_variance, rel=1e-8)
    e = EWMA(0.94)
    f = e.forecast(r, 5)
    assert np.all(f == f[0]) and f[0] == pytest.approx(0.94 * e.variance(r)[-1] + 0.06 * r[-1] ** 2)
    assert ARCH(1.0, 0.6).kurtosis == math.inf


def test_garch_fit_recovers_parameters_within_standard_errors():
    true = GARCH(omega=2e-6, alpha=0.08, beta=0.9, mu=3e-4)
    r = true.simulate(5000, rng=8).returns
    fit = GARCH.fit(r)
    assert fit.converged and fit.nobs == 5000 and fit.n_params == 4
    for name, value in true.params.items():
        assert abs(fit.params[name] - value) < 4 * fit.std_errors[name], name
    assert fit.loglik >= true.loglikelihood(r)
    assert fit.aic < fit.bic


def test_gjr_and_egarch_fits_find_the_leverage_effect():
    r = GJRGARCH(omega=0.02, alpha=0.02, gamma=0.15, beta=0.88).simulate(5000, rng=9).returns
    gjr = GJRGARCH.fit(r)
    assert abs(gjr.params["gamma"] - 0.15) < 4 * gjr.std_errors["gamma"]
    egarch_true = EGARCH(omega=0.0, alpha=0.15, gamma=-0.1, beta=0.95)
    r = egarch_true.simulate(5000, rng=10).returns
    eg = EGARCH.fit(r)
    for name in ("alpha", "gamma", "beta"):
        assert abs(eg.params[name] - egarch_true.params[name]) < 4 * eg.std_errors[name], name


def test_arch_and_ewma_fits():
    r = ARCH(omega=0.5, alpha=0.5).simulate(4000, rng=11).returns
    fit = ARCH.fit(r)
    assert abs(fit.params["alpha"] - 0.5) < 4 * fit.std_errors["alpha"]
    lam = EWMA.fit(GARCH(0.01, 0.05, 0.94).simulate(3000, rng=12).returns).params["lam"]
    assert 0.85 < lam < 0.9999
    with pytest.raises(ValueError):
        GARCH.fit(np.zeros(10))


def test_arch_lm_size_and_power():
    rng = np.random.default_rng(13)
    reject = [arch_lm_test(rng.standard_normal(500), lags=5).rejects() for _ in range(400)]
    assert abs(np.mean(reject) - 0.05) < 4 * np.sqrt(0.05 * 0.95 / 400)
    res = arch_lm_test(ARCH(0.5, 0.5).simulate(2000, rng=14).returns, lags=1)
    assert res.df == 1 and res.pvalue < 1e-6


def test_simulate_path_properties_and_repr():
    path = GARCH(0.05, 0.1, 0.85).simulate(100, burn=10, rng=15)
    assert path.returns.shape == path.variance.shape == (100,)
    np.testing.assert_allclose(path.volatility**2, path.variance)
    assert repr(EWMA(0.94)) == "EWMA(lam=0.94)"
    z = GARCH(0.05, 0.1, 0.85).standardized_residuals(path.returns)
    assert z.shape == (100,)
