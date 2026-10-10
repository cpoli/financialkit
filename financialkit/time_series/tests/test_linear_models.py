"""Tests for Yule-Walker, partial autocorrelations, ARMA simulation and estimation, and order selection."""

import numpy as np
import pytest

from financialkit.time_series import (
    ar_cycle_period,
    arma_acf,
    arma_psi_weights,
    fit_arma,
    ljung_box,
    partial_autocorrelation,
    select_arma_order,
    simulate_arma,
    yule_walker,
)


def test_yule_walker_recovers_yules_sunspot_ar2():
    """Asymptotically Var(phi_hat_i) = (1 - phi_2^2) / n for an AR(2) (Box and Jenkins 1970, Sec. 7.2)."""
    phi = np.array([1.34, -0.65])
    n = 20_000
    res = yule_walker(simulate_arma(ar=phi, n=n, rng=0), 2)
    se = np.sqrt((1 - phi[1] ** 2) / n)
    assert np.all(np.abs(res.ar - phi) < 4 * se)
    assert abs(res.sigma2 - 1.0) < 4 * np.sqrt(2 / n)
    assert res.ma.size == 0


def test_ar_cycle_period_matches_ar2_formula():
    phi1, phi2 = 1.34, -0.65
    exact = 2 * np.pi / np.arccos(phi1 / (2 * np.sqrt(-phi2)))
    assert ar_cycle_period([phi1, phi2]) == pytest.approx(exact, rel=1e-12)
    assert ar_cycle_period([0.5, 0.2]) == np.inf


def test_pacf_is_last_yule_walker_coefficient():
    x = simulate_arma(ar=[0.5, 0.2], ma=[0.3], n=2000, rng=1)
    pacf = partial_autocorrelation(x, 5)
    for k in range(1, 6):
        assert pacf[k] == pytest.approx(yule_walker(x, k).ar[-1], rel=1e-9)


def test_psi_weights_and_acf_of_arma11_closed_form():
    """ARMA(1,1): psi_j = (phi + theta) phi^(j-1) and
    rho_1 = (1 + phi theta)(phi + theta) / (1 + 2 phi theta + theta^2), rho_k = phi rho_(k-1)
    (Box and Jenkins 1970, Sec. 3.4.3)."""
    phi, theta = 0.6, 0.3
    psi = arma_psi_weights([phi], [theta], 6)
    np.testing.assert_allclose(psi[1:], (phi + theta) * phi ** np.arange(5), rtol=1e-12)
    rho = arma_acf([phi], [theta], nlags=4)
    rho1 = (1 + phi * theta) * (phi + theta) / (1 + 2 * phi * theta + theta**2)
    np.testing.assert_allclose(rho[1:], rho1 * phi ** np.arange(4), rtol=1e-10)


def test_simulated_ma1_sample_acf_cuts_off():
    from financialkit.time_series import autocorrelation

    n = 50_000
    x = simulate_arma(ma=[0.5], n=n, rng=2)
    rho = autocorrelation(x, 3)
    assert abs(rho[1] - 0.5 / 1.25) < 4 / np.sqrt(n)
    assert np.all(np.abs(rho[2:]) < 4 * np.sqrt((1 + 2 * 0.4**2) / n))


def test_fit_ar1_equals_ols_and_forecast_is_exact():
    """Pure AR fits are conditional least squares; the AR(1) h-step forecast is
    mu + phi^h (x_T - mu) with variance sigma^2 (1 - phi^(2h)) / (1 - phi^2)."""
    x = simulate_arma(ar=[0.8], n=500, mean=2.0, rng=3)
    res = fit_arma(x, 1, 0)
    X = np.column_stack([np.ones(499), x[:-1]])
    c, phi = np.linalg.lstsq(X, x[1:], rcond=None)[0]
    assert res.ar[0] == pytest.approx(phi, rel=1e-10)
    assert res.mean == pytest.approx(c / (1 - phi), rel=1e-10)
    mean, se = res.forecast(5)
    h = np.arange(1, 6)
    np.testing.assert_allclose(mean, res.mean + phi**h * (x[-1] - res.mean), rtol=1e-10)
    np.testing.assert_allclose(se**2, res.sigma2 * (1 - phi ** (2 * h)) / (1 - phi**2), rtol=1e-10)


def test_fit_arma11_recovers_parameters():
    """Asymptotic variances (Box and Jenkins 1970, Sec. 7.2.6):
    Var(phi_hat) = (1 - phi^2)(1 + phi theta)^2 / (n (phi + theta)^2), and the same with
    1 - theta^2 for theta_hat."""
    phi, theta, n = 0.7, -0.3, 10_000
    x = simulate_arma(ar=[phi], ma=[theta], n=n, mean=0.5, rng=4)
    res = fit_arma(x, 1, 1)
    common = (1 + phi * theta) ** 2 / (n * (phi + theta) ** 2)
    assert abs(res.ar[0] - phi) < 4 * np.sqrt((1 - phi**2) * common)
    assert abs(res.ma[0] - theta) < 4 * np.sqrt((1 - theta**2) * common)
    assert abs(res.sigma2 - 1) < 4 * np.sqrt(2 / n)
    assert not ljung_box(res.residuals, 10, fitted_params=2).rejects(0.01)
    assert res.n_params == 4 and res.aic < res.bic


def test_ma1_forecast_reverts_to_mean_after_one_step():
    res = fit_arma(simulate_arma(ma=[0.6], n=2000, mean=1.0, rng=5), 0, 1)
    mean, se = res.forecast(4)
    np.testing.assert_allclose(mean[1:], res.mean, rtol=1e-12)
    np.testing.assert_allclose(se[1:] ** 2, res.sigma2 * (1 + res.ma[0] ** 2), rtol=1e-12)


def test_fit_arma_validation():
    x = simulate_arma(ar=[0.5], n=50, rng=6)
    with pytest.raises(ValueError):
        fit_arma(x, 2, 0, start=1)
    with pytest.raises(ValueError):
        fit_arma(x[:4], 2, 1)
    assert fit_arma(x, 0, 0, include_mean=False).ar.size == 0


def test_order_selection_bic_finds_true_order_and_loglik_is_nested():
    x = simulate_arma(ar=[0.5, -0.3], n=4000, rng=7)
    sel = select_arma_order(x, max_p=5, max_q=1)
    assert sel.best("bic") == (2, 0)
    ll = -0.5 * (sel.aic[:, 0] - 2 * (np.arange(6) + 2))
    assert np.all(np.diff(ll) >= -1e-8)  # nested AR models on a common sample
