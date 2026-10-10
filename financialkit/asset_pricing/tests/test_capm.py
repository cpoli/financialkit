"""Tests for the CAPM, Jensen's alpha, Black's zero-beta CAPM and the GRS test."""

import numpy as np
import pytest
from scipy import stats

from financialkit.asset_pricing import (
    capm_equilibrium,
    capm_expected_return,
    estimate_beta,
    grs_test,
    jensen_alpha,
    simulate_factor_model,
    zero_beta_capm,
)
from financialkit.constants import DEFAULT_ATOL

MU = np.array([0.05, 0.08, 0.11, 0.14])
VOL = np.array([0.10, 0.16, 0.22, 0.30])
CORR = np.array([[1.0, 0.3, 0.2, 0.1], [0.3, 1.0, 0.4, 0.3], [0.2, 0.4, 1.0, 0.5], [0.1, 0.3, 0.5, 1.0]])
COV = CORR * np.outer(VOL, VOL)


def test_capm_expected_return_broadcasts():
    np.testing.assert_allclose(capm_expected_return([0.0, 1.0, 2.0], 0.03, 0.08), [0.03, 0.08, 0.13])


def test_equilibrium_puts_every_asset_on_the_sml():
    eq = capm_equilibrium(MU, COV, 0.02)
    assert eq.market_weights.sum() == pytest.approx(1.0)
    assert eq.market_weights @ eq.betas == pytest.approx(1.0)
    np.testing.assert_allclose(eq.sml_errors, 0.0, atol=DEFAULT_ATOL)
    assert eq.market_premium == pytest.approx(eq.market_return - 0.02)


def test_market_maximises_the_sharpe_ratio():
    eq = capm_equilibrium(MU, COV, 0.02)
    best = (eq.market_return - 0.02) / eq.market_vol
    for w in np.random.default_rng(0).dirichlet(np.ones(4), 300):
        assert (w @ MU - 0.02) / np.sqrt(w @ COV @ w) <= best + 1e-12


def test_estimated_beta_and_jensen_alpha_recover_the_truth():
    panel = simulate_factor_model(5000, [0.5, 1.0, 1.5], [0.006], [[0.002]], idio_vol=0.03, alphas=[0.0, 0.002, -0.001], rng=1)
    b = estimate_beta(panel.excess_returns, panel.factors[:, 0])
    res = jensen_alpha(panel.excess_returns, panel.factors[:, 0])
    np.testing.assert_allclose(b, res.betas[:, 0])
    assert np.all(np.abs(res.alphas - panel.alphas) < 4 * res.alpha_se)
    assert np.all(np.abs(res.betas[:, 0] - [0.5, 1.0, 1.5]) < 0.05)


def test_zero_beta_portfolio():
    """Black (1972): z is uncorrelated with m, has the stated mean, and every asset lies on the line through it."""
    rz = 0.03
    res = zero_beta_capm(MU, COV, rz)
    assert res.zero_beta_weights.sum() == pytest.approx(1.0)
    assert res.zero_beta_weights @ COV @ res.market_weights == pytest.approx(0.0, abs=1e-14)
    assert res.zero_beta_rate == pytest.approx(rz)
    np.testing.assert_allclose(MU, rz + res.betas * (res.market_return - rz), atol=DEFAULT_ATOL)


def test_zero_beta_is_minimum_variance_among_zero_beta_portfolios():
    res = zero_beta_capm(MU, COV, 0.03)
    v = res.zero_beta_weights @ COV @ res.zero_beta_weights
    # Perturb along directions keeping both constraints.
    A = np.column_stack([np.ones(4), COV @ res.market_weights])
    null = np.linalg.svd(A.T)[2][2:]
    for d in null:
        for eps in (-0.1, 0.1):
            w = res.zero_beta_weights + eps * d
            assert w @ COV @ w >= v


def test_grs_single_asset_equals_scaled_t_squared():
    """With N = K = 1, GRS's F is the square of the OLS t-statistic of alpha (Gibbons-Ross-Shanken 1989, Sec. 3)."""
    panel = simulate_factor_model(120, [1.2], [0.006], [[0.002]], alphas=[0.004], rng=3)
    T = 120
    grs = grs_test(panel.excess_returns, panel.factors)
    t = jensen_alpha(panel.excess_returns, panel.factors[:, 0]).alpha_t[0]
    assert grs.statistic == pytest.approx(t**2)
    assert grs.p_value == pytest.approx(stats.f.sf(grs.statistic, 1, T - 2))


def test_grs_size_and_power():
    rng = np.random.default_rng(4)
    betas = np.linspace(0.5, 1.5, 10)
    p_null = np.array([grs_test(*_ef(simulate_factor_model(120, betas, [0.006], [[0.002]], rng=rng))).p_value for _ in range(400)])
    size = np.mean(p_null < 0.05)
    assert abs(size - 0.05) < 4 * np.sqrt(0.05 * 0.95 / 400)
    alt = simulate_factor_model(120, betas, [0.006], [[0.002]], alphas=0.01, rng=rng)
    assert grs_test(*_ef(alt)).reject()
    with pytest.raises(ValueError):
        grs_test(alt.excess_returns[:11], alt.factors[:11])


def _ef(panel):
    return panel.excess_returns, panel.factors
