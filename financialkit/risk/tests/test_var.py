"""Tests for historical, parametric, Cornish-Fisher, Monte Carlo, hybrid and filtered VaR."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_ATOL, DEFAULT_RTOL
from financialkit.risk import (
    cornish_fisher_var,
    delta_normal_var,
    filtered_historical_var,
    historical_var,
    hybrid_var,
    monte_carlo_var,
    parametric_es,
    parametric_var,
    rolling_historical_var,
    weighted_tail,
)

HULL_COV = np.array([[0.02**2, 0.3 * 0.02 * 0.01], [0.3 * 0.02 * 0.01, 0.01**2]])


def test_historical_var_is_an_order_statistic_and_es_dominates():
    x = np.random.default_rng(0).standard_normal(1000)
    res = historical_var(x, 0.99)
    assert res.var == pytest.approx(-np.sort(x)[9])
    assert res.es >= res.var
    assert res.weights.sum() == pytest.approx(1.0)


def test_rolling_var_matches_one_window_at_a_time():
    x = np.random.default_rng(1).standard_normal(400)
    roll = rolling_historical_var(x, window=250, alpha=0.99)
    assert roll.shape == (150,)
    for t in (0, 77, 149):
        assert roll[t] == historical_var(x[t : t + 250], 0.99).var


def test_parametric_var_hull_microsoft():
    """Hull, 10th ed., Ch. 22: $10m at 2% daily vol, ten-day 99% VaR of $1.47m (= sqrt(10) x one-day)."""
    one_day = parametric_var(0.0, 200_000.0, 0.99)
    assert one_day == pytest.approx(465_300, rel=1e-4)
    assert parametric_var(0.0, 200_000.0, 0.99, horizon=10) == pytest.approx(np.sqrt(10) * one_day, rel=DEFAULT_RTOL)


def test_parametric_var_t_and_broadcasting():
    alpha = np.array([0.95, 0.99, 0.999])
    v_n = parametric_var(0.001, 0.02, alpha)
    v_t = parametric_var(0.001, 0.02, alpha, distribution="t", df=4)
    assert v_n.shape == (3,)
    # Same variance, fatter tails: the t VaR overtakes the normal far out.
    assert v_t[-1] > v_n[-1]
    with pytest.raises(ValueError):
        parametric_var(0.0, 1.0, 0.99, distribution="t", df=2)


def test_delta_normal_var_hull_two_assets():
    """Hull, 10th ed., Ch. 22: sigma_P = 220,227 a day, ten-day 99% VaR of about $1.62 million."""
    v = delta_normal_var([10e6, 5e6], HULL_COV, 0.99, horizon=10)
    assert v / (stats.norm.ppf(0.99) * np.sqrt(10)) == pytest.approx(220_227, rel=1e-5)
    assert v == pytest.approx(1.62e6, rel=1e-3)


def test_cornish_fisher_reduces_to_normal_and_beats_it_on_skewed_data():
    np.testing.assert_allclose(cornish_fisher_var(0.0, 1.0, 0.0, 0.0, [0.95, 0.99]), stats.norm.ppf([0.95, 0.99]), rtol=DEFAULT_RTOL)
    # A loss-skewed P&L: minus a gamma(8) variable. The CF quantile is closer to the truth.
    dist = stats.gamma(8.0)
    mu, var, skew, kurt = (float(m) for m in dist.stats(moments="mvsk"))
    exact = dist.ppf(0.99) - mu
    cf = cornish_fisher_var(0.0, np.sqrt(var), -skew, kurt, 0.99)
    normal = parametric_var(0.0, np.sqrt(var), 0.99)
    assert abs(cf - exact) < 0.1 * abs(normal - exact)


def test_monte_carlo_var_of_a_linear_portfolio_matches_delta_normal():
    w = np.array([10e6, 5e6])
    res = monte_carlo_var(lambda dX: dX @ w, HULL_COV, 0.99, n_paths=200_000, ci_level=0.9999, rng=3)
    exact = delta_normal_var(w, HULL_COV, 0.99)
    assert res.ci[0] < exact < res.ci[1]
    assert res.ci[0] < res.var < res.ci[1]
    assert res.es == pytest.approx(parametric_es(0.0, exact / stats.norm.ppf(0.99), 0.99), rel=0.02)  # SE of the tail mean is ~0.5%


def test_monte_carlo_var_checks_pnl_shape():
    with pytest.raises(ValueError):
        monte_carlo_var(lambda dX: dX, np.eye(2), n_paths=100, rng=0)


def test_hybrid_with_lam_one_is_historical_and_weights_decay():
    x = np.random.default_rng(4).standard_normal(500)
    h = historical_var(x, 0.99)
    b = hybrid_var(x, 0.99, lam=1.0)
    assert (b.var, b.es) == pytest.approx((h.var, h.es), rel=DEFAULT_RTOL, abs=DEFAULT_ATOL)
    w = hybrid_var(x, 0.99, lam=0.98).weights
    assert w[-1] / w[-2] == pytest.approx(1 / 0.98) and w.sum() == pytest.approx(1.0)


def test_hybrid_reacts_to_a_volatility_jump():
    rng = np.random.default_rng(5)
    x = np.concatenate([0.01 * rng.standard_normal(400), 0.03 * rng.standard_normal(50)])
    assert hybrid_var(x, 0.99, 0.97).var > 1.25 * historical_var(x, 0.99).var


def test_filtered_historical_scales_standardized_residuals():
    rng = np.random.default_rng(6)
    sigma = np.linspace(0.01, 0.04, 1000)
    z = rng.standard_normal(1000)
    res = filtered_historical_var(sigma * z, sigma, 0.02, 0.99)
    assert res.var == pytest.approx(0.02 * weighted_tail(z, 0.99)[0])
    with pytest.raises(ValueError):
        filtered_historical_var(z, -sigma, 0.02)


def test_alpha_is_validated():
    with pytest.raises(ValueError):
        historical_var([1.0, 2.0], 1.0)
