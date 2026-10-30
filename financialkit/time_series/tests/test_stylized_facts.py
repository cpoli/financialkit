"""Tests for returns, return statistics, tail survival and the Hill estimator."""

import numpy as np
import pytest

from financialkit.time_series import hill_estimator, log_returns, return_statistics, simple_returns, tail_survival


def test_log_and_simple_returns_agree():
    p = np.array([100.0, 103.0, 98.0, 101.5])
    np.testing.assert_allclose(np.exp(log_returns(p)) - 1, simple_returns(p), rtol=1e-12)
    with pytest.raises(ValueError):
        log_returns([1.0, 0.0])


def test_return_statistics_of_normal_sample():
    """Skewness and excess kurtosis have asymptotic standard errors sqrt(6/n) and sqrt(24/n)."""
    n = 50_000
    s = return_statistics(np.random.default_rng(0).normal(0.001, 0.02, n))
    assert s.n_obs == n
    assert abs(s.mean - 0.001) < 4 * 0.02 / np.sqrt(n)
    assert abs(s.skewness) < 4 * np.sqrt(6 / n)
    assert abs(s.excess_kurtosis) < 4 * np.sqrt(24 / n)


def test_return_statistics_detects_fat_tails():
    """A Student t with 5 degrees of freedom has excess kurtosis 6."""
    s = return_statistics(np.random.default_rng(1).standard_t(5, 200_000))
    assert s.excess_kurtosis > 3
    assert s.jarque_bera_p_value < 1e-10


def test_tail_survival_is_decreasing_with_last_point_one_over_n():
    u, p = tail_survival(np.random.default_rng(2).standard_normal(1000))
    assert np.all(np.diff(u) >= 0) and np.all(np.diff(p) < 0)
    assert p[-1] == pytest.approx(1 / 1000)
    u_up, _ = tail_survival(np.array([-2.0, -1.0, 1.0, 3.0]), side="upper")
    np.testing.assert_array_equal(u_up, [1.0, 3.0])


@pytest.mark.parametrize("alpha", [1.5, 3.0])
def test_hill_recovers_pareto_index(alpha):
    """For an exact Pareto tail the Hill estimator is the MLE, with s.e. alpha / sqrt(k)."""
    x = np.random.default_rng(3).pareto(alpha, 200_000) + 1.0
    res = hill_estimator(x, k=[500, 2000, 10_000])
    assert np.all(np.abs(res.alpha - alpha) < 4 * res.std_error)


def test_hill_lower_tail_and_validation():
    x = -(np.random.default_rng(4).pareto(2.5, 100_000) + 1.0)
    res = hill_estimator(x, k=5000, side="lower")
    assert abs(res.alpha[0] - 2.5) < 4 * res.std_error[0]
    with pytest.raises(ValueError):
        hill_estimator(np.arange(1.0, 11.0), k=10)
    default = hill_estimator(np.arange(1.0, 101.0))
    assert default.k[0] == 2 and default.k[-1] == 24
