"""Tests for the Kupiec, Christoffersen and Basel traffic-light backtests."""

import numpy as np
import pytest
from scipy import stats

from financialkit.risk import basel_traffic_light, christoffersen_independence, christoffersen_test, kupiec_test, var_exceptions


def hits(x, n):
    h = np.zeros(n, dtype=bool)
    h[np.linspace(0, n - 1, x).astype(int)] = True
    return h


def test_kupiec_statistic_by_hand():
    res = kupiec_test(hits(12, 250))
    p_hat = 12 / 250
    lr = 2 * (12 * np.log(p_hat / 0.01) + 238 * np.log((1 - p_hat) / 0.99))
    assert res.statistic == pytest.approx(lr)
    assert res.expected == pytest.approx(2.5) and res.n_exceptions == 12


def test_kupiec_non_rejection_region():
    """Kupiec (1995), Table 3 (as reproduced in Jorion, *Value at Risk*, 3rd ed., Table 6-2): 99% VaR, T = 1000, 4 < N < 17."""
    accept = [x for x in range(30) if not kupiec_test(hits(x, 1000)).reject()]
    assert (accept[0], accept[-1]) == (5, 16)


def test_kupiec_with_no_exceptions_is_finite():
    assert np.isfinite(kupiec_test(np.zeros(250, dtype=bool)).statistic)


def test_christoffersen_detects_clustering():
    spread = hits(10, 1000)
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True
    assert not christoffersen_independence(spread).reject()
    assert christoffersen_independence(clustered).reject(0.001)
    # Same count, so the same unconditional statistic; conditional coverage adds the clustering.
    assert kupiec_test(clustered).statistic == pytest.approx(kupiec_test(spread).statistic)
    cc = christoffersen_test(clustered)
    assert cc.df == 2 and cc.statistic == pytest.approx(kupiec_test(clustered).statistic + christoffersen_independence(clustered).statistic)


def test_christoffersen_size_under_the_null():
    """Under a correct model, the 5% test rejects about 5% of the time (binomial 4-sigma band)."""
    rng = np.random.default_rng(0)
    m = 400
    rejections = sum(christoffersen_independence(rng.random(1000) < 0.05, 0.95).reject() for _ in range(m))
    assert abs(rejections / m - 0.05) < 4 * np.sqrt(0.05 * 0.95 / m) + 0.02  # + chi2 small-sample bias


def test_basel_zones():
    """Basel Committee (1996), Table 2: 0-4 green, 5-9 yellow with plus factors 0.40-0.85, 10+ red."""
    zones = [basel_traffic_light(k).zone for k in range(12)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 2
    assert [basel_traffic_light(k).multiplier for k in (4, 5, 7, 9, 10)] == [3.0, 3.4, 3.65, 3.85, 4.0]
    assert basel_traffic_light(4).cumulative_probability == pytest.approx(stats.binom.cdf(4, 250, 0.01))
    assert np.isnan(basel_traffic_light(3, n_obs=500).multiplier)


def test_var_exceptions():
    np.testing.assert_array_equal(var_exceptions([-3.0, -1.0, 2.0], [2.0, 2.0, 2.0]), [True, False, False])
