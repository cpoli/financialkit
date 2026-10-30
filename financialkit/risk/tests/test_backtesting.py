"""Tests for the Kupiec, Christoffersen, Basel traffic-light and Acerbi-Szekely backtests."""

import numpy as np
import pytest
from scipy import stats

from financialkit.risk import (
    acerbi_szekely_test,
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    christoffersen_test,
    kupiec_test,
    parametric_es,
    parametric_var,
    var_exceptions,
)


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


def test_backtests_reject_an_empty_hit_series():
    with pytest.raises(ValueError, match="empty"):
        kupiec_test([])


VAR975, ES975 = parametric_var(0.0, 1.0, 0.975), parametric_es(0.0, 1.0, 0.975)


def test_acerbi_szekely_statistics_by_hand():
    x = np.array([-3.0, 0.5, -2.0, 1.0])
    # Exceptions on days 0 and 2 (VaR 1.5): X I / ES = -1.5 and -1.0.
    assert acerbi_szekely_z1(x, 1.5, 2.0) == pytest.approx(-0.25)
    assert acerbi_szekely_z2(x, 1.5, 2.0, alpha=0.5) == pytest.approx(-2.5 / 2 + 1)
    assert np.isnan(acerbi_szekely_z1(np.ones(4), 1.5, 2.0))
    with pytest.raises(ValueError, match="positive"):
        acerbi_szekely_z2(x, 1.5, 0.0)
    with pytest.raises(ValueError, match="empty"):
        acerbi_szekely_z1([], 1.5, 2.0)


def test_acerbi_szekely_z2_critical_values():
    """Acerbi and Szekely (2014): Z2 has mean 0 and, for 97.5% ES over 250 days, a 5% critical value of about -0.70."""
    sims = np.random.default_rng(0).standard_normal((20_000, 250))
    z = acerbi_szekely_z2(sims, VAR975, ES975)
    assert abs(z.mean()) < 4 * z.std() / np.sqrt(z.size)
    # The 5% quantile of 20,000 draws has a standard error of about 0.004.
    assert np.quantile(z, 0.05) == pytest.approx(-0.70, abs=0.02)
    assert np.nanmean(acerbi_szekely_z1(sims, VAR975, ES975)) == pytest.approx(0.0, abs=0.01)


def test_acerbi_szekely_test_p_values():
    rng = np.random.default_rng(1)
    sims = rng.standard_normal((5000, 250))
    good = acerbi_szekely_test(rng.standard_normal(250), VAR975, ES975, simulated=sims)
    assert 0.0 < good.p_value <= 1.0 and good.n_obs == 250 and good.expected == pytest.approx(6.25)
    # Volatility understated by a third: the tail is far heavier than forecast.
    bad = acerbi_szekely_test(1.5 * rng.standard_normal(250), VAR975, ES975, simulated=sims)
    assert bad.reject(0.01) and bad.statistic < -1.0
    for test in ("z1", "z2"):
        res = acerbi_szekely_test(1.5 * rng.standard_normal(250), VAR975, ES975, test=test, simulated=sims)
        assert res.test == test and res.statistic < 0
    assert np.isnan(acerbi_szekely_test(np.zeros(250), VAR975, ES975).p_value)
    with pytest.raises(ValueError, match="test"):
        acerbi_szekely_test(np.zeros(250), VAR975, ES975, test="z3")
    with pytest.raises(ValueError, match="shape"):
        acerbi_szekely_test(np.zeros(250), VAR975, ES975, simulated=sims[:, :10])
