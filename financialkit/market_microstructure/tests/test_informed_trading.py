"""Tests for PIN and VPIN."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_ATOL
from financialkit.market_microstructure import bulk_volume_classification, fit_pin, pin_loglik, simulate_pin, vpin


def test_pin_loglik_matches_direct_mixture():
    """Easley, Kiefer, O'Hara and Paperman (1996), eq. (2)."""
    a, d, mu, eb, es = 0.3, 0.6, 20.0, 15.0, 12.0
    B, S = np.array([14, 40, 10]), np.array([33, 11, 13])
    pb = ps = stats.poisson.pmf
    like = (1 - a) * pb(B, eb) * ps(S, es) + a * d * pb(B, eb) * ps(S, es + mu) + a * (1 - d) * pb(B, eb + mu) * ps(S, es)
    assert pin_loglik([a, d, mu, eb, es], B, S) == pytest.approx(np.log(like).sum(), rel=1e-12)


def test_fit_pin_recovers_parameters():
    true = dict(alpha=0.4, delta=0.5, mu=40.0, eps_b=30.0, eps_s=30.0)
    B, S = simulate_pin(2000, **true, rng=0)
    res = fit_pin(B, S)
    pin = 0.4 * 40 / (0.4 * 40 + 60)
    assert abs(res.pin - pin) < 4 * res.pin_se
    assert res.loglik >= pin_loglik(list(true.values()), B, S)


def test_bulk_volume_classification():
    buy = bulk_volume_classification([-1.0, 0.0, 2.0], [10.0, 10.0, 10.0], sigma=1.0)
    np.testing.assert_allclose(buy, 10 * stats.norm.cdf([-1.0, 0.0, 2.0]))


def test_vpin_buckets_conserve_volume():
    rng = np.random.default_rng(1)
    dp = rng.standard_normal(500)
    v = rng.integers(50, 150, 500).astype(float)
    res = vpin(dp, v, bucket_volume=1000.0, window=10)
    n = int(v.sum() // 1000)
    assert res.vpin.size == n and np.all(np.isnan(res.vpin[:9])) and not np.isnan(res.vpin[9])
    np.testing.assert_allclose(res.buy_volume + res.sell_volume, 1000.0)
    assert res.buy_volume.sum() == pytest.approx(bulk_volume_classification(dp, v)[: res.bucket_end[-1] + 1].sum(), abs=1000)
    np.testing.assert_allclose(res.vpin[9], res.imbalance[:10].mean())


def test_vpin_extremes_and_straddling():
    # Two bars of 150: bucket 1 is the first bar plus half the second.
    res = vpin([5.0, -5.0], [150.0, 150.0], bucket_volume=100.0, window=1, sigma=1.0)
    np.testing.assert_allclose(res.buy_volume, [100.0, 50.0, 0.0], atol=1e-4)  # Phi(5) = 1 - 2.9e-7
    assert res.bucket_end.tolist() == [0, 1, 1]
    flat = vpin(np.zeros(20), np.full(20, 10.0), bucket_volume=50.0, window=2, sigma=1.0)
    np.testing.assert_allclose(flat.vpin[1:], 0.0, atol=DEFAULT_ATOL)
    with pytest.raises(ValueError):
        vpin([0.0], [1.0], bucket_volume=10.0)
