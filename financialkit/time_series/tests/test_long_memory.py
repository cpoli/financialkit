"""Tests for rescaled-range analysis and fractional Gaussian noise."""

import numpy as np
import pytest

from financialkit.time_series import anis_lloyd_expected_rs, fgn_autocovariance, hurst_exponent, rescaled_range, simulate_fgn


@pytest.mark.parametrize("n", [10, 40])
def test_anis_lloyd_expectation_matches_simulation(n):
    rng = np.random.default_rng(n)
    rs = np.array([rescaled_range(rng.standard_normal(n)) for _ in range(20_000)])
    assert abs(rs.mean() - anis_lloyd_expected_rs(n)) < 4 * rs.std() / np.sqrt(rs.size)


def test_anis_lloyd_vectorizes_and_validates():
    np.testing.assert_allclose(anis_lloyd_expected_rs([10, 40]), [anis_lloyd_expected_rs(10), anis_lloyd_expected_rs(40)])
    with pytest.raises(ValueError):
        anis_lloyd_expected_rs(2)


def test_fgn_covariance_matches_theory():
    """Lag-0, 1 and 5 sample covariances over independent replications agree with
    gamma(k) = (|k+1|^2H - 2|k|^2H + |k-1|^2H) / 2 within four standard errors."""
    H, n_rep = 0.75, 4000
    rng = np.random.default_rng(0)
    X = np.array([simulate_fgn(64, H, rng=rng) for _ in range(n_rep)])
    gamma = fgn_autocovariance(H, 5)
    for k in (0, 1, 5):
        prod = X[:, 10] * X[:, 10 + k]
        assert abs(prod.mean() - gamma[k]) < 4 * prod.std() / np.sqrt(n_rep)
    with pytest.raises(ValueError):
        fgn_autocovariance(1.0, 3)


def test_hurst_corrected_is_one_half_for_independent_returns():
    rng = np.random.default_rng(1)
    est = np.array([hurst_exponent(rng.standard_normal(4096)).hurst_corrected for _ in range(30)])
    assert abs(est.mean() - 0.5) < 4 * est.std(ddof=1) / np.sqrt(est.size)
    raw = hurst_exponent(rng.standard_normal(4096)).hurst
    assert raw > 0.5  # the uncorrected slope is biased upward in small windows


def test_hurst_orders_persistent_and_antipersistent_noise():
    rng = np.random.default_rng(2)
    h = {H: hurst_exponent(simulate_fgn(8192, H, rng=rng)).hurst_corrected for H in (0.3, 0.5, 0.8)}
    assert h[0.3] < 0.4 < h[0.5] < 0.6 < h[0.8]
    with pytest.raises(ValueError):
        hurst_exponent(np.ones(20))
