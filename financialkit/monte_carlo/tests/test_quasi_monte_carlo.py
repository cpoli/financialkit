"""Tests for financialkit.monte_carlo.systems.quasi_monte_carlo."""

import itertools

import numpy as np
import pytest

from financialkit.monte_carlo import (
    gbm_path_estimate,
    halton,
    koksma_hlawka_bound,
    l2_star_discrepancy,
    randomized_qmc_estimate,
    sobol,
    star_discrepancy,
    van_der_corput,
)
from financialkit.options import black_scholes_price


def brute_star_discrepancy(x):
    """D* by checking every corner from the coordinate grid, open and closed boxes."""
    best = 0.0
    for a, b in itertools.product(np.append(x[:, 0], 1.0), np.append(x[:, 1], 1.0)):
        o = np.mean((x[:, 0] < a) & (x[:, 1] < b))
        c = np.mean((x[:, 0] <= a) & (x[:, 1] <= b))
        best = max(best, a * b - o, c - a * b)
    return best


def test_van_der_corput_base_3():
    np.testing.assert_allclose(van_der_corput(5, base=3), [1 / 3, 2 / 3, 1 / 9, 4 / 9, 7 / 9])
    with pytest.raises(ValueError):
        van_der_corput(3, base=1)


def test_halton_matches_van_der_corput_per_coordinate():
    x = halton(16, 2)
    np.testing.assert_allclose(x[:, 0], van_der_corput(16, 2, start=0))
    np.testing.assert_allclose(x[:, 1], van_der_corput(16, 3, start=0))


def test_sobol_is_a_net():
    x = sobol(64, 2, scramble=False)
    # (0, m, 2)-net in base 2: every 8x8 cell holds exactly one point.
    counts = np.histogram2d(x[:, 0], x[:, 1], bins=8, range=[[0, 1], [0, 1]])[0]
    assert np.all(counts == 1)


def test_star_discrepancy_1d_closed_form():
    N = 16
    assert star_discrepancy((np.arange(N) + 0.5) / N) == pytest.approx(0.5 / N)
    assert star_discrepancy(np.zeros(4)) == pytest.approx(1.0)


def test_star_discrepancy_2d_matches_brute_force():
    for seed in range(3):
        x = np.random.default_rng(seed).random((30, 2))
        assert star_discrepancy(x) == pytest.approx(brute_star_discrepancy(x))
    with pytest.raises(ValueError):
        star_discrepancy(np.random.default_rng(0).random((5, 3)))


def test_low_discrepancy_beats_random():
    n = 512
    assert star_discrepancy(sobol(n, 2, rng=0)) < 0.5 * star_discrepancy(np.random.default_rng(0).random((n, 2)))
    assert l2_star_discrepancy(halton(n, 5, scramble=True, rng=0)) < l2_star_discrepancy(np.random.default_rng(0).random((n, 5)))
    assert l2_star_discrepancy(sobol(n, 2, rng=1)) <= star_discrepancy(sobol(n, 2, rng=1))


def test_koksma_hlawka_bound_holds():
    """f(x, y) = x y has Hardy-Krause variation 3 and integral 1/4."""
    for n in (64, 256, 1024):
        for x in (sobol(n, 2, rng=n), halton(n, 2, scramble=True, rng=n), np.random.default_rng(n).random((n, 2))):
            assert abs(np.mean(x[:, 0] * x[:, 1]) - 0.25) <= koksma_hlawka_bound(x, 3.0)


@pytest.mark.parametrize("sequence", ["sobol", "halton"])
def test_randomized_qmc_unbiased_and_accurate(sequence):
    f = lambda u: np.prod(1 + (u - 0.5), axis=1)  # noqa: E731  integral 1
    res = randomized_qmc_estimate(f, 1024, dim=4, n_replications=16, sequence=sequence, rng=2)
    assert abs(res.estimate - 1.0) < 4 * res.std_error
    mc_se = np.sqrt((13 / 12) ** 4 - 1) / np.sqrt(res.n_samples)
    assert res.std_error < 0.1 * mc_se


@pytest.mark.parametrize("sampler", ["pseudo", "sobol", "bridge"])
def test_gbm_path_estimate_european(sampler):
    exact = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    res = gbm_path_estimate(lambda p: np.maximum(p[:, -1] - 100.0, 0.0), 100.0, 1.0, 0.05, 0.2, n_steps=16, n_paths=1024, sampler=sampler, rng=3)
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_bridge_beats_plain_sobol_in_high_dimension():
    asian = lambda p: np.maximum(p[:, 1:].mean(axis=1) - 100.0, 0.0)  # noqa: E731
    se = {s: gbm_path_estimate(asian, 100.0, 1.0, 0.05, 0.2, 64, 1024, s, n_replications=16, rng=4).std_error for s in ("pseudo", "sobol", "bridge")}
    assert se["bridge"] < se["sobol"] < se["pseudo"]
    with pytest.raises(ValueError):
        gbm_path_estimate(asian, 100.0, 1.0, 0.05, 0.2, 4, 16, "halton")
