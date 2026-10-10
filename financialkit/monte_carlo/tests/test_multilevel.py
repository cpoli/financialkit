"""Tests for financialkit.monte_carlo.systems.multilevel."""

import numpy as np
import pytest

from financialkit.monte_carlo import mlmc_level, multilevel_monte_carlo
from financialkit.options import black_scholes_price

ARGS = (100.0, 100.0, 1.0, 0.05, 0.2)


def test_level_variances_decay():
    v = [mlmc_level(l, 20_000, *ARGS, rng=l).var() for l in range(1, 6)]
    # Euler for GBM has strong order 1/2 but a Lipschitz payoff gives V_l ~ M^{-l} (Giles 2008, Sec. 4).
    slope = np.polyfit(np.arange(1, 6), np.log2(v), 1)[0]
    assert -1.4 < slope < -0.7


def test_level_zero_is_one_step_euler_payoff():
    y0 = mlmc_level(0, 200_000, *ARGS, rng=1)
    assert y0.min() == 0.0 and abs(y0.mean() - black_scholes_price(*ARGS)) < 0.5  # one Euler step: O(T) weak bias


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mlmc_reaches_target_accuracy(option_type):
    eps = 0.05
    res = multilevel_monte_carlo(*ARGS, option_type=option_type, eps=eps, rng=2)
    exact = black_scholes_price(*ARGS, option_type=option_type)
    # MSE = variance + bias^2 <= eps^2; allow 3 eps for the sampling tail.
    assert abs(res.estimate - exact) < 3 * eps
    assert res.std_error**2 <= 0.5 * eps**2 * 1.01
    assert res.cost < res.standard_mc_cost
    assert res.level_samples[0] > res.level_samples[-1]


def test_mlmc_rejects_bad_eps():
    with pytest.raises(ValueError):
        multilevel_monte_carlo(*ARGS, eps=0.0)
