"""Tests for the ``paths=`` and ``discount=`` entry point of the exotic Monte Carlo pricers."""

import numpy as np
import pytest

from financialkit.exotic_options import arithmetic_asian_mc, barrier_option_mc, longstaff_schwartz, lookback_option_mc
from financialkit.exotic_options.utils.numerics import spot_paths
from financialkit.integrators import antithetic_increments, brownian_increments, gbm_paths

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.01
N_STEPS, N_PATHS = 20, 4_000


def _gbm(rng, n=N_STEPS, m=N_PATHS, dW=None):
    return gbm_paths(S, r - q, sigma, T, n, m, rng=rng, dW=dW)[1]


def _same(a, b):
    assert a.estimate == b.estimate and a.std_error == b.std_error and a.n_samples == b.n_samples


@pytest.mark.parametrize("antithetic", [True, False])
def test_longstaff_schwartz_gbm_paths_reproduce_todays_price(antithetic):
    m = N_PATHS
    dW = antithetic_increments(T, N_STEPS, m, rng=1) if antithetic else brownian_increments(T, N_STEPS, m, rng=1)
    today = longstaff_schwartz(S, K, T, r, sigma, q, n_steps=N_STEPS, n_paths=m, antithetic=antithetic, rng=1)
    on_paths = longstaff_schwartz(S, K, T, r, q=q, paths=_gbm(None, dW=dW))
    assert on_paths.estimate == today.estimate
    np.testing.assert_array_equal(on_paths.exercise_boundary, today.exercise_boundary)
    np.testing.assert_array_equal(on_paths.times, today.times)
    if not antithetic:  # supplied paths are treated as independent, so only then do the errors agree
        assert on_paths.std_error == today.std_error


@pytest.mark.parametrize("control_variate", [True, False])
def test_arithmetic_asian_gbm_paths_reproduce_todays_price(control_variate):
    today = arithmetic_asian_mc(S, K, T, r, sigma, q, n_fixings=12, n_paths=N_PATHS, control_variate=control_variate, rng=2)
    _same(arithmetic_asian_mc(S, K, T, r, sigma, q, paths=_gbm(2, n=12), control_variate=control_variate), today)


def test_arithmetic_asian_on_paths_skips_the_gbm_control_by_default():
    plain = arithmetic_asian_mc(S, K, T, r, sigma, q, n_fixings=12, n_paths=N_PATHS, control_variate=False, rng=2)
    _same(arithmetic_asian_mc(S, K, T, r, paths=_gbm(2, n=12)), plain)


@pytest.mark.parametrize(("H", "barrier_type", "continuous"), [(90.0, "down-and-out", True), (110.0, "up-and-in", False)])
def test_barrier_gbm_paths_reproduce_todays_price(H, barrier_type, continuous):
    today = barrier_option_mc(S, K, H, T, r, sigma, q, barrier_type=barrier_type, n_steps=N_STEPS, n_paths=N_PATHS, continuous=continuous, rng=3)
    sig = sigma if continuous else None
    _same(barrier_option_mc(S, K, H, T, r, sig, q, barrier_type=barrier_type, continuous=continuous, paths=_gbm(3)), today)


@pytest.mark.parametrize(("strike", "option_type", "continuous"), [(None, "call", True), (100.0, "put", False)])
def test_lookback_gbm_paths_reproduce_todays_price(strike, option_type, continuous):
    today = lookback_option_mc(S, T, r, sigma, q, option_type, strike, n_steps=N_STEPS, n_paths=N_PATHS, continuous=continuous, rng=4)
    gen = np.random.default_rng(4)  # the bridge draws continue the generator that made the paths
    paths = _gbm(gen)
    _same(lookback_option_mc(S, T, r, sigma, q, option_type, strike, continuous=continuous, paths=paths, rng=gen), today)


def test_discount_factors_replace_the_constant_rate():
    """D(0, t) = exp(-r t) reproduces the default; a 1% higher rate scales a European-style payoff by exp(-0.01 T)."""
    paths = _gbm(5)
    times = np.linspace(0.0, T, N_STEPS + 1)
    base = barrier_option_mc(S, K, 90.0, T, r, sigma, paths=paths)
    flat = barrier_option_mc(S, K, 90.0, T, r, sigma, paths=paths, discount=np.exp(-r * times))
    assert flat.estimate == pytest.approx(base.estimate, rel=1e-12)
    per_path = np.tile(np.exp(-(r + 0.01) * times), (N_PATHS, 1))
    for price in (
        lambda D: barrier_option_mc(S, K, 90.0, T, r, sigma, paths=paths, discount=D),
        lambda D: lookback_option_mc(S, T, r, paths=paths, continuous=False, discount=D),
        lambda D: arithmetic_asian_mc(S, K, T, r, paths=paths, discount=D),
    ):
        assert price(per_path).estimate == pytest.approx(price(None).estimate * np.exp(-0.01 * T), rel=1e-12)


def test_longstaff_schwartz_discounts_step_by_step():
    dW = brownian_increments(T, N_STEPS, N_PATHS, rng=6)
    paths = _gbm(None, dW=dW)
    times = np.linspace(0.0, T, N_STEPS + 1)
    base = longstaff_schwartz(S, K, T, r, paths=paths)
    flat = longstaff_schwartz(S, K, T, r, paths=paths, discount=np.exp(-r * times))
    rows = longstaff_schwartz(S, K, T, r, paths=paths, discount=np.tile(np.exp(-r * times), (N_PATHS, 1)))
    assert flat.estimate == pytest.approx(base.estimate, rel=1e-12)
    assert rows.estimate == pytest.approx(flat.estimate, rel=1e-12)


def test_spot_paths_validation():
    good = [[S, 101.0, 99.0]]
    with pytest.raises(ValueError, match="discount needs paths"):
        spot_paths(S, T, r, sigma, 2, 1, discount=[1.0, 0.99, 0.98])
    with pytest.raises(ValueError, match="sigma is required"):
        spot_paths(S, T, r, None, 2, 1)
    with pytest.raises(ValueError, match="shape"):
        spot_paths(S, T, r, None, 2, 1, paths=[S, 101.0])
    with pytest.raises(ValueError, match="start at the spot"):
        spot_paths(S, T, r, None, 2, 1, paths=[[99.0, 101.0, 99.0]])
    with pytest.raises(ValueError, match="does not broadcast"):
        spot_paths(S, T, r, None, 2, 1, paths=good, discount=[1.0, 0.99])
    with pytest.raises(ValueError, match="D\\(0, 0\\) = 1"):
        spot_paths(S, T, r, None, 2, 1, paths=good, discount=[0.99, 0.98, 0.97])


def test_pricers_need_sigma_only_where_the_model_needs_it():
    paths = _gbm(7)
    with pytest.raises(ValueError, match="sigma is required"):
        longstaff_schwartz(S, K, T, r)
    with pytest.raises(ValueError, match="needs sigma for the bridge"):
        barrier_option_mc(S, K, 90.0, T, r, paths=paths)
    with pytest.raises(ValueError, match="needs sigma for the bridge"):
        lookback_option_mc(S, T, r, paths=paths)
    with pytest.raises(ValueError, match="control variate needs sigma"):
        arithmetic_asian_mc(S, K, T, r, paths=paths, control_variate=True)
