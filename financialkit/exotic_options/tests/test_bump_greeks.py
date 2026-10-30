"""Barrier deltas, one bump_greek call each for the closed form and the Monte Carlo pricer."""

import numpy as np

from financialkit.exotic_options import barrier_option_mc, barrier_price
from financialkit.options import black_scholes_greeks, bump_greek

ARGS = (100.0, 100.0, 90.0, 1.0, 0.05, 0.25)


def test_barrier_deltas_satisfy_in_out_parity():
    """Knock-in + knock-out = vanilla with no rebate, so their deltas add up to the Black-Scholes delta."""
    out = bump_greek(barrier_price, "S", *ARGS, barrier_type="down-and-out")
    into = bump_greek(barrier_price, "S", *ARGS, barrier_type="down-and-in")
    assert abs(out + into - black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.25).delta) < 1e-8


def test_barrier_mc_delta_matches_the_closed_form():
    """Ten independent common-random-number estimates with the Brownian-bridge
    survival weight, which is smooth in S; their mean is within four standard
    errors of the Reiner-Rubinstein delta."""
    exact = bump_greek(barrier_price, "S", *ARGS, barrier_type="down-and-out")
    est = [bump_greek(barrier_option_mc, "S", *ARGS, n_steps=50, n_paths=20_000, rng=seed, h=1e-3, relative=True) for seed in range(10)]
    assert abs(np.mean(est) - exact) < 4 * np.std(est, ddof=1) / np.sqrt(len(est))
