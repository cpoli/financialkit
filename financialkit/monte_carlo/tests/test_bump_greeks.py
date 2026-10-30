"""Finite-difference Greeks of a Monte Carlo pricer on common random numbers, against pathwise Greeks."""

import pytest

from financialkit.monte_carlo import european_option_mc, pathwise_greeks
from financialkit.options import black_scholes_greeks, bump_greek

ARGS = (100.0, 105.0, 0.75, 0.04, 0.25)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_common_random_numbers_reproduce_pathwise_greeks(option_type):
    """On the same normals, the central difference of the sample mean tends to the
    pathwise estimator as h -> 0; they differ only on the few paths that end within
    the bump of the strike, far less than one standard error."""
    kw = {"q": 0.01, "option_type": option_type, "n_paths": 100_000, "rng": 3}
    pw = pathwise_greeks(*ARGS, **kw)
    delta = bump_greek(european_option_mc, "S", *ARGS, **kw, h=1e-4, relative=True)
    vega = bump_greek(european_option_mc, "sigma", *ARGS, **kw)
    assert abs(delta - pw.delta) < 0.05 * pw.delta_se
    assert abs(vega - pw.vega) < 0.05 * pw.vega_se


def test_generator_rng_is_shared_by_every_bump():
    pw = pathwise_greeks(*ARGS, n_paths=100_000, rng=0)
    exact = black_scholes_greeks(*ARGS).delta
    delta = bump_greek(european_option_mc, "S", *ARGS, n_paths=100_000, rng=None, h=1e-4, relative=True)
    # without common random numbers this would be noise of order S * SE(price) / h
    assert abs(delta - exact) < 4 * pw.delta_se
