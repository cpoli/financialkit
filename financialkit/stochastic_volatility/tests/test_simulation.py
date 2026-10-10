"""Tests for Heston simulation: truncated Euler schemes and Andersen's QE."""

import numpy as np
import pytest

from financialkit.stochastic_volatility import HESTON_SCHEMES, HestonModel, heston_mc_price, heston_paths

MILD = HestonModel(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
ANDERSEN_I = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)


@pytest.mark.parametrize("scheme", HESTON_SCHEMES)
def test_schemes_price_mild_heston(scheme):
    """With the Feller condition satisfied every scheme converges; 50 steps leave a bias below the MC error."""
    res = heston_mc_price(MILD, 100.0, 100.0, 1.0, 0.03, n_steps=50, n_paths=40_000, scheme=scheme, rng=1)
    assert abs(res.estimate - MILD.price(100.0, 100.0, 1.0, 0.03)) < 4 * res.std_error


@pytest.mark.parametrize("scheme", HESTON_SCHEMES)
def test_forward_is_a_martingale(scheme):
    paths = heston_paths(MILD, 100.0, 1.0, 0.05, 0.02, n_steps=20, n_paths=40_000, scheme=scheme, rng=2)
    ST = paths.spot[:, -1]
    assert abs(ST.mean() - 100 * np.exp(0.03)) < 4 * ST.std() / np.sqrt(ST.size)
    assert paths.spot.shape == paths.variance.shape == (40_000, 21)


def test_qe_variance_is_non_negative_and_moment_matched():
    paths = heston_paths(ANDERSEN_I, 100.0, 1.0, 0.0, n_steps=4, n_paths=100_000, scheme="qe", rng=3)
    assert paths.variance.min() >= 0.0
    vT = paths.variance[:, -1]
    assert abs(vT.mean() - ANDERSEN_I.variance_mean(1.0)) < 4 * vT.std() / np.sqrt(vT.size)


def test_qe_beats_full_truncation_on_andersen_case_i():
    """Andersen (2008), Case I with two steps per year over T = 10: QE's bias is a fraction of full
    truncation's (Andersen's Tables 3-4 and Lord, Koekkoek and van Dijk 2010 report the same ordering)."""
    exact = ANDERSEN_I.price(100.0, 100.0, 10.0, 0.0)
    qe = heston_mc_price(ANDERSEN_I, 100.0, 100.0, 10.0, 0.0, n_steps=20, n_paths=20_000, scheme="qe", rng=4)
    ft = heston_mc_price(ANDERSEN_I, 100.0, 100.0, 10.0, 0.0, n_steps=20, n_paths=20_000, scheme="full_truncation", rng=4)
    assert abs(qe.estimate - exact) + 3 * qe.std_error < abs(ft.estimate - exact) - 3 * ft.std_error


def test_full_truncation_has_smallest_euler_bias():
    """Lord, Koekkoek and van Dijk (2010), Table 2: full truncation beats partial truncation and Higham-Mao."""
    exact = ANDERSEN_I.price(100.0, 100.0, 10.0, 0.0)
    bias = {s: heston_mc_price(ANDERSEN_I, 100.0, 100.0, 10.0, 0.0, n_steps=40, n_paths=20_000, scheme=s, rng=5) for s in HESTON_SCHEMES[:3]}
    ft = bias["full_truncation"]
    for s in ("partial_truncation", "higham_mao"):
        assert abs(ft.estimate - exact) + 3 * ft.std_error < abs(bias[s].estimate - exact)


def test_invalid_scheme():
    with pytest.raises(ValueError):
        heston_paths(MILD, 100.0, 1.0, 0.0, scheme="exact")
