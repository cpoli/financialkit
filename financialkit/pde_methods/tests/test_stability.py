"""Tests for the explicit stability limit, the diffusion number and von Neumann analysis."""

import numpy as np
import pytest

from financialkit.pde_methods import ExplicitScheme, diffusion_number, explicit_stability, log_explicit_price, von_neumann_amplification


def test_explicit_limit_separates_bounded_from_blowing_up():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    limit = explicit_stability(np.linspace(0.0, 400.0, 101), 1.0, sigma, r).limit
    n_stable = int(np.ceil(T / (0.97 * limit)))
    n_unstable = int(np.floor(T / (1.5 * limit)))
    stable = ExplicitScheme(100, n_stable, s_max=400.0).price(S, K, T, r, sigma)
    unstable = ExplicitScheme(100, n_unstable, s_max=400.0).price(S, K, T, r, sigma)
    assert stable.stability.stable and not unstable.stability.stable
    assert np.abs(stable.V).max() < 400.0
    assert np.abs(unstable.V).max() > 1e6


def test_explicit_limit_scales_like_the_square_of_the_spot_step():
    l1 = explicit_stability(np.linspace(0.0, 400.0, 101), 1.0, 0.2, 0.0).limit
    l2 = explicit_stability(np.linspace(0.0, 400.0, 201), 1.0, 0.2, 0.0).limit
    assert l1 / l2 == pytest.approx(4.0, rel=0.05)


def test_diffusion_number():
    res = diffusion_number(0.3, 0.01, 0.03)
    assert res.number == pytest.approx(1.0) and res.stable
    assert not diffusion_number(0.3, 0.012, 0.03).stable


@pytest.mark.parametrize("lam", [0.5, 1.0, 1.5, 3.0])
def test_von_neumann_explicit_limit_is_unit_diffusion_number(lam):
    sigma, dx = 0.2, 0.02
    dt = lam * dx**2 / sigma**2
    res = von_neumann_amplification(sigma, 0.0, dt, dx, theta=0.0, q=-0.5 * sigma**2)  # nu = 0: pure diffusion
    np.testing.assert_allclose(np.abs(res.g[-1]), abs(1.0 - 2.0 * lam), atol=1e-12)  # g(pi) = 1 - 2 lambda
    assert res.stable == (lam <= 1.0)


@pytest.mark.parametrize("theta", [0.5, 0.75, 1.0])
def test_von_neumann_implicit_schemes_are_unconditionally_stable(theta):
    for dt in (1e-4, 1e-2, 1.0):
        assert von_neumann_amplification(0.3, 0.05, dt, 0.01, theta=theta).stable


def test_crank_nicolson_damps_stiff_modes_less_than_implicit():
    cn = von_neumann_amplification(0.2, 0.0, 0.1, 0.01, theta=0.5)
    im = von_neumann_amplification(0.2, 0.0, 0.1, 0.01, theta=1.0)
    assert abs(cn.g[-1]) > 0.95 and cn.g[-1].real < 0  # g -> -1: undamped oscillation
    assert abs(im.g[-1]) < 0.05
    with pytest.raises(ValueError):
        von_neumann_amplification(0.2, 0.0, 0.1, 0.01, theta=-0.1)


def test_log_explicit_scheme_blows_up_beyond_unit_diffusion_number():
    res = log_explicit_price(100.0, 100.0, 1.0, 0.05, 0.2, n_time=200, dx=0.2 * np.sqrt(0.6 / 200))
    assert not res.stability.stable
    assert not np.isfinite(res.price) or abs(res.price) > 1e6
