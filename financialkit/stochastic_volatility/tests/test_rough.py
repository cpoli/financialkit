"""Tests for the Volterra kernels, rough Bergomi, fractional Brownian motion and the roughness estimate."""

import numpy as np
import pytest
from scipy.integrate import quad

from financialkit.options import black_scholes_price
from financialkit.stochastic_volatility import (
    RoughBergomiModel,
    fbm_paths,
    hybrid_scheme_variance,
    rough_bergomi_mc_price,
    rough_bergomi_paths,
    volatility_roughness,
    volterra_covariance,
    volterra_paths,
)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3, 0.7])
def test_volterra_covariance_matches_the_kernel_integrals(H):
    """Bayer-Friz-Gatheral's hypergeometric form against quadrature of 2H int (u-s)^a (v-s)^a ds."""
    u, v = 0.3, 1.0
    C = volterra_covariance(H, [u, v])
    a = H - 0.5
    exact = 2 * H * quad(lambda s: (u - s) ** a * (v - s) ** a, 0, u, limit=200)[0]
    assert C[0, 1] == pytest.approx(exact, rel=1e-8)
    cross = np.sqrt(2 * H) * quad(lambda s: (v - s) ** a, 0, u)[0]
    assert C[1, 2] == pytest.approx(cross, rel=1e-8)
    np.testing.assert_allclose(np.diag(C), [u ** (2 * H), v ** (2 * H), u, v], rtol=1e-12)
    assert C[2, 3] == u


def test_volterra_covariance_is_positive_definite_for_rough_H():
    C = volterra_covariance(0.05, np.linspace(0.01, 1.0, 100))
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > 0


def test_hybrid_scheme_variance_is_close_to_exact_and_riemann_is_not():
    """Bennedsen et al. (2017), Section 3: the hybrid scheme fixes the Riemann sum's bias at small H."""
    H, T, n = 0.07, 1.0, 100
    t = np.linspace(0, T, n + 1)
    hyb = hybrid_scheme_variance(H, T, n)
    riem = hybrid_scheme_variance(H, T, n, "riemann")
    assert np.max(np.abs(hyb[1:] / t[1:] ** (2 * H) - 1)) < 0.01
    assert riem[-1] < 0.6
    # Near the singularity the first step is integrated exactly.
    assert hyb[1] == pytest.approx((T / n) ** (2 * H), rel=1e-12)


def test_brownian_case_is_exact_for_every_scheme():
    """With H = 1/2 the kernel is 1 and the Volterra process is the Brownian motion itself.

    The Cholesky scheme is left out: its joint covariance is then singular.
    """
    for scheme in ("hybrid", "riemann"):
        _, X, dW = volterra_paths(0.5, 1.0, 20, 50, scheme, rng=3)
        np.testing.assert_allclose(X[:, 1:], np.cumsum(dW, axis=1), atol=1e-12)
    np.testing.assert_allclose(hybrid_scheme_variance(0.5, 2.0, 10), np.linspace(0, 2, 11), rtol=1e-13)


@pytest.mark.parametrize("scheme", ["hybrid", "cholesky"])
def test_volterra_paths_have_the_right_moments(scheme):
    H, n_paths = 0.1, 40_000
    t, X, dW = volterra_paths(H, 1.0, 50, n_paths, scheme, rng=11)
    for i in (10, 50):
        var = X[:, i].var()
        # The sample variance of a normal has standard error var * sqrt(2 / n).
        assert abs(var - t[i] ** (2 * H)) < 4 * t[i] ** (2 * H) * np.sqrt(2 / n_paths) + (0.01 if scheme == "hybrid" else 0)
    cov = np.mean(X[:, -1] * dW.sum(axis=1))
    exact = np.sqrt(2 * H) / (H + 0.5)
    assert abs(cov - exact) < 4 * np.sqrt((1.0 + exact**2) / n_paths)
    assert dW.shape == (n_paths, 50)


def test_rough_bergomi_variance_and_spot_are_martingale_corrected():
    m = RoughBergomiModel(H=0.1, eta=1.5, rho=-0.7, xi0=0.04)
    p = rough_bergomi_paths(m, 100.0, 1.0, 0.03, q=0.01, n_steps=50, n_paths=40_000, scheme="cholesky", rng=5)
    v = p.variance[:, -1]
    assert abs(v.mean() - 0.04) < 4 * v.std() / np.sqrt(v.size)
    ST = p.spot[:, -1]
    assert abs(ST.mean() - 100 * np.exp(0.02)) < 4 * ST.std() / np.sqrt(ST.size)
    assert p.scheme == "cholesky" and p.times.shape == (51,)


def test_rough_bergomi_without_vol_of_vol_is_black_scholes():
    m = RoughBergomiModel(H=0.1, eta=0.0, rho=-0.9, xi0=0.04)
    res = rough_bergomi_mc_price(m, 100.0, 105.0, 1.0, 0.02, option_type="put", n_steps=10, n_paths=40_000, rng=2)
    exact = black_scholes_price(100.0, 105.0, 1.0, 0.02, 0.2, option_type="put")
    assert abs(res.estimate - exact) < 4 * res.std_error


def test_hybrid_and_cholesky_prices_agree():
    """Bennedsen et al. (2017), Section 4: hybrid-scheme rough Bergomi prices match exact sampling."""
    m = RoughBergomiModel(H=0.07, eta=1.9, rho=-0.9, xi0=0.235**2)
    a = rough_bergomi_mc_price(m, 1.0, 1.0, 0.5, 0.0, n_steps=50, n_paths=40_000, scheme="hybrid", rng=7)
    b = rough_bergomi_mc_price(m, 1.0, 1.0, 0.5, 0.0, n_steps=50, n_paths=40_000, scheme="cholesky", rng=8)
    assert abs(a.estimate - b.estimate) < 4 * np.hypot(a.std_error, b.std_error)


@pytest.mark.parametrize("H", [0.1, 0.5, 0.8])
def test_fbm_has_fractional_covariance(H):
    n_paths = 20_000
    t, B = fbm_paths(H, 2.0, 32, n_paths, rng=4)
    assert B.shape == (n_paths, 33) and np.all(B[:, 0] == 0)
    i, j = 16, 32
    exact = 0.5 * (t[i] ** (2 * H) + t[j] ** (2 * H) - (t[j] - t[i]) ** (2 * H))
    cov = np.mean(B[:, i] * B[:, j])
    assert abs(cov - exact) < 4 * np.sqrt((t[i] ** (2 * H) * t[j] ** (2 * H) + exact**2) / n_paths)


@pytest.mark.parametrize("H", [0.1, 0.14, 0.5])
def test_volatility_roughness_recovers_the_hurst_exponent(H):
    """Gatheral, Jaisson and Rosenbaum (2018), Section 2: zeta_q = q H for fractional Brownian log-volatility."""
    _, B = fbm_paths(H, 1.0, 8000, 1, rng=9)
    est = volatility_roughness(B[0])
    # The estimate is biased low for long lags on one path; 0.03 is a few standard deviations here.
    assert abs(est.H - H) < 0.03
    np.testing.assert_allclose(est.zeta / est.q, H, atol=0.04)
    assert est.moments.shape == (5, 50)


def test_validation():
    with pytest.raises(ValueError, match="H must lie"):
        RoughBergomiModel(H=0.0, eta=1.0, rho=0.0, xi0=0.04)
    with pytest.raises(ValueError, match="eta >= 0"):
        RoughBergomiModel(H=0.1, eta=1.0, rho=0.0, xi0=0.0)
    with pytest.raises(ValueError, match="rho must lie"):
        RoughBergomiModel(H=0.1, eta=1.0, rho=1.5, xi0=0.04)
    with pytest.raises(ValueError, match="scheme must be one of"):
        volterra_paths(0.1, 1.0, 10, 10, scheme="euler")
    with pytest.raises(ValueError, match="scheme must be"):
        hybrid_scheme_variance(0.1, 1.0, 10, scheme="cholesky")
    with pytest.raises(ValueError, match="T must be positive"):
        fbm_paths(0.1, 0.0, 10, 10)
    with pytest.raises(ValueError, match="at least 1"):
        volterra_paths(0.1, 1.0, 0, 10)
    with pytest.raises(ValueError, match="times must be positive"):
        volterra_covariance(0.1, [0.0, 1.0])
    with pytest.raises(ValueError, match="shorter than the series"):
        volatility_roughness(np.zeros(10), lags=[1, 20])
    with pytest.raises(ValueError, match="at least two lags"):
        volatility_roughness(np.arange(100.0), lags=[1])


def test_rough_bergomi_alpha_is_h_minus_a_half():
    assert RoughBergomiModel(H=0.1, eta=1.9, rho=-0.9, xi0=0.04).alpha == pytest.approx(-0.4)
