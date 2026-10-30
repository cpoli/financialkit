"""Tests for the sample, single-index, fundamental factor and Ledoit-Wolf covariance estimators."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.portfolio import fundamental_factor_covariance, ledoit_wolf, sample_covariance, single_index_covariance


def _factor_returns(n, d, rng):
    m = 0.04 * rng.standard_normal(n)
    beta = np.linspace(0.5, 1.5, d)
    X = np.outer(m, beta) + 0.03 * rng.standard_normal((n, d))
    return X, m, beta


def test_single_index_recovers_betas_and_structure():
    rng = np.random.default_rng(0)
    X, m, beta = _factor_returns(20_000, 5, rng)
    res = single_index_covariance(X, m)
    np.testing.assert_allclose(res.betas, beta, atol=0.02)
    true = 0.04**2 * np.outer(beta, beta) + 0.03**2 * np.eye(5)
    np.testing.assert_allclose(res.cov, true, atol=1e-4)
    assert res.n_params == 16 and sample_covariance(X).n_params == 15
    with pytest.raises(ValueError):
        single_index_covariance(X, m[:-1])


def _lw_identity_reference(X):
    """Loop-by-loop transcription of Ledoit and Wolf (2004a), Sec. 3.3."""
    n, p = X.shape
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / n
    I = np.eye(p)
    norm2 = lambda A: np.trace(A @ A.T) / p
    m = np.trace(S @ I) / p
    d2 = norm2(S - m * I)
    b_bar2 = sum(norm2(np.outer(Y[k], Y[k]) - S) for k in range(n)) / n**2
    b2 = min(b_bar2, d2)
    return b2 / d2 * m * I + (d2 - b2) / d2 * S


def test_ledoit_wolf_identity_matches_paper():
    X = np.random.default_rng(1).standard_normal((25, 10)) @ np.diag(np.linspace(1, 3, 10))
    np.testing.assert_allclose(ledoit_wolf(X).cov, _lw_identity_reference(X), rtol=DEFAULT_RTOL)


def test_ledoit_wolf_constant_correlation_against_direct_sums():
    """Ledoit and Wolf (2004b), Appendix B, with the sums written out."""
    rng = np.random.default_rng(2)
    X, _, _ = _factor_returns(40, 6, rng)
    n, p = X.shape
    Y = X - X.mean(axis=0)
    S = Y.T @ Y / n
    s = np.sqrt(np.diag(S))
    r_bar = sum(S[i, j] / (s[i] * s[j]) for i in range(p) for j in range(p) if i != j) / (p * (p - 1))
    F = np.array([[S[i, i] if i == j else r_bar * s[i] * s[j] for j in range(p)] for i in range(p)])
    pi = sum(np.mean((Y[:, i] * Y[:, j] - S[i, j]) ** 2) for i in range(p) for j in range(p))
    theta = lambda i, j: np.mean((Y[:, i] ** 2 - S[i, i]) * (Y[:, i] * Y[:, j] - S[i, j]))
    rho = sum(np.mean((Y[:, i] ** 2 - S[i, i]) ** 2) for i in range(p))
    rho += sum(r_bar / 2 * (s[j] / s[i] * theta(i, j) + s[i] / s[j] * theta(j, i)) for i in range(p) for j in range(p) if i != j)
    gamma = np.sum((F - S) ** 2)
    delta = max(0.0, min(1.0, (pi - rho) / gamma / n))
    res = ledoit_wolf(X, "constant_correlation")
    assert res.shrinkage == pytest.approx(delta, rel=DEFAULT_RTOL)
    np.testing.assert_allclose(res.cov, delta * F + (1 - delta) * S, rtol=DEFAULT_RTOL)


def test_shrinkage_vanishes_with_data_and_improves_small_samples():
    rng = np.random.default_rng(3)
    true = np.diag(np.linspace(1, 4, 15))
    L = np.linalg.cholesky(true)
    assert ledoit_wolf(rng.standard_normal((20_000, 15)) @ L.T).shrinkage < 0.05
    loss_lw, loss_s = [], []
    for _ in range(50):
        X = rng.standard_normal((20, 15)) @ L.T
        loss_lw.append(np.sum((ledoit_wolf(X).cov - true) ** 2))
        loss_s.append(np.sum((sample_covariance(X).cov - true) ** 2))
    assert np.mean(loss_lw) < np.mean(loss_s)
    with pytest.raises(ValueError):
        ledoit_wolf(X, "nope")
    with pytest.raises(ValueError):
        sample_covariance([1.0, 2.0])


def test_shrinkage_lowers_the_condition_number_and_keeps_unit_correlations():
    """Ledoit and Wolf (2004): shrinking toward the identity pulls the eigenvalues together."""
    X = np.random.default_rng(11).standard_normal((40, 20))
    sample, shrunk = sample_covariance(X), ledoit_wolf(X)
    for res in (sample, shrunk):
        np.testing.assert_allclose(np.diag(res.corr), 1.0, rtol=1e-14)
        ev = np.linalg.eigvalsh(res.cov)
        assert res.condition_number == pytest.approx(ev[-1] / ev[0])
    assert shrunk.condition_number < 0.5 * sample.condition_number


def test_fundamental_factor_regression_is_weighted_least_squares_per_period():
    rng = np.random.default_rng(3)
    n, d, k = 12, 30, 4
    X = rng.standard_normal((n, d, k))
    R = rng.standard_normal((n, d))
    w = rng.uniform(0.5, 2.0, d)
    res = fundamental_factor_covariance(R, X, weights=w)
    for t in range(n):
        f_t, *_ = np.linalg.lstsq(np.sqrt(w)[:, None] * X[t], np.sqrt(w) * R[t], rcond=None)
        np.testing.assert_allclose(res.factor_returns[t], f_t, rtol=1e-10)
        # Weighted residuals are orthogonal to every exposure.
        np.testing.assert_allclose(X[t].T @ (w * res.residuals[t]), 0.0, atol=1e-12)
    np.testing.assert_allclose(res.cov, X[-1] @ res.factor_cov @ X[-1].T + np.diag(res.specific_var), rtol=DEFAULT_RTOL)
    wp = np.full(d, 1.0 / d)
    assert sum(res.risk_decomposition(wp)) == pytest.approx(wp @ res.cov @ wp, rel=1e-12)
    assert res.n_params == k * (k + 1) // 2 + d


def test_fundamental_factor_model_recovers_its_inputs():
    rng = np.random.default_rng(4)
    n, d = 4000, 300
    industry = np.eye(3)[rng.integers(0, 3, d)]
    X = np.column_stack([industry, rng.standard_normal(d)])
    F = np.diag([0.04, 0.05, 0.06, 0.02]) ** 2
    spec = rng.uniform(0.05, 0.1, d) ** 2
    f = rng.standard_normal((n, 4)) @ np.sqrt(F)
    R = f @ X.T + rng.standard_normal((n, d)) * np.sqrt(spec)
    res = fundamental_factor_covariance(R, X)
    # Sampling error of a variance from n draws is about sqrt(2 / n) = 2.2% of it; allow 4 standard errors.
    np.testing.assert_allclose(np.diag(res.factor_cov), np.diag(F), rtol=0.1)
    np.testing.assert_allclose(res.specific_var, spec, rtol=0.1)
    # With fewer periods than assets the sample matrix is singular; the factor model is not.
    short = fundamental_factor_covariance(R[:200], X)
    assert np.linalg.eigvalsh(sample_covariance(R[:200]).cov)[0] < 1e-12 < np.linalg.eigvalsh(short.cov)[0]


def test_fundamental_factor_covariance_validates_inputs():
    R = np.zeros((10, 3))
    with pytest.raises(ValueError):
        fundamental_factor_covariance(R, np.ones((4, 1)))
    with pytest.raises(ValueError):
        fundamental_factor_covariance(R, np.ones((3, 3)))
    with pytest.raises(ValueError):
        fundamental_factor_covariance(R, np.ones((3, 1)), weights=[1.0, -1.0, 1.0])
