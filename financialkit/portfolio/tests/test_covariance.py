"""Tests for the sample, single-index and Ledoit-Wolf covariance estimators."""

import numpy as np
import pytest

from financialkit.constants import DEFAULT_RTOL
from financialkit.portfolio import ledoit_wolf, sample_covariance, single_index_covariance


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
