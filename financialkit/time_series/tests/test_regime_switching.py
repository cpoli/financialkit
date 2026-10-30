"""Tests for Hamilton's filter, Kim's smoother and the EM fit of a Markov-switching model."""

import itertools

import numpy as np
import pytest
from scipy.stats import norm

from financialkit.time_series import fit_markov_switching, hamilton_filter, kim_smoother, simulate_markov_switching

P = np.array([[0.9, 0.1], [0.3, 0.7]])
MU, SIGMA = np.array([0.2, -0.5]), np.array([0.6, 1.8])


def test_filter_and_smoother_match_enumeration_of_all_regime_paths():
    x = np.array([0.3, -2.0, 0.1, 1.5, -0.4, -3.0, 0.2])
    p0 = np.array([0.6, 0.4])  # before the first observation
    filt, pred, ll = hamilton_filter(x, MU, SIGMA, P, p0)
    sm = kim_smoother(filt, pred, P)
    n = x.size
    total, marg = 0.0, np.zeros((n, 2))
    for s0 in range(2):
        for path in itertools.product(range(2), repeat=n):
            states = (s0, *path)
            w = p0[s0] * np.prod([P[states[t], states[t + 1]] for t in range(n)])
            w *= np.prod(norm.pdf(x, MU[list(path)], SIGMA[list(path)]))
            total += w
            marg[np.arange(n), list(path)] += w
    assert ll == pytest.approx(np.log(total), rel=1e-12)
    np.testing.assert_allclose(sm, marg / total, rtol=1e-10)
    np.testing.assert_allclose(sm[-1], filt[-1], rtol=1e-12)


def test_ergodic_probabilities_and_durations():
    x, s = simulate_markov_switching(40_000, MU, SIGMA, P, rng=0)
    pi1 = (1 - P[1, 1]) / (2 - P[0, 0] - P[1, 1])
    assert abs(np.mean(s == 0) - pi1) < 0.02  # regime spells of ~10 periods leave ~4000 independent spells
    res = fit_markov_switching(x[:5000])
    assert res.ergodic_probs[0] == pytest.approx((1 - res.transition[1, 1]) / (2 - res.transition[0, 0] - res.transition[1, 1]))
    np.testing.assert_allclose(res.expected_duration, 1 / (1 - np.diag(res.transition)))
    np.testing.assert_allclose(res.transition.sum(axis=1), 1.0)


def test_em_recovers_the_parameters_and_classifies_the_regimes():
    P2 = np.array([[0.98, 0.02], [0.05, 0.95]])
    x, s = simulate_markov_switching(4000, [0.05, -0.2], [0.7, 2.0], P2, rng=1)
    res = fit_markov_switching(x)
    assert res.converged and res.n_regimes == 2
    np.testing.assert_allclose(res.sigma, [0.7, 2.0], rtol=0.1)
    np.testing.assert_allclose(np.diag(res.transition), np.diag(P2), atol=0.02)
    assert np.mean(res.most_likely_regime() == s) > 0.9
    # EM reaches at least the likelihood of the true parameters.
    assert res.loglik >= hamilton_filter(x, [0.05, -0.2], [0.7, 2.0], P2)[2] - 1e-6
    assert res.aic < res.bic and res.n_params == 6
    again = fit_markov_switching(x, init={"mu": res.mu, "sigma": res.sigma, "transition": res.transition})
    assert again.loglik == pytest.approx(res.loglik, abs=1e-3)


def test_impossible_observation_gives_minus_infinite_loglik():
    _, _, ll = hamilton_filter([1e6], [0.0, 0.0], [1.0, 1.0], P)
    assert ll == -np.inf
