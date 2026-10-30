"""Tests for Hawkes order flow, the Cont-Stoikov-Talreja book and order-flow imbalance."""

import numpy as np
import pytest
from scipy import stats

from financialkit.constants import DEFAULT_RTOL
from financialkit.market_microstructure import (
    cst_mid_up_probability,
    fit_hawkes,
    fit_ofi_impact,
    hawkes_compensator,
    hawkes_intensity,
    hawkes_loglik,
    order_flow_imbalance,
    simulate_cst,
    simulate_hawkes,
)

MU, ALPHA, BETA = 0.8, 1.2, 2.0
LAM = 1.92 / np.arange(1, 9) ** 0.52
THETA = np.array([0.71, 0.81, 0.68, 0.56, 0.47, 0.47, 0.47, 0.47])


def test_hawkes_event_count_matches_the_stationary_rate():
    """E N(T) = mu T / (1 - n) up to an O(1) start-up term; Var N(T) ~ mu T / (1 - n)^3 (Hawkes 1971)."""
    T, n = 5000.0, ALPHA / BETA
    N = simulate_hawkes(MU, ALPHA, BETA, T, rng=3).size
    assert abs(N - MU * T / (1 - n)) < 4 * np.sqrt(MU * T / (1 - n) ** 3)
    with pytest.raises(ValueError):
        simulate_hawkes(MU, 2.0, 2.0, T)


def test_hawkes_loglik_and_intensity_match_brute_force_sums():
    t = simulate_hawkes(MU, ALPHA, BETA, 100.0, rng=0)
    lag = np.subtract.outer(t, t)
    lam_brute = MU + ALPHA * np.where(lag > 0, np.exp(-BETA * np.maximum(lag, 0)), 0.0).sum(axis=1)
    np.testing.assert_allclose(hawkes_intensity(t, t, MU, ALPHA, BETA), lam_brute, rtol=DEFAULT_RTOL)
    compensator_T = MU * 100.0 + ALPHA / BETA * np.sum(1 - np.exp(-BETA * (100.0 - t)))
    assert hawkes_loglik(t, 100.0, MU, ALPHA, BETA) == pytest.approx(np.log(lam_brute).sum() - compensator_T, rel=1e-12)
    # Just after an event the intensity has jumped by alpha.
    assert hawkes_intensity(t, t[5] + 1e-12, MU, ALPHA, BETA) == pytest.approx(lam_brute[5] + ALPHA, rel=1e-9)


def test_hawkes_residuals_are_unit_exponentials():
    """Ogata (1988): under the true model the compensator increments are i.i.d. Exp(1)."""
    t = simulate_hawkes(MU, ALPHA, BETA, 2000.0, rng=1)
    residuals = np.diff(hawkes_compensator(t, MU, ALPHA, BETA), prepend=0.0)
    assert stats.kstest(residuals, "expon").pvalue > 0.01
    # The same events under a Poisson model are overdispersed: a clear rejection.
    assert stats.kstest(np.diff(t, prepend=0.0) * t.size / 2000.0, "expon").pvalue < 1e-6


def test_fit_hawkes_recovers_the_parameters():
    """The fitted parameters lie within four standard errors from the observed information."""
    T = 3000.0
    t = simulate_hawkes(MU, ALPHA, BETA, T, rng=2)
    fit = fit_hawkes(t, T)
    theta_hat = np.array([fit.mu, fit.alpha, fit.beta])
    assert fit.loglik >= hawkes_loglik(t, T, MU, ALPHA, BETA)

    h = 1e-4 * theta_hat
    H = np.empty((3, 3))
    for i in range(3):
        for j in range(3):

            def ll(si, sj, i=i, j=j):
                x = theta_hat.copy()
                x[i] += si * h[i]
                x[j] += sj * h[j]
                return hawkes_loglik(t, T, *x)

            H[i, j] = (ll(1, 1) - ll(1, -1) - ll(-1, 1) + ll(-1, -1)) / (4 * h[i] * h[j])
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    assert np.all(np.abs(theta_hat - [MU, ALPHA, BETA]) < 4 * se)
    assert fit.branching_ratio == pytest.approx(fit.alpha / fit.beta)
    assert fit.mean_intensity == pytest.approx(fit.mu / (1 - fit.branching_ratio))
    with pytest.raises(ValueError):
        fit_hawkes([1.0, 0.5, 2.0], 3.0)


def test_cst_probability_is_a_negative_binomial_race_without_arrivals_or_cancellations():
    """With lambda = theta = 0 each queue is a Poisson(mu) countdown, so the ask empties first
    when at least b of the first a + b - 1 market orders are buys."""
    p = cst_mid_up_probability(0.0, 1.0, 0.0, n_max=12)
    a, b = np.meshgrid(np.arange(1, 13), np.arange(1, 13), indexing="ij")
    np.testing.assert_allclose(p[1:, 1:], stats.binom.sf(b - 1, a + b - 1, 0.5), atol=1e-12)


def test_cst_probability_is_symmetric_and_monotone():
    p = cst_mid_up_probability(LAM[0], 0.94, THETA[0], n_max=30)
    np.testing.assert_allclose(p[1:, 1:] + p[1:, 1:].T, 1.0, atol=1e-12)
    assert np.all(np.diff(p[1:20, 1:20], axis=0) > 0) and np.all(np.diff(p[1:20, 1:20], axis=1) < 0)
    with pytest.raises(ValueError):
        cst_mid_up_probability(1.0, 0.0, 0.0)


def test_cst_probability_matches_a_monte_carlo_race_of_the_queues():
    lam, mu, theta, n_paths = LAM[0], 0.94, THETA[0], 20000
    rng = np.random.default_rng(0)
    q = np.tile([[2, 4]], (n_paths, 1))  # bid queue 2, ask queue 4
    live = np.ones(n_paths, dtype=bool)
    while live.any():
        idx = np.flatnonzero(live)
        down = mu + theta * q[idx]
        rates = np.column_stack([np.full(idx.size, lam), np.full(idx.size, lam), down])
        u = rng.random(idx.size) * rates.sum(axis=1)
        cum = np.cumsum(rates, axis=1)
        which = (u[:, None] >= cum).sum(axis=1)  # 0, 1: a bid or ask arrives; 2, 3: a bid or ask leaves
        q[idx[which < 2], which[which < 2]] += 1
        q[idx[which >= 2], which[which >= 2] - 2] -= 1
        live[idx] = np.all(q[idx] > 0, axis=1)
    up = q[:, 1] == 0
    p = cst_mid_up_probability(lam, mu, theta, n_max=60)[2, 4]
    assert abs(up.mean() - p) < 4 * np.sqrt(p * (1 - p) / n_paths)


def test_simulate_cst_clock_and_quotes():
    book = simulate_cst(LAM, 0.94, THETA, 20000, n_ticks=200, rng=1)
    T = book.times[-1]
    assert np.all(book.spread >= 1) and np.all(np.diff(book.times) > 0)
    assert np.all(book.bid_size > 0) and np.all(book.ask_size > 0)
    # Market buys arrive at rate mu whenever there are asks, which is always here: a Poisson count.
    n_mb = np.sum(book.events == 2)
    assert abs(n_mb - 0.94 * T) < 4 * np.sqrt(0.94 * T)
    # A market buy that takes the last order at the ask moves the ask up.
    k = np.flatnonzero((book.events == 2) & (book.ask_size[:-1] == 1))
    assert k.size and np.all(book.best_ask[k + 1] > book.best_ask[k])
    with pytest.raises(ValueError):
        simulate_cst(LAM, 0.94, THETA[:3], 10)


def test_simulate_cst_price_moves_follow_the_queue_race():
    """By the strong Markov property, from each one-tick-spread state reached by a mid move the
    next mid move is up with probability p(a, b), so sum(up - p) is a martingale."""
    book = simulate_cst(LAM, 0.94, THETA, 30000, n_ticks=200, rng=2)
    mid = book.mid
    moves = np.flatnonzero(np.diff(mid) != 0) + 1  # indices right after each mid change
    starts = moves[book.spread[moves] == 1][:-1]
    next_move = moves[np.searchsorted(moves, starts, side="right")]
    up = mid[next_move] > mid[next_move - 1]
    p = cst_mid_up_probability(LAM[0], 0.94, THETA[0], n_max=60)[book.bid_size[starts], book.ask_size[starts]]
    assert starts.size > 500
    assert abs(np.sum(up - p)) < 4 * np.sqrt(np.sum(p * (1 - p)))


def test_order_flow_imbalance_rules():
    """Cont, Kukanov and Stoikov (2014), eq. (2): each case of a quote update."""
    e = order_flow_imbalance([10, 11, 10, 10, 10], [5, 2, 4, 4, 4], [12, 12, 12, 11, 13], [3, 3, 3, 6, 8])
    # bid up (+2 new); bid down (-2 old); ask down (-6 new); ask up (+6 old, the new queue does not count)
    np.testing.assert_array_equal(e, [2.0, -2.0, -6.0, 6.0])
    with pytest.raises(ValueError):
        order_flow_imbalance([1, 2], [1], [3, 4], [1, 1])


def test_order_flow_imbalance_of_book_events():
    """Without a quote change, a buy-side event at the best quotes adds one and a sell-side event subtracts one."""
    book = simulate_cst(LAM, 0.94, THETA, 5000, n_ticks=200, rng=3)
    e = order_flow_imbalance(book.best_bid, book.bid_size, book.best_ask, book.ask_size)
    same = (np.diff(book.best_bid) == 0) & (np.diff(book.best_ask) == 0)
    buy_side = np.isin(book.events, [0, 2, 5])  # limit buy, market buy, cancelled sell
    assert np.all(np.isin(e[same], [-1.0, 0.0, 1.0]))
    assert np.all(e[same & buy_side] >= 0) and np.all(e[same & ~buy_side] <= 0)
    assert np.any(e[same & buy_side] == 1) and np.any(e[same & ~buy_side] == -1)


def test_fit_ofi_impact():
    rng = np.random.default_rng(0)
    ofi = rng.normal(0, 10, 400)
    fit = fit_ofi_impact(0.2 + 0.05 * ofi + rng.normal(0, 0.1, 400), ofi)
    assert np.all(np.abs(fit.coef - [0.2, 0.05]) < 4 * fit.std_errors)
    with pytest.raises(ValueError):
        fit_ofi_impact([1.0, 2.0], [1.0])
