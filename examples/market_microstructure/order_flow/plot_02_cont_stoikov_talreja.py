r"""
Cont, Stoikov and Talreja: a Markov limit order book
====================================================

Rama Cont, Sasha Stoikov and Rishi Talreja (2010) modelled a limit order
book as a continuous-time Markov chain of unit orders. Limit orders
arrive :math:`i` ticks from the opposite best quote at rate
:math:`\lambda(i) = k / i^{a}`, market orders at rate :math:`\mu`, and
each resting order is cancelled at rate :math:`\theta(i)`. Everything
about the book follows from these rates: its average shape, and the
probability that the next mid-price move is up given the queues at the
best bid and ask. The rates below are illustrative, of the shape the
paper estimated (a power law with :math:`a \approx 0.5`).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure import cst_mid_up_probability, simulate_cst

i = np.arange(1, 11)
lam = 1.92 / i**0.52
theta = np.array([0.71, 0.81, 0.68, 0.56, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47])
mu = 0.94
book = simulate_cst(lam, mu, theta, 30000, n_ticks=300, rng=0)

# %%
# Prices and the book's shape
# ---------------------------
# Far from the best quotes, each level behaves like an infinite-server
# queue with arrival rate :math:`\lambda(i)` and service rate
# :math:`\theta(i)`, so its average depth is close to
# :math:`\lambda(i)/\theta(i)`. Market orders thin the levels nearest the
# spread.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
s = slice(0, 3000)
ax1.step(book.times[s], book.best_ask[s], where="post", color="C1", lw=1, label="best ask")
ax1.step(book.times[s], book.best_bid[s], where="post", color="C0", lw=1, label="best bid")
ax1.set_xlabel("time")
ax1.set_ylabel("price (ticks)")
ax1.legend()
ax2.bar(i, book.mean_depth, color="C0", alpha=0.6, label="simulated")
ax2.plot(i, lam / theta, "C3o-", label=r"$\lambda(i)/\theta(i)$")
ax2.set_xlabel("ticks from the opposite best quote")
ax2.set_ylabel("average orders")
ax2.legend()
fig.tight_layout()
print(f"spread of one tick {np.mean(book.spread == 1):.0%} of events")
assert np.all(np.abs(book.mean_depth[3:] - (lam / theta)[3:]) < 0.25 * (lam / theta)[3:])

# %%
# Which way will the price move?
# ------------------------------
# With a one-tick spread the best bid and ask queues are independent
# birth-death processes, and the mid moves up when the ask queue empties
# first. :func:`~financialkit.market_microstructure.cst_mid_up_probability`
# solves for that probability; the dots are the simulated book's
# frequencies after each price move.

p = cst_mid_up_probability(lam[0], mu, theta[0], n_max=60)
mid = book.mid
moves = np.flatnonzero(np.diff(mid) != 0) + 1
starts = moves[book.spread[moves] == 1][:-1]
up = mid[moves[np.searchsorted(moves, starts, side="right")]] > mid[starts]
a, b = book.bid_size[starts], book.ask_size[starts]

fig, ax = plt.subplots(figsize=(6, 4))
for k, bid_queue in enumerate([1, 2, 4]):
    asks = np.arange(1, 9)
    ax.plot(asks, p[bid_queue, asks], f"C{k}-", label=f"bid queue {bid_queue}")
    freq = [up[(a == bid_queue) & (b == q)].mean() if np.sum((a == bid_queue) & (b == q)) >= 30 else np.nan for q in asks]
    ax.plot(asks, freq, f"C{k}o")
ax.axhline(0.5, color="0.6", lw=0.8)
ax.set_xlabel("ask queue")
ax.set_ylabel("P(next mid move is up)")
ax.legend()
fig.tight_layout()
pred = p[a, b]
print(f"{starts.size} price moves: {up.mean():.3f} up, predicted {pred.mean():.3f}")
assert abs(np.sum(up - pred)) < 4 * np.sqrt(np.sum(pred * (1 - pred)))

plt.show()
