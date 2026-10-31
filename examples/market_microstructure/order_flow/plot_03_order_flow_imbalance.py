r"""
Cont, Kukanov and Stoikov: order-flow imbalance
===============================================

Rama Cont, Arseniy Kukanov and Sasha Stoikov (2014) asked which events in
the book move the price. Their order-flow imbalance (OFI) adds the
orders that arrive at the best bid and subtracts those that arrive at
the best ask, counting limit orders, cancellations and market orders
alike. Over short intervals the price change is linear in it, with a
slope inversely proportional to the depth at the best quotes,

.. math::

   \Delta P_k = \beta\,\mathrm{OFI}_k + \varepsilon_k, \qquad \beta \propto \frac{1}{\text{depth}},

and it explains far more of the price change than the imbalance of
trades alone. Here the book is simulated with
:func:`~financialkit.market_microstructure.simulate_cst`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure import fit_ofi_impact, order_flow_imbalance, simulate_cst

i = np.arange(1, 11)
theta = np.array([0.71, 0.81, 0.68, 0.56, 0.47, 0.47, 0.47, 0.47, 0.47, 0.47])


def interval_regressions(scale, n_intervals=100, rng=0):
    """Price change, OFI and trade imbalance over equal intervals of a book with limit order rates scale * lambda."""
    book = simulate_cst(scale * 1.92 / i**0.52, 0.94, theta, 6000, n_ticks=300, rng=rng)
    e = order_flow_imbalance(book.best_bid, book.bid_size, book.best_ask, book.ask_size)
    trades = np.select([book.events == 2, book.events == 3], [1.0, -1.0], 0.0)
    dt = book.times[-1] / n_intervals
    k = (book.times[1:] // dt).astype(int)
    n = n_intervals
    keep = k < n  # the last event ends the last interval
    ofi, ti = np.bincount(k[keep], e[keep], n), np.bincount(k[keep], trades[keep], n)
    mid = book.mid[np.searchsorted(book.times, dt * np.arange(n + 1), side="right") - 1]
    depth = 0.5 * np.mean(book.bid_size + book.ask_size)
    return np.diff(mid), ofi, ti, depth


dp, ofi, ti, depth = interval_regressions(1.0)
fit_ofi, fit_ti = fit_ofi_impact(dp, ofi), fit_ofi_impact(dp, ti)
print(f"{dp.size} intervals; R^2 on OFI {fit_ofi.r_squared:.2f}, on trade imbalance {fit_ti.r_squared:.2f}")
assert fit_ofi.r_squared > fit_ti.r_squared

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, x, fit, name in [(ax1, ofi, fit_ofi, "order-flow imbalance"), (ax2, ti, fit_ti, "trade imbalance")]:
    ax.plot(x, dp, "C0.", ms=3, alpha=0.5)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, fit.coef[0] + fit.coef[1] * xs, "C3-", label=f"$R^2$ = {fit.r_squared:.2f}")
    ax.set_xlabel(name)
    ax.legend()
ax1.set_ylabel("mid-price change (ticks)")
fig.tight_layout()

# %%
# Deeper books move less
# ----------------------
# Scaling up every limit order rate deepens the book, and the same
# imbalance moves the price less. Across stocks the paper found the slope
# close to proportional to 1/depth. In this simulated book it falls
# faster: the levels behind the best quotes deepen too, so an emptied
# queue moves the price by one tick, and queues empty more rarely.

scales = [1.0, 1.5, 2.0]
slopes, depths = [], []
for s in scales:
    dp_s, ofi_s, _, depth_s = interval_regressions(s, rng=1)
    slopes.append(fit_ofi_impact(dp_s, ofi_s).coef[1])
    depths.append(depth_s)
    print(f"rates x{s:g}: depth at best {depth_s:.2f}, slope {slopes[-1]:.3f} tick/order, slope x depth {slopes[-1] * depth_s:.2f}")
assert slopes[0] > slopes[1] > slopes[2]

fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(depths, slopes, "C0o-", label="fitted slope")
ax.loglog(depths, slopes[0] * depths[0] / np.array(depths), "k--", label="1 / depth")
ax.set_xlabel("average depth at the best quotes")
ax.set_ylabel("OFI slope (ticks per order)")
ax.legend()
fig.tight_layout()

plt.show()
