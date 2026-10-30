r"""
Avellaneda and Stoikov: market making in a limit order book
===========================================================

Marco Avellaneda and Sasha Stoikov (2008) let a market maker quote
around a Brownian mid price :math:`s`, with fills arriving at rate
:math:`A e^{-k\delta}` at distance :math:`\delta`. A trader with
exponential utility first computes the *reservation price*
:math:`r = s - q\gamma\sigma^2(T - t)`, the mid price adjusted for its
inventory :math:`q`, and then quotes a spread
:math:`\gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln(1 + \gamma/k)` around it.
The example reproduces the comparison of their Table 1: the inventory
strategy gives up a little expected profit for a much smaller variance.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure import AvellanedaStoikov, SymmetricQuotes, simulate_market_making
from financialkit.market_microstructure.visualizers import plot_market_making

s0, sigma, T, n_steps, A, k = 100.0, 2.0, 1.0, 200, 140.0, 1.5

# %%
# One trading session
# -------------------

res = simulate_market_making(AvellanedaStoikov(0.1, sigma, k, T), s0, sigma, T, n_steps, A, k, n_paths=1, rng=5)
axes = plot_market_making(res)
axes[0].set_title(f"inventory strategy, gamma = 0.1: profit {res.pnl[0]:.1f}")

# %%
# Inventory against symmetric quoting
# -----------------------------------
#
# The symmetric strategy quotes the same average spread centred on the
# mid price. Paper values for :math:`\gamma = 0.1`: profit 62.94 (sd 5.89)
# against 67.21 (sd 13.43).

print(f"{'gamma':>6s} {'strategy':>10s} {'spread':>7s} {'profit':>7s} {'sd':>6s} {'q(T)':>6s} {'sd q':>6s}")
for g in (0.01, 0.1, 0.5):
    inv = simulate_market_making(AvellanedaStoikov(g, sigma, k, T), s0, sigma, T, n_steps, A, k, n_paths=1000, rng=6)
    half = float(np.mean(inv.ask - inv.bid)) / 2
    sym = simulate_market_making(SymmetricQuotes(half), s0, sigma, T, n_steps, A, k, n_paths=1000, rng=6)
    assert inv.pnl.std() < 0.75 * sym.pnl.std() and inv.inventory[:, -1].std() < sym.inventory[:, -1].std()
    for name, r in [("inventory", inv), ("symmetric", sym)]:
        q = r.inventory[:, -1]
        print(f"{g:6.2f} {name:>10s} {2 * half:7.2f} {r.pnl.mean():7.2f} {r.pnl.std():6.2f} {q.mean():6.2f} {q.std():6.2f}")

plt.show()
