r"""
Cox-Ross-Rubinstein: the binomial tree converges to Black-Scholes
=================================================================

John Cox, Stephen Ross and Mark Rubinstein (1979) showed that the
Black-Scholes price is the limit of a much simpler model: in each short
period the stock moves up by :math:`u = e^{\sigma\sqrt{\Delta t}}` or
down by :math:`d = 1/u`, and the option is replicated one step at a
time. The risk-neutral probability :math:`p = (e^{r\Delta t} - d)/(u -
d)` falls out of the replication, and backward induction prices any
payoff. The error shrinks like :math:`1/n`, oscillating as the strike
moves between nodes.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import CRRTree, black_scholes_price
from financialkit.options.visualizers import plot_tree_convergence

# %%
# One-period replication
# ----------------------
#
# A call struck at 100 on a stock at 100 that moves to 110 or 90 in a
# year, with r = 5%: hold :math:`\Delta` shares and borrow so the
# portfolio matches the call in both states.

S, K, u, d, r = 100.0, 100.0, 1.1, 0.9, 0.05
cu, cd = max(S * u - K, 0), max(S * d - K, 0)
delta = (cu - cd) / (S * u - S * d)
bond = (cu - delta * S * u) * np.exp(-r)
p = (np.exp(r) - d) / (u - d)
print(f"replicate with {delta:.2f} shares and {bond:.3f} in bonds: price {delta * S + bond:.4f}")
print(f"risk-neutral expectation with p = {p:.4f}: {np.exp(-r) * (p * cu + (1 - p) * cd):.4f}")

# %%
# Convergence to Black-Scholes
# ----------------------------

S, K, T, r, sigma = 100.0, 95.0, 1.0, 0.05, 0.2
exact = black_scholes_price(S, K, T, r, sigma)
steps = np.arange(5, 301, 1)
prices = np.array([CRRTree(n).price(S, K, T, r, sigma).price for n in steps])
print(f"Black-Scholes {exact:.5f}; CRR with 300 steps {prices[-1]:.5f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(steps, prices, lw=1, label="CRR tree")
axes[0].axhline(exact, color="k", ls=":", label="Black-Scholes")
axes[0].set_xlabel("time steps n")
axes[0].set_ylabel("call price")
axes[0].legend()
plot_tree_convergence(steps, prices, exact, ax=axes[1], label="CRR error")
axes[1].loglog(steps, 1.0 / steps, "k:", label="1/n")
axes[1].legend()
fig.tight_layout()

plt.show()
