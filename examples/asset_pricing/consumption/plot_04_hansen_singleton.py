r"""
Hansen-Singleton: the Euler equation estimated
==============================================

Lars Peter Hansen and Kenneth Singleton (1982) estimated the
consumption-based model without solving it. Whatever drives
consumption and returns, a power-utility investor's first-order
condition says that the pricing error
:math:`\beta G_{t+1}^{-\gamma} R_{t+1} - 1` cannot be predicted by
anything known at :math:`t`. Each asset times each instrument is a
moment condition, and GMM estimates the time preference :math:`\beta`
and the risk aversion :math:`\gamma` directly.

The data are 1,000 periods from a Lucas tree with :math:`\beta = 0.97`,
:math:`\gamma = 4`, and consumption growth of 2% with a volatility of
3.6%, close to the US figures. The assets are the tree and the riskless
bond. The instruments are a constant and last period's consumption growth.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import LucasTree, hansen_singleton

path = LucasTree(0.97, 4.0, 0.02, 0.036).simulate(1000, rng=0)
G = path.growth[1:]
R = np.column_stack([path.equity_return, path.risk_free])[1:]
Z = np.column_stack([np.ones(G.size), path.growth[:-1]])

res = hansen_singleton(G, R, Z)
(b, g), (se_b, se_g) = res.params, res.std_errors
print(f"beta = {b:.3f} ({se_b:.3f}), gamma = {g:.2f} ({se_g:.2f}), J = {res.j_stat:.2f} on {res.df} df, p = {res.j_pvalue:.2f}")
# The truth lies within three standard errors, and the overidentifying
# restrictions are not rejected.
assert abs(b - 0.97) < 3 * se_b and abs(g - 4.0) < 3 * se_g
assert res.j_pvalue > 0.05
# Smooth consumption leaves risk aversion poorly identified even in 1,000 periods.
assert se_g > 1.0

# %%
# A ridge, not a peak
# -------------------
#
# The GMM objective :math:`T g_T' W g_T` is nearly flat along a ridge in
# :math:`(\beta, \gamma)`. A higher risk aversion raises the riskless
# rate the model predicts, and a higher :math:`\beta` brings it back.
# The estimates of the two parameters are therefore strongly correlated.


def objective(beta, gamma):
    e = beta * G[:, None] ** -gamma * R - 1.0
    gbar = (e[:, :, None] * Z[:, None, :]).reshape(G.size, -1).mean(axis=0)
    return G.size * gbar @ res.weighting @ gbar


bb, gg = np.meshgrid(np.linspace(0.85, 1.05, 81), np.linspace(-2, 12, 81))
obj = np.vectorize(objective)(bb, gg)
corr = res.cov[0, 1] / (se_b * se_g)
print(f"correlation of the two estimates: {corr:.2f}")
assert corr > 0.9

fig, ax = plt.subplots(figsize=(7, 4.5))
cs = ax.contourf(bb, gg, np.log10(obj + 1e-3), levels=20, cmap="viridis_r")
fig.colorbar(cs, ax=ax, label="log10 GMM objective")
ax.plot(0.97, 4.0, "*", color="C3", ms=14, label="truth")
ax.plot(b, g, "o", color="w", mec="k", label="estimate")
ax.set_xlabel("beta")
ax.set_ylabel("gamma")
ax.legend(loc="upper left")
fig.tight_layout()

# %%
# The equity premium, estimated
# -----------------------------
#
# Now give the stock an extra 6% a year that the model cannot justify,
# about the gap that Mehra and Prescott later documented. GMM can still
# fit the two unconditional Euler equations, but only with a risk
# aversion near 40 and a time preference far below one. Those are the
# implausible estimates that Hansen and Singleton's successors found in
# US data, and that the equity premium and riskless rate puzzles are
# about.

R_puzzle = R * np.array([1.06, 1.0])
puzzle = hansen_singleton(G, R_puzzle, Z)
print(f"with a 6% premium: beta = {puzzle.params[0]:.2f}, gamma = {puzzle.params[1]:.1f}")
assert puzzle.params[1] > 20 and puzzle.params[0] < 0.8

plt.show()
