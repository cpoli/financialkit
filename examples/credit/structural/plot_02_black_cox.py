r"""
Black and Cox: default at the first passage
===========================================

Black and Cox (1976) let safety covenants force default the first time
the asset value touches a barrier, not only at maturity. The reflection
principle gives the first-passage probability in closed form. Here it is
checked against simulated asset paths (with a Brownian-bridge
correction for crossings between grid points) and compared with
Merton's maturity-only default probability.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.credit import black_cox_default_probability, first_passage_crossing_probability
from financialkit.integrators import gbm_paths

V0, K, mu, sigma, T = 100.0, 70.0, 0.05, 0.25, 5.0
t, paths = gbm_paths(V0, mu, sigma, T=T, n_steps=100, n_paths=4000, rng=0)

# %%
# Paths that touch the barrier
# ----------------------------

hit_grid = (paths <= K).any(axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
for i in range(60):
    ax.plot(t, paths[i], color="C3" if hit_grid[i] else "C0", lw=0.6, alpha=0.7)
ax.axhline(K, color="k", lw=2, label="barrier K")
ax.set_xlabel("years")
ax.set_ylabel("asset value")
ax.set_title("red: defaulted before maturity")
ax.legend()

# %%
# First passage versus Merton
# ---------------------------

horizons = np.linspace(0.1, T, 50)
bc = black_cox_default_probability(V0, K, horizons, mu, sigma)
merton = stats.norm.cdf((np.log(K / V0) - (mu - 0.5 * sigma**2) * horizons) / (sigma * np.sqrt(horizons)))
p_bridge = first_passage_crossing_probability(paths, K, sigma, T / 100)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(horizons, bc, lw=2, label="Black-Cox first passage")
ax.plot(horizons, merton, lw=2, ls="--", label="Merton: below K at the horizon")
ax.errorbar(T, p_bridge.mean(), yerr=4 * p_bridge.std() / np.sqrt(p_bridge.size), fmt="o", color="k", label="Monte Carlo, bridge-corrected")
ax.plot(T, hit_grid.mean(), "x", color="C3", ms=8, label="Monte Carlo, grid only (biased low)")
ax.set_xlabel("horizon (years)")
ax.set_ylabel("default probability")
ax.legend()

plt.show()
