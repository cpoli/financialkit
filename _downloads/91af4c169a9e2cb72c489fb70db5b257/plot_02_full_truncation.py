r"""
Lord, Koekkoek and van Dijk's full truncation
=============================================

An Euler step for the square-root variance can go negative, and the
repairs proposed in the literature differ in where they take the
positive part: in the diffusion only (partial truncation, Deelstra and
Delbaen 1998), as an absolute value in the diffusion (Higham and Mao
2005), or in both the drift and the diffusion while letting the
variance itself go negative (full truncation). Lord, Koekkoek and van
Dijk (2010) put them in one framework and found full truncation the
least biased. All three schemes here run through
:func:`financialkit.integrators.euler_maruyama` with correlated increments.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.stochastic_volatility import HestonModel, heston_mc_price, heston_paths
from financialkit.stochastic_volatility.visualizers import plot_heston_paths

m = HestonModel(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
S, K, T, r = 100.0, 100.0, 10.0, 0.0
exact = m.price(S, K, T, r)

# Higham-Mao's absolute value keeps feeding large variances back in, so
# its estimates are both biased and very noisy.

steps = np.array([10, 20, 40, 80])
fig, ax = plt.subplots(figsize=(7, 4))
for scheme in ("full_truncation", "partial_truncation", "higham_mao"):
    res = [heston_mc_price(m, S, K, T, r, n_steps=n, n_paths=20_000, scheme=scheme, rng=3) for n in steps]
    bias = np.array([x.estimate - exact for x in res])
    se = np.array([x.std_error for x in res])
    print(f"{scheme:>18}: bias {np.round(bias, 2)}, standard error {np.round(se, 2)}")
    ax.errorbar(T / steps, bias, yerr=2 * se, marker="o", ms=3, capsize=2, label=scheme)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_ylim(-15, 15)
ax.set_xlabel("time step")
ax.set_ylabel("price bias (exact 13.0847)")
ax.legend()

# %%
# Full truncation lets the variance go negative
# ---------------------------------------------
#
# The state :math:`v` itself crosses zero, but only :math:`v^+` enters
# the coefficients, so it drifts back up at rate :math:`\kappa\theta`.

paths = heston_paths(m, S, 2.0, r, n_steps=200, n_paths=8, scheme="full_truncation", rng=4)
print(f"lowest simulated variance: {paths.variance.min():.4f}")
plot_heston_paths(paths)

plt.show()
