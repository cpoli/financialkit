r"""
Owen scrambling: randomized QMC
===============================

A deterministic point set gives one number and no error bar. Art Owen
(1995) randomly permuted the digits of a net, nested so that the result
is still a net: every scrambled point is uniform on the cube, so the
estimate is unbiased, independent scramblings give an honest standard
error, and for smooth integrands the root-mean-square error falls like
:math:`N^{-3/2}`, faster than the unscrambled sequence's :math:`N^{-1}`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.monte_carlo import monte_carlo_integrate, randomized_qmc_estimate, sobol
from financialkit.monte_carlo.visualizers import plot_error_convergence, plot_point_set

# %%
# Scrambling keeps the net structure
# ----------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
plot_point_set(sobol(64, 2, scramble=False), ax=axes[0], grid=8, title="unscrambled Sobol")
plot_point_set(sobol(64, 2, rng=1), ax=axes[1], grid=8, title="Owen-scrambled Sobol")

# %%
# Error bars and convergence
# --------------------------

d = 3
f = lambda u: np.prod(np.exp(u) / (np.e - 1), axis=1)  # noqa: E731  integral 1
ns = 2 ** np.arange(5, 14)
rqmc = [randomized_qmc_estimate(f, n, d, n_replications=16, rng=n) for n in ns]
mc = [monte_carlo_integrate(f, n, d, rng=n) for n in ns]
# Per-replication RMSE = standard error times sqrt(replications).
plot_error_convergence(ns, [r.std_error * 4 for r in rqmc], ax=axes[2], label="scrambled Sobol (RMSE)", rates=(1.5,))
plot_error_convergence(ns, [m.std_error for m in mc], ax=axes[2], label="Monte Carlo (SE)", rates=(0.5,))
res = rqmc[-1]
print(f"N = {ns[-1]} x 16 scramblings: {res.estimate:.8f} +/- {res.std_error:.1e}, exact 1")
fig.tight_layout()
plt.show()
