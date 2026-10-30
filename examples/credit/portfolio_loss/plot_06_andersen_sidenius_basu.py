r"""
Andersen, Sidenius and Basu: the loss recursion
===============================================

Real CDO pools mix names with different default probabilities,
correlations and notionals, so the binomial formula of the homogeneous
pool no longer applies. Andersen, Sidenius and Basu (2003) kept the
one-factor copula and noticed that, given the factor, defaults are
independent, so the loss distribution can be built exactly by adding
one name at a time:

.. math:: f^{(k+1)}_l = (1 - p_{k+1}(M))\,f^{(k)}_l + p_{k+1}(M)\,f^{(k)}_{l - u_{k+1}},

with :math:`u_k` the name's loss in units. Averaging over the factor by
Gauss-Hermite quadrature gives the portfolio loss distribution with no
simulation, at a cost linear in the number of names.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.credit import asb_loss_distribution, one_factor_loss_distribution, tranche_loss

# %%
# A mixed pool
# ------------
# 100 names: default probabilities from 0.5% to 6%, three correlation
# buckets and notionals of 1, 2 or 4 (total 200). Each name loses 60% of
# its notional, so a unit of loss is 0.6 of notional and 0.3% of the pool.

rng = np.random.default_rng(42)
n = 100
p = rng.uniform(0.005, 0.06, n)
rho = rng.choice([0.15, 0.3, 0.45], n)
notional = rng.choice([1, 2, 4], n, p=[0.5, 0.3, 0.2])
unit = 0.6 / notional.sum()
asb = asb_loss_distribution(p, rho, notional, unit=unit)
print(f"expected loss {asb.mean:.4%} (exactly sum p_i LGD_i: {0.6 * p @ notional / notional.sum():.4%})")
print(f"99.9% loss quantile {asb.quantile(0.999):.2%}")
assert np.isclose(asb.mean, 0.6 * p @ notional / notional.sum())

# %%
# The recursion against Monte Carlo
# ---------------------------------
# 100,000 draws of the same copula, and the homogeneous pool with the
# average probability and correlation that one would otherwise use.

sims = 100_000
M = rng.standard_normal((sims, 1))
Z = np.sqrt(rho) * M + np.sqrt(1 - rho) * rng.standard_normal((sims, n))
mc = unit * ((Z < stats.norm.ppf(p)) @ notional)
homogeneous = one_factor_loss_distribution(p.mean(), rho.mean(), n, lgd=0.6)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(asb.losses, 1 - np.cumsum(asb.pmf), lw=2, label="ASB recursion")
grid = asb.losses[::4]
tail_mc = np.array([np.mean(mc > x) for x in grid])
ax.semilogy(grid, tail_mc, "o", ms=3, label="Monte Carlo, 100,000 draws")
ax.semilogy(homogeneous.losses, 1 - np.cumsum(homogeneous.pmf), "--", label="homogeneous pool, average p and rho")
ax.set_xlim(0, 0.15)
ax.set_ylim(1e-5, 1)
ax.set_xlabel("portfolio loss")
ax.set_ylabel("P(loss > x)")
ax.legend()

# the recursion agrees with simulation within 4 standard errors at every level
tail_asb = np.array([asb.pmf[asb.losses > x].sum() for x in grid])
se = np.sqrt(tail_asb * (1 - tail_asb) / sims)
assert np.all(np.abs(tail_mc - tail_asb) < 4 * se + 1e-12)

# %%
# Tranche losses of the bespoke pool
# ----------------------------------
# With the distribution in hand, any tranche's expected loss is a sum.
# Replacing the pool by its average name misprices every tranche, here
# by a few percent each, although the pool's expected loss is the same.

for a, d in ((0.0, 0.03), (0.03, 0.07), (0.07, 0.15)):
    el = asb.pmf @ tranche_loss(asb.losses, a, d)
    el_h = homogeneous.pmf @ tranche_loss(homogeneous.losses, a, d)
    print(f"{100 * a:g}-{100 * d:g}%: expected tranche loss {el:.4f} (homogeneous {el_h:.4f})")
    assert abs(el_h / el - 1) > 0.02

plt.show()
