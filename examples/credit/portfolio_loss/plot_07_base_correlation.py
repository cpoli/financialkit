r"""
Base correlation
================

After 2003 dealers quoted index tranches by the flat copula correlation
that reprices each one, its *compound* correlation. The equity and
senior tranches have unique compound correlations, but a mezzanine
tranche's spread first rises and then falls with correlation, so a
quote can have two compound correlations or none. McGinty, Beinstein,
Ahluwalia and Watts (2004) at J.P. Morgan quoted *base* correlations
instead: the correlation of each equity tranche :math:`[0, K]`, whose
expected loss falls monotonically with correlation. A tranche
:math:`[A, D]` is the difference of the equity tranches :math:`[0, D]`
and :math:`[0, A]`, each at its own correlation, so the skew is
bootstrapped from the bottom of the capital structure up.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import HazardCurve, base_correlation, base_correlation_spread, compound_correlations, tranche_spread

# %%
# The mezzanine problem
# ---------------------
# A five-year pool with a 1.2% hazard rate. The 3-7% tranche's spread is
# hump-shaped in correlation: a quote below the hump has two compound
# correlations, a quote above it none.

curve = HazardCurve.flat(0.012)
rhos = np.linspace(0.01, 0.95, 25)
mezz = np.array([tranche_spread(0.03, 0.07, curve, rho, 5.0, 0.03).par_spread for rho in rhos])
quote = 0.0340
roots = compound_correlations(quote, 0.03, 0.07, curve, 5.0, 0.03, n_grid=10)
none = compound_correlations(0.06, 0.03, 0.07, curve, 5.0, 0.03, n_grid=10)
print("compound correlations of a 340 bp 3-7% quote:", np.round(roots, 3))
print("compound correlations of a 600 bp quote:", none)
assert roots.size == 2 and none.size == 0

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rhos, 1e4 * mezz, lw=2)
ax.axhline(1e4 * quote, color="k", ls=":", label="quote, 340 bp")
ax.plot(roots, [1e4 * quote] * roots.size, "o", color="C3", label="two compound correlations")
ax.set_xlabel("flat copula correlation")
ax.set_ylabel("3-7% par spread (bp)")
ax.legend()

# %%
# Bootstrapping the skew
# ----------------------
# Quotes for the standard capital structure, here generated from a
# rising skew, are turned back into base correlations one tranche at a
# time, each a unique root.

K = np.array([0.03, 0.07, 0.10, 0.15, 0.30])
skew = np.array([0.15, 0.25, 0.32, 0.40, 0.55])
A, RA = np.r_[0.0, K[:-1]], np.r_[0.0, skew[:-1]]
quotes = [base_correlation_spread(a, d, ra, rd, curve, 5.0, 0.03).par_spread for a, d, ra, rd in zip(A, K, RA, skew, strict=True)]
bc = base_correlation(K, quotes, curve, 5.0, 0.03)
for a, d, s, rho in zip(A, K, quotes, bc.base_correlations, strict=True):
    print(f"{100 * a:>4g}-{100 * d:<3g}%: {1e4 * s:7.1f} bp -> base correlation {rho:.3f}")
assert np.allclose(bc.base_correlations, skew, atol=1e-8)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(100 * K, bc.base_correlations, "o-", lw=2, label="base correlation")
compound = [compound_correlations(s, a, d, curve, 5.0, 0.03, n_grid=10) for a, d, s in zip(A, K, quotes, strict=True)]
for d, c in zip(K, compound, strict=True):
    ax.plot([100 * d] * c.size, c, "x", color="C3", ms=8)
ax.plot([], [], "x", color="C3", label="compound correlation(s) of the same quote")
ax.set_xlabel("detachment (%)")
ax.set_ylabel("correlation")
ax.legend()

# %%
# Pricing a bespoke tranche
# -------------------------
# A 5-12% tranche is not quoted. Interpolating the skew at 5% and 12%
# prices it consistently with every quoted tranche.

rho_a, rho_d = bc.correlation(0.05), bc.correlation(0.12)
bespoke = base_correlation_spread(0.05, 0.12, rho_a, rho_d, curve, 5.0, 0.03)
print(f"5-12%: base correlations {rho_a:.3f} and {rho_d:.3f}, par spread {1e4 * bespoke.par_spread:.1f} bp")
assert quotes[2] < bespoke.par_spread < quotes[1]

plt.show()
