r"""
Litterman-Scheinkman: level, slope, curvature
=============================================

Litterman and Scheinkman (1991) ran a principal component analysis of
Treasury returns and found that three factors explain almost all of the
variation. They named them level, steepness (slope) and curvature. The
script repeats the analysis on monthly changes in the U.S. Treasury
constant-maturity yields from 3 months to 10 years (FRED, H.15) since
1981, and shows the three loadings and their shares of variance.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.datasets import load_treasury_yields
from financialkit.fixed_income import yield_curve_pca
from financialkit.fixed_income.visualizers import plot_pca_loadings

curve = load_treasury_yields()
tenors = curve.tenors
changes = np.diff(curve.yields, axis=0)
print(f"{changes.shape[0]} monthly changes, {curve.dates[1]} to {curve.dates[-1]}")
pca = yield_curve_pca(changes, tenors)
level, slope = pca.loadings[:, 0], pca.loadings[:, 1]
assert np.all(np.sign(level) == np.sign(level[0]))  # level moves every tenor the same way
assert np.all(np.diff(slope) > 0) or np.all(np.diff(slope) < 0)  # slope tilts the curve

# %%
# Loadings of the first three components
# --------------------------------------

plot_pca_loadings(pca)

# %%
# Share of variance explained
# ---------------------------

fig, ax = plt.subplots()
ax.bar(np.arange(1, tenors.size + 1), 100 * pca.explained_variance_ratio)
ax.plot(np.arange(1, tenors.size + 1), 100 * np.cumsum(pca.explained_variance_ratio), "ko-", label="cumulative")
ax.set_xlabel("component")
ax.set_ylabel("variance explained (%)")
ax.legend()
print("level, slope, curvature explain", ", ".join(f"{100 * v:.1f}%" for v in pca.explained_variance_ratio[:3]))
print("first three components explain", f"{100 * pca.explained_variance_ratio[:3].sum():.1f}%")
assert pca.explained_variance_ratio[0] > 0.75
assert pca.explained_variance_ratio[:3].sum() > 0.98
plt.show()
