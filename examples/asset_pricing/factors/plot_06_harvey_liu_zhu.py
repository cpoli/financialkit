r"""
Harvey-Liu-Zhu: the factor zoo and multiple testing
===================================================

By 2012 more than 300 factors had been published as explaining the
cross-section of expected returns, each with a t-statistic above 2.
Campbell Harvey, Yan Liu and Heng Zhu (2016) pointed out that a 5% test
applied 300 times will find about 15 factors by luck alone. Methods that
control the family-wise error rate (Bonferroni, Holm) or the false
discovery rate (Benjamini-Hochberg-Yekutieli, valid under any
dependence) raise the hurdle. Their recommendation was that a new
factor should clear a t-statistic of about 3.0.

The zoo here is 316 long-short factors over 40 years of monthly
returns. Thirty have a real premium, with a Sharpe ratio of 0.6 a year,
and the other 286 have none.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.asset_pricing import adjusted_p_values, t_hurdle

rng = np.random.default_rng(0)
M, T, n_true = 316, 480, 30
true = np.arange(M) < n_true
premia = np.where(true, 0.6 / np.sqrt(12) * 0.03, 0.0)
returns = premia + 0.03 * rng.standard_normal((T, M))
t = returns.mean(axis=0) / returns.std(axis=0, ddof=1) * np.sqrt(T)

p_values = 2 * stats.norm.sf(np.abs(t))
rows = []
for label, method, level in [("|t| > 1.96", None, 0.05), ("Bonferroni 5%", "bonferroni", 0.05), ("Holm 5%", "holm", 0.05), ("BHY 1%", "bhy", 0.01)]:
    if method is None:
        found, hurdle = np.abs(t) > 1.96, 1.96
    else:
        found, hurdle = adjusted_p_values(p_values, method) <= level, t_hurdle(t, level, method)
    false = int((found & ~true).sum())
    rows.append((label, hurdle, int(found.sum()), false))
    print(f"{label:>14}: hurdle {hurdle:.2f}, discoveries {found.sum():3d}, false {false:2d} ({false / max(found.sum(), 1):.0%})")

single, bonf, holm, bhy = rows
# One test per factor finds roughly 5% of the 286 nulls (about 14) by chance.
assert single[3] >= 8
# Every correction keeps false discoveries to a handful at most and
# lifts the hurdle near or above Harvey, Liu and Zhu's 3.0.
assert all(r[3] <= 2 for r in (bonf, holm, bhy))
assert all(r[1] > 2.8 for r in (bonf, holm, bhy))

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(-4, 8, 49)
ax.hist(t[~true], bins=bins, alpha=0.6, label="no premium")
ax.hist(t[true], bins=bins, alpha=0.6, label="true premium")
for (label, hurdle, *_), c in zip(rows, ["k", "C3", "C2", "C4"], strict=True):
    ax.axvline(hurdle, color=c, ls="--", label=f"{label} ({hurdle:.2f})")
ax.set_xlabel("t-statistic of the factor premium")
ax.set_ylabel("factors")
ax.legend(fontsize=8)
fig.tight_layout()

# %%
# The hurdle rises with the number of factors tried
# -------------------------------------------------
#
# The Bonferroni hurdle depends only on the number of tests. With 316
# factors it is 3.78, the figure in Harvey, Liu and Zhu's Table 1 for
# 2012.

counts = np.arange(1, 400)
hurdles = [t_hurdle(np.zeros(n), 0.05, "bonferroni") for n in counts]
assert round(t_hurdle(np.zeros(316), 0.05, "bonferroni"), 2) == 3.78
assert np.isclose(hurdles[0], 1.96, atol=0.005)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(counts, hurdles, label="Bonferroni, 5%")
ax.axhline(3.0, color="C3", ls="--", label="Harvey-Liu-Zhu's recommendation")
ax.axhline(1.96, color="0.5", ls=":", label="a single test")
ax.set_xlabel("number of factors tested")
ax.set_ylabel("t-statistic hurdle")
ax.legend()
fig.tight_layout()

plt.show()
