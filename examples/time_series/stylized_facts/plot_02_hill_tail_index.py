r"""
Hill tail index
===============

If the tail is a power law, :math:`P(X > u) \approx C u^{-\alpha}`, the
log-excesses of the largest observations over a threshold are
exponential with mean :math:`1/\alpha`. Bruce Hill (1975) turned this
into the estimator

.. math:: \hat\alpha_k = \Bigl[\frac1k\sum_{i=1}^{k}\ln X_{(i)} - \ln X_{(k+1)}\Bigr]^{-1},

using only the :math:`k` largest order statistics. The tail index decides
which moments exist (those of order below :math:`\alpha`), and so whether
the variance and kurtosis that risk models rely on are finite.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.datasets import load_market_daily
from financialkit.time_series import hill_estimator
from financialkit.time_series.visualizers import plot_hill

# %%
# The Hill plot on a known tail
# -----------------------------
# For an exact Pareto tail the estimate is flat in :math:`k`. For a Student
# t with :math:`\nu = 4`, only the far tail is a power law, so the estimate
# drifts once :math:`k` reaches into the body: choosing :math:`k` trades
# variance (small :math:`k`) against bias (large :math:`k`).

rng = np.random.default_rng(1975)
n = 20_000
pareto = rng.pareto(3.0, n) + 1.0
student = np.abs(rng.standard_t(4, n))

hill_pareto = hill_estimator(pareto, k=np.arange(20, 3000))
hill_student = hill_estimator(student, k=np.arange(20, 3000))
assert abs(np.median(hill_pareto.alpha) - 3) < 0.15  # flat around the truth
assert hill_student.alpha[-1] < hill_student.alpha[0] - 0.5  # biased down into the body

fig, ax = plt.subplots()
plot_hill(hill_pareto, ax=ax, label=r"Pareto, $\alpha$ = 3")
plot_hill(hill_student, ax=ax, label=r"|t(4)|, $\alpha$ = 4")
for a in (3, 4):
    ax.axhline(a, color="k", ls="--", lw=0.8)
ax.set_ylim(1, 7)

# %%
# The tails of the U.S. stock market
# ----------------------------------
# Daily log returns on the CRSP value-weighted market since 1926 (French's
# data library). Both tails have :math:`\alpha` near 3, the "cubic law" of
# Jansen and de Vries (1991) and Gabaix et al. (2003): the variance is
# finite, but the fourth moment, and so the sample kurtosis, is not
# reliably so.

r = np.log1p(load_market_daily().returns)
k = np.arange(50, 2000)
fig, ax = plt.subplots()
plot_hill(hill_estimator(r, k=k, side="lower"), ax=ax, label="losses (left tail)")
plot_hill(hill_estimator(r, k=k, side="upper"), ax=ax, label="gains (right tail)")
ax.axhline(3, color="k", ls="--", lw=0.8)
ax.set_ylim(1, 5)
ax.set_title("U.S. market, daily log returns, 1926 onwards")
alpha_250 = {side: hill_estimator(r, k=250, side=side).alpha[0] for side in ("lower", "upper")}
for side, a in alpha_250.items():
    print(f"{side} tail: alpha at k = 250 is {a:.2f}")
    assert 2.5 < a < 3.5
plt.show()
