r"""
Hawkes: self-exciting order flow
================================

Alan Hawkes (1971) defined a point process whose intensity jumps by
:math:`\alpha` at every event and decays back at rate :math:`\beta`,

.. math::

   \lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta (t - t_i)},

so each event makes the next more likely. Each event triggers on average
:math:`n = \alpha/\beta` others, and the process is stationary when
:math:`n < 1`, with mean rate :math:`\mu/(1 - n)`. Order arrivals in
electronic markets cluster this way, with fitted branching ratios that
are often close to one. Here a simulated stream of orders is fitted by
maximum likelihood, and Ogata's residual test tells it apart from a
Poisson stream with the same rate.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.market_microstructure import fit_hawkes, hawkes_compensator, hawkes_intensity, simulate_hawkes

mu, alpha, beta, T = 0.5, 1.6, 2.0, 2000.0
t = simulate_hawkes(mu, alpha, beta, T, rng=7)
print(f"{t.size} orders, expected {mu * T / (1 - alpha / beta):.0f}")

# %%
# Clustering
# ----------
# The intensity spikes with every order and the orders bunch together.
# A Poisson stream at the same mean rate is spread evenly.

grid = np.linspace(0.0, 100.0, 2000)
poisson = np.sort(np.random.default_rng(7).uniform(0.0, T, t.size))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax1.plot(grid, hawkes_intensity(t, grid, mu, alpha, beta), "C0-", lw=1)
ax1.plot(t[t < 100], np.zeros(np.sum(t < 100)), "k|", ms=12)
ax1.set_ylabel("intensity")
ax1.set_title("Hawkes")
ax2.plot(poisson[poisson < 100], np.zeros(np.sum(poisson < 100)), "k|", ms=12)
ax2.set_title("Poisson, same rate")
ax2.set_yticks([])
ax2.set_xlabel("time")
fig.tight_layout()

# Counts in unit windows: the Hawkes counts are overdispersed.
counts = np.bincount(t.astype(int), minlength=int(T))
print(f"variance / mean of counts per unit time: {counts.var() / counts.mean():.2f} (Poisson: 1)")

# %%
# Maximum likelihood
# ------------------

fit = fit_hawkes(t, T)
print(f"mu = {fit.mu:.3f} ({mu}), alpha = {fit.alpha:.3f} ({alpha}), beta = {fit.beta:.3f} ({beta})")
print(f"branching ratio {fit.branching_ratio:.3f} ({alpha / beta})")
assert abs(fit.branching_ratio - alpha / beta) < 0.1

# %%
# Residual analysis
# -----------------
# Under the right model the compensator increments
# :math:`\Lambda(t_i) - \Lambda(t_{i-1})` are unit exponentials (Ogata,
# 1988). The fitted Hawkes model passes; a Poisson model of the same
# orders fails, because it ignores the clustering.

res_hawkes = np.diff(hawkes_compensator(t, fit.mu, fit.alpha, fit.beta), prepend=0.0)
res_poisson = np.diff(t, prepend=0.0) * t.size / T
q = stats.expon.ppf((np.arange(t.size) + 0.5) / t.size)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(q, np.sort(res_hawkes), "C0.", ms=3, label="Hawkes residuals")
ax.plot(q, np.sort(res_poisson), "C3.", ms=3, label="Poisson residuals")
ax.plot([0, q.max()], [0, q.max()], "k--", lw=1)
ax.set_xlabel("Exp(1) quantile")
ax.set_ylabel("residual quantile")
ax.legend()
fig.tight_layout()
p_hawkes, p_poisson = stats.kstest(res_hawkes, "expon").pvalue, stats.kstest(res_poisson, "expon").pvalue
print(f"Kolmogorov-Smirnov p-values: Hawkes {p_hawkes:.3f}, Poisson {p_poisson:.1e}")
assert p_hawkes > 0.01 > p_poisson

plt.show()
