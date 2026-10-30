r"""
Barraquand moment matching
==========================

A finite sample of normals never has mean exactly 0 and variance exactly
1, and the payoff inherits those sampling errors. Jérôme Barraquand
(1995), pricing multi-asset options, simply rescaled the sample,

.. math:: \tilde z_i = \frac{z_i - \bar z}{s_z},

so the first two moments are exact ("quadratic resampling"). The draws
become dependent, so error bars come from independent replications.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.monte_carlo import moment_match, moment_matching_estimate, sample_mean, terminal_prices, vanilla_payoff
from financialkit.monte_carlo.visualizers import plot_estimator_comparison
from financialkit.options import black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
exact = black_scholes_price(S, K, T, r, sigma)


def call(z):
    return np.exp(-r * T) * vanilla_payoff(terminal_prices(S, T, r, sigma, z), K)


# %%
# Matched samples have exact moments
# ----------------------------------

z = np.random.default_rng(0).standard_normal(500)
w = moment_match(z)
print(f"raw:     mean {z.mean():+.4f}, std {z.std():.4f}")
print(f"matched: mean {w.mean():+.4f}, std {w.std():.4f}")
assert np.isclose(w.mean(), 0.0, atol=1e-12) and np.isclose(w.std(), 1.0)

# %%
# Twenty replications of 2000 draws each
# --------------------------------------

n, reps = 2000, 20
gen = np.random.default_rng(1)
crude = sample_mean([call(gen.standard_normal(n)).mean() for _ in range(reps)], method="crude", n_evaluations=n * reps)
matched = moment_matching_estimate(call, n, n_replications=reps, rng=2)
print(f"crude {crude.estimate:.4f} +/- {crude.std_error:.4f}; matched {matched.estimate:.4f} +/- {matched.std_error:.4f}; exact {exact:.4f}")
assert abs(matched.estimate - exact) < 3 * matched.std_error and matched.std_error < crude.std_error

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_estimator_comparison([crude, matched], exact=exact, ax=axes[0])

# %%
# The gain is largest for small samples
# -------------------------------------

ns = np.array([100, 300, 1000, 3000, 10000])
ratio = []
for k in ns:
    g = np.random.default_rng(int(k))
    c = sample_mean([call(g.standard_normal(k)).mean() for _ in range(reps)], n_evaluations=k * reps)
    ratio.append(moment_matching_estimate(call, int(k), n_replications=reps, rng=int(k) + 1).variance_ratio(c))
assert min(ratio) > 1 and ratio[0] > ratio[-1]
axes[1].semilogx(ns, ratio, "o-")
axes[1].set_xlabel("draws per replication")
axes[1].set_ylabel("variance reduction factor")
fig.tight_layout()
plt.show()
