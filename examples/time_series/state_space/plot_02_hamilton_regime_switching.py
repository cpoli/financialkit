r"""
Hamilton's regime-switching model
=================================

James Hamilton (1989) let the growth rate of U.S. output switch between
expansion and recession according to a hidden two-state Markov chain, and
gave the filter that turns each new observation into updated probabilities
of the two states. Applied to returns with a switching mean *and*
variance, the same model separates calm bull markets from turbulent bear
markets. Here a two-regime model is estimated by the EM algorithm on
simulated daily returns, and Kim's smoother recovers the hidden regimes.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.constants import TRADING_DAYS_PER_YEAR
from financialkit.time_series import fit_markov_switching, simulate_markov_switching

# %%
# Bull and bear regimes
# ---------------------
# The calm regime has a 15% annual volatility and lasts a hundred days on
# average; the turbulent one has 45% volatility, a negative drift, and
# lasts about a month. Volatilities and durations are well identified;
# drifts, as always with returns, are not.

ann = TRADING_DAYS_PER_YEAR
mu = np.array([0.10, -0.30]) / ann
sigma = np.array([0.15, 0.45]) / np.sqrt(ann)
P = np.array([[0.99, 0.01], [0.04, 0.96]])
r, regime = simulate_markov_switching(2500, mu, sigma, P, rng=1989)

fit = fit_markov_switching(r)
for j, name in enumerate(("calm", "turbulent")):
    print(
        f"{name:9s} drift {ann * fit.mu[j]:+.2f} (true {ann * mu[j]:+.2f}), "
        f"vol {np.sqrt(ann) * fit.sigma[j]:.2f} (true {np.sqrt(ann) * sigma[j]:.2f}), "
        f"duration {fit.expected_duration[j]:5.1f} days (true {1 / (1 - P[j, j]):5.1f})"
    )
assert np.allclose(fit.sigma, sigma, rtol=0.1)
assert np.allclose(np.diag(fit.transition), np.diag(P), atol=0.015)

# %%
# Filtered and smoothed regime probabilities
# ------------------------------------------
# The filter uses data up to each day and reacts to a turbulent spell only
# after it has started; the smoother uses the whole sample and dates the
# switches more sharply.

accuracy = np.mean(fit.most_likely_regime() == regime)
print(f"days classified correctly by the smoothed probabilities: {accuracy:.1%}")
assert accuracy > 0.9

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
days = np.arange(r.size)
axes[0].plot(days, 100 * r, lw=0.5, color="0.3")
axes[0].fill_between(days, -12, 12, where=regime == 1, color="C3", alpha=0.15, label="true turbulent regime")
axes[0].set_ylim(-12, 12)
axes[0].set_ylabel("daily return (%)")
axes[0].legend(loc="lower left", fontsize=8)
axes[1].plot(days, fit.filtered_probs[:, 1], color="C0", lw=0.7, label="filtered")
axes[1].plot(days, fit.smoothed_probs[:, 1], color="C3", lw=1.0, label="smoothed")
axes[1].set_ylabel("P(turbulent)")
axes[1].set_xlabel("day")
axes[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
plt.show()
