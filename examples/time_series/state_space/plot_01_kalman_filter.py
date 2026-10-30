r"""
Kalman filter and the state-space model
=======================================

Rudolf Kalman (1960) wrote a dynamic system as a hidden state that evolves
linearly, :math:`x_t = F x_{t-1} + w_t`, and is seen only through noisy
measurements, :math:`y_t = H x_t + v_t`. His recursion updates the
Gaussian mean and covariance of the state with each new measurement,
weighting the news by a *gain* that balances the noise in the state
against the noise in the measurement. Here the state is an efficient
price that follows a random walk, and the measurements are trade prices
contaminated by microstructure noise: the *local-level* model.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import fit_local_level, kalman_filter, local_level_steady_state_gain

# %%
# An efficient price seen through noise
# -------------------------------------
# The efficient log price moves by 0.1% a period; each observed price adds
# an independent 0.2% error (bid-ask bounce, stale quotes).

rng = np.random.default_rng(1960)
n, s_eta, s_eps = 600, 0.001, 0.002
price = np.cumsum(s_eta * rng.standard_normal(n))
y = price + s_eps * rng.standard_normal(n)

res = kalman_filter(y, F=1.0, H=1.0, Q=s_eta**2, R=s_eps**2, x0=0.0, P0=1.0)
smoothed, smoothed_cov = res.smooth()
filtered = res.filtered_state[:, 0]


def rmse(x):
    return 1e4 * np.sqrt(np.mean((x - price) ** 2))


print(f"RMSE in bp: observed {rmse(y):.1f}, filtered {rmse(filtered):.1f}, smoothed {rmse(smoothed[:, 0]):.1f}")
assert rmse(smoothed[:, 0]) < rmse(filtered) < rmse(y)

fig, ax = plt.subplots(figsize=(8, 3.8))
sl = slice(200, 300)
t = np.arange(n)[sl]
ax.plot(t, 1e4 * y[sl], ".", color="0.6", ms=4, label="observed price")
ax.plot(t, 1e4 * price[sl], "k", lw=1, label="efficient price")
ax.plot(t, 1e4 * filtered[sl], "C0", label="filtered $\\hat x_{t|t}$")
band = 1.96e4 * np.sqrt(smoothed_cov[sl, 0, 0])
ax.plot(t, 1e4 * smoothed[sl, 0], "C3", label="smoothed $\\hat x_{t|n}$")
ax.fill_between(t, 1e4 * smoothed[sl, 0] - band, 1e4 * smoothed[sl, 0] + band, color="C3", alpha=0.15)
ax.set_xlabel("period")
ax.set_ylabel("log price (bp)")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()

# %%
# The gain settles at its steady state
# ------------------------------------
# Starting from a vague prior, the gain falls quickly to the solution of
# the Riccati equation. In the steady state the filter is an exponentially
# weighted moving average of the prices, with weight :math:`K` on the
# newest one.

q = s_eta**2 / s_eps**2
K_ss = local_level_steady_state_gain(q)
print(f"signal-to-noise q = {q:.2f}, steady-state gain K = {K_ss:.4f}, final gain {res.gain[-1, 0, 0]:.4f}")
assert abs(res.gain[-1, 0, 0] - K_ss) < 1e-10

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(res.gain[:30, 0, 0], "o-", ms=3)
ax.axhline(K_ss, color="C3", ls="--", label="Riccati steady state")
ax.set_xlabel("period")
ax.set_ylabel("Kalman gain $K_t$")
ax.legend()
fig.tight_layout()

# %%
# Maximum likelihood by the prediction-error decomposition
# --------------------------------------------------------
# The innovations :math:`y_t - \hat x_{t|t-1}` are independent with known
# variances, so the filter also gives the exact likelihood. Maximizing it
# recovers the two noise levels from the observed prices alone.

fit = fit_local_level(y)
print(f"sigma_eta: true {1e4 * s_eta:.1f} bp, estimated {1e4 * np.sqrt(fit.sigma2_level):.1f} bp")
print(f"sigma_eps: true {1e4 * s_eps:.1f} bp, estimated {1e4 * np.sqrt(fit.sigma2_noise):.1f} bp")
assert abs(np.sqrt(fit.sigma2_level) / s_eta - 1) < 0.3
assert abs(np.sqrt(fit.sigma2_noise) / s_eps - 1) < 0.15
plt.show()
