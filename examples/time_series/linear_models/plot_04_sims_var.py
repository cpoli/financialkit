r"""
Sims's vector autoregression and impulse responses
==================================================

Christopher Sims (1980) called the identifying restrictions of the large
macroeconometric models "incredible" and proposed an alternative: regress
each variable on lags of *all* the variables, and read the dynamics off
the responses to orthogonalized shocks. The shocks come from a Cholesky
factorization of the residual covariance, so the ordering of the
variables says which shocks can move which variables within the period.
Here a three-variable system of output growth, inflation and a policy
rate is simulated from a known VAR(2), estimated, and summarized by its
impulse responses with bootstrap bands and its variance decomposition.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.time_series import VectorAutoregressionResult, fit_var, simulate_var

# %%
# A small macro system
# --------------------
# The rate responds to inflation and output (a Taylor-type rule), and a
# higher rate lowers output growth and inflation with a lag.

names = ["output growth", "inflation", "policy rate"]
A1 = np.array([[0.4, 0.0, -0.20], [0.10, 0.6, -0.05], [0.15, 0.25, 0.70]])
A2 = np.array([[0.1, 0.0, 0.00], [0.00, 0.2, 0.00], [0.00, 0.00, 0.10]])
Sigma = np.array([[1.0, 0.2, 0.1], [0.2, 0.5, 0.05], [0.1, 0.05, 0.3]])
y = simulate_var([A1, A2], 400, sigma=Sigma, rng=1980)

aic = {p: fit_var(y, p, start=6).aic for p in range(1, 7)}
best = min(aic, key=lambda p: aic[p])
print("AIC by order:", {p: round(v, 1) for p, v in aic.items()}, "-> p =", best)
res = fit_var(y, 2)
print("stable:", res.is_stable())
assert res.is_stable() and best in (2, 3)

# %%
# Orthogonalized impulse responses
# --------------------------------
# A residual bootstrap gives 90% bands: resample the fitted residuals,
# rebuild the series from the fitted model, refit and recompute the
# responses.

h = 16
true = VectorAutoregressionResult(np.stack([A1, A2]), np.zeros(3), Sigma, np.empty((0, 3)), 0.0, 1, 0, y).impulse_response(h)
irf = res.impulse_response(h)
rng = np.random.default_rng(0)
boot = []
for _ in range(200):
    u = res.residuals[rng.integers(0, res.n_obs, res.n_obs)]
    yb = y.copy()
    for t in range(2, y.shape[0]):
        yb[t] = res.intercept + res.coefs[0] @ yb[t - 1] + res.coefs[1] @ yb[t - 2] + u[t - 2]
    boot.append(fit_var(yb, 2).impulse_response(h))
lo, hi = np.percentile(boot, [5, 95], axis=0)
inside = np.mean((true >= lo) & (true <= hi))
print(f"true responses inside the 90% bands: {inside:.0%}")
assert inside > 0.75

fig, axes = plt.subplots(3, 3, figsize=(9, 7), sharex=True)
for a in range(3):
    for b in range(3):
        ax = axes[a, b]
        ax.fill_between(range(h + 1), lo[:, a, b], hi[:, a, b], color="C0", alpha=0.2)
        ax.plot(irf[:, a, b], "C0", label="estimated")
        ax.plot(true[:, a, b], "k--", lw=1, label="true")
        ax.axhline(0, color="0.5", lw=0.6)
        if a == 0:
            ax.set_title(f"shock: {names[b]}", fontsize=9)
        if b == 0:
            ax.set_ylabel(names[a], fontsize=9)
axes[0, 0].legend(fontsize=7)
for ax in axes[2]:
    ax.set_xlabel("quarters")
fig.tight_layout()

# %%
# Forecast-error variance decomposition
# -------------------------------------
# The share of each variable's forecast-error variance due to each shock.
# At horizon one the first-ordered variable moves only with its own
# shock; over time the policy rate is driven increasingly by inflation.

fevd = res.fevd(h)
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for a, ax in enumerate(axes):
    ax.stackplot(range(1, h + 1), fevd[:, a, :].T, labels=names, alpha=0.8)
    ax.set_title(names[a], fontsize=9)
    ax.set_xlabel("horizon")
axes[0].set_ylabel("share of variance")
axes[2].legend(fontsize=7, loc="lower left")
fig.tight_layout()
print("1-step decomposition of output growth:", fevd[0, 0].round(3))
assert fevd[0, 0, 0] == 1.0 and np.allclose(fevd.sum(axis=2), 1.0)
plt.show()
