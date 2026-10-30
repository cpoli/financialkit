r"""
Engle's dynamic conditional correlation (DCC-GARCH)
===================================================

Diversification fails when it is needed most: correlations between assets
rise in sell-offs. Multivariate GARCH models can capture this, but with a
free covariance matrix their parameters grow with the fourth power of the
number of assets. Robert Engle (2002) split the problem in two. Fit a
univariate GARCH to each asset, standardize the returns, and let the
correlation matrix of the standardized returns follow a scalar GARCH-like
recursion with just two parameters, its intercept fixed at the sample
correlation (*correlation targeting*).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.constants import TRADING_DAYS_PER_YEAR
from financialkit.time_series import GARCH, fit_dcc, simulate_dcc

# %%
# Two assets with moving correlation
# ----------------------------------
# Daily returns in percent; each has GARCH(1,1) volatility, and their
# correlation follows DCC(1,1) around a long-run 0.4.

ann = TRADING_DAYS_PER_YEAR
stock = GARCH(omega=0.02, alpha=0.09, beta=0.89)
bond_proxy = GARCH(omega=0.01, alpha=0.05, beta=0.93)
Rbar = np.array([[1.0, 0.4], [0.4, 1.0]])
r, R_true = simulate_dcc([stock, bond_proxy], a=0.06, b=0.92, Rbar=Rbar, n=2500, rng=2002)

fit = fit_dcc(r)
print(f"DCC: a = {fit.a:.3f} (true 0.06), b = {fit.b:.3f} (true 0.92), persistence {fit.persistence:.3f}")
for name, f in zip(("asset 1", "asset 2"), fit.univariate, strict=True):
    print(f"{name}: GARCH alpha = {f.params['alpha']:.3f}, beta = {f.params['beta']:.3f}")
assert abs(fit.a - 0.06) < 0.03 and abs(fit.b - 0.92) < 0.04

# %%
# Tracking the correlation
# ------------------------
# DCC against a 60-day rolling correlation and the constant correlation
# of Bollerslev (1990).

rho = fit.correlation[:, 0, 1]
w = 60
z = r - r.mean(axis=0)
rolling = np.full(r.shape[0], np.nan)
for t in range(w, r.shape[0]):
    rolling[t] = np.corrcoef(z[t - w : t].T)[0, 1]
constant = np.corrcoef(r.T)[0, 1]
true = R_true[:, 0, 1]
sl = slice(w, None)
errors = {
    "DCC": np.mean(np.abs(rho[sl] - true[sl])),
    "60-day rolling": np.mean(np.abs(rolling[sl] - true[sl])),
    "constant": np.mean(np.abs(constant - true[sl])),
}
for k, v in errors.items():
    print(f"mean absolute error, {k:15s}: {v:.3f}")
assert errors["DCC"] < errors["60-day rolling"] < errors["constant"]

fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
days = np.arange(r.shape[0])
axes[0].plot(days, np.sqrt(ann) * fit.volatility, lw=0.7)
axes[0].set_ylabel("volatility (% a year)")
axes[0].legend(["asset 1", "asset 2"], fontsize=8)
axes[1].plot(days, true, "k", lw=0.8, label="true $\\rho_t$")
axes[1].plot(days, rolling, color="0.6", lw=0.8, label="60-day rolling")
axes[1].plot(days, rho, "C3", lw=1.0, label="DCC")
axes[1].axhline(constant, color="C0", ls="--", lw=1, label="constant")
axes[1].set_ylabel("correlation")
axes[1].set_xlabel("day")
axes[1].legend(fontsize=8, ncol=4, loc="lower left")
fig.tight_layout()

# %%
# What a constant correlation misses in portfolio risk
# ----------------------------------------------------
# The volatility of a 60/40 portfolio with the DCC covariance, against the
# same univariate volatilities joined by the constant correlation. The
# constant model understates risk whenever :math:`\rho_t` is above its
# average and overstates it when it is below.

wts = np.array([0.6, 0.4])
H = fit.covariance
port_dcc = np.sqrt(ann * np.einsum("i,tij,j->t", wts, H, wts))
d = fit.volatility
H_const = d[:, :, None] * np.array([[1.0, constant], [constant, 1.0]]) * d[:, None, :]
port_const = np.sqrt(ann * np.einsum("i,tij,j->t", wts, H_const, wts))
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(port_const, port_dcc, s=3, alpha=0.4)
lim = [port_const.min(), port_const.max()]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("portfolio vol, constant correlation (%)")
ax.set_ylabel("portfolio vol, DCC (%)")
fig.tight_layout()
gap = port_dcc / port_const - 1
print(f"constant correlation misstates portfolio volatility by {100 * gap.min():+.0f}% to {100 * gap.max():+.0f}%")
assert gap.max() - gap.min() > 0.1 and np.corrcoef(rho - constant, gap)[0, 1] > 0.9
plt.show()
