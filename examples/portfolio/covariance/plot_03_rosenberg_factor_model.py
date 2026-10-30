r"""
Rosenberg's fundamental factor risk model (Barra)
=================================================

Sharpe's single index cannot tell a bank from a utility once their betas
match. Barr Rosenberg (1974) let the observable characteristics of a
company, its industry, size and value, define its exposures :math:`X` to
a handful of common factors. Each month a cross-sectional regression of
the stock returns on those exposures estimates the factor returns
:math:`f_t`, and the covariance matrix is rebuilt from them:

.. math:: r_t = X f_t + u_t, \qquad \Sigma = XFX' + \operatorname{diag}(s^2).

For 100 stocks the sample covariance has 5050 free numbers; this model
with five industries and two styles has 128. Rosenberg's firm, Barra,
sold such models to most of the industry. Here the returns are drawn
from a factor model, so the estimates can be compared with the truth.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.portfolio import fundamental_factor_covariance, ledoit_wolf, min_variance_portfolio, sample_covariance

rng = np.random.default_rng(11)
d, n = 100, 120
industry = np.eye(5)[np.arange(d) % 5]
styles = rng.standard_normal((d, 2))
styles -= styles.mean(axis=0)
X = np.column_stack([industry, styles])
names = [f"industry {i + 1}" for i in range(5)] + ["size", "value"]
F = np.diag([0.05, 0.06, 0.04, 0.07, 0.05, 0.02, 0.015]) ** 2 + 0.6 * 0.05**2 * np.outer(np.r_[np.ones(5), 0, 0], np.r_[np.ones(5), 0, 0])
spec = rng.uniform(0.04, 0.10, d) ** 2
true = X @ F @ X.T + np.diag(spec)
L = np.linalg.cholesky(F)


def simulate(n_months):
    f = rng.standard_normal((n_months, 7)) @ L.T
    return f, f @ X.T + rng.standard_normal((n_months, d)) * np.sqrt(spec)


f, R = simulate(n)
model = fundamental_factor_covariance(R, X)
print(f"parameters: sample {d * (d + 1) // 2}, factor model {model.n_params}")

# %%
# Factor returns from cross-sectional regressions
# -----------------------------------------------
# Each month's regression on 100 stocks recovers that month's factor
# returns, up to the specific noise averaged over the cross-section. The
# style factors have the smaller estimation errors, because every stock
# carries a style exposure but only 20 belong to each industry.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for j, color in ((0, "C0"), (5, "C1")):
    axes[0].scatter(f[:, j], model.factor_returns[:, j], s=10, color=color, label=names[j])
lim = np.abs(f[:, [0, 5]]).max()
axes[0].plot([-lim, lim], [-lim, lim], "k:", lw=1)
axes[0].set_xlabel("true monthly factor return")
axes[0].set_ylabel("estimated by cross-sectional regression")
axes[0].legend()
corr = [np.corrcoef(f[:, j], model.factor_returns[:, j])[0, 1] for j in range(7)]
error = np.std(model.factor_returns - f, axis=0)
print("correlation of estimated and true factor returns:", np.round(corr, 3))
print("estimation error (monthly):", np.round(error, 4))
assert min(corr) > 0.85 and max(error[5:]) < min(error[:5])

# %%
# Factor and specific risk
# ------------------------
# The model splits any portfolio's variance into a factor part and a
# specific part. An equal-weighted portfolio is almost all industry
# risk, a long-short value bet that is industry-neutral is mostly style
# risk, and a portfolio of five stocks keeps much of its specific risk.

value_bet = X[:, 6] - industry @ np.linalg.lstsq(industry, X[:, 6], rcond=None)[0]
portfolios = {
    "equal weight": np.full(d, 1.0 / d),
    "value long-short": value_bet / np.abs(value_bet).sum(),
    "five stocks": np.eye(d)[:5].sum(axis=0) / 5,
}
shares = np.array([model.risk_decomposition(w) for w in portfolios.values()])
shares /= shares.sum(axis=1, keepdims=True)
axes[1].barh(list(portfolios), shares[:, 0], label="factor")
axes[1].barh(list(portfolios), shares[:, 1], left=shares[:, 0], label="specific")
axes[1].set_xlabel("share of variance")
axes[1].legend(loc="lower right")
fig.tight_layout()
assert shares[0, 0] > 0.9 and shares[2, 1] > shares[0, 1]

# %%
# Better minimum-variance portfolios, and honest risk forecasts
# -------------------------------------------------------------
# With 120 months for 100 stocks the sample covariance is nearly
# singular: its minimum-variance portfolio looks far safer in sample
# than it is. The factor model's portfolio is both less risky in truth
# and forecasts its own risk almost exactly; Ledoit-Wolf shrinkage
# towards the identity sits in between.

estimators = {
    "sample": lambda R: sample_covariance(R).cov,
    "Ledoit-Wolf": lambda R: ledoit_wolf(R).cov,
    "factor model": lambda R: fundamental_factor_covariance(R, X).cov,
}
trials = {k: [] for k in estimators}
for _ in range(20):
    _, R = simulate(n)
    for k, est in estimators.items():
        C = est(R)
        w = min_variance_portfolio(C).weights
        trials[k].append((np.sqrt(w @ C @ w), np.sqrt(w @ true @ w)))
best = min_variance_portfolio(true).volatility
fig, ax = plt.subplots(figsize=(6, 4))
for k, vals in trials.items():
    pred, real = np.array(vals).T * np.sqrt(12)
    ax.scatter(pred, real, label=k, s=18)
    print(f"{k:13s} forecast {pred.mean():.2%}  true {real.mean():.2%}")
ax.axhline(best * np.sqrt(12), color="k", ls=":", label="true optimum")
lo, hi = ax.get_xlim()
ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, label="forecast = true")
ax.set_xlabel("forecast volatility (annual)")
ax.set_ylabel("true volatility (annual)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
mean = {k: np.array(v).mean(axis=0) for k, v in trials.items()}
assert mean["factor model"][1] < mean["Ledoit-Wolf"][1] < mean["sample"][1]
assert abs(mean["factor model"][0] / mean["factor model"][1] - 1) < 0.1 < 1 - mean["sample"][0] / mean["sample"][1]
