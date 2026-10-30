r"""
Goldfarb-Iyengar robust portfolio selection
===========================================

A mean-variance optimizer treats estimated expected returns and factor
loadings as exact, so it bets hardest on the assets whose estimates
happen to be most flattering. Donald Goldfarb and Garud Iyengar (2003)
let each input range over the confidence region its regression gives
and maximized the worst case instead. For a box around each mean and an
ellipsoid around each asset's factor loadings the worst case is explicit,

.. math:: \max_{\mathbf 1'w = 1}\; w'\hat\mu - \gamma_\mu'|w|
          - \frac{\gamma}{2}\left[\left(\|F^{1/2}\hat B'w\| + \rho'|w|\right)^2 + w'Dw\right],

a convex problem in which estimation error acts as a charge on each
position. Here eight assets follow a two-factor model; an investor
estimates :math:`\mu_i` and :math:`B_i` from five years of monthly data
by time-series regression and sizes the uncertainty sets from their
standard errors at 95% confidence.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.portfolio import robust_portfolio

rng = np.random.default_rng(5)
d, k, n, gamma = 8, 2, 60, 3.0
B = rng.uniform(0.4, 1.4, (d, k))
F = np.array([[0.0016, 0.0002], [0.0002, 0.0009]])
spec = rng.uniform(0.03, 0.06, d) ** 2
mu = np.full(d, 0.008) + 0.002 * rng.standard_normal(d)
true_cov = B @ F @ B.T + np.diag(spec)
L = np.linalg.cholesky(F)
z_mu, z_B = stats.norm.ppf(0.975), np.sqrt(stats.chi2.ppf(0.95, k))


def estimate(n_months):
    """Regress each asset on the factors: intercepts, loadings, specific variances and the uncertainty sets."""
    f = rng.standard_normal((n_months, k)) @ L.T
    R = mu + f @ B.T + rng.standard_normal((n_months, d)) * np.sqrt(spec)
    A = np.column_stack([np.ones(n_months), f])
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    s2 = ((R - A @ coef) ** 2).sum(axis=0) / (n_months - k - 1)
    Finv_n = np.linalg.inv(A.T @ A)
    F_hat = np.atleast_2d(np.cov(f, rowvar=False))
    # Loading errors have covariance s_i^2 (A'A)^{-1}; their F-norm is about s_i sqrt(k / n).
    rho = z_B * np.sqrt(s2 * np.trace(F_hat @ Finv_n[1:, 1:]) / k)
    return coef[0], coef[1:].T, F_hat, s2, z_mu * np.sqrt(s2 * Finv_n[0, 0]), rho


def true_utility(w):
    return w @ mu - 0.5 * gamma * w @ true_cov @ w


m_hat, B_hat, F_hat, s2, g, rho = estimate(n)
nominal = robust_portfolio(m_hat, B_hat, F_hat, s2, gamma)
robust = robust_portfolio(m_hat, B_hat, F_hat, s2, gamma, mu_halfwidth=g, loading_radius=rho)
for name, res in (("nominal", nominal), ("robust", robust)):
    print(f"{name:8s} weights {np.round(res.weights, 2)}")
    print(f"         promised utility {res.nominal_return - 0.5 * gamma * res.nominal_volatility**2:.5f}, true {true_utility(res.weights):.5f}")

# %%
# Concentrated bets against hedged ones
# -------------------------------------
# The estimated means spread far more than the true ones, and the
# nominal portfolio follows them into a few assets. The robust one
# charges each asset its mean's confidence half-width, so it keeps a
# broader portfolio and promises less.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(d)
axes[0].bar(x - 0.2, nominal.weights, 0.4, label="nominal")
axes[0].bar(x + 0.2, robust.weights, 0.4, label="robust")
ax2 = axes[0].twinx()
ax2.errorbar(x, 12 * m_hat, yerr=12 * g, fmt="ko", ms=4, capsize=3, label=r"estimate $\pm\,\gamma_\mu$")
ax2.plot(x, 12 * mu, "r_", ms=14, mew=2, label="true mean")
ax2.set_ylabel("expected return (annual)")
axes[0].set_xlabel("asset")
axes[0].set_ylabel("weight")
axes[0].set_ylim(0, 1.3)
axes[0].legend(loc="upper left")
ax2.legend(loc="upper right", fontsize=8)
effective_n = [1 / np.sum(res.weights**2) for res in (nominal, robust)]
print(f"effective number of assets: nominal {effective_n[0]:.1f}, robust {effective_n[1]:.1f}")
assert effective_n[1] > effective_n[0]

# %%
# What the investor actually gets
# -------------------------------
# Repeating the estimation 150 times, the robust portfolio's true
# utility is higher on average and less dispersed: guarding against
# the worst case of the estimates protects against the estimates
# themselves.

draws = []
for _ in range(150):
    m_hat, B_hat, F_hat, s2, g, rho = estimate(n)
    w_nom = robust_portfolio(m_hat, B_hat, F_hat, s2, gamma).weights
    w_rob = robust_portfolio(m_hat, B_hat, F_hat, s2, gamma, mu_halfwidth=g, loading_radius=rho).weights
    draws.append((true_utility(w_nom), true_utility(w_rob)))
draws = 12 * np.array(draws)
best = 12 * max(true_utility(robust_portfolio(mu, B, F, spec, gamma).weights), 0)
bins = np.linspace(draws.min(), best, 30)
axes[1].hist(draws[:, 0], bins, alpha=0.6, label=f"nominal, mean {draws[:, 0].mean():.4f}")
axes[1].hist(draws[:, 1], bins, alpha=0.6, label=f"robust, mean {draws[:, 1].mean():.4f}")
axes[1].axvline(best, color="k", ls=":", label="true optimum")
axes[1].set_xlabel("true mean-variance utility (annual)")
axes[1].set_ylabel("estimation draws")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
print(f"mean true utility: nominal {draws[:, 0].mean():.4f}, robust {draws[:, 1].mean():.4f}")
assert draws[:, 1].mean() > draws[:, 0].mean() and draws[:, 1].std() < draws[:, 0].std()
