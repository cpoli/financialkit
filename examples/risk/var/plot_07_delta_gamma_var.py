r"""
Rouvinez: delta-gamma VaR
=========================

A delta-normal VaR sees only the first derivatives of an option book,
so it misses the convexity that dominates the loss of a short-option
position. The second-order approximation

.. math:: \Delta V \approx \theta h + \delta^\top \Delta S + \tfrac12 \Delta S^\top \Gamma\, \Delta S,
   \qquad \Delta S \sim N(0, \Sigma h),

is no longer normal. It used to be fitted by its moments, through the
Cornish-Fisher expansion. Rouvinez (1997) noted that its characteristic
function is known in closed form, a product of noncentral chi-squared
terms, and inverted it numerically. That gives the exact quantile of the
quadratic approximation at the cost of one integral.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.options import black_scholes_greeks, black_scholes_price
from financialkit.risk import delta_gamma_cdf, delta_gamma_var, delta_normal_var, monte_carlo_var

# %%
# A short-option book on two stocks
# ---------------------------------
#
# The book is short 100 at-the-money calls on stock A and short 150
# at-the-money puts on stock B, five weeks from expiry, and has hedged
# 90% of their delta with the stocks. The stocks have a correlation of
# 0.5 and the horizon is ten days. What is left is mostly short gamma,
# the case where the profit and loss is most skewed.

S0 = np.array([100.0, 50.0])
vol, rho, r, T, h = np.array([0.30, 0.40]), 0.5, 0.03, 0.1, 10 / 252
K = S0.copy()
units = np.array([-100.0, -150.0])
kinds = ["call", "put"]


g = [black_scholes_greeks(S0[i], K[i], T, r, vol[i], option_type=kinds[i]) for i in range(2)]
shares = -0.9 * units * np.array([gi.delta for gi in g])


def book(S, t):
    options = sum(u * black_scholes_price(S[..., i], K[i], t, r, vol[i], option_type=kinds[i]) for i, u in enumerate(units))
    return options + S @ shares


delta = units * np.array([gi.delta for gi in g]) + shares
gamma = np.diag(units * np.array([gi.gamma for gi in g]))
theta = float(sum(u * gi.theta for u, gi in zip(units, g, strict=True)))
corr = np.array([[1.0, rho], [rho, 1.0]])
cov = corr * np.outer(S0 * vol, S0 * vol)  # annual covariance of price changes
print(f"delta {np.round(delta, 1)}, gamma {np.round(np.diag(gamma), 2)}, theta {theta:,.0f} per year")

# %%
# Four VaRs at 99%
# ----------------
#
# Full revaluation by Monte Carlo is the benchmark: it reprices both
# options at the scenario prices with ten fewer days to expiry.

v0 = book(S0, T)
mc = monte_carlo_var(lambda dS: book(S0 + dS, T - h) - v0, cov, 0.99, horizon=h, n_paths=200_000, rng=1)
dn = delta_normal_var(delta, cov, 0.99, horizon=h)
cf = delta_gamma_var(delta, gamma, cov, 0.99, horizon=h, theta=theta, method="cornish_fisher")
dg = delta_gamma_var(delta, gamma, cov, 0.99, horizon=h, theta=theta)
print(f"delta-normal             {dn:8,.0f}")
print(f"delta-gamma, Cornish-F.  {cf:8,.0f}")
print(f"delta-gamma, exact       {dg:8,.0f}")
print(f"full revaluation         {mc.var:8,.0f}  (95% CI {mc.ci[0]:,.0f} to {mc.ci[1]:,.0f})")

# %%
# The delta-normal VaR sees only the residual delta and is almost zero.
# Cornish-Fisher overshoots in the skewed tail. The exact inversion is
# the true quantile of the quadratic, as a simulation of the quadratic
# confirms. What separates it from full revaluation is the third-order
# term the Taylor expansion leaves out.

rng = np.random.default_rng(2)
x = rng.multivariate_normal(np.zeros(2), cov * h, 400_000)
quad = theta * h + x @ delta + 0.5 * np.einsum("ni,ij,nj->n", x, gamma, x)
q_mc = -np.quantile(quad, 0.01)
print(f"simulated quadratic      {q_mc:8,.0f}")
assert abs(dg - q_mc) < 0.01 * dg
assert abs(dg - mc.var) < abs(dn - mc.var) and abs(dg - mc.var) < abs(cf - mc.var)

# %%
# The left tail of the distributions
# ----------------------------------
#
# The exact delta-gamma distribution function follows the
# full-revaluation tail closely. The normal one, built from the delta
# alone, puts far too little mass on large losses.

grid = np.linspace(-1.6 * mc.var, 0.0, 60)
full = np.sort(mc.scenarios)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(grid, np.searchsorted(full, grid) / full.size, "k", lw=2, label="full revaluation (Monte Carlo)")
ax.semilogy(grid, delta_gamma_cdf(grid, delta, gamma, cov, h, theta), "C0", label="delta-gamma, exact (Rouvinez)")
ax.semilogy(grid, delta_gamma_cdf(grid, delta, np.zeros((2, 2)), cov, h), "C3--", label="delta-normal")
ax.axhline(0.01, color="0.5", lw=0.8)
ax.set_xlabel("ten-day P&L")
ax.set_ylabel("P(P&L ≤ x)")
ax.set_ylim(1e-4, 1)
ax.legend()
fig.tight_layout()

plt.show()
