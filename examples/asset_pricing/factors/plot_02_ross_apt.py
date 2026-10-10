r"""
Ross's arbitrage pricing theory
===============================

Stephen Ross (1976) derived linear pricing without an equilibrium
model. If returns follow a factor model
:math:`r_i = E[r_i] + \beta_i' f + \varepsilon_i` with diversifiable
:math:`\varepsilon_i`, then any asset off the plane
:math:`E[r_i] = r_f + \beta_i'\lambda` can be combined with others into
a portfolio with no cost and no factor risk that still earns a positive
expected return. In a large market its idiosyncratic risk diversifies
away, and the portfolio becomes an arbitrage. The pricing errors
therefore have to be small for all but finitely many assets.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import apt_arbitrage_portfolio, apt_expected_returns

rng = np.random.default_rng(1976)
lam = np.array([0.05, 0.02])
idio = 0.25

sizes = np.array([10, 30, 100, 300, 1000, 3000])
sharpe = []
for n in sizes:
    B = np.column_stack([rng.uniform(0.5, 1.5, n), rng.normal(0, 1, n)])
    mu = apt_expected_returns(B, lam, 0.03) + rng.normal(0, 0.01, n)  # 1% pricing errors
    w = apt_arbitrage_portfolio(mu, B)
    sharpe.append((w @ mu) / (idio * np.linalg.norm(w)))
    if n == 100:
        print(f"n = 100: cost {w.sum():.1e}, factor exposure {np.abs(w @ B).max():.1e}, expected return {w @ mu:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(sizes, sharpe, "o-", label="arbitrage portfolio")
ax.loglog(sizes, sharpe[0] * np.sqrt(sizes / sizes[0]), "k:", label="~ sqrt(n)")
ax.set_xlabel("number of assets")
ax.set_ylabel("Sharpe ratio of the zero-cost, zero-beta portfolio")
ax.set_title("constant pricing errors allow unbounded Sharpe ratios")
ax.legend()
fig.tight_layout()

# %%
# When pricing errors die out
# ---------------------------
#
# If only a handful of assets are mispriced, the Sharpe ratio stays bounded.

sharpe_sparse = []
for n in sizes:
    B = np.column_stack([rng.uniform(0.5, 1.5, n), rng.normal(0, 1, n)])
    err = np.zeros(n)
    err[:5] = 0.01
    mu = apt_expected_returns(B, lam, 0.03) + err
    w = apt_arbitrage_portfolio(mu, B)
    sharpe_sparse.append((w @ mu) / (idio * np.linalg.norm(w)))
ax.loglog(sizes, sharpe_sparse, "s-", label="five mispriced assets")
ax.legend()

plt.show()
