r"""
Campbell-Shiller log-linear present value
=========================================

John Campbell and Robert Shiller (1988) linearised the log return
around the average dividend-price ratio,

.. math::

   r_{t+1} \approx k + dp_t - \rho\, dp_{t+1} + \Delta d_{t+1},
   \qquad \rho = \frac{1}{1 + e^{\overline{dp}}},

and iterated it forward into an identity that holds in every model:
:math:`dp_t \approx \text{const} + \sum_j \rho^{j-1} (r_{t+j} - \Delta d_{t+j})`.
A dividend-price ratio that moves must forecast either returns or
dividend growth. In long-run regressions on :math:`dp_t` the two
coefficients differ by one, so they measure how much of the variation
comes from each source.

The simulation lets the log dividend-price ratio follow an AR(1) around
a 4% dividend yield, and computes exact returns from the prices.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.asset_pricing import campbell_shiller_decomposition, campbell_shiller_rho, log_linear_return

rng = np.random.default_rng(0)
T, phi, dp_bar = 5000, 0.94, np.log(0.04)


def simulate(dividend_loading):
    """dp is AR(1); dividend growth loads on last period's dp (zero: unpredictable)."""
    dp = np.empty(T)
    dp[0] = dp_bar
    shocks = rng.standard_normal((2, T))
    for t in range(1, T):
        dp[t] = dp_bar + phi * (dp[t - 1] - dp_bar) + 0.05 * shocks[0, t]
    dd = 0.02 + 0.1 * shocks[1]
    dd[1:] += dividend_loading * (dp[:-1] - dp_bar)
    r = np.full(T, np.nan)
    r[1:] = dd[1:] + np.log1p(np.exp(-dp[1:])) + dp[:-1]
    return r, dd, dp


r, dd, dp = simulate(0.0)
rho = campbell_shiller_rho(dp.mean())
approx = log_linear_return(dp[:-1], dp[1:], dd[1:], rho)
err = approx - r[1:]
print(f"rho = {rho:.4f}; linearisation error: mean {err.mean():.5f}, std {err.std():.5f} against return std {r[1:].std():.3f}")
# The approximation error is two orders of magnitude smaller than the returns.
assert err.std() < 0.01 * r[1:].std()

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(r[1:], approx, ".", ms=2, alpha=0.4)
lim = [r[1:].min(), r[1:].max()]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("exact log return")
ax.set_ylabel("log-linear return")
fig.tight_layout()

# %%
# Where does the variation in dp come from?
# ------------------------------------------
#
# In the first economy dividend growth is unpredictable, so all the
# variation in the dividend-price ratio is news about returns: the
# long-run return coefficient is one and the dividend coefficient zero.
# In the second, a high dividend yield also forecasts lower dividend
# growth, which takes some of the burden. The two long-run coefficients
# still differ by one. Cochrane (2008) found that in US data the return
# coefficient is close to one, as in the first economy.

results = {}
for name, loading in [("returns only", 0.0), ("returns and dividends", -0.06)]:
    r, dd, dp = simulate(loading)
    pv = campbell_shiller_decomposition(r, dd, dp)
    results[name] = pv
    print(f"{name:>22}: b_r = {pv.b_r:.3f}, b_d = {pv.b_d:+.3f}, phi = {pv.phi:.3f}")
    print(f"{'':>22}  long run: returns {pv.long_run_return:.2f}, dividends {pv.long_run_dividend:+.2f}")
    # The present-value identity: the long-run coefficients differ by one.
    assert abs(pv.long_run_return - pv.long_run_dividend - 1.0) < 0.05
assert abs(results["returns only"].long_run_dividend) < 0.15
assert results["returns and dividends"].long_run_dividend < -0.2

fig, ax = plt.subplots(figsize=(6, 3.5))
names = list(results)
ax.bar(np.arange(2) - 0.2, [results[n].long_run_return for n in names], 0.4, label="expected returns")
ax.bar(np.arange(2) + 0.2, [-results[n].long_run_dividend for n in names], 0.4, label="expected dividend growth")
ax.set_xticks(range(2), names)
ax.set_ylabel("share of Var(dp)")
ax.legend()
fig.tight_layout()

plt.show()
