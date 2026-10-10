r"""
Lee and Ready's trade classification
====================================

Trade records do not say who initiated a trade. Charles Lee and Mark
Ready (1991) signed trades above the midquote as buys and below it as
sells, used the tick test for trades at the midquote, and compared each
trade with the quote of *five seconds earlier*: on the NYSE the quote
revision a trade caused was often time-stamped before the trade itself,
so the contemporaneous quote had already moved toward the trade price.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure import align_quotes, lee_ready, quote_rule, tick_test

rng = np.random.default_rng(2)

# %%
# A market with early-stamped quote updates
# -----------------------------------------
#
# Trades arrive every 15 seconds on average, 15% of them at the midquote.
# Each one moves the midquote by 4 cents in its direction, plus noise,
# and the new quote is recorded two seconds *before* the trade that
# caused it. Without the delay, the quote rule compares trades with a
# quote that has already moved past them.

n, half = 3000, 0.05
t = np.cumsum(3.0 + rng.exponential(12.0, n))
side = rng.choice([-1.0, 1.0], n)
step = 0.04 * side + 0.02 * rng.standard_normal(n)
mid_before = 20 + np.concatenate([[0.0], np.cumsum(step)[:-1]])
at_mid = rng.random(n) < 0.15
price = np.where(at_mid, mid_before, mid_before + side * rng.uniform(0.3, 1.0, n) * half)
mid_after = mid_before + step
quote_t = np.concatenate([[0.0], t - 2.0])
quote_mid = np.concatenate([[mid_before[0]], mid_after])

results = {}
for delay in (0.0, 5.0):
    b, a = align_quotes(t, quote_t, quote_mid - half, quote_mid + half, delay=delay)
    results[f"quote rule, {delay:g}s"] = quote_rule(price, b, a)
    results[f"Lee-Ready, {delay:g}s"] = lee_ready(price, b, a)
results["tick test"] = tick_test(price)

for name, est in results.items():
    print(f"{name:17s} correct {np.mean(est == side):.1%}")

# %%
# Accuracy by rule
# ----------------

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(list(results), [np.mean(e == side) for e in results.values()], color=["C0", "C2", "C0", "C2", "C1"])
ax.set_xlim(0.4, 1.0)
ax.set_xlabel("fraction of trades correctly signed")
fig.tight_layout()

plt.show()
