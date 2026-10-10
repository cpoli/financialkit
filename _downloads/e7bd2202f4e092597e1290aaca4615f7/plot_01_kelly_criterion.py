r"""
The Kelly criterion
===================

John Kelly (1956), at Bell Labs, asked how a gambler with a private wire
should bet. Staking the fraction :math:`f` of wealth on a bet won with
probability :math:`p` at odds :math:`b`, wealth grows at the rate
:math:`g(f) = p\log(1 + bf) + q\log(1 - f)` per bet, maximized by
:math:`f^* = p - q/b`. The Kelly bettor eventually outgrows any other
fixed-fraction strategy; betting more than twice Kelly makes growth
negative even though every bet is favourable.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.portfolio import kelly_fraction, kelly_growth_rate, kelly_wealth_paths

p = 0.55
f_star = kelly_fraction(p)
print(f"Kelly fraction for an even-money bet with p = {p}: {f_star:.2f}")

# %%
# Growth rate against the fraction staked
# ---------------------------------------

f = np.linspace(0, 0.25, 300)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(f, kelly_growth_rate(f, p))
ax.axvline(f_star, color="C2", ls="--", label="Kelly")
ax.axvline(2 * f_star, color="C3", ls=":", label="twice Kelly")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xlabel("fraction of wealth staked")
ax.set_ylabel("expected log growth per bet")
ax.legend()
fig.tight_layout()

# %%
# A thousand bets, three bettors
# ------------------------------
#
# The same coin flips for each bettor. Half Kelly gives up a quarter of
# the growth for much smaller swings; double Kelly goes nowhere.

n_bets, n_paths = 1000, 200
fig, ax = plt.subplots(figsize=(7, 4))
for mult, color in [(0.5, "C0"), (1.0, "C2"), (2.0, "C3")]:
    W = kelly_wealth_paths(mult * f_star, p, n_bets, n_paths, rng=0)
    ax.semilogy(np.median(W, axis=0), color=color, lw=2, label=f"{mult:g} x Kelly (median)")
    ax.fill_between(np.arange(n_bets + 1), *np.percentile(W, [10, 90], axis=0), color=color, alpha=0.15)
    print(f"{mult:g} x Kelly: median final wealth {np.median(W[:, -1]):.3g}, P(lose money) {np.mean(W[:, -1] < 1):.2f}")
ax.set_xlabel("bet")
ax.set_ylabel("wealth")
ax.legend()
fig.tight_layout()
plt.show()
