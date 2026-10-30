r"""
Obizhaeva and Wang: transient impact and a refilling book
=========================================================

Anna Obizhaeva and Jiang Wang (2013) derived execution costs from the
order book itself. A purchase of :math:`x` shares walks up a book of
density :math:`1/\kappa` and displaces the ask by :math:`\kappa x`; new
sell orders then refill the book, and the displacement decays at the
resilience rate :math:`\rho`. Impact is transient, so the trader should
wait for the book to recover, but not too long. The optimal schedule
buys a block at the start, trades steadily at the rate the book refills,
and buys a second, equal block at the end:

.. math::

   x_0 = x_T = \frac{X}{\rho T + 2}, \qquad \dot x_t = \frac{\rho X}{\rho T + 2}, \qquad
   C = \frac{\kappa X^2}{\rho T + 2} + \frac{\lambda X^2}{2}.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.market_microstructure import obizhaeva_wang, transient_impact_cost

X, T, N = 1e5, 1.0, 20  # buy 100,000 shares over a day, at 21 dates
kappa, rho = 1e-5, 5.0  # $1 of displacement per 100,000 shares; the book refills in ~1/5 day

res = obizhaeva_wang(X, T, N, kappa, rho)
twap = np.full(N + 1, X / (N + 1))
print(f"end blocks {res.trades[0]:,.0f} shares, interior trades {res.trades[1]:,.0f}")

# %%
# The schedule and the book
# -------------------------
# Between dates the displacement decays exponentially. Under the optimal
# schedule each interior trade restores exactly what the book recovered,
# so the ask stays at a constant distance above its steady state.


def displacement_path(trades, n_fine=50):
    t, d = [], []
    dates = np.linspace(0.0, T, trades.size)
    D = 0.0
    for k, x in enumerate(trades):
        t += [dates[k], dates[k]]
        d += [D, D + kappa * x]
        D += kappa * x
        if k < trades.size - 1:
            s = np.linspace(0.0, T / N, n_fine)[1:]
            t += list(dates[k] + s)
            d += list(D * np.exp(-rho * s))
            D *= np.exp(-rho * T / N)
    return np.array(t), np.array(d)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax1.bar(res.times, res.trades, width=0.03, color="C0", label="Obizhaeva-Wang")
ax1.plot(res.times, twap, "C1_", ms=12, mew=2, label="TWAP")
ax1.set_ylabel("shares bought")
ax1.legend()
for trades, c, name in [(res.trades, "C0", "Obizhaeva-Wang"), (twap, "C1", "TWAP")]:
    ax2.plot(*displacement_path(trades), c, lw=1, label=name)
ax2.set_xlabel("time (days)")
ax2.set_ylabel("ask displacement ($)")
ax2.legend()
fig.tight_layout()
assert np.allclose(res.deviation_before[1:], res.deviation_before[1])

# %%
# Cost against resilience
# -----------------------
# A book that never refills (:math:`\rho = 0`) charges
# :math:`\kappa X^2/2` whatever the schedule; the faster it refills, the
# more there is to gain from spreading the order out, and the
# continuous-time formula is reached already with 21 dates.

rhos = np.linspace(0.0, 30.0, 61)
optimal = [obizhaeva_wang(X, T, N, kappa, r).expected_cost for r in rhos]
uniform = [transient_impact_cost(twap, T, kappa, r) for r in rhos]
block = transient_impact_cost([X], T, kappa, rho)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rhos, optimal, "C0-", label="Obizhaeva-Wang, 21 dates")
ax.plot(rhos, kappa * X**2 / (rhos * T + 2), "k--", lw=1, label=r"continuous limit $\kappa X^2/(\rho T + 2)$")
ax.plot(rhos, uniform, "C1-", label="TWAP")
ax.axhline(block, color="C3", label="one block")
ax.set_xlabel(r"resilience $\rho$ (per day)")
ax.set_ylabel("impact cost ($)")
ax.legend()
fig.tight_layout()
print(f"rho = {rho}: optimal ${res.expected_cost:,.0f}, TWAP ${transient_impact_cost(twap, T, kappa, rho):,.0f}, one block ${block:,.0f}")
assert np.all(np.array(optimal) <= np.array(uniform) + 1e-9) and np.isclose(optimal[0], block)

plt.show()
