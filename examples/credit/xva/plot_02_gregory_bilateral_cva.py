r"""
Gregory: bilateral CVA and DVA
==============================

Unilateral CVA treats the bank as default-free, so the two sides of a
trade disagree on its price: each charges the other for credit risk.
Gregory (2009) put the bank's own default into the price. If the bank
defaults while it owes money, it pays only a recovery, a gain worth

.. math:: \text{DVA} = (1 - R_B) \sum_i E[D(0, t_i) V(t_i)^-]\,Q_C(t_i)\,[Q_B(t_{i-1}) - Q_B(t_i)].

Counting a default only if the other party has not defaulted first
makes the two sides agree: the bank's DVA is the counterparty's CVA.
Gregory also showed the uncomfortable consequences: a bank whose own
credit worsens reports a profit, and the gain can only be realized by
defaulting.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import HazardCurve, bilateral_cva, swap_exposure
from financialkit.credit.visualizers import plot_exposure_profile
from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, forward_swap_rate

# %%
# Both sides of one swap
# ----------------------
# Bank B pays fixed on a ten-year swap to counterparty C. On the same
# Hull-White paths, C's values are B's with the sign flipped, so B's
# positive exposure is C's negative exposure.

curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
hw = HullWhiteModel(curve, a=0.05, sigma=0.01)
K = forward_swap_rate(curve, 0.0, 10.0, 2)
bank = swap_exposure(hw, K, 10.0, notional=100.0, n_paths=4000, rng=11)
cpty = swap_exposure(hw, K, 10.0, notional=100.0, payer=False, n_paths=4000, rng=11)
ax = plot_exposure_profile(bank, pfe_level=None, label="bank pays fixed")
ax.set_title("the bank's exposure is the counterparty's negative exposure")

Q_bank, Q_cpty = HazardCurve.flat(0.01), HazardCurve.flat(0.03)
b = bilateral_cva(bank, Q_cpty, Q_bank, recovery_counterparty=0.4, recovery_own=0.4)
c = bilateral_cva(cpty, Q_bank, Q_cpty, recovery_counterparty=0.4, recovery_own=0.4)
print(f"bank:         CVA {b.cva:.4f}  DVA {b.dva:.4f}  adjustment {b.bva:+.4f}")
print(f"counterparty: CVA {c.cva:.4f}  DVA {c.dva:.4f}  adjustment {c.bva:+.4f}")
assert np.isclose(b.cva, c.dva) and np.isclose(b.dva, c.cva) and np.isclose(b.bva, -c.bva)

# %%
# First to default
# ----------------
# Without the survival factors both defaults are counted, although only
# the first one happens, so CVA and DVA are each overstated, and more so
# the riskier the other party is.

hazards = np.linspace(0.0, 0.1, 11)
ftd = [bilateral_cva(bank, Q_cpty, HazardCurve.flat(h)) for h in hazards]
naive = [bilateral_cva(bank, Q_cpty, HazardCurve.flat(h), first_to_default=False) for h in hazards]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(1e4 * hazards, [r.cva for r in ftd], lw=2, color="C0", label="CVA, first to default")
ax.plot(1e4 * hazards, [r.cva for r in naive], "--", lw=2, color="C0", label="CVA, both defaults counted")
ax.plot(1e4 * hazards, [r.dva for r in ftd], lw=2, color="C1", label="DVA, first to default")
ax.plot(1e4 * hazards, [r.dva for r in naive], "--", lw=2, color="C1", label="DVA, both defaults counted")
ax.set_xlabel("bank's own hazard rate (bp)")
ax.set_ylabel("adjustment per 100 notional")
ax.legend()
assert all(n.cva > f.cva and n.dva > f.dva for n, f in zip(naive[1:], ftd[1:], strict=True))

# %%
# Profiting from one's own decline
# --------------------------------
# As the bank's own hazard rate rises, its DVA grows and the trade's value
# to the bank rises with it: the adjustment increases with own risk.

bva = np.array([r.bva for r in ftd])
print(f"own hazard 0 -> 10%: bilateral adjustment {bva[0]:+.4f} -> {bva[-1]:+.4f}")
assert np.all(np.diff(bva) > 0)

plt.show()
