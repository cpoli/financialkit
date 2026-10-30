r"""
Brigo and Masetti: unilateral CVA from exposure profiles
========================================================

A bank that is owed money by a counterparty loses the positive part of
the trade's value if the counterparty defaults. With rates and default
independent, the credit valuation adjustment is the discounted expected
positive exposure integrated against the default probabilities,

.. math:: \text{CVA} = (1 - R) \sum_i E[D(0, t_i) V(t_i)^+]\,[Q(t_{i-1}) - Q(t_i)].

Brigo and Masetti (2005) observed that for an interest rate swap,
:math:`E[D(0, T_i) V(T_i)^+]` after a payment date is a swaption on the
remaining swap, so the CVA of a swap is a strip of swaptions. Here the
exposure comes from Longstaff-Schwartz regression on Hull-White paths
fitted to a Nelson-Siegel curve, and the swaptions from Jamshidian's
closed form in the same model.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.credit import ExposureProfile, HazardCurve, swap_cva_swaptions, swap_exposure, unilateral_cva
from financialkit.credit.visualizers import plot_exposure_profile
from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, forward_swap_rate, jamshidian_swaption

# %%
# Exposure of a ten-year payer swap
# ---------------------------------
# The swap is struck at the par rate, so it is worth zero today, but the
# upward-sloping curve makes paying fixed more likely to end in the money.
# The regression uses only realized cash flows; the exact profile revalues
# the swap on the same paths with Hull-White bond prices.

curve = NelsonSiegelCurve(0.04, -0.015, 0.01, 2.0)
hw = HullWhiteModel(curve, a=0.05, sigma=0.01)
K = forward_swap_rate(curve, 0.0, 10.0, 2)
reg = swap_exposure(hw, K, 10.0, notional=100.0, n_paths=4000, rng=7)
exact = swap_exposure(hw, K, 10.0, notional=100.0, n_paths=4000, rng=7, method="exact")
print(f"par rate {K:.4%}; mean discount factor to 10y {reg.discount[:, -1].mean():.4f} vs curve {curve.discount(10.0):.4f}")

ax = plot_exposure_profile(reg, n_paths=40, label="regression")
ax.plot(exact.times, exact.epe, "k:", label="EPE, exact revaluation")
pay = np.arange(1, 20) / 2
swaptions = np.array([jamshidian_swaption(hw, T, 10.0 - T, K, 2, notional=100.0).price for T in pay])
ax.plot(pay, swaptions, "o", color="C3", ms=4, label="payer swaptions (Jamshidian)")
ax.set_ylim(-6, 8)
ax.set_title("10y payer swap, notional 100, Hull-White a = 5%, sigma = 1%")
ax.legend(fontsize=8)

# at the payment dates the exposure is the swaption, within 4 standard errors
on_pay = np.arange(4, 80, 4)
assert np.all(np.abs(exact.epe[on_pay] - swaptions) < 4 * exact.epe_std_error[on_pay])

# %%
# CVA as a strip of swaptions
# ---------------------------
# Defaults in :math:`(T_{i-1}, T_i]` are settled on the exposure at
# :math:`T_i`. On the payment dates the regression profile and the
# swaption strip agree; the full exposure grid also sees the coming
# coupon between payment dates, and gives a somewhat larger CVA.

counterparty = HazardCurve([1.0, 3.0, 5.0, 10.0], [0.01, 0.015, 0.02, 0.025])
strip = swap_cva_swaptions(hw, K, 10.0, counterparty, recovery=0.4, notional=100.0)
on_payments = ExposureProfile(reg.times[::4], reg.values[:, ::4], reg.discount[:, ::4], reg.method)
cva_pay = unilateral_cva(on_payments, counterparty)
cva_grid = unilateral_cva(reg, counterparty)
print(f"swaption strip: {strip:.4f}; regression on payment dates: {cva_pay:.4f}; on the full grid: {cva_grid:.4f}")
assert abs(cva_pay / strip - 1) < 0.05 and strip < cva_grid < 1.25 * strip

# %%
# The receiver side
# -----------------
# Receiving fixed on the same swap is a strip of receiver swaptions, which
# are cheaper on this curve, so the receiver's CVA is smaller.

receiver = swap_cva_swaptions(hw, K, 10.0, counterparty, notional=100.0, payer=False)
hazards = np.linspace(0.0, 0.06, 13)
fig, ax = plt.subplots(figsize=(7, 4))
for payer, name in ((True, "pay fixed"), (False, "receive fixed")):
    ax.plot(1e4 * hazards, [swap_cva_swaptions(hw, K, 10.0, HazardCurve.flat(h), notional=100.0, payer=payer) for h in hazards], lw=2, label=name)
ax.set_xlabel("counterparty hazard rate (bp)")
ax.set_ylabel("CVA per 100 notional")
ax.legend()
print(f"receiver CVA {receiver:.4f} < payer CVA {strip:.4f}")
assert receiver < strip

plt.show()
