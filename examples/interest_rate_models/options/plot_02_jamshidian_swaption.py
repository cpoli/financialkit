r"""
Jamshidian's decomposition: swaptions in one-factor models
==========================================================

An option on a coupon bond is not a portfolio of options on its
coupons, in general. Jamshidian (1989) noticed that in a one-factor
model it is: every zero-coupon bond price falls with the short rate, so
the bond is above its strike exactly when the short rate is below a
critical :math:`r^*`, and then each zero is above
:math:`K_i = P(T, t_i; r^*)`. Hence

.. math:: \max\left(\sum_i c_i P(T, t_i) - K, 0\right) = \sum_i c_i \max(P(T, t_i) - K_i, 0).

A payer swaption is a put on a coupon bond paying the fixed rate, so it
too is a sum of zero-coupon bond options.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HullWhiteModel, forward_swap_rate, jamshidian_swaption, path_discount_factors, swaption_schedule
from financialkit.interest_rate_models.visualizers import plot_jamshidian

curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)
hw = HullWhiteModel(curve, 0.1, 0.01)
expiry, tenor = 2.0, 5.0
S = forward_swap_rate(curve, expiry, tenor, 2)

# %%
# The decomposition of an at-the-money payer swaption
# ---------------------------------------------------

res = jamshidian_swaption(hw, expiry, tenor, S, freq=2, payer=True, notional=100.0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_jamshidian(res, ax=axes[0])
print(f"forward swap rate {100 * S:.3f}%, critical short rate r* = {100 * res.critical_rate:.3f}%")

# %%
# Against Monte Carlo across strikes
# ----------------------------------

times, r = hw.simulate(expiry, 100, 20_000, rng=0)
D = path_discount_factors(times, r)[:, -1]
sched = swaption_schedule(expiry, tenor, 2)
P = hw.zero_coupon_bond(sched[None, :], t=expiry, state=r[:, -1:])
K = S + np.linspace(-0.015, 0.015, 7)
closed = [jamshidian_swaption(hw, expiry, tenor, k, notional=100.0).price for k in K]
pay = 100 * D[:, None] * np.maximum(1 - (0.5 * K[None, :] * P.sum(axis=1)[:, None] + P[:, -1:]), 0.0)
z = (pay.mean(0) - closed) / (pay.std(0) / np.sqrt(D.size))
print(f"largest |MC - Jamshidian| / SE: {np.abs(z).max():.2f}")
assert np.abs(z).max() < 3
axes[1].plot(100 * K, closed, "k-", label="Jamshidian")
axes[1].errorbar(100 * K, pay.mean(0), yerr=1.96 * pay.std(0) / np.sqrt(D.size), fmt="o", ms=3, capsize=2, label="Monte Carlo")
axes[1].set_xlabel("fixed rate (%)")
axes[1].set_ylabel("payer swaption (per 100 notional)")
axes[1].legend()
fig.tight_layout()

plt.show()
