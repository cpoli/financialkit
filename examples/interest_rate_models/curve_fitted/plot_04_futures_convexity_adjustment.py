r"""
The futures convexity adjustment
================================

A Eurodollar futures contract is settled every day, a forward rate
agreement only at the end. Gains on the futures are reinvested when
rates are high and financed when they are low, so the futures rate is
the risk-neutral expectation of the future rate and exceeds the forward
rate. With the Ho-Lee and Hull-White models (1990) the gap has a closed
form; under Ho-Lee

.. math:: \text{futures rate} - \text{forward rate} = \tfrac12\sigma^2 t_1 t_2.

It is negligible for short contracts and worth tens of basis points for
long ones, which is why long-dated futures quotes cannot be used as
forward rates when bootstrapping.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve
from financialkit.interest_rate_models import HoLeeModel, HullWhiteModel, futures_convexity_adjustment

curve = NelsonSiegelCurve(0.05, -0.02, 0.01, 2.0)

# %%
# Hull's example
# --------------
#
# Hull (Example 6.4): sigma = 1.2%, an 8-year contract on 3-month rates.

hull = 1e4 * futures_convexity_adjustment(8.0, 8.25, 0.012)
print(f"Ho-Lee adjustment: {hull:.1f} bp (Hull: 47.5 bp)")
assert abs(hull - 47.5) < 0.05  # Hull's figure to the digit he prints

# %%
# The adjustment grows with the square of maturity
# ------------------------------------------------

t1 = np.linspace(0.25, 10, 40)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for a in (0.0, 0.03, 0.1):
    axes[0].plot(t1, 1e4 * futures_convexity_adjustment(t1, t1 + 0.25, 0.012, a), label="Ho-Lee" if a == 0 else f"Hull-White a = {a}")
axes[0].set_xlabel("futures expiry t1 (years)")
axes[0].set_ylabel("futures - forward (bp)")
axes[0].legend()

# %%
# The futures rate is the risk-neutral expected rate
# --------------------------------------------------
#
# Simulate the short rate to :math:`t_1`, compute the 3-month rate there
# from the model's bond price, and average it.

expiries = np.array([1.0, 3.0, 5.0, 8.0])
for name, model, a in (("Ho-Lee", HoLeeModel(curve, 0.012), 0.0), ("Hull-White", HullWhiteModel(curve, 0.1, 0.012), 0.1)):
    gap, err = [], []
    for k, t in enumerate(expiries):
        _, r = model.simulate(t, 8, 40_000, rng=k)
        R = -np.log(model.zero_coupon_bond(t + 0.25, t=t, state=r[:, -1])) / 0.25
        gap.append(R.mean() - curve.forward_rate(t, t + 0.25))
        err.append(R.std() / np.sqrt(R.size))
    z = (np.array(gap) - futures_convexity_adjustment(expiries, expiries + 0.25, 0.012, a)) / np.array(err)
    assert np.abs(z).max() < 3  # the simulated gap matches the closed form within 3 standard errors
    axes[1].errorbar(expiries, 1e4 * np.array(gap), yerr=1.96e4 * np.array(err), fmt="o", capsize=3, label=f"{name} Monte Carlo")
    axes[1].plot(t1, 1e4 * futures_convexity_adjustment(t1, t1 + 0.25, 0.012, a), lw=1)
axes[1].set_xlabel("futures expiry t1 (years)")
axes[1].set_ylabel("E[R] - forward (bp)")
axes[1].legend()
fig.tight_layout()

plt.show()
