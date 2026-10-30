r"""
Richard-Roll: MBS prepayment under a short-rate model
=====================================================

A mortgage pass-through passes the homeowners' payments to the investor,
prepayments included, and homeowners prepay most when rates have
fallen. Richard and Roll (1989) wrote the conditional prepayment rate
as the product of four effects: a refinancing incentive (an S-curve in
the gap between the mortgage rate and the refinancing rate), seasoning
(young loans prepay slowly), seasonality (people move in summer) and
burnout (once the eager refinancers are gone the rest prepay less).

Because prepayments depend on the path of rates, the security is valued
by simulation. Here the short rate is Hull-White fitted to today's
curve, the refinancing rate is the model's 10-year yield on each path,
and each path's cash flows are discounted with its own short rates.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.fixed_income import NelsonSiegelCurve, ShiftedCurve
from financialkit.interest_rate_models import HullWhiteModel, mbs_price, richard_roll_cpr
from financialkit.interest_rate_models.systems.mbs import SEASONALITY

curve = NelsonSiegelCurve(0.055, -0.015, 0.0, 2.0)
wac, coupon = 0.075, 0.07  # mortgage rate, and the pass-through rate after servicing

# %%
# The four effects
# ----------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
gap = np.linspace(-0.02, 0.05, 200)
for factor, ls in ((1.0, "-"), (0.5, "--"), (0.2, ":")):
    axes[0].plot(100 * gap, 100 * richard_roll_cpr(wac, wac - gap, 60, 6, factor), ls, label=f"pool factor {factor}")
axes[0].set_xlabel("WAC - 10-year rate (%)")
axes[0].set_ylabel("CPR (%)")
axes[0].set_title("Refinancing S-curve, with burnout")
axes[0].legend()
inset = axes[0].inset_axes([0.62, 0.12, 0.35, 0.3])
inset.bar(np.arange(1, 13), SEASONALITY, color="C2")
inset.set_title("seasonality", fontsize=8)
inset.tick_params(labelsize=7)

# %%
# Expected prepayments along the simulated paths
# ----------------------------------------------
# Starting in January with new loans: the CPR ramps up over 30 months,
# wiggles with the seasons, and is higher when rates start lower.

for shift in (-0.01, 0.0, 0.01):
    res = mbs_price(HullWhiteModel(ShiftedCurve(curve, shift), 0.1, 0.01), wac, coupon, rng=0)
    axes[1].plot(res.times, 100 * res.expected_cpr, label=f"curve {100 * shift:+.0f}%, price {res.price:.2f}")
axes[1].set_xlabel("years")
axes[1].set_ylabel("expected CPR (%)")
axes[1].set_title("Hull-White paths, Richard-Roll prepayments")
axes[1].legend()

# %%
# Negative convexity
# ------------------
# As rates fall a bond with the same schedule and no prepayment keeps
# gaining, but the pass-through gains less and less: the homeowners'
# prepayment option, which the investor is short, moves into the money.
# The price curve is concave around today's rates. In a deep rally, where
# nearly everyone who can refinance does, the pass-through behaves like a
# short bond again and the curvature turns positive.

shifts = np.linspace(-0.025, 0.025, 11)
mbs, bullet, se = [], [], []
for shift in shifts:
    model = HullWhiteModel(ShiftedCurve(curve, shift), 0.1, 0.01)
    res = mbs_price(model, wac, coupon, n_paths=1000, rng=1)
    mbs.append(res.price)
    se.append(res.std_error)
    bullet.append(mbs_price(model, wac, coupon, n_paths=1000, prepayment=lambda *args: 0.0, rng=1).price)
mbs, bullet = np.array(mbs), np.array(bullet)
axes[2].errorbar(100 * shifts, mbs, yerr=4 * np.array(se), fmt="o-", ms=3, label="pass-through (Richard-Roll)")
axes[2].plot(100 * shifts, bullet, "s--", ms=3, label="no prepayment")
axes[2].set_xlabel("parallel curve shift (%)")
axes[2].set_ylabel("price per 100")
axes[2].set_title("Price against rates")
axes[2].legend()
fig.tight_layout()

curvature = np.diff(mbs, 2)
print(f"pass-through gains {mbs[0] - mbs[5]:.2f} for a 2.5% rally; without prepayment {bullet[0] - bullet[5]:.2f}")
assert np.all(np.diff(mbs) < 0)
assert mbs[0] - mbs[5] < 0.5 * (bullet[0] - bullet[5])  # prepayments cap the rally
assert np.all(curvature[4:8] < 0) and np.all(np.diff(bullet, 2) > 0)  # negative convexity around today's rates

plt.show()
