r"""
Acerbi-Szekely: backtesting expected shortfall
==============================================

Expected shortfall was long said to be impossible to backtest, because
it is not *elicitable*: no scoring function is minimised by the true ES.
Acerbi and Szekely (2014) showed that this does not matter for a test,
and proposed statistics with expectation zero under a correct model:

.. math::

   Z_1 = \frac{1}{N_T}\sum_t \frac{X_t I_t}{\mathrm{ES}_t} + 1, \qquad
   Z_2 = \sum_t \frac{X_t I_t}{T (1 - \alpha)\, \mathrm{ES}_t} + 1 ,

where :math:`I_t` flags a VaR exception. :math:`Z_1` checks how large
the exceptions are, given how many there were. :math:`Z_2` checks their
number and size together. Its critical values hardly depend on the
distribution: about -0.70 at 5% for a 97.5% ES over 250 days.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.risk import acerbi_szekely_test, acerbi_szekely_z1, acerbi_szekely_z2, kupiec_test, parametric_es, parametric_var

# %%
# A normal model against three realities
# --------------------------------------
#
# The bank forecasts a standard normal profit and loss and reports its
# 97.5% VaR and ES. Each of 2,000 simulated years of 250 days is drawn
# from one of three realities. In the first the model is right. In the
# second the tails are Student-t with three degrees of freedom, scaled
# so that the 97.5% VaR is still exactly right, so only the losses
# beyond the VaR are underestimated. In the third the volatility is 20%
# higher than forecast.

alpha, T, m = 0.975, 250, 2000
var, es = parametric_var(0.0, 1.0, alpha), parametric_es(0.0, 1.0, alpha)
rng = np.random.default_rng(0)
worlds = {
    "model is right": rng.standard_normal((m, T)),
    "t(3) tails, VaR right": rng.standard_t(3, (m, T)) * var / stats.t.ppf(alpha, 3),
    "volatility 20% higher": 1.2 * rng.standard_normal((m, T)),
}

# Critical values at 5%, simulated under the model.
z1_null = acerbi_szekely_z1(worlds["model is right"], var, es)
z1_crit = np.quantile(z1_null[np.isfinite(z1_null)], 0.05)
z2_crit = np.quantile(acerbi_szekely_z2(worlds["model is right"], var, es, alpha), 0.05)
print(f"5% critical values: Z1 {z1_crit:.2f}, Z2 {z2_crit:.2f}")
assert abs(z2_crit + 0.70) < 0.05

power = {}
for name, X in worlds.items():
    kup = np.mean([kupiec_test(x < -var, alpha).reject() for x in X])
    z1 = acerbi_szekely_z1(X, var, es)
    z2 = acerbi_szekely_z2(X, var, es, alpha)
    power[name] = (kup, np.mean(z1 < z1_crit), np.mean(z2 < z2_crit))
    print(f"{name:22s}: rejection rate  Kupiec {power[name][0]:.0%},  Z1 {power[name][1]:.0%},  Z2 {power[name][2]:.0%}")

# %%
# Only the ES tests see a tail that is wrong beyond the VaR. Kupiec's
# count of exceptions cannot, because the count is right. Z1 is the
# sharper test of tail shape. Z2 also responds to the number of
# exceptions, so it catches the understated volatility.

assert power["t(3) tails, VaR right"][0] < 0.10 < 0.5 < power["t(3) tails, VaR right"][1]
assert power["volatility 20% higher"][2] > max(power["volatility 20% higher"][:2])

fig, ax = plt.subplots(figsize=(7, 3.8))
bins = np.linspace(-3, 1, 60)
for name, X in worlds.items():
    ax.hist(acerbi_szekely_z2(X, var, es, alpha), bins=bins, histtype="step", lw=1.5, density=True, label=name)
ax.axvline(z2_crit, color="k", ls="--", lw=1, label=f"5% critical value {z2_crit:.2f}")
ax.set_xlabel("$Z_2$ over one year")
ax.set_ylabel("density")
ax.legend(loc="upper left")
fig.tight_layout()

# %%
# One year's backtest with a simulated p-value
# --------------------------------------------
#
# In practice the bank has one realised year. It simulates the statistic
# under its own model to get a p-value.

year = worlds["t(3) tails, VaR right"][0]
for test in ("z1", "z2"):
    res = acerbi_szekely_test(year, var, es, alpha, test=test, simulated=worlds["model is right"])
    print(f"{test}: statistic {res.statistic:+.2f}, p = {res.p_value:.3f}, {res.n_exceptions} exceptions (expected {res.expected:.2f})")

plt.show()
