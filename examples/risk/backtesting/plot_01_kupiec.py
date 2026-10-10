r"""
Kupiec's proportion-of-failures test
====================================

A 99% VaR should be exceeded on about 1% of days. Kupiec (1995) turned
"about" into a likelihood-ratio test: with :math:`x` exceptions in
:math:`n` days and :math:`\hat p = x/n`,

.. math::

   LR_{uc} = -2 \ln \frac{(1 - p)^{n - x} p^x}{(1 - \hat p)^{n - x} \hat p^x} \sim \chi^2_1.

He also showed how weak the test is on a year of data: a model that
is badly wrong is often not rejected.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from financialkit.risk import kupiec_test, parametric_var, var_exceptions

# %%
# Non-rejection regions
# ---------------------

for n in (250, 500, 1000):
    accepted = [x for x in range(60) if not kupiec_test(np.arange(n) < x, 0.99).reject()]
    print(f"n = {n:4d}: a 99% VaR passes with {accepted[0]} to {accepted[-1]} exceptions (expected {n // 100})")

# %%
# Testing a model that underestimates volatility
# ----------------------------------------------
#
# Returns have a 1.2% daily volatility; the model assumes 1%. With 250
# days the test catches the error only about half the time; with four
# years, almost always.

rng = np.random.default_rng(0)
var = parametric_var(0.0, 0.01, 0.99)
power = {}
for n in (250, 1000):
    pnl = 0.012 * rng.standard_normal((1000, n))
    power[n] = np.mean([kupiec_test(var_exceptions(row, var)).reject() for row in pnl])
    print(f"n = {n:4d}: rejection rate {power[n]:.0%}")

pnl = 0.012 * rng.standard_normal(1000)
res = kupiec_test(var_exceptions(pnl, var))
print(f"one sample: {res.n_exceptions} exceptions vs {res.expected:.0f} expected, LR = {res.statistic:.2f}, p = {res.p_value:.4f}")

x = np.arange(0, 25)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(x, [kupiec_test(np.arange(1000) < k).statistic for k in x], ".-", label="LR statistic, n = 1000")
ax.axhline(3.841, color="C3", ls="--", label="5% critical value")
ax.set_xlabel("number of exceptions")
ax.legend()
fig.tight_layout()

plt.show()
