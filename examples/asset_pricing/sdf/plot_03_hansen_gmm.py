r"""
Hansen's generalized method of moments
======================================

Lars Peter Hansen (1982) gave asset pricing its estimator. A model
says that some pricing errors have mean zero, here those of a linear
discount factor :math:`m = 1 - b f` on ten excess returns,
:math:`E[R^e_i (1 - b f)] = 0`. GMM picks :math:`b` to make the ten
sample means as small as possible in the metric :math:`g_T' W g_T`.
There are ten moments and only one parameter, and the leftover
nine are a free test of the model: :math:`J = T g_T' \hat S^{-1} g_T`
is :math:`\chi^2_9` when the model is right.

The returns are simulated from a one-factor model with
:math:`E[f] = 0.6\%` and :math:`\sigma(f) = 4.5\%` a month, so the
true discount factor has :math:`b = E[f] / E[f^2]`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.asset_pricing import gmm, simulate_factor_model

mu_f, sd_f, T = 0.006, 0.045, 240
betas = np.linspace(0.5, 1.5, 10)
b_true = mu_f / (sd_f**2 + mu_f**2)


def estimate(alphas=None, rng=None, weighting="two-step"):
    panel = simulate_factor_model(T, betas, [mu_f], [[sd_f**2]], idio_vol=0.03, alphas=alphas, rng=rng)
    R, f = panel.excess_returns, panel.factors[:, 0]
    return gmm(lambda th: R * (1.0 - th[0] * f)[:, None], [0.0], weighting=weighting)


first = estimate(rng=0, weighting="identity")
second = estimate(rng=0)
print(f"true b = {b_true:.3f}")
print(f"identity weighting: b = {first.params[0]:.3f} (se {first.std_errors[0]:.3f}), J = {first.j_stat:.1f}")
print(f"two-step efficient: b = {second.params[0]:.3f} (se {second.std_errors[0]:.3f}), J = {second.j_stat:.1f}, p = {second.j_pvalue:.2f}")
# Both estimates lie within three standard errors of the truth, and the
# model is not rejected.
for res in (first, second):
    assert abs(res.params[0] - b_true) < 3 * res.std_errors[0]
assert second.j_pvalue > 0.05

# %%
# The J test
# ----------
#
# Over 400 samples the J statistics follow the :math:`\chi^2_9`
# distribution, and the test rejects about 5% of the time. When half
# the assets carry a pricing error of 0.5% a month, J moves to the
# right and the test rejects almost always.

rng = np.random.default_rng(1)
J_null = np.array([estimate(rng=rng).j_stat for _ in range(400)])
alphas = np.where(np.arange(10) < 5, 0.005, 0.0)
J_alt = np.array([estimate(alphas, rng=rng).j_stat for _ in range(400)])
crit = stats.chi2.ppf(0.95, 9)
size, power = (J_null > crit).mean(), (J_alt > crit).mean()
print(f"rejection rate: {size:.1%} when the model holds, {power:.1%} with mispricing")
# The size is 5% to within three binomial standard errors.
assert abs(size - 0.05) < 3 * np.sqrt(0.05 * 0.95 / 400)
assert power > 0.9

fig, ax = plt.subplots(figsize=(7, 4))
x = np.linspace(0, 60, 300)
ax.hist(J_null, bins=40, range=(0, 60), density=True, alpha=0.6, label="J, model holds")
ax.hist(J_alt, bins=40, range=(0, 60), density=True, alpha=0.6, label="J, half the assets mispriced")
ax.plot(x, stats.chi2.pdf(x, 9), "k", label=r"$\chi^2_9$")
ax.axvline(crit, color="C3", ls="--", label="5% critical value")
ax.set_xlabel("J statistic")
ax.legend()
fig.tight_layout()
plt.show()
