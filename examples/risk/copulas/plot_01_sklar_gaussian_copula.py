r"""
Sklar's theorem and the Gaussian copula
=======================================

Sklar (1959) showed that every joint distribution is its marginals
glued together by a copula :math:`C`:

.. math:: F(x_1, x_2) = C\big(F_1(x_1), F_2(x_2)\big).

Risk managers can therefore model each asset's return on its own and
choose the dependence separately. Here a Gaussian copula with
correlation 0.7 joins a fat-tailed Student-t equity return to a skewed
credit loss, and the copula is recovered from the joint sample by
ranks alone.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from financialkit.risk import apply_marginals, copula_kendall_tau, gaussian_copula_sample, pseudo_observations
from financialkit.risk.visualizers import plot_copula

corr = np.array([[1.0, 0.7], [0.7, 1.0]])
u = gaussian_copula_sample(corr, 3000, rng=0)

# %%
# One copula, any marginals
# -------------------------

equity = stats.t(4, scale=0.01)
credit = stats.lognorm(0.8, scale=0.005)
x = apply_marginals(u, [equity.ppf, credit.ppf])

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
plot_copula(u, ax=axes[0], title="Gaussian copula, rho = 0.7")
axes[1].scatter(x[:, 0], x[:, 1], s=2, alpha=0.4)
axes[1].set_xlabel("equity return (t, 4 df)")
axes[1].set_ylabel("credit loss (lognormal)")
axes[1].set_title("joint distribution")
plot_copula(pseudo_observations(x), ax=axes[2], title="copula recovered from ranks")
fig.tight_layout()

# %%
# Rank correlation is a property of the copula
# --------------------------------------------
#
# Pearson correlation changes with the marginals; Kendall's tau does
# not, and for the Gaussian copula it equals :math:`\tfrac{2}{\pi}\arcsin\rho`.

pearson_u, pearson_x = np.corrcoef(u.T)[0, 1], np.corrcoef(x.T)[0, 1]
tau_u, tau_x = stats.kendalltau(*u.T).statistic, stats.kendalltau(*x.T).statistic
print(f"Pearson, uniforms {pearson_u:.3f}  vs  Pearson, t/lognormal {pearson_x:.3f}")
print(f"Kendall, uniforms {tau_u:.3f}  vs  Kendall, t/lognormal {tau_x:.3f}")
print(f"theory 2/pi arcsin(0.7) = {copula_kendall_tau(0.7):.3f}")
# the marginals move Pearson but leave Kendall untouched; tau's standard error is about 0.01 here
assert pearson_u - pearson_x > 0.05
assert np.isclose(tau_u, tau_x) and abs(tau_u - copula_kendall_tau(0.7)) < 0.03

plt.show()
