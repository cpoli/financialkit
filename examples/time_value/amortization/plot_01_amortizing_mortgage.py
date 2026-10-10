r"""
The fully amortizing mortgage
=============================

Before the 1930s a typical American home loan ran five to ten years,
paid interest only, and ended in a balloon payment that had to be
refinanced -- which became impossible in the Depression. The Home
Owners' Loan Corporation (1933) and then the FHA (1934) replaced it with
the long-term, level-payment, fully amortizing mortgage: each monthly
payment covers the interest on the balance and repays a little
principal, so the loan is gone at the last payment.
"""

# %%
import matplotlib.pyplot as plt

from financialkit.time_value import amortization_schedule, plot_amortization

# %%
# A 30-year loan of 200,000 at 6%
# -------------------------------

s = amortization_schedule(200_000.0, 0.06, 360)
print(f"monthly payment: {s.payment:,.2f}")
print(f"first payment: interest {s.interest[0]:,.2f}, principal {s.principal[0]:,.2f}")
print(f"last payment:  interest {s.interest[-1]:,.2f}, principal {s.principal[-1]:,.2f}")
print(f"total interest over 30 years: {s.total_interest:,.0f}")
half = int((s.balance > 100_000).sum())
print(f"half the principal is repaid only after {half / 12:.1f} years")

# %%
# Interest first, principal later
# -------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_amortization(s, ax=axes[0])
for years in (15, 30):
    loan = amortization_schedule(200_000.0, 0.06, 12 * years)
    axes[1].plot(loan.periods / 12, loan.balance, label=f"{years}-year: payment {loan.payment:,.0f}")
axes[1].set_xlabel("years")
axes[1].set_ylabel("outstanding balance")
axes[1].set_title("15- vs 30-year mortgage")
axes[1].legend()
fig.tight_layout()

plt.show()
