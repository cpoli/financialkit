financialkit.fixed_income
=========================


.. include:: /_generated/nav/fixed_income.rst

Bond cash flows, prices and yield to maturity (``scipy.optimize.brentq``),
accrued interest and STRIPS; Macaulay, modified and Fisher-Weil duration,
convexity and key rate durations; Lutz forward rates; zero curves
bootstrapped from bond prices or par rates, Carleton-Cooper discount
factors and McCulloch's spline discount function (``numpy.linalg.lstsq``);
Nelson-Siegel and Svensson fits (``scipy.optimize.least_squares``) and the
principal components of curve moves (``numpy.linalg.eigh``); Redington
immunization and the Fong-Vasicek :math:`M^2` bound; and interest-rate and
currency swaps. Bonds and swaps also run on dated schedules from
``time_value.coupon_schedule``, with SIA odd-coupon pricing and accrual.
Overnight rates such as SOFR compound in arrears under the ARRC lookback,
observation-shift and lockout conventions; OIS discount curves pair with
separate forecasting curves; Treasury bond futures have conversion
factors and a cheapest-to-deliver bond; and mortgage pass-throughs pay
out under the PSA prepayment benchmark.

.. automodule:: financialkit.fixed_income
   :members:
   :undoc-members:
