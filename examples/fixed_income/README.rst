Examples
========

This gallery walks through every public feature of ``financialkit.fixed_income``:
bond prices and yields, duration and convexity, yield-curve construction,
immunization, swaps, Treasury futures and mortgage pass-throughs. Each script illustrates one dated breakthrough from
the :doc:`history page </history/fixed_income_breakthroughs>`.

See also the narrative tutorial:

- :doc:`/tutorials/yield_curve_end_to_end`

Each script in this gallery is self-contained and can be run directly with
``python examples/fixed_income/<section>/<script>.py``.

Sections
--------

- **bonds** -- yield to maturity, accrued interest, and a coupon bond as a
  portfolio of STRIPS.
- **duration** -- Macaulay, modified and Fisher-Weil duration, convexity,
  and key rate durations.
- **curves** -- forward rates, McCulloch splines, bootstrapping,
  Nelson-Siegel and Svensson fits, the principal components of curve
  moves, and OIS discounting with a separate forecasting curve.
- **immunization** -- Redington's conditions and the Fong-Vasicek
  :math:`M^2` bound.
- **swaps** -- the IBM-World Bank currency swap, the plain-vanilla
  interest-rate swap, and SOFR compounded in arrears.
- **futures** -- Treasury bond futures: conversion factors and the
  cheapest-to-deliver bond.
- **mbs** -- mortgage pass-through cash flows under the PSA prepayment
  benchmark.
