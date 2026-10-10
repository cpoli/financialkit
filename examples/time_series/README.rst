Examples
========

This gallery walks through every public feature of ``financialkit.time_series``:
the stylized facts of returns, tests of the random walk, ARMA models and
order selection, unit roots and cointegration, long memory, and
conditional-volatility models. Each script illustrates one dated
breakthrough from the :doc:`history page </history/time_series_breakthroughs>`.
All data are synthetic and seeded; no market data is downloaded.

Each script in this gallery is self-contained and can be run directly with
``python examples/time_series/<section>/<script>.py``.

Sections
--------

- **linear_models** -- Yule's autoregression, Box-Jenkins ARMA and
  Akaike's information criterion.
- **long_memory** -- Hurst's rescaled range.
- **efficiency** -- Kendall's serial correlations, Fama's runs test, the
  Ljung-Box portmanteau test and Lo-MacKinlay variance ratios.
- **stylized_facts** -- Mandelbrot's fat tails and the Hill tail index.
- **unit_roots** -- the Dickey-Fuller test and Engle-Granger cointegration.
- **volatility** -- Parkinson's range, ARCH, GARCH, EGARCH, GJR-GARCH,
  RiskMetrics EWMA and realized volatility.
