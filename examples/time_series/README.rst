Examples
========

This gallery walks through every public feature of ``financialkit.time_series``:
the stylized facts of returns, tests of the random walk, ARMA and VAR
models, unit roots and cointegration, long memory, state-space and
regime-switching models, and conditional-volatility models. Each script illustrates one dated
breakthrough from the :doc:`history page </history/time_series_breakthroughs>`.
All data are synthetic and seeded; no market data is downloaded.

See also the narrative tutorial:

- :doc:`/tutorials/from_returns_to_risk`

Each script in this gallery is self-contained and can be run directly with
``python examples/time_series/<section>/<script>.py``.

Sections
--------

- **linear_models** -- Yule's autoregression, Box-Jenkins ARMA,
  Akaike's information criterion and Sims's VAR.
- **long_memory** -- Hurst's rescaled range.
- **efficiency** -- Kendall's serial correlations, Fama's runs test, the
  Ljung-Box portmanteau test and Lo-MacKinlay variance ratios.
- **stylized_facts** -- Mandelbrot's fat tails and the Hill tail index.
- **unit_roots** -- the Dickey-Fuller test, Engle-Granger and Johansen
  cointegration, and the KPSS test.
- **state_space** -- the Kalman filter and Hamilton's regime switching.
- **volatility** -- Parkinson's range, ARCH, GARCH, EGARCH, GJR-GARCH,
  RiskMetrics EWMA, realized volatility, Corsi's HAR and DCC-GARCH.
