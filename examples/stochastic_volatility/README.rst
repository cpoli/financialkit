Examples
========

This gallery walks through every public feature of
``financialkit.stochastic_volatility``: local volatility, stochastic
volatility, jumps, Fourier pricing, smile parametrisations and Heston
simulation. Each script illustrates one dated breakthrough from the
:doc:`history page </history/stochastic_volatility_breakthroughs>`.

Each script in this gallery is self-contained and can be run directly with
``python examples/stochastic_volatility/<section>/<script>.py``.

Sections
--------

- **local_vol** -- Cox's CEV model and Dupire's local volatility.
- **jumps** -- Merton's jump diffusion, the Madan-Seneta and
  Madan-Carr-Chang variance gamma, Bates and Kou.
- **heston** -- Hull and White's mixing formula and the Heston model.
- **fourier** -- Carr-Madan's FFT, Lewis's single integral and the COS method.
- **smile** -- Hagan's SABR, Gatheral's SVI and Lee's moment formula.
- **simulation** -- Andersen's QE scheme and full-truncation Euler.
