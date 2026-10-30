Examples
========

This gallery walks through every public feature of
``financialkit.stochastic_volatility``: local volatility, stochastic
volatility, jumps, Fourier pricing, smile parametrisations, Heston
simulation and rough volatility. Each script illustrates one dated breakthrough from the
:doc:`history page </history/stochastic_volatility_breakthroughs>`.

See also the narrative tutorials:

- :doc:`/tutorials/one_option_five_pricers`
- :doc:`/tutorials/where_the_smile_comes_from`

Each script in this gallery is self-contained and can be run directly with
``python examples/stochastic_volatility/<section>/<script>.py``.

Sections
--------

- **local_vol** -- Cox's CEV model and Dupire's local volatility.
- **jumps** -- Merton's jump diffusion, the Madan-Seneta and
  Madan-Carr-Chang variance gamma, Bates and Kou.
- **heston** -- Hull and White's mixing formula and the Heston model.
- **fourier** -- Carr-Madan's FFT, Lewis's single integral and the COS method.
- **smile** -- Hagan's SABR, Gatheral's SVI, Lee's moment formula and SSVI.
- **simulation** -- Andersen's QE scheme and full-truncation Euler.
- **rough** -- rough Bergomi, the hybrid scheme and the roughness of
  volatility.
