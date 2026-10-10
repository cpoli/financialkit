Examples
========

This gallery walks through every public feature of
``financialkit.interest_rate_models``: equilibrium and affine short-rate
models, models fitted to today's curve, bond options, caps, floors and
swaptions, and the Heath-Jarrow-Morton framework. Each script illustrates
one dated breakthrough from the
:doc:`history page </history/interest_rate_models_breakthroughs>`.

Each script in this gallery is self-contained and can be run directly with
``python examples/interest_rate_models/<section>/<script>.py``.

Sections
--------

- **short_rate** -- Feynman-Kac bond prices by simulation, Feller's
  boundary condition, Merton's Gaussian rate, Vasicek, Cox-Ingersoll-Ross,
  Duffie-Kan affine models and the two-factor G2++.
- **curve_fitted** -- Ho-Lee, Hull-White, the Black-Derman-Toy tree and
  the futures convexity adjustment.
- **options** -- Jamshidian's bond options and swaptions, caps and floors
  by Black's formula, and Black swaptions.
- **hjm** -- the Heath-Jarrow-Morton drift condition.
