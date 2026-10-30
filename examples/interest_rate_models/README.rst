Examples
========

This gallery walks through every public feature of
``financialkit.interest_rate_models``: equilibrium and affine short-rate
models, models fitted to today's curve, bond options, caps, floors and
swaptions, the Heath-Jarrow-Morton framework, callable bonds and
mortgage-backed securities. Each script illustrates
one dated breakthrough from the
:doc:`history page </history/interest_rate_models_breakthroughs>`.

See also the narrative tutorial:

- :doc:`/tutorials/yield_curve_end_to_end`

Each script in this gallery is self-contained and can be run directly with
``python examples/interest_rate_models/<section>/<script>.py``.

Sections
--------

- **short_rate** -- Feynman-Kac bond prices by simulation, Feller's
  boundary condition, Merton's Gaussian rate, Vasicek, Cox-Ingersoll-Ross,
  Duffie-Kan affine models and the two-factor G2++.
- **curve_fitted** -- Ho-Lee, Hull-White, the Black-Derman-Toy tree, the
  futures convexity adjustment, Hull-White calibration and the Hull-White
  trinomial tree for callable bonds and the OAS.
- **options** -- Jamshidian's bond options and swaptions, caps and floors
  by Black's formula, Black swaptions and the Lyashenko-Mercurio forward
  market model for backward-looking rates.
- **mbs** -- mortgage pass-throughs with Richard-Roll prepayments.
- **hjm** -- the Heath-Jarrow-Morton drift condition.
