Examples
========

This gallery walks through every public feature of
``financialkit.pde_methods``: the explicit, implicit and Crank-Nicolson
schemes, their stability and accuracy, American options as linear
complementarity problems, and a two-asset ADI solver. Each script
illustrates one dated breakthrough from the
:doc:`history page </history/pde_methods_breakthroughs>`.

Each script in this gallery is self-contained and can be run directly with
``python examples/pde_methods/<section>/<script>.py``.

Sections
--------

- **schemes** -- Crank-Nicolson, Laasonen's implicit scheme, the Thomas
  algorithm, the heat-equation transform, Schwartz's explicit scheme,
  the explicit scheme as a trinomial tree, and barrier options.
- **stability** -- the Courant-Friedrichs-Lewy condition, von Neumann
  analysis and Rannacher smoothing.
- **accuracy** -- Richardson extrapolation and Tavella-Randall grids.
- **american** -- Cryer's projected SOR, the Brennan-Schwartz algorithm
  and the penalty method.
- **multi_factor** -- the Peaceman-Rachford ADI scheme for two assets.
