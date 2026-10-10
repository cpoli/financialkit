Examples
========

This gallery walks through every public feature of
``financialkit.exotic_options``: barrier options from Merton's image
formula to discrete monitoring, binary options, Asian options,
lookbacks and variance swaps, compound, forward-start and chooser
options, options on two assets, and American options by least-squares
Monte Carlo. Each script illustrates one dated breakthrough from the
:doc:`history page </history/exotic_options_breakthroughs>`.

Each script in this gallery is self-contained and can be run directly with
``python examples/exotic_options/<section>/<script>.py``.

Sections
--------

- **barriers** -- Merton's down-and-out call, the Reiner-Rubinstein
  formulas, Ikeda-Kunitomo double barriers and the Broadie-Glasserman-Kou
  correction for discrete monitoring.
- **binaries** -- cash-or-nothing, asset-or-nothing, gap and one-touch
  options.
- **asian** -- Kemna-Vorst geometric averages and control variates, and
  Turnbull-Wakeman moment matching.
- **path_dependent** -- floating and fixed lookbacks and variance swaps.
- **compound** -- Geske compound options, forward starts and choosers.
- **multi_asset** -- Margrabe exchange options, Stulz options on the
  maximum and minimum, and quantos.
- **american** -- Longstaff-Schwartz least-squares Monte Carlo.
