"""
cloelib – The Library for the Cosmology Likelihood for Observables in Euclid.

`cloelib` is a flexible and efficient library designed to compute cosmological observables for the
CLOE (Cosmology Likelihood for Observables in Euclid) project. It is built for seamless integration
with Boltzmann solvers and JAX-based frameworks, enabling automatic differentiation and modularity
for the next generation of cosmological analyses.

Key Features:
- **Intuitive & User-Friendly**: Generate Euclid-like observables (e.g., power spectra, window functions,
  and tracer statistics) rapidly.
- **Automatic Differentiation**: Includes a toy example with `JAX` for gradient-based computations.
- **Modular & Extensible**:
    - Easily interface with external Boltzmann solvers or emulators via Python `Protocols` following
      the cosmology.API.
    - Core structure enables defining Background & Perturbation models, choosing observables via
      Tracer or SpectroPower protocols, and computing final summary statistics like angular power spectra
      or Legendre multipoles.

License:
This project is licensed under the MIT LICENSE.
"""

import jax

# Enable double precision in JAX
jax.config.update("jax_enable_x64", True)

__all__: list[str] = []
