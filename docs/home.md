# cloelib – The Library for the Cosmology Likelihood for Observables in Euclid

**cloelib** is a flexible and efficient library designed to compute cosmological observables for the **CLOE** (_Cosmology Likelihood for Observables in Euclid_) project. It is built for seamless integration with **Boltzmann solvers** and **JAX-based frameworks**, enabling automatic differentiation and modularity for the next generation of cosmological analyses.

---

## Features

- **Intuitive & User-Friendly** – Fast generation of **Euclid-like** observables (e.g., power spectra, window functions, and tracer statistics).

- **Automatic Differentiation** – Supporting **`JAX`** for gradient-based computations.

- **Modular & Extensible** – Easily interface with external Boltzmann solvers or emulators via Python `Protocols` following the [cosmology.API](https://cosmology.readthedocs.io/projects/api/latest/#). The core structure enables defining Background & Perturbation models, choosing observables via Tracer or SpectroPower protocols, and computing final summary statistics such as angular power spectra or Legendre multipoles. Check the [guide](code_structure/index.md) to learn more about its architecture.

---

## Supported External Codes

--8<-- "README.md:supported-codes"

---

## Contributors

Contributions to this project follow the [all-contributors](https://allcontributors.org) specification. All forms of contributions are appreciated, including code, documentation, and more.

--8<-- "README.md:contributors"

## License

This project is licensed under the [MIT License](https://github.com/cloe-org/cloelib?tab=MIT-1-ov-file).
