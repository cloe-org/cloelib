# cloelib – The Library for the Cosmology Likelihood for Observables in Euclid

**cloelib** is a flexible and efficient library designed to compute cosmological observables for the **CLOE** (_Cosmology Likelihood for Observables in Euclid_) project. It is built for seamless integration with **Boltzmann solvers** and **JAX-based frameworks**, enabling automatic differentiation and modularity for the next generation of cosmological analyses.

---

## ✨ Features

🔹 **Intuitive & User-Friendly** – Fast generation of **Euclid-like** observables (e.g., power spectra, window functions, and tracer statistics).

🔹 **Automatic Differentiation** – Supporting **`JAX`** for gradient-based computations.

🔹 **Modular & Extensible** – Easily interface with external **Boltzmann solvers** or **emulators** via **Python `Protocols`** following the [cosmology.API](https://cosmology.readthedocs.io/projects/api/latest/#). Core structure enables defining **Background & Perturbation** models, choosing observables via **Tracer** or **SpectroPower** protocols, and computing final summary statistics like **angular power spectra** or **Legendre multipoles**.

---

## 📂 Supported external codes

`cloelib` interfaces with the following external cosmological codes, each used by a specific internal module for its calculations:

| Background                                        | Perturbations                                                      | SpectroPower                                                       |
| ------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| [camb](https://camb.readthedocs.io)               | [camb](https://camb.readthedocs.io)                                | [comet-emu](https://comet-emu.readthedocs.io/en/latest/index.html) |
| [class](https://github.com/lesgourg/class_public) | [class](https://github.com/lesgourg/class_public)                  | `PBJ` (not publicly available)                                     |
| NA                                                | [HMCode2020emu](https://github.com/MariaTsedrik/HMcode2020Emu.git) | NA                                                                 |

We do not provide installation support for `PBJ`.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

For contributions to this project, we follow the [all-contributors](https://allcontributors.org) specification. We appreciate all forms of contributions, including code, documentation, and more. Please refer to the [README](README.md) for detailed guidelines on how to contribute and be recognized for your efforts!
