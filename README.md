# cloelib – The Library for the Cosmology Likelihood for Observables in Euclid  

**cloelib** is a flexible and efficient library designed to compute cosmological observables for the **CLOE** (*Cosmology Likelihood for Observables in Euclid*) project. It is built for seamless integration with **Boltzmann solvers** and **JAX-based frameworks**, enabling automatic differentiation and modularity for the next generation of cosmological analyses.  

We welcome feedback from the **Euclid community** and beyond to refine and improve this library!  

[![CI](https://github.com/cloe-org/cloelib/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/cloe-org/cloelib/actions/workflows/ci.yaml)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)

---

## 📖 Table of Contents  
- [✨ Features](#-features)  
- [📂 Supported external codes](#-supported-external-codes) 
- [🚀 Installation](#-installation)  
- [📊 Usage](#-usage)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  
- [🙏 Acknowledgements](#-acknowledgements)  

---

## ✨ Features  

🔹 **Intuitive & User-Friendly** – Generate **Euclid-like** observables (e.g., power spectra, window functions, and tracer statistics) in just **3 minutes**!  

🔹 **Automatic Differentiation** – Includes a **toy-example with `JAX`** for gradient-based computations.  

🔹 **Modular & Extensible** –  
- Easily interface with external **Boltzmann solvers** or **emulators** via **Python `Protocols`** following the cosmology.API.
- Core structure enables defining **Background & Perturbation** models, choosing observables via **Tracer** or **SpectroPower** protocols, and computing final summary statistics like **angular power spectra** or **Legendre multipoles**.  

---

## 📂 Supported external codes

`cloelib` interfaces with the following external codes, each used by a specific internal module for its calculations:

| Background                                           | Perturbations                                         | SpectroPower                                         |
|------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------|
| [camb](https://camb.readthedocs.io)                   | [camb](https://camb.readthedocs.io)                    | [comet-emu](https://comet-emu.readthedocs.io/en/latest/index.html) |
| [class](https://github.com/lesgourg/class_public)     | [class](https://github.com/lesgourg/class_public)      | `PBJ` (not publicly available)                       |
| NA    | [HMCode2020emu](https://github.com/MariaTsedrik/HMcode2020Emu.git)       | NA                       |
| NA    | [FlamingoBaryonResponseEmulator](https://github.com/FLAMINGOSIM/FlamingoBaryonResponseEmulator.git)  | NA                        |
We do not provide installation support for `PBJ` and `class`.

---

## 🚀 Installation  

To install `cloelib` source code, clone the repository and install it via `pip`:  
```sh
pip install .
```

You can also install (some) supported dependencies:

```sh
pip install .[camb,hmcode2020emu,comet-emu,FlamingoBaryonResponseEmulator]
```

**Note:** Some shells or terminals may not interpret the brackets correctly. If you encounter an error, try adding quotation marks:

 ```sh
pip install ."[camb,hmcode2020emu,comet-emu,FlamingoBaryonResponseEmulator]"
```

**Note:** We do not offer installation support for `PBJ` and `CLASS`. For installation instructions, please refer to the official documentation of each package.

To work with the latest stable release of the code, move to the latest tag by typing: 
 ```sh
 git checkout name-latest-release
 ```
 with name-latest-release the latest name that appears in "Releases".

---

## 📊 Usage  

Explore the **tutorials** in the `cloe-org/playground` repository for examples on how to compute cosmological observables and other key quantities!  

---

## 🤝 Contributing  

Please review the organization's general contribution guidelines and the specific guidelines for this repository in the [CONTRIBUTING.md](CONTRIBUTING.md) file. Once you're familiar with the guidelines, follow these steps:

1️⃣ Create a new branch:  
   ```sh
   git checkout -b feature/your-feature-name
   ```  
2️⃣ Implement your changes following project style guidelines.  
3️⃣ Commit your modifications:  
   ```sh
   git commit -m "Add feature: [brief description]"  
   ```  
4️⃣ Push your branch:  
   ```sh
   git push origin feature/your-feature-name  
   ```  
5️⃣ Open a **pull request** and contribute to the project!  

---

## 📜 License  

This project is licensed under the **MIT LICENSE** – see the [LICENSE](LICENSE) file for details.  

---

## 🙏 Acknowledgements  

🔭 Inspired by the pioneering work of the **Euclid Consortium** CLOE software and the **`jaxcosmo`** project. 

👩‍💻🧑‍💻 Authored by M. Bonici, G. Cañas-Herrera, P. Carrilho, S. Casas, C. Moretti, and A. Pezzotta (listed in alphabetical order).

🎯 With technical advice from S. Farrens and N. Tessore.

## 🤝 Contributors

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://alexhall.space"><img src="https://avatars.githubusercontent.com/u/59091484?v=4?s=100" width="100px;" alt="Alex Hall"/><br /><sub><b>Alex Hall</b></sub></a><br /><a href="#bug-ahallcosmo" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/llinke1"><img src="https://avatars.githubusercontent.com/u/42432333?v=4?s=100" width="100px;" alt="Laila Linke"/><br /><sub><b>Laila Linke</b></sub></a><br /><a href="#code-llinke1" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DavidNavarroG"><img src="https://avatars.githubusercontent.com/u/29857945?v=4?s=100" width="100px;" alt="David Navarro Gironés"/><br /><sub><b>David Navarro Gironés</b></sub></a><br /><a href="#doc-DavidNavarroG" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stefanodavini"><img src="https://avatars.githubusercontent.com/u/206831738?v=4?s=100" width="100px;" alt="stefanodavini"/><br /><sub><b>stefanodavini</b></sub></a><br /><a href="#code-stefanodavini" title="Code">💻</a> <a href="#doc-stefanodavini" title="Documentation">📖</a> <a href="#test-stefanodavini" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#maintenance-gcanasherrera" title="Maintenance">🚧</a> <a href="#ideas-gcanasherrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-gcanasherrera" title="Bug reports">🐛</a> <a href="#content-gcanasherrera" title="Content">🖋</a> <a href="#data-gcanasherrera" title="Data">🔣</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a> <a href="#infra-gcanasherrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-gcanasherrera" title="Project Management">📆</a> <a href="#question-gcanasherrera" title="Answering Questions">💬</a> <a href="#test-gcanasherrera" title="Tests">⚠️</a> <a href="#talk-gcanasherrera" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://marcobonici.github.io/"><img src="https://avatars.githubusercontent.com/u/58727599?v=4?s=100" width="100px;" alt="Marco Bonici"/><br /><sub><b>Marco Bonici</b></sub></a><br /><a href="#code-marcobonici" title="Code">💻</a> <a href="#maintenance-marcobonici" title="Maintenance">🚧</a> <a href="#ideas-marcobonici" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-marcobonici" title="Bug reports">🐛</a> <a href="#content-marcobonici" title="Content">🖋</a> <a href="#doc-marcobonici" title="Documentation">📖</a> <a href="#infra-marcobonici" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-marcobonici" title="Project Management">📆</a> <a href="#question-marcobonici" title="Answering Questions">💬</a> <a href="#test-marcobonici" title="Tests">⚠️</a> <a href="#talk-marcobonici" title="Talks">📢</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chiaramoretti"><img src="https://avatars.githubusercontent.com/u/12472732?v=4?s=100" width="100px;" alt="Chiara Moretti"/><br /><sub><b>Chiara Moretti</b></sub></a><br /><a href="#code-chiaramoretti" title="Code">💻</a> <a href="#maintenance-chiaramoretti" title="Maintenance">🚧</a> <a href="#ideas-chiaramoretti" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-chiaramoretti" title="Bug reports">🐛</a> <a href="#content-chiaramoretti" title="Content">🖋</a> <a href="#doc-chiaramoretti" title="Documentation">📖</a> <a href="#talk-chiaramoretti" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaPezzotta"><img src="https://avatars.githubusercontent.com/u/29603598?v=4?s=100" width="100px;" alt="AndreaPezzotta"/><br /><sub><b>AndreaPezzotta</b></sub></a><br /><a href="#code-AndreaPezzotta" title="Code">💻</a> <a href="#maintenance-AndreaPezzotta" title="Maintenance">🚧</a> <a href="#ideas-AndreaPezzotta" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-AndreaPezzotta" title="Bug reports">🐛</a> <a href="#content-AndreaPezzotta" title="Content">🖋</a> <a href="#data-AndreaPezzotta" title="Data">🔣</a> <a href="#doc-AndreaPezzotta" title="Documentation">📖</a> <a href="#talk-AndreaPezzotta" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="#code-santiagocasas" title="Code">💻</a> <a href="#maintenance-santiagocasas" title="Maintenance">🚧</a> <a href="#ideas-santiagocasas" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PedroCarrilho"><img src="https://avatars.githubusercontent.com/u/60090062?v=4?s=100" width="100px;" alt="Pedro Carrilho"/><br /><sub><b>Pedro Carrilho</b></sub></a><br /><a href="#code-PedroCarrilho" title="Code">💻</a> <a href="#maintenance-PedroCarrilho" title="Maintenance">🚧</a> <a href="#ideas-PedroCarrilho" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-PedroCarrilho" title="Bug reports">🐛</a> <a href="#content-PedroCarrilho" title="Content">🖋</a> <a href="#data-PedroCarrilho" title="Data">🔣</a> <a href="#doc-PedroCarrilho" title="Documentation">📖</a> <a href="#talk-PedroCarrilho" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://ntessore.page"><img src="https://avatars.githubusercontent.com/u/3993688?v=4?s=100" width="100px;" alt="Nicolas Tessore"/><br /><sub><b>Nicolas Tessore</b></sub></a><br /><a href="#tool-ntessore" title="Tools">🔧</a> <a href="#mentoring-ntessore" title="Mentoring">🧑‍🏫</a> <a href="#code-ntessore" title="Code">💻</a> <a href="#review-ntessore" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://sfarrens.github.io"><img src="https://avatars.githubusercontent.com/u/6851839?v=4?s=100" width="100px;" alt="Samuel Farrens"/><br /><sub><b>Samuel Farrens</b></sub></a><br /><a href="#tool-sfarrens" title="Tools">🔧</a> <a href="#mentoring-sfarrens" title="Mentoring">🧑‍🏫</a> <a href="#code-sfarrens" title="Code">💻</a> <a href="#review-sfarrens" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josecolomanadal"><img src="https://avatars.githubusercontent.com/u/83759085?v=4?s=100" width="100px;" alt="Jose Coloma Nadal"/><br /><sub><b>Jose Coloma Nadal</b></sub></a><br /><a href="#bug-josecolomanadal" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/caspervedder"><img src="https://avatars.githubusercontent.com/u/187176614?v=4?s=100" width="100px;" alt="Casper Vedder"/><br /><sub><b>Casper Vedder</b></sub></a><br /><a href="#code-caspervedder" title="Code">💻</a> <a href="#ideas-caspervedder" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-caspervedder" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

