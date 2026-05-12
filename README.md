# cloelib – The Library for the Cosmology Likelihood for Observables in Euclid

**cloelib** is a flexible and efficient library designed to compute cosmological observables for the **CLOE** (_Cosmology Likelihood for Observables in Euclid_) project. It is built for seamless integration with **Boltzmann solvers** and **JAX-based frameworks**, enabling automatic differentiation and modularity for the next generation of cosmological analyses.

We welcome feedback from the **Euclid community** and beyond to refine and improve this library!

[![CI](https://github.com/cloe-org/cloelib/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/cloe-org/cloelib/actions/workflows/ci.yaml)
[![Docs](https://github.com/cloe-org/cloelib/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/cloe-org/cloelib/actions/workflows/docs.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-blue?logo=pytest)](https://docs.pytest.org/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-purple?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Code Style: Prettier](https://img.shields.io/badge/code%20style-prettier-ff69b4.svg?logo=prettier&logoColor=white)](https://prettier.io/)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-8A2BE2?logo=mypy&logoColor=white)](https://mypy.readthedocs.io/)
[![PyPI version](https://img.shields.io/pypi/v/cloelib.svg?logo=pypi&logoColor=white)](https://pypi.org/project/cloelib/)

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

<!-- --8<-- [start:supported-codes] -->

`cloelib` interfaces with the following external codes, each used by a specific internal module for its calculations:

| Background                                        | Perturbations                                                                 | SpectroPower                                                       |
| ------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [camb](https://camb.readthedocs.io)               | [camb](https://camb.readthedocs.io)                                           | [comet-emu](https://comet-emu.readthedocs.io/en/latest/index.html) |
| [class](https://github.com/lesgourg/class_public) | [class](https://github.com/lesgourg/class_public)                             | [pbjcosmo](https://chiaramoretti.gitlab.io/pbj/)                   |
| NA                                                | [HMCode2020emu](https://github.com/MariaTsedrik/HMcode2020Emu.git)            | NA                                                                 |
| NA                                                | [cosmopower-jax](https://github.com/dpiras/cosmopower-jax.git)                | NA                                                                 |
| NA                                                | [euclidemu2](https://github.com/PedroCarrilho/EuclidEmulator2/tree/pywrapper) | NA                                                                 |
| NA                                                | [BACCOemu](https://bitbucket.org/rangulo/baccoemu/)                           | NA                                                                 |
| NA                                                | [e-MANTIS](https://gitlab.obspm.fr/e-mantis/e-mantis)                         | NA                                                                 |

<!-- --8<-- [end:supported-codes] -->

### Optional Dependencies

Several optional dependencies enhance **cloelib** capabilities:

- **`pyinstrument`** – Time profiling for performance optimization
- **`pylevin`**, **`mpmath`** – Required for specific observational probes (i.e: COSEBIs)
- **`tensorflow`** – Needed for certain emulator backends (i.e: `HMCode2020emu`)

These are not included in the default installation but can be installed as shown below.

---

## 🚀 Installation

**Quick Start** 🎯

1. **Set up your environment** – Create a fresh conda/mamba environment using the [cloe-org-environments](https://github.com/cloe-org/cloe-org-environments) repository. Then, clone `cloelib`:

```sh
git clone https://github.com/cloe-org/cloelib.git
cd cloelib
```

2. **Check out the latest release** (optional but recommended):

```sh
git checkout <latest-tag>  # Find tags in "Releases"
```

3. **Install cloelib** – Get the core library up and running:

```sh
pip install .
```

4. **Add optional superpowers** – Enhance with external dependencies and tools:

```sh
pip install .[camb,classy,hmcode2020emu,comet-emu,euclidemu2,pylevin,mpmath,tensorflow,pyinstrument,baccoemu,emantis,pbjcosmo]
```

> **💡 Pro Tip:** Some shells struggle with brackets. Try quotes if needed:
>
> ```sh
> pip install ."[camb,classy,hmcode2020emu,comet-emu,euclidemu2,pylevin,mpmath,tensorflow,pyinstrument,baccoemu,emantis,pbjcosmo]"
> ```

You're all set! 🎉 Ready to compute cosmological observables.

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

<!-- --8<-- [start:contributors] -->

## 🤝 Contributors

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://alexhall.space"><img src="https://avatars.githubusercontent.com/u/59091484?v=4?s=100" width="100px;" alt="Alex Hall"/><br /><sub><b>Alex Hall</b></sub></a><br /><a href="#bug-ahallcosmo" title="Bug reports">🐛</a> <a href="#ideas-ahallcosmo" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-ahallcosmo" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a> <a href="#projectManagement-itutusaus" title="Project Management">📆</a> <a href="#mentoring-itutusaus" title="Mentoring">🧑‍🏫</a> <a href="#promotion-itutusaus" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/llinke1"><img src="https://avatars.githubusercontent.com/u/42432333?v=4?s=100" width="100px;" alt="Laila Linke"/><br /><sub><b>Laila Linke</b></sub></a><br /><a href="#code-llinke1" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DavidNavarroG"><img src="https://avatars.githubusercontent.com/u/29857945?v=4?s=100" width="100px;" alt="David Navarro Gironés"/><br /><sub><b>David Navarro Gironés</b></sub></a><br /><a href="#doc-DavidNavarroG" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stefanodavini"><img src="https://avatars.githubusercontent.com/u/206831738?v=4?s=100" width="100px;" alt="stefanodavini"/><br /><sub><b>stefanodavini</b></sub></a><br /><a href="#code-stefanodavini" title="Code">💻</a> <a href="#doc-stefanodavini" title="Documentation">📖</a> <a href="#test-stefanodavini" title="Tests">⚠️</a> <a href="#ideas-stefanodavini" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-stefanodavini" title="Tools">🔧</a></td>
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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josecolomanadal"><img src="https://avatars.githubusercontent.com/u/83759085?v=4?s=100" width="100px;" alt="Jose Coloma Nadal"/><br /><sub><b>Jose Coloma Nadal</b></sub></a><br /><a href="#bug-josecolomanadal" title="Bug reports">🐛</a> <a href="#code-josecolomanadal" title="Code">💻</a> <a href="#ideas-josecolomanadal" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/caspervedder"><img src="https://avatars.githubusercontent.com/u/187176614?v=4?s=100" width="100px;" alt="Casper Vedder"/><br /><sub><b>Casper Vedder</b></sub></a><br /><a href="#code-caspervedder" title="Code">💻</a> <a href="#ideas-caspervedder" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-caspervedder" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://benbose.com/"><img src="https://avatars.githubusercontent.com/u/45853389?v=4?s=100" width="100px;" alt="Ben Bose"/><br /><sub><b>Ben Bose</b></sub></a><br /><a href="#code-nebblu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/didamarkovic"><img src="https://avatars.githubusercontent.com/u/9950748?v=4?s=100" width="100px;" alt="Dida Markovic"/><br /><sub><b>Dida Markovic</b></sub></a><br /><a href="#ideas-didamarkovic" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-didamarkovic" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/m-aguena"><img src="https://avatars.githubusercontent.com/u/12038660?v=4?s=100" width="100px;" alt="Michel Aguena"/><br /><sub><b>Michel Aguena</b></sub></a><br /><a href="#code-m-aguena" title="Code">💻</a> <a href="#test-m-aguena" title="Tests">⚠️</a> <a href="#ideas-m-aguena" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ippoppi"><img src="https://avatars.githubusercontent.com/u/50492103?v=4?s=100" width="100px;" alt="Filippo Oppizzi"/><br /><sub><b>Filippo Oppizzi</b></sub></a><br /><a href="#code-ippoppi" title="Code">💻</a> <a href="#ideas-ippoppi" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-ippoppi" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jaimeruizzapatero.net/"><img src="https://avatars.githubusercontent.com/u/39957598?v=4?s=100" width="100px;" alt="Jaime RZ"/><br /><sub><b>Jaime RZ</b></sub></a><br /><a href="#ideas-JaimeRZP" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://arthurmloureiro.github.io"><img src="https://avatars.githubusercontent.com/u/6471279?v=4?s=100" width="100px;" alt="Arthur Loureiro"/><br /><sub><b>Arthur Loureiro</b></sub></a><br /><a href="#code-arthurmloureiro" title="Code">💻</a> <a href="#ideas-arthurmloureiro" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-arthurmloureiro" title="Documentation">📖</a> <a href="#review-arthurmloureiro" title="Reviewed Pull Requests">👀</a> <a href="#bug-arthurmloureiro" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pburger112"><img src="https://avatars.githubusercontent.com/u/51719634?v=4?s=100" width="100px;" alt="Pierre Burger"/><br /><sub><b>Pierre Burger</b></sub></a><br /><a href="#code-pburger112" title="Code">💻</a> <a href="#ideas-pburger112" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/davidesciotti"><img src="https://avatars.githubusercontent.com/u/84071067?v=4?s=100" width="100px;" alt="Davide Sciotti"/><br /><sub><b>Davide Sciotti</b></sub></a><br /><a href="#bug-davidesciotti" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GabrieleParimbelli"><img src="https://avatars.githubusercontent.com/u/43963112?v=4?s=100" width="100px;" alt="GabrieleParimbelli"/><br /><sub><b>GabrieleParimbelli</b></sub></a><br /><a href="#bug-GabrieleParimbelli" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zahrabaghkhani"><img src="https://avatars.githubusercontent.com/u/47903409?v=4?s=100" width="100px;" alt="Zahra Baghkhani"/><br /><sub><b>Zahra Baghkhani</b></sub></a><br /><a href="#code-zahrabaghkhani" title="Code">💻</a> <a href="#ideas-zahrabaghkhani" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-zahrabaghkhani" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fabriceroy"><img src="https://avatars.githubusercontent.com/u/29073232?v=4?s=100" width="100px;" alt="Fabrice Roy"/><br /><sub><b>Fabrice Roy</b></sub></a><br /><a href="#doc-fabriceroy" title="Documentation">📖</a> <a href="#code-fabriceroy" title="Code">💻</a> <a href="#ideas-fabriceroy" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pltaylor16"><img src="https://avatars.githubusercontent.com/u/22646673?v=4?s=100" width="100px;" alt="pltaylor16"/><br /><sub><b>pltaylor16</b></sub></a><br /><a href="#code-pltaylor16" title="Code">💻</a> <a href="#doc-pltaylor16" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rreischke"><img src="https://avatars.githubusercontent.com/u/31727230?v=4?s=100" width="100px;" alt="Robert Reischke"/><br /><sub><b>Robert Reischke</b></sub></a><br /><a href="#ideas-rreischke" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-rreischke" title="Mentoring">🧑‍🏫</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AngusWright"><img src="https://avatars.githubusercontent.com/u/5625880?v=4?s=100" width="100px;" alt="Angus H. Wright"/><br /><sub><b>Angus H. Wright</b></sub></a><br /><a href="#ideas-AngusWright" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-AngusWright" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/matteobaratto"><img src="https://avatars.githubusercontent.com/u/75221958?v=4?s=100" width="100px;" alt="Matteo Baratto "/><br /><sub><b>Matteo Baratto </b></sub></a><br /><a href="#code-matteobaratto" title="Code">💻</a> <a href="#ideas-matteobaratto" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MariaTsedrik"><img src="https://avatars.githubusercontent.com/u/93711395?v=4?s=100" width="100px;" alt="Maria Tsedrik"/><br /><sub><b>Maria Tsedrik</b></sub></a><br /><a href="#code-MariaTsedrik" title="Code">💻</a> <a href="#ideas-MariaTsedrik" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arsouki"><img src="https://avatars.githubusercontent.com/u/162714090?v=4?s=100" width="100px;" alt="Arghavan Souki"/><br /><sub><b>Arghavan Souki</b></sub></a><br /><a href="#bug-arsouki" title="Bug reports">🐛</a> <a href="#doc-arsouki" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ryusei172525"><img src="https://avatars.githubusercontent.com/u/48290932?v=4?s=100" width="100px;" alt="Ryusei Kano"/><br /><sub><b>Ryusei Kano</b></sub></a><br /><a href="#code-ryusei172525" title="Code">💻</a> <a href="#ideas-ryusei172525" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.ph.ed.ac.uk/people/linus-thummel"><img src="https://avatars.githubusercontent.com/u/142009018?v=4?s=100" width="100px;" alt="Linus Thummel"/><br /><sub><b>Linus Thummel</b></sub></a><br /><a href="#code-cosmicLinux" title="Code">💻</a> <a href="#doc-cosmicLinux" title="Documentation">📖</a> <a href="#ideas-cosmicLinux" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-cosmicLinux" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://lisagoh.github.io"><img src="https://avatars.githubusercontent.com/u/74064921?v=4?s=100" width="100px;" alt="Lisa Goh"/><br /><sub><b>Lisa Goh</b></sub></a><br /><a href="#code-LisaGoh" title="Code">💻</a> <a href="#doc-LisaGoh" title="Documentation">📖</a> <a href="#ideas-LisaGoh" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zsirap"><img src="https://avatars.githubusercontent.com/u/50758399?v=4?s=100" width="100px;" alt="zsirap"/><br /><sub><b>zsirap</b></sub></a><br /><a href="#code-zsirap" title="Code">💻</a> <a href="#ideas-zsirap" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-zsirap" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ktanidis"><img src="https://avatars.githubusercontent.com/u/60473500?v=4?s=100" width="100px;" alt="Konstantinos Tanidis"/><br /><sub><b>Konstantinos Tanidis</b></sub></a><br /><a href="#code-ktanidis" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://chaitanyachawak.github.io/"><img src="https://avatars.githubusercontent.com/u/55046588?v=4?s=100" width="100px;" alt="Chaitanya"/><br /><sub><b>Chaitanya</b></sub></a><br /><a href="#code-ChaitanyaChawak" title="Code">💻</a> <a href="#bug-ChaitanyaChawak" title="Bug reports">🐛</a> <a href="#review-ChaitanyaChawak" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/raphkou"><img src="https://avatars.githubusercontent.com/u/61792335?v=4?s=100" width="100px;" alt="raphkou"/><br /><sub><b>raphkou</b></sub></a><br /><a href="#bug-raphkou" title="Bug reports">🐛</a> <a href="#code-raphkou" title="Code">💻</a> <a href="#ideas-raphkou" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-raphkou" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ivansladoljev"><img src="https://avatars.githubusercontent.com/u/144113061?v=4?s=100" width="100px;" alt="Ivan Sladoljev"/><br /><sub><b>Ivan Sladoljev</b></sub></a><br /><a href="#code-ivansladoljev" title="Code">💻</a> <a href="#ideas-ivansladoljev" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NastassiaG"><img src="https://avatars.githubusercontent.com/u/107264848?v=4?s=100" width="100px;" alt="NastassiaG"/><br /><sub><b>NastassiaG</b></sub></a><br /><a href="#code-NastassiaG" title="Code">💻</a> <a href="#bug-NastassiaG" title="Bug reports">🐛</a> <a href="#ideas-NastassiaG" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/martincrocce"><img src="https://avatars.githubusercontent.com/u/29067049?v=4?s=100" width="100px;" alt="Martin Crocce"/><br /><sub><b>Martin Crocce</b></sub></a><br /><a href="#projectManagement-martincrocce" title="Project Management">📆</a> <a href="#mentoring-martincrocce" title="Mentoring">🧑‍🏫</a> <a href="#promotion-martincrocce" title="Promotion">📣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CarmelitaCarbone"><img src="https://avatars.githubusercontent.com/u/17458225?v=4?s=100" width="100px;" alt="CarmelitaCarbone"/><br /><sub><b>CarmelitaCarbone</b></sub></a><br /><a href="#projectManagement-CarmelitaCarbone" title="Project Management">📆</a> <a href="#mentoring-CarmelitaCarbone" title="Mentoring">🧑‍🏫</a> <a href="#promotion-CarmelitaCarbone" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://garico92.github.io/website/"><img src="https://avatars.githubusercontent.com/u/78367368?v=4?s=100" width="100px;" alt="Giovanni"/><br /><sub><b>Giovanni</b></sub></a><br /><a href="#code-garico92" title="Code">💻</a> <a href="#ideas-garico92" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-garico92" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JegerBroxterman"><img src="https://avatars.githubusercontent.com/u/115992884?v=4?s=100" width="100px;" alt="Jeger Broxterman"/><br /><sub><b>Jeger Broxterman</b></sub></a><br /><a href="#code-JegerBroxterman" title="Code">💻</a> <a href="#ideas-JegerBroxterman" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.matthieuschaller.com"><img src="https://avatars.githubusercontent.com/u/42518815?v=4?s=100" width="100px;" alt="Matthieu Schaller"/><br /><sub><b>Matthieu Schaller</b></sub></a><br /><a href="#ideas-MatthieuSchaller" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joezuntz"><img src="https://avatars.githubusercontent.com/u/220537?v=4?s=100" width="100px;" alt="joezuntz"/><br /><sub><b>joezuntz</b></sub></a><br /><a href="#mentoring-joezuntz" title="Mentoring">🧑‍🏫</a> <a href="#ideas-joezuntz" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KlaraBertmann"><img src="https://avatars.githubusercontent.com/u/153739278?v=4?s=100" width="100px;" alt="KlaraBertmann"/><br /><sub><b>KlaraBertmann</b></sub></a><br /><a href="#code-KlaraBertmann" title="Code">💻</a> <a href="#ideas-KlaraBertmann" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/annmapo"><img src="https://avatars.githubusercontent.com/u/18014895?v=4?s=100" width="100px;" alt="Anna Porredon"/><br /><sub><b>Anna Porredon</b></sub></a><br /><a href="#ideas-annmapo" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-annmapo" title="Mentoring">🧑‍🏫</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/emiliobellini"><img src="https://avatars.githubusercontent.com/u/6113149?v=4?s=100" width="100px;" alt="emiliobellini"/><br /><sub><b>emiliobellini</b></sub></a><br /><a href="#maintenance-emiliobellini" title="Maintenance">🚧</a> <a href="#ideas-emiliobellini" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://miguelzuma.github.io/"><img src="https://avatars.githubusercontent.com/u/5014027?v=4?s=100" width="100px;" alt="Miguel Zumalacarregui"/><br /><sub><b>Miguel Zumalacarregui</b></sub></a><br /><a href="#maintenance-miguelzuma" title="Maintenance">🚧</a> <a href="#ideas-miguelzuma" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/inigosaezcasares"><img src="https://avatars.githubusercontent.com/u/56702840?v=4?s=100" width="100px;" alt="Iñigo Sáez Casares"/><br /><sub><b>Iñigo Sáez Casares</b></sub></a><br /><a href="#code-inigosaezcasares" title="Code">💻</a> <a href="#ideas-inigosaezcasares" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/neelcosmo"><img src="https://avatars.githubusercontent.com/u/151792852?v=4?s=100" width="100px;" alt="Neel Shah"/><br /><sub><b>Neel Shah</b></sub></a><br /><a href="#code-neelcosmo" title="Code">💻</a> <a href="#ideas-neelcosmo" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FelicitasKeil"><img src="https://avatars.githubusercontent.com/u/70713596?v=4?s=100" width="100px;" alt="Felicitas Keil"/><br /><sub><b>Felicitas Keil</b></sub></a><br /><a href="#code-FelicitasKeil" title="Code">💻</a> <a href="#ideas-FelicitasKeil" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jipdebuck"><img src="https://avatars.githubusercontent.com/u/236796982?v=4?s=100" width="100px;" alt="Jip de Buck"/><br /><sub><b>Jip de Buck</b></sub></a><br /><a href="#userTesting-jipdebuck" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/HironaoMiyatake"><img src="https://avatars.githubusercontent.com/u/1507529?v=4?s=100" width="100px;" alt="Hironao Miyatake"/><br /><sub><b>Hironao Miyatake</b></sub></a><br /><a href="#ideas-HironaoMiyatake" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- --8<-- [end:contributors] -->
