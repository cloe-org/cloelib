# cloelib – The Library for the Cosmology Likelihood for Observables in Euclid  

🚀 **cloelib** is a flexible and efficient library designed to compute cosmological observables for the **CLOE** (*Cosmology Likelihood for Observables in Euclid*) project. It is built for seamless integration with **Boltzmann solvers** and **JAX-based frameworks**, enabling automatic differentiation and modularity for the next generation of cosmological analyses.  

🛠️ **Work in Progress** – We welcome feedback from the **Euclid community** and beyond to refine and improve this library!  

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

---

## 🚀 Installation  

To install `cloelib` source code, clone the repository and install it via `pip`:  
```sh
pip install .
```

You can also install (some) supported dependencies:

```sh
pip install .[camb,hmcode2020emu,comet-emu]
```

We do not provide installation support for `PBJ` and `CLASS`. For instructions on installing these packages, please refer to the [`playground`](https://github.com/cloe-org/playground) repository.

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

This project is licensed under the **GNU LESSER GENERAL PUBLIC LICENSE** – see the [LICENSE](LICENSE) file for details.  

---

## 🙏 Acknowledgements  

🔭 Inspired by the pioneering work of the **Euclid Consortium** and the **`jaxcosmo`** project. 

🎯 With technical advice from S. Farrens and N. Tessore.
