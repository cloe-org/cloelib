# cloelib – The Library for the Cosmology Likelihood for Observables in Euclid  

🚀 **cloelib** is a flexible and efficient library designed to compute cosmological observables for the **CLOE** (*Cosmology Likelihood for Observables in Euclid*) project. It is built for seamless integration with **Boltzmann solvers** and **JAX-based frameworks**, enabling automatic differentiation and modularity for the next generation of cosmological analyses for surveys like Euclid.  

🛠️ **Work in Progress** – We welcome feedback from the **Euclid community** and beyond to refine and improve this library!  

---

## 📖 Table of Contents  
- [✨ Features](#-features)  
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

## 🚀 Installation  

Clone the repository and install it via `pip`:  
```sh
pip install .
```

### ✅ Prerequisites  
Ensure you have the following dependencies installed:  
- `python`  
- `jax`, `jaxlib`  
- `interpax`  

---

## 📊 Usage  

Explore the **tutorials** in the `cloe-org/playground` repository for examples on how to compute cosmological observables and other key quantities!  

---

## 🤝 Contributing  

We encourage contributions! Please review the organization's general contribution guidelines along with the specific guidelines for this repository. Then, follow these steps:

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

This project is licensed under the **GLG License** – see the [LICENSE](LICENSE) file for details.  

---

## 🙏 Acknowledgements  

🔭 Inspired by the pioneering work of the **Euclid Consortium** and the **`jaxcosmo`** project.  

🎯 **Join us in shaping the future of cosmological inference!** 🚀  
