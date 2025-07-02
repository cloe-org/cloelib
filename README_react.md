# ReACTEmu: Modified Gravity Nonlinear Boost Module

`ReACTEmu` is a Python module for computing the **nonlinear matter power spectrum boost** in **modified gravity (MG)** and **dark energy** models. It leverages pre-trained emulators based on **ReACT** and the **Halo Model Reaction framework** ([arXiv:1812.05594](https://arxiv.org/abs/1812.05594)).

These emulators were trained on data from ReACT for a variety of MG models including:

- **f(R)** gravity
- **DGP** (Dvali–Gabadadze–Porrati) braneworld model
- General parameterizations such as:
  - Linder's γ
  - Interacting dark energy (IDE)
  - Horndeski-like \( q_1 \) models

---

## 🚀 Features

- Computes redshift- and scale-dependent nonlinear boost factors \( B(k, z) \)
- Fast spline-based interpolation over the emulator grid
- Currently has a constant extrapolation in \( k \) 

---

## 🧠 Core Classes

- **`MGemuNonlinearBoost`**  
  Computes and interpolates the nonlinear boost from the emulator given:
  - A background cosmology
  - Linear perturbation power spectrum
  - Redshift array
  - MG model and parameters

- **`BoostedPerturbations`**  
  Applies the boost factor \( B(k, z) \) to any base nonlinear spectrum \( P_{\Lambda\text{CDM}}(k,z) \) to compute:
  \[
  P_{\text{MG}}(k,z) = B(k, z) \cdot P_{\Lambda\text{CDM}}(k,z)
  \]

---

## 📦 Requirements

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `CAMB` (via `cloelib`) for background and linear spectra
- [`MGEmu`](https://github.com/nebblu/MGEmus) installed and accessible
- `tensorflow` and `tensorflow-probability` with `tf_keras` if required by emulator backend
- `pytest` for running tests

To install the necessary packages:

```bash
pip install numpy scipy matplotlib camb pytest
pip install "tensorflow-probability[tf]"


### 🧪 Testing and Validation

A `pytest` test suite is included in `tests/test_ReACTEmu.py`. It performs the following:

- ✅ Verifies correct initialization and interface behavior  
- 📊 Loads external validation files for **f(R)** and **DGP** models  
- ❌ Fails if the predicted nonlinear boost differs by more than **0.5%** in the range  
  \( k \in [0.01,\ 3] \ \text{Mpc}^{-1} \)

#### 🔧 Run the test suite

To execute the tests from the repository root:

```bash
pytest tests/test_ReACTEmu.py



### 📊 Boost Validation Notebook

A notebook `ReACTEmu_ValidationPlot.ipynb` is included to:

- 📥 Load external benchmark boost data  
- 🔍 Compare emulator predictions against the benchmarks  
- 📈 Plot absolute and relative errors across redshifts and scales  

Use this notebook to visually inspect the accuracy of the ReACTEmu implementation.

---

### 📚 References

- Halo model reaction, [arXiv:1812.05594](https://arxiv.org/abs/1812.05594)  
- ReACT, [arXiv:2005.12184](https://arxiv.org/abs/2005.12184)  
- ReACT Emulator Repository: [https://github.com/nebblu/MGEmus](https://github.com/nebblu/MGEmus)
