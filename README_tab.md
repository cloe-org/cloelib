# TabulatedBoost_cosmology: Beyond-LCDM Nonlinear Boost Module from tabulated data

`TabulatedBoost_cosmology` is a CLOE module for interpolating tabulated **nonlinear matter power spectrum boost** data which gives the **Beyond-LCDM** correction to the nonlinear power spectrum. It takes as input a 2d array .txt or .dat file with columns referencing different z snapshots and rows k. This data can be from simulations for example.

The data file should have the following column format: [k, B(k,z1), B(k,z2) ...] where B(k,z) is the nonlinear boost and k is the wavemode in \[h/Mpc\].

---

## 🚀 Features

- Interpolates in redshift- and scale- a tabulated nonlinear boost factor \( B(k, z) \)
- Adopts CLOE k extrapolation scheme
- Allows for multiple redshift extrapolation methods (see module for details)

---

## 🧠 Core Classes

- **`TabulatedNonlinearBoost`**
  Interpolates the nonlinear boost from the tabulated data file given:
  - A background cosmology (to convert k in \[h/Mpc\] to \[1/Mpc\]).
  - Linear perturbations for the k-grid
  - Output redshift array
  - Path to the tabulated data file
  - Redshift array of the tablulated data
  - Redshift extrapolation policy (various are allowed - see module file)

- **`BoostedPerturbations`**
  Applies the boost factor $B(k, z)$ to any base nonlinear spectrum $P_{\Lambda\text{CDM}}(k,z)$ to compute:

$P_{\text{MG}}(k,z) = B(k, z) \cdot P_{\Lambda\text{CDM}}(k,z)$

## Also calculated linear growth factor based on this boost.

## 📦 Requirements

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- `CAMB` (via `cloelib`) for background and linear spectra

To install the necessary packages:

```bash
pip install ."
```

### 📊 Boost Validation Notebook

An example notebook `TabulatedBoost.ipynb` is included in playground/cosmology. You can download the required DAKAR2 tabulated boost data there.

- 📥 Download and load external benchmark boost data
- 📈 Plot power spectra and test extrapolation schemes
- 🔍 Compare boost predictions against halo model reaction emulator
