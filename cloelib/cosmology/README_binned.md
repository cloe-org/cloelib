# Modified Gravity Euclid Module (CLOE Integration)

This module implements modified gravity (MG) perturbations for Euclid-like analyses using a binned MG emulator. It is designed to be fully compatible with the `cloelib` cosmology and likelihood framework.

The core component is the `MGPerturbations` class, which wraps an emulator-based boost model and applies it to standard ΛCDM linear and nonlinear matter power spectra.



---


## Installation


You need to clone the parent repo that contains the emulator files and down the models from Zenodo. 


```python
https://github.com/sankarshana16/mg_binned_boost_emulator.git
```

You can find the models in 

```
[Zenodo Dataset](https://zenodo.org/records/19625918)
```
After downloading, place all files in:

```bash
mg_binned_boost_emulator/models/
```

---

## 🚀 Features

- Emulator-based MG boost applied to:
  - Linear matter power spectrum
  - Nonlinear matter power spectrum
- Fully compatible with `cloelib` `Perturbations` interface
- Lazy loading of emulator (loaded once per session)
- Supports redshift-binned MG modifications
- Provides:
  - \( P(k, z) \)
  - Growth factor \( D(z, k) \)
  - Growth rate \( f(z, k) \)
  - \( \sigma_8 \)
  - Lensing modification \( \Sigma(z) \)

---

## 📦 Class Overview

### `MGPerturbations`

Main wrapper class implementing MG perturbations.

---

### Initialization

    MGPerturbations(
        background,
        linearperturbations,
        nonlinearperturbations,
        redshifts,
        mu,
        eta,
        bin_index,
        model_dir="./models"
    )

#### Inputs

| Parameter | Description |
|----------|-------------|
| `background` | `cloelib` background cosmology |
| `linearperturbations` | Linear ΛCDM perturbations |
| `nonlinearperturbations` | Nonlinear ΛCDM (e.g. HMCode) |
| `redshifts` | Array of redshifts |
| `mu`, `eta` | Modified gravity parameters |
| `bin_index` | Active MG redshift bin |
| `model_dir` | Path to emulator models |

---

## ⚙️ How It Works

1. Convert `cloelib` cosmology → emulator format  
2. Load MG emulator (cached globally)  
3. Predict:
   - Nonlinear boost
   - Linear boost  
4. Apply boosts to ΛCDM spectra:  
   \( P_{\rm MG} = \text{boost} \times P_{\Lambda\rm CDM} \)  
5. Build spline interpolators for fast evaluation  

---

## 📊 Available Methods

### Matter Power Spectrum

    matter_power_spectrum(zs, ks)

Returns:

\( P(k, z) \)

---

### Growth Factor

    growth_factor(zs, ks)

\[
D(z, k) = \sqrt{\frac{P_{\rm lin}(z, k)}{P_{\rm lin}(0, k)}}
\]

---

### Growth Rate

    growth_rate(zs, ks)

\[
f(z, k) = \frac{d \ln D}{d \ln a}
\]

Computed numerically via:

\[
f = -(1+z)\frac{d\ln D}{dz}
\]

---

### Sigma8

    sigma8_0()

\[
\sigma_8^2 = \frac{1}{2\pi^2} \int dk\, k^2 P_{\rm lin}(k, z=0) W^2(kR)
\]

with:

- \( R = 8\, \mathrm{Mpc}/h \)
- Top-hat window function

---

### Lensing Modification

    Sigma(zs)

\[
\Sigma = \frac{\mu(1 + \eta)}{2}
\]

Applied only within the active redshift bin.

Outside the bin:

\[
\Sigma = 1
\]

---

## 📦 Redshift Binning

| bin_index | Redshift range |
|----------|----------------|
| 0 | 0.00 – 0.43 |
| 1 | 0.43 – 0.91 |
| 2 | 0.91 – 1.47 |
| 3 | 1.47 – 2.15 |
| 4 | 2.15 – 3.00 |

---

## ⚡ Performance

- Emulator is loaded **once per Python session**
- Subsequent likelihood calls reuse cached model
- Designed for fast MCMC / nested sampling

---

## ⚠️ Assumptions & Limitations

- Flat cosmology only (`Omega_k = 0`)
- Emulator validity range must be respected
- Not implemented:
  - `matter_power_spectrum_cb`

---

## 🧪 Example Usage

    mg = MGPerturbations(
        background=background,
        linearperturbations=lp,
        nonlinearperturbations=nlp,
        redshifts=z,
        mu=1.0,
        eta=1.0,
        bin_index=1,
    )

    Pk = mg.matter_power_spectrum(z, k)

---

## 📁 Dependencies

- numpy
- scipy
- cloelib
- MG emulator (`MGEmulator`)

---

## 🧑‍💻 Author

Sankarshana Srinivasan

---

## 📌 TODO

- Add baryon/CDM split (`P_cb`)
- Validate \( \sigma_8 \) normalization (h-consistency)
