# Summary Statistics: Final Data Products

The **Summary Statistics** module produces final data products for likelihood analysis.

This module completes the cosmological pipeline by converting tracers and power spectra into statistical quantities for comparison with observations. Currently, `cloelib` supports observables for large-scale structure multi-probe experiments. Observable outputs are formatted — or can be requested to be formatted — following the [cosmolib](https://github.com/astro-ph/cosmolib) data format, which is used by downstream packages such as the reading and writing package [euclidlib](https://github.com/euclidlib/euclidlib). To understand the format, we kindly request the user to check [`euclidlib` docs](https://euclidlib.readthedocs.io/en/latest/format.html).

To see examples of usage, please check the example notebooks for observable computation at [`playground`](https://github.com/cloe-org/playground/tree/main/tutorials/observables).

## Overview

**Location**: `cloelib/summary_statistics/`

This module computes final statistical quantities for likelihood evaluation, including:

- $C_\ell$: Angular power spectra for photometric surveys, either full sky or convolved with the mask
- $\xi_+(\theta)$, $\xi_-(\theta)$, $w(\theta)$, $\gamma_T$, $\gamma_\times$: Angular two-point photometric correlation functions
- COSEBIs: Complete Orthogonal Sets of E/B-Integrals for photometric surveys as in [Asgari et al., 2018.](https://arxiv.org/pdf/1201.2669)
- $P_\ell(k)$: Legendre multipoles for spectroscopic surveys, either full sky or convolved with the mask
- $\xi(r)$: Two-point correlation function from Legendre multipoles, also supported as polar two-point correlation function
- $\alpha_\parallel$, $\alpha_\perp$: BAO distortion parameters for spectroscopic surveys

These quantities are directly measurable and form the basis for cosmological parameter inference.

!!! warning cloelib does not use internal interpolations. Keep redshift and wavenumber arrays to a maximum of 1500 elements for optimal performance. Otherwise, memory problems may arise.

## Performance Tips

`cloelib` does not use internal interpolations. Keep redshift and wavenumber arrays to a maximum of 1500 elements for optimal performance. Otherwise, memory problems may arise.

## Performance Tips

For expensive summary-statistic evaluations, prefer moderate redshift and wavenumber grids, especially when scanning parameter space repeatedly. In practice, keeping these arrays at or below roughly 1500 elements avoids unnecessary memory pressure in the current implementation.

## Available Summary Statistics

### For Photometric Surveys (Using Tracers)

#### AngularTwoPoint

Compute angular power spectra $C_\ell$, pseudo-$C_\ell$ and COSEBIs from two tracers.

**Location**: `cloelib/summary_statistics/angular_two_point.py`

**What it does**:

- Takes two `Tracer` objects
- Integrates over redshift using the Limber approximation with advanced vectorisation across tomographic bins

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
import numpy as np

# Set up cosmology
bg = CAMBBackground(H0=67.5, Omega_b0=0.0492, ...)
pert = CAMBPerturbations(background=bg)

# Create tracers
z = np.linspace(0.01, 3.0, 100)
dndz = np.exp(-((z - 0.7) / 0.3)**2)
dndz = dndz / np.trapz(dndz, z)

tracer1 = ShearTracer(perturbations=pert, dndz=dndz[np.newaxis, :], z=z, nuisance_params={...})
tracer2 = ShearTracer(perturbations=pert, dndz=dndz[np.newaxis, :], z=z, nuisance_params={...})

# Compute angular power spectrum
two_point = AngularTwoPoint(tracer1=tracer1, tracer2=tracer2)

ells = np.logspace(1, 3, 20)  # ℓ from 10 to 1000
C_ell = two_point.get_Cl(ells=ells)  # Shape: (1, 1, 20) for single bins

print(f"C_ℓ at ℓ=100: {C_ell[0, 0, 10]:.2e}")
```

**Cross-Correlations**:

You can correlate different tracers to calculate different statistics:

```python
# Shear-shear (cosmic shear)
shear_tracer = ShearTracer(...)
C_shear_shear = AngularTwoPoint(shear_tracer, shear_tracer).get_Cl(ells)

# Position-position (galaxy clustering)
pos_tracer = PositionsTracer(...)
C_gg = AngularTwoPoint(pos_tracer, pos_tracer).get_Cl(ells)

# Shear-position (galaxy-galaxy lensing)
C_g_shear = AngularTwoPoint(pos_tracer, shear_tracer).get_Cl(ells)
```

**Tomographic Bins**:

```python
# Multiple redshift bins
dndz_bins = np.array([
    np.exp(-((z - 0.5) / 0.2)**2),
    np.exp(-((z - 1.0) / 0.3)**2),
    np.exp(-((z - 1.5) / 0.4)**2),
])

# Normalise! cloelib will always expect the n(z) normalised
dndz_bins = dndz_bins / np.trapz(dndz_bins, z, axis=1)[:, np.newaxis]

tracer = ShearTracer(perturbations=pert, dndz=dndz_bins, z=z, ...)

# Auto and cross-correlations
two_point = AngularTwoPoint(tracer, tracer)
C_ell = two_point.get_Cl(ells)
```

**COSEBIs**:
Complete Orthogonal Sets of E/B-Integrals are specialized statistics for cosmic shear.
They are available both as methods of `AngularTwoPoint` (which injects a software provenance tag automatically) and as **standalone module-level functions** that can be used without instantiating the class, useful when $C_\ell$ or $\xi(\pm)$ are already computed, even from real or simulated measurements.

**Standalone functions** (no tracers required):

```python
from cloelib.summary_statistics.angular_two_point import (
    get_cosebis_from_cl,
    get_cosebis_from_2pcf,
)

# From angular power spectra
cosebis = get_cosebis_from_cl(cells, ells, w_ell, ns)

# From two-point correlation functions
cosebis = get_cosebis_from_2pcf(twopcf, theta, T_plus, T_minus, ns)
```

**Class method** (`get_cosebis` — computes $C_\ell$ internally, software provenance tag included automatically):

```python
two_point = AngularTwoPoint(tracer1, tracer2)
cosebis = two_point.get_cosebis(ells, nl, ks, w_ell, ns)
```

!!! note
Only `get_cosebis` is available as a class method. `get_cosebis_from_2pcf` is standalone only — use it directly when $\xi_\pm(\theta)$ are already available.

Both interfaces require optional dependencies (`pylevin`, `mpmath`). The COSEBIs kernels (`w_ell`, `T_plus`, `T_minus`) must be precomputed using helpers from `cloelib.auxiliary.cosebi_helpers`.

#### AngularCorrelationFunction

Protocol to set up how to compute real-space angular correlation functions $\xi(\theta)$.

**Location**: `cloelib/summary_statistics/angular_correlation_function.py`

#### AngularCorrelationFunctionWigner

Alternative implementation using Wigner 3j symbols for computing real-space angular correlation functions $\xi(\theta)$ from harmonic angular power spectra.

**Location**: `cloelib/summary_statistics/angular_correlation_function_wigner.py`

### For Spectroscopic Surveys (Using SpectroPower)

#### LegendreMultipoles

Compute multipoles $P_\ell(k)$ from 2D power spectrum $P(k, \mu)$.

**Location**: `cloelib/summary_statistics/legendre_multipoles.py`

**What it does**:

- Takes a `SpectroPower` object
- Integrates $P(k, \mu)$ over $\mu$ with Legendre polynomials
- Outputs $P_0(k)$, $P_2(k)$, $P_4(k)$, ...
- Allows to further compute two-point correlation functions, in cartesian and polar coordinates.

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_spectro
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles

bg = CAMBBackground(H0=67.5, ...)
spectro = CometEFT_spectro(background=bg, z_pk=1.0)

# Compute multipoles
leg_multi = LegendreMultipoles(spectro_power=spectro)

k = np.logspace(-2, 0, 50)  # k in h/Mpc
multipoles = leg_multi.power_multipoles(
    k=k,
    ells=[0, 2, 4],  # Which multipoles
    z=1.0,
    bias=2.0,
)

P0 = multipoles[0]  # Monopole
P2 = multipoles[2]  # Quadrupole
P4 = multipoles[4]  # Hexadecapole

print(f"Monopole at k=0.1: {P0[20]:.2e}")
```

#### BAOAlphas

Compute BAO distortion parameters $\alpha_\parallel$ and $\alpha_\perp$.

**Location**: `cloelib/summary_statistics/bao_alphas.py`

**What it does**: Compute alpha parameters for the BAO analysis

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.summary_statistics.bao_alphas import BaryonAcousticOscillations
import numpy as np

# True and fiducial cosmologies
bg = CAMBBackground(H0=67.5, ...)
bg_fid = CAMBBackground(H0=67.5, ...)  # Fiducial cosmology

zs = np.array([0.5, 1.0, 1.5])

bao = BaryonAcousticOscillations(
    background=bg,
    background_fiducial=bg_fid,
    redshifts=zs,
)

# Access alphas at each redshift
alphas_z05 = bao.alphas_dict[0.5]

alpha_parallel = alphas_z05['alpha_par']      # Along LOS
alpha_perpendicular = alphas_z05['alpha_perp'] # Transverse
alpha_iso = alphas_z05['alpha_iso']            # Isotropic dilation
alpha_AP = alphas_z05['alpha_AP']              # Alcock-Paczynski ratio
```

## Next Steps

The pipeline is now complete.
From here:

- [API Reference](../api.md) – Full technical documentation
- [Contributing Guide](../contributing.md) – General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground) – Real usage examples
- [Back to Overview](index.md) – Review the architecture

Or return to any component:

- [Background](background.md)
- [Perturbations](perturbations.md)
- [Observables](observables.md)
