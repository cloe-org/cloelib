# Photometric Surveys Summary Statistics (Using Tracers)

## AngularTwoPoint

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

## AngularCorrelationFunction

Protocol to set up how to compute real-space angular correlation functions $\xi(\theta)$.

**Location**: `cloelib/summary_statistics/angular_correlation_function.py`

## AngularCorrelationFunctionWigner

Alternative implementation using Wigner 3j symbols for computing real-space angular correlation functions $\xi(\theta)$ from harmonic angular power spectra.

**Location**: `cloelib/summary_statistics/angular_correlation_function_wigner.py`

## Next Steps

The pipeline is now complete.
From here:

- [API Reference](../../api.md) – Full technical documentation
- [Contributing Guide](../../contributing.md) – General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground) – Real usage examples
- [Back to Overview](../index.md) – Review the architecture

Or return to any component:

- [Background](../background.md)
- [Perturbations](../perturbations.md)
- [Observables](../observables.md)
