# Tracer Protocol (Photometric Observables)

**Protocol Definition**: `cloelib.observables.photo.tracer.Tracer`

Tracers define window functions for photometric surveys—how galaxies are distributed in redshift and how they trace the matter field.

## Required Property

- **`perturbations`**: Reference to a Perturbations object

Tracers need perturbations to compute power spectra and growth.

## Required Methods

### `get_window(z)`

Compute the window function W(z) at given redshifts.

**Returns**: Window function values, shape depends on number of redshift bins

### `_window_integrand(z, zprime)`

Window integrand for Limber integration.

Used internally by summary statistics calculators.

### `_get_prefactor(ell)`

Compute prefactor for Limber approximation.

Handles different tracer types (shear has extra factors, galaxy clustering doesn't).

## Existing Tracer Implementations

### ShearTracer

For weak gravitational lensing (cosmic shear) measurements.

**Location**: `cloelib/observables/photo/shear.py`

**What it does**:

- Computes lensing window function W^κ(z)
- Includes lensing efficiency
- Handles intrinsic alignments
- Applies nuisance parameters (multiplicative bias, photo-z errors)

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBPerturbations
from cloelib.observables.photo import ShearTracer
import numpy as np

# Set up cosmology
bg = CAMBBackground(H0=67.5, Omega_b0=0.0492, ...)
pert = CAMBPerturbations(background=bg)

# Define redshift distribution (normalized.)
z = np.linspace(0.01, 3.0, 100)
dndz = np.exp(-((z - 0.7) / 0.3)**2)  # Gaussian n(z)
dndz = dndz / np.trapz(dndz, z)  # Normalize

# Multiple tomographic bins
dndz_bins = np.array([
    np.exp(-((z - 0.5) / 0.2)**2),
    np.exp(-((z - 1.0) / 0.3)**2),
])
dndz_bins = dndz_bins / np.trapz(dndz_bins, z, axis=1)[:, np.newaxis]

# Nuisance parameters
nuisance = {
    'multiplicative_bias_1': 0.0,
    'multiplicative_bias_2': 0.0,
    'dz_shear_1': 0.0,  # Photo-z bias
    'dz_shear_2': 0.0,
    'AIA': 1.0,         # Intrinsic alignment amplitude
    'CIA': 0.0164,      # IA normalization
    'EtaIA': -0.41,     # IA redshift evolution
}

# Create tracer
tracer = ShearTracer(
    perturbations=pert,
    dndz=dndz_bins,
    z=z,
    nuisance_params=nuisance,
)

# Get window function
window = tracer.get_window(z)  # Shape: (n_bins, len(z))
```

**Special Features**:

- Lensing efficiency calculation
- Intrinsic alignment modeling (NLA model)
- Photo-z error handling
- Multiplicative shear bias

**Intrinsic alignment models**: `ShearTracer` accepts an `ia_model` keyword
(default `"NLA"`, reproducing the behavior above exactly). `ia_model="TATT"`
switches to the Tidal Alignment + Tidal Torquing model (Blazek et al. 2019;
Navarro-Gironés et al. 2026, arXiv:2602.16448) instead, reading
`nuisance_params["AIA"/"A2IA"/"bTA"]` (and optionally `"EtaIA"/"Eta2IA"/
"z0IA"`) - everything downstream (`AngularTwoPoint.get_Cl`, `get_pseudo_Cl`,
`get_cosebis`) is unchanged either way:

```python
from cloelib.observables.photo import ShearTracer

tracer = ShearTracer(
    perturbations=pert,
    dndz=dndz_bins,
    z=z,
    nuisance_params={**nuisance, 'A2IA': 0.4, 'bTA': -0.83},
    ia_model="TATT",
)
```

TATT's ten one-loop perturbation-theory kernels come from a
`tatt_loop_computer` (default: `PlaceholderTATTLoopComputer`, illustrative
kernels with no extra dependency). For real kernels, pass
`PBJTATTLoopComputer` (computed via the `fast-pt` package's
`FASTPT.IA_ta`/`.IA_tt`/`.IA_mix`, an optional dependency -
`pip install cloelib[fastpt]`), constructed from the _same_ `perturbations`
object already passed above:

```python
from cloelib.observables.photo.shear import PBJTATTLoopComputer

tracer = ShearTracer(
    perturbations=pert,
    dndz=dndz_bins,
    z=z,
    nuisance_params={**nuisance, 'A2IA': 0.4, 'bTA': -0.83},
    ia_model="TATT",
    tatt_loop_computer=PBJTATTLoopComputer(pert),
)
```

`TATTContribution`, `PlaceholderTATTLoopComputer`, and `PBJTATTLoopComputer`
all live in `cloelib.observables.photo.shear`, alongside `ShearTracer`
itself.

### PositionsTracer

For galaxy clustering (galaxy positions) measurements.

**Location**: `cloelib/observables/photo/positions.py`

**What it does**:

- Computes galaxy clustering window function W^g(z)
- Handles galaxy bias
- Applies magnification bias
- Handles photo-z uncertainties

**Example**:

```python
from cloelib.observables.photo import PositionsTracer

# Nuisance for clustering
nuisance = {
    'bias_1': 1.5,      # Galaxy bias
    'bias_2': 1.8,
    'dz_clustering_1': 0.0,
    'dz_clustering_2': 0.0,
    'magnification_bias_1': 0.0,
    'magnification_bias_2': 0.0,
}

tracer = PositionsTracer(
    perturbations=pert,
    dndz=dndz_bins,
    z=z,
    nuisance_params=nuisance,
)

window = tracer.get_window(z)
```

### CMBLensingTracer

For CMB weak gravitational lensing (convergence) measurements.

**Location**: `cloelib/observables/cmb.py`

**What it does**:

- Computes lensing window function W^κ(z)

**Example**:

```python
from cloelib.observables.cmb import CMBLensingTracer

z = np.arange(1e-3,pert.background.z_star,0.01)
tracer = CMBLensingTracer(
    perturbations=pert,
    z=z,
)

window = tracer.get_window(z)
```

## Adding Your Own Tracer

To add a new type of photometric observable, follow these steps.

### Step 1: Create Your Tracer Class

```python
# cloelib/observables/my_new_tracer.py
from cloelib.observables.photo.tracer import Tracer
from cloelib.cosmology.cosmology import Perturbations
import numpy as np
import jax.numpy as jnp

class CMBLensingTracer:
    """Tracer for CMB lensing convergence."""

    def __init__(
        self,
        perturbations: Perturbations,
        z_cmb: float = 1100.0,  # CMB redshift
    ):
        """
        Initialize CMB lensing tracer.

        Args:
            perturbations: Perturbations object
            z_cmb: Redshift of last scattering surface
        """
        self.perturbations = perturbations
        self.background = perturbations.background
        self.z_cmb = z_cmb
        self.prefact_toggle = 1  # Include lensing prefactor

    def get_window(self, z: np.ndarray) -> np.ndarray:
        """
        Compute CMB lensing window function.

        The window peaks at z ~ z_cmb/2 (halfway to CMB).
        """
        # Comoving distances
        chi_z = self.background.comoving_distance(z)
        chi_cmb = self.background.comoving_distance(self.z_cmb)

        # CMB lensing efficiency: peaks halfway
        efficiency = (chi_cmb - chi_z) / chi_cmb * chi_z / chi_cmb
        efficiency[z >= self.z_cmb] = 0.0  # No lensing beyond CMB

        # Hubble parameter
        H_z = self.background.hubble_parameter(z, units="1/Mpc")

        # Window function
        window = 1.5 * self.background.Omega_m(0.0) * (1 + z) * H_z * efficiency

        return window

    def _window_integrand(self, z: np.ndarray, zprime: np.ndarray) -> np.ndarray:
        """Window integrand for Limber integration."""
        # This is used by AngularTwoPoint
        # Usually just returns get_window for the z argument
        return self.get_window(z)

    def _get_prefactor(self, ell: np.ndarray) -> np.ndarray:
        """Prefactor for Limber approximation."""
        # CMB lensing has same prefactor as shear
        return ell * (ell + 1)
```

### Step 2: Add to Package

Update `cloelib/observables/__init__.py`:

```python
from cloelib.observables.my_new_tracer import CMBLensingTracer

__all__ = [
    # ... existing exports
    "CMBLensingTracer",
]
```

### Step 3: Write Tests

```python
# tests/test_my_new_tracer.py
import pytest
import numpy as np
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBPerturbations
from cloelib.observables.my_new_tracer import CMBLensingTracer

def test_cmb_lensing_tracer():
    """Test CMB lensing tracer."""
    bg = CAMBBackground(H0=67.5, ...)
    pert = CAMBPerturbations(background=bg)

    tracer = CMBLensingTracer(perturbations=pert)

    z = np.linspace(0.1, 2.0, 50)
    window = tracer.get_window(z)

    # Check shape
    assert window.shape == z.shape

    # Window should peak somewhere between 0 and z_cmb
    peak_idx = np.argmax(window)
    assert 0 < z[peak_idx] < tracer.z_cmb

    # Window should be zero beyond z_cmb
    z_high = np.array([1200.0])
    assert tracer.get_window(z_high)[0] == 0.0
```

## Tips & Tricks

### Protocol Compliance

Always verify your implementation:

```python
from cloelib.observables.photo.tracer import Tracer

assert isinstance(my_tracer, Tracer)
```

### Nuisance Parameters

Keep nuisance parameters in a dictionary:

```python
nuisance = {
    'bias': 1.5,
    'm_bias': 0.01,
    'photo_z_error': 0.03,
}

tracer = MyTracer(perturbations=pert, nuisance_params=nuisance)
```

This makes it easy to vary parameters in MCMC!

### Performance

These calculations are invoked frequently during likelihood evaluation:

```python
from functools import lru_cache

class MyTracer:
    @lru_cache(maxsize=128)
    def _get_window_cached(self, z_tuple):
        z = np.array(z_tuple)
        return self._compute_window(z)

    def get_window(self, z):
        return self._get_window_cached(tuple(z.flat))
```

### JAX Compatibility

If using JAX, avoid Python control flow:

```python
# ❌ Bad (won't JIT)
if z > 1.0:
    result = compute_high_z(z)
else:
    result = compute_low_z(z)

# Good (JIT-able)
result = jnp.where(z > 1.0, compute_high_z(z), compute_low_z(z))
```

## Next Steps

Ready to compute final statistics with your observables?

- [Summary Statistics](../summary_statistics/index.md) – Combine tracers into C_ℓ and multipoles
- [Perturbations](../perturbations.md) – Review structure formation
- [Background](../background.md) – Review the foundation
- [API Reference](../../api.md) – Full technical details
- [Back to Observables](index.md) – Review all oobservables
- [Back to Overview](../index.md) – Review the architecture
