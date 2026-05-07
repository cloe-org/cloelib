# Observables: Connecting Theory to Observations

The **Observables** module connects theoretical predictions to observational measurements.

This module handles the transformation from theoretical quantities to observable measurements, accounting for survey-specific effects.

## Overview

This module computes survey-specific quantities including selection functions, window functions, and bias parameters that distinguish real observations from idealized theoretical predictions.

This module addresses:

- Window functions for weak lensing surveys and CMB lensing
- Galaxy bias modeling and corrections
- Redshift-space power spectra P(k, μ)

## Two Flavors of Observables

**cloelib** has two types of observable protocols, each serving different purposes:

### **Tracer Protocol**

For photometric observables (angular correlations, weak lensing)

### **SpectroPower Protocol**

For spectroscopic observables (3D clustering, redshift-space distortions)

---

## Part 1: Tracer Protocol (Photometric Observables)

**Protocol Definition**: `cloelib.observables.tracer.Tracer`

Tracers define window functions for photometric surveys—how galaxies are distributed in redshift and how they trace the matter field.

### Required Property

- **`perturbations`**: Reference to a Perturbations object

Tracers need perturbations to compute power spectra and growth.

### Required Methods

#### `get_window(z)`

Compute the window function W(z) at given redshifts.

**Returns**: Window function values, shape depends on number of redshift bins

#### `_window_integrand(z, zprime)`

Window integrand for Limber integration.

Used internally by summary statistics calculators.

#### `_get_prefactor(ell)`

Compute prefactor for Limber approximation.

Handles different tracer types (shear has extra factors, galaxy clustering doesn't).

### Existing Tracer Implementations

#### ShearTracer

For weak gravitational lensing (cosmic shear) measurements.

**Location**: `cloelib/observables/photo.py`

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

#### PositionsTracer

For galaxy clustering (galaxy positions) measurements.

**Location**: `cloelib/observables/photo.py`

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

#### CMBLensingTracer

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

### Adding Your Own Tracer

To add a new type of photometric observable, follow these steps.

#### Step 1: Create Your Tracer Class

```python
# cloelib/observables/my_new_tracer.py
from cloelib.observables.tracer import Tracer
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

#### Step 2: Add to Package

Update `cloelib/observables/__init__.py`:

```python
from cloelib.observables.my_new_tracer import CMBLensingTracer

__all__ = [
    # ... existing exports
    "CMBLensingTracer",
]
```

#### Step 3: Write Tests

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

---

## Part 2: SpectroPower Protocol (Spectroscopic Observables)

**Protocol Definition**: `cloelib.observables.spectro.SpectroPower`

SpectroPower handles 3D power spectra with redshift-space distortions—what you measure in spectroscopic galaxy surveys.

### Required Property

- **`background`**: Reference to Background object

Note: SpectroPower uses Background directly, not Perturbations. This is because emulators often bypass traditional perturbation calculations.

### Required Attribute

- **`NLcode`**: String identifying the non-linear code/emulator

### Required Methods

#### `Pk2d_rsd(k, mu, **args)`

Compute 2D power spectrum P(k, μ) with redshift-space distortions.

**Inputs**:

- `k`: Wavenumbers (1D array)
- `mu`: Cosine of angle to line-of-sight (1D array)
- `**args`: Additional parameters (redshift, cosmological parameters, etc.)

**Returns**: P(k, μ), shape (len(k), len(mu))

#### `Pk2d_term_rsd(k, mu, **args)`

Compute individual terms of the perturbation theory expansion.

Useful for checking contributions of different terms.

### Existing SpectroPower Implementations

#### CometEFT_spectro

Fast emulator using [comet-emu](https://comet-emu.readthedocs.io) with EFT model.

**Location**: `cloelib/observables/CometEFT_spectro.py`

**When to use**: Fast predictions for clustering, MCMC sampling

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower

bg = CAMBBackground(H0=67.5, ...)

RSD_parameters = {'b1': 1.412, ...} # Biases and counterterms
redshift = 1.0
spectro = CometEFT_spectro(
    bg,
    RSD_parameters,
    redshift
)

k = np.logspace(-2, 0, 50)  # k in 1/Mpc
mu = np.linspace(0, 1, 20)  # μ from 0 (perpendicular) to 1 (parallel)

P_k_mu = spectro.Pk2d_rsd(k, mu)  # Shape: (50, 20)
```

#### CometVDG_spectro

Comet emulator with VDG (velocity divergence - galaxy) model.

**Location**: `cloelib/observables/CometVDG_spectro.py`

**When to use**: Alternative RSD modeling

#### PBJ_spectro

Perturbation theory code (not publicly available).

**Location**: `cloelib/observables/PBJ_spectro.py`

### Adding Your Own SpectroPower

To add a new SpectroPower implementation, follow these steps.

#### Step 1: Create Your Class

```python
# cloelib/observables/my_spectro_power.py
from cloelib.observables.spectro import SpectroPower
from cloelib.cosmology.cosmology import Background
import numpy as np

class MyEmulatorSpectro:
    """Interface to MyEmulator for spectroscopic power spectra."""

    NLcode: str = "MyEmulator"

    def __init__(
        self,
        background: Background,
        z_pk: float,
        model: str = "standard",
    ):
        """
        Initialize spectro power emulator.

        Args:
            background: Background cosmology
            z_pk: Redshift for power spectrum
            model: Which RSD model to use
        """
        self._background = background
        self.z_pk = z_pk
        self.model = model

        # Initialize emulator with cosmological parameters
        self._emulator = MyEmulator(
            Omega_m=background.Omega_m(0.0),
            sigma8=0.8,  # You might need to compute this.
            h=background.h,
            # ... other parameters
        )

    @property
    def background(self) -> Background:
        """Return background object."""
        return self._background

    def Pk2d_rsd(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        **args
    ) -> np.ndarray:
        """
        Compute 2D power spectrum with RSD.

        Args:
            k: Wavenumbers in h/Mpc
            mu: Cosine of angle to LOS
            **args: Can include z, bias, f, etc.

        Returns:
            P(k, μ) in (Mpc/h)³, shape (len(k), len(mu))
        """
        # Extract parameters from args
        z = args.get('z', self.z_pk)
        bias = args.get('bias', 1.0)

        # Get growth rate
        f = self._compute_growth_rate(z)

        # Compute real-space power
        P_real = self._emulator.get_power(k, z)

        # Apply Kaiser formula (simple RSD model)
        # P(k, μ) = P_real(k) * (b + f μ²)²
        beta = f / bias
        kaiser = (bias + f * mu**2)**2

        # Broadcast to 2D
        P_k_mu = P_real[:, np.newaxis] * kaiser[np.newaxis, :]

        return P_k_mu

    def Pk2d_term_rsd(
        self,
        k: np.ndarray,
        mu: np.ndarray,
        **args
    ) -> np.ndarray:
        """Compute individual terms (if supported)."""
        # Return dict of different contributions
        return {
            'tree': self._compute_tree_level(k, mu),
            'one_loop': self._compute_one_loop(k, mu),
        }

    def _compute_growth_rate(self, z: float) -> float:
        """Helper to compute f(z)."""
        # Can use approximations or call Perturbations if available
        Omega_m_z = self.background.Omega_m(z)
        return Omega_m_z**0.55  # Approximate f ≈ Ωₘ^0.55
```

#### Step 2: Handle Different Input Styles

SpectroPower implementations often need to handle various inputs:

```python
def Pk2d_rsd(self, k, mu, **args):
    """
    Flexible input handling.

    args can contain:
        - z: redshift (default: self.z_pk)
        - bias: galaxy bias (default: 1.0)
        - f: growth rate (default: computed from cosmology)
        - sigma_v: velocity dispersion (default: 0.0)
    """
    z = args.get('z', self.z_pk)
    bias = args.get('bias', 1.0)
    f = args.get('f', self._compute_f(z))

    # Use these to compute P(k, μ)
    ...
```

#### Step 3: Write Tests

```python
# tests/test_my_spectro_power.py
import pytest
import numpy as np
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.my_spectro_power import MyEmulatorSpectro

def test_my_emulator_init():
    """Test initialization."""
    bg = CAMBBackground(H0=67.5, ...)
    spectro = MyEmulatorSpectro(background=bg, z_pk=1.0)

    assert spectro.background is bg
    assert spectro.z_pk == 1.0
    assert spectro.NLcode == "MyEmulator"

def test_pk2d_rsd():
    """Test P(k, μ) calculation."""
    bg = CAMBBackground(...)
    spectro = MyEmulatorSpectro(background=bg, z_pk=1.0)

    k = np.array([0.1, 0.2, 0.5])
    mu = np.array([0.0, 0.5, 1.0])

    P_k_mu = spectro.Pk2d_rsd(k, mu, bias=2.0)

    # Check shape
    assert P_k_mu.shape == (len(k), len(mu))

    # Check positivity
    assert np.all(P_k_mu > 0)

    # Check RSD enhancement at μ=1 (line of sight)
    # Should have P(k, μ=1) > P(k, μ=0) due to Kaiser effect
    assert np.all(P_k_mu[:, -1] > P_k_mu[:, 0])

def test_with_different_parameters():
    """Test parameter flexibility."""
    bg = CAMBBackground(...)
    spectro = MyEmulatorSpectro(background=bg, z_pk=0.5)

    k = np.logspace(-2, 0, 20)
    mu = np.linspace(0, 1, 10)

    # Test with different biases
    P1 = spectro.Pk2d_rsd(k, mu, bias=1.0)
    P2 = spectro.Pk2d_rsd(k, mu, bias=2.0)

    # Higher bias should give higher power
    assert np.all(P2 > P1)
```

---

## Tips & Tricks for Both Protocols

### Protocol Compliance

Always verify your implementation:

```python
from cloelib.observables.tracer import Tracer
from cloelib.observables.spectro import SpectroPower

# For Tracer
assert isinstance(my_tracer, Tracer)

# For SpectroPower
assert isinstance(my_spectro, SpectroPower)
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

- [Summary Statistics](summary_statistics.md) – Combine tracers into C_ℓ and multipoles
- [Perturbations](perturbations.md) – Review structure formation
- [Background](background.md) – Review the foundation
- [API Reference](../api.md) – Full technical details
