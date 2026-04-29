# Perturbations: Structure Formation

The **Perturbations** module computes structure formation in the universe.

Building on the Background module, Perturbations calculates how galaxies, clusters, and cosmic web structures form and evolve.

## Overview

This module computes the growth and evolution of density fluctuations over cosmic time, including:

- Matter power spectra at various redshifts and scales
- Growth rates of structures at different epochs
- Key cosmological parameters like σ₈

This module bridges smooth background cosmology and the observed clustered universe.

## The Perturbations Protocol

**Protocol Definition**: `cloelib.cosmology.cosmology.Perturbations`

The Perturbations protocol defines what every perturbation calculator must provide. This module requires a Background instance as it depends on cosmological distances and densities.

!!! warning "Units convention"
All wavenumbers within `cloelib` are **always in 1/Mpc** (not h/Mpc). Power spectra are therefore in **Mpc³**. Any class implementing the `Perturbations` protocol **must** adopt this convention. If the underlying code uses h/Mpc internally, the implementation wrapper is responsible for converting inputs and outputs.

### Required Property

- **`background`**: Reference to the associated Background object

This is key. Perturbations _always_ needs a Background to compute distances, densities, etc.

### Required Methods

#### `matter_power_spectrum(zs, ks)`

Compute the matter power spectrum for total matter Pmm(k, z).

**Inputs**:

- `zs`: Redshifts (can be array)
- `ks`: Wavenumbers in **1/Mpc** (can be array)

**Returns**: Power spectrum in **Mpc³**

**Note**: Can be linear or non-linear depending on implementation.

#### `matter_power_spectrum_cb(zs, ks)`

Compute the CDM+baryons matter power spectrum Pcb(k, z).

**Inputs**:

- `zs`: Redshifts (can be array)
- `ks`: Wavenumbers in **1/Mpc** (can be array)

**Returns**: Power spectrum in **Mpc³**

**Note**: Can be linear or non-linear depending on implementation.

#### `growth_factor(zs, ks)`

Calculate the linear growth factor D(z, k).

Normalized such that D(z=0) ≈ 1 in matter-dominated era.

#### `growth_rate(zs, ks=None)`

Calculate the linear growth rate f(z) = d ln D / d ln a.

For scale-independent models, `ks` can be `None`.

#### `sigma8_0()`

Compute σ₈ at redshift z=0.

The RMS matter fluctuation in 8 Mpc/h spheres—a key cosmological parameter.

## Existing Implementations

### CAMBPerturbations

Interfaces with [CAMB](https://camb.readthedocs.io) for perturbation calculations.

**Location**: `cloelib/cosmology/camb_cosmology.py`

**When to use**: accurate, production-ready, but sometimes slow

**Features**:

- Linear and non-linear power spectra
- Multiple non-linear models (Halofit, HMCode, ...)
- Neutrino effects
- Modified gravity

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBPerturbations

# First, create background
bg = CAMBBackground(
    H0=67.5,
    Omega_b0=0.0492,
    Omega_cdm0=0.2650,
    As=2.1e-9,
    ns=0.965,
    # ... other parameters
)

# Then create perturbations
pert = CAMBPerturbations(
    background=bg,
    # other parameters
)

# Compute power spectrum
z = np.array([0.0, 0.5, 1.0])
k = np.logspace(-3, 1, 100)  # k in 1/Mpc
P_k_z = pert.matter_power_spectrum(z, k)

# Get σ₈
sigma8 = pert.sigma8_0()
print(f"σ₈ = {sigma8:.4f}")
```

### CLASSPerturbations

Interfaces with [CLASS](https://github.com/lesgourg/class_public).

**Location**: `cloelib/cosmology/class_cosmology.py`

**When to use**: CLASS-specific features, comparison studies

**Example**:

```python
from cloelib.cosmology.class_cosmology import CLASSBackground, CLASSPerturbations

bg = CLASSBackground(...)
pert = CLASSPerturbations(
    background=bg,
    # other parameters
)
```

### HMCode2020EmuPerturbations

Fast emulator for non-linear power spectra using [HMCode2020Emu](https://github.com/MariaTsedrik/HMcode2020Emu.git).

**Location**: `cloelib/cosmology/HMcode2020Emu_cosmology.py`

**When to use**: Fast non-linear predictions, MCMC sampling

**Features**:

- Lightning-fast emulator
- Accurate non-linear P(k)
- Limited parameter range

### EE2Perturbations

Simulation-based emulator for non-linear power spectra using [euclidemu2](https://github.com/PedroCarrilho/EuclidEmulator2/tree/pywrapper).

**Location**: `cloelib/cosmology/EE2_cosmology.py`

**When to use**: Fast non-linear predictions based on simulations, MCMC sampling

**Features**:

- Lightning-fast (emulator.)
- Accurate DM-only non-linear boost for P_mm
- More limited parameter range

### BACCOemuPerturbations

Accurate and fast emulators of the linear, non-linear, and baryonic power spectra using [BACCOemu](https://bitbucket.org/rangulo/baccoemu/src/master/).

**Location**: `cloelib/cosmology/baccoemu_cosmology.py`

**When to use**: Predictions for the matter clustering, in the linear and non linear regime. Predictions of baryonic effects.

**Features**:

- Fast predictions of linear power spectra, growth factors, growth rates, and amplitude of fluctuations;
- Accurate total and cold non-linear matter power spectrum emulated from high-resolution simulations;
- Inclusion of baryonic effects through baryonification;
- Large cosmological parameter range;
- Neural network evaluation with JAX;

### TabulatedBoost / TabulatedBoostedPerturbations

Lightweight wrapper for applying a tabulated beyond-LCDM nonlinear boost to an existing nonlinear matter power spectrum.

**Location**: `cloelib/cosmology/TabulatedBoost_cosmology.py`

**When to use**: You already have boost data from simulations or another external pipeline and want to interpolate it onto the `cloelib` perturbation grid.

**Features**:

- Reads tabulated boost files with columns `[k, B(z_1), B(z_2), ...]`
- Converts input `k` values from `h/Mpc` to the internal `1/Mpc` convention
- Reuses `cloelib`'s wavenumber extrapolation machinery
- Supports `power_law`, `freeze`, `one`, and `taper` high-redshift policies
- Exposes a `TabulatedBoostedPerturbations` wrapper with boosted `matter_power_spectrum`, `growth_factor`, and `sigma8_0`

**Input format**:

- Column 1: `k` in `h/Mpc`
- Remaining columns: boost values `B(k, z_i)` at each tabulated redshift
- `z_cols` must list the redshifts corresponding to those boost columns, in the same order as the file

**Example**:

```python
import numpy as np

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
)
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuNonLinearPerturbations
from cloelib.cosmology.TabulatedBoost_cosmology import (
    TabulatedNonlinearBoost,
    TabulatedBoostedPerturbations,
)

zs = np.linspace(0.0, 4.0, 100)
z_cols = [
    3.017980,
    2.479559,
    2.161320,
    2.013288,
    1.609499,
    1.259818,
    1.000000,
    0.823800,
    0.677100,
    0.552300,
    0.444200,
    0.349200,
    0.264800,
    0.188900,
    0.120200,
    0.057540,
    0.000031,
    0.0,
]

background = CAMBBackground(...)
linear_perturbations = CAMBLinearPerturbations(background, zs)
nonlinear_perturbations = HMemuNonLinearPerturbations(
    background,
    linear_perturbations,
    zs,
)

tabulated_boost = TabulatedNonlinearBoost(
    background,
    linear_perturbations,
    zs,
    "m11_p01_boosts.txt",
    z_cols,
    high_z_policy="freeze",
)
boosted_perturbations = TabulatedBoostedPerturbations(
    linear_perturbations,
    nonlinear_perturbations,
    tabulated_boost.MGboost_interp,
)
```

**Usage notes**:

- The boost file is assumed to be defined relative to a LCDM baseline spectrum.
- `TabulatedNonlinearBoost` sorts `z_cols` internally, so the input columns do not have to be pre-sorted.
- A small synthetic test can cover interpolation behavior, but realistic validation still depends on the physical boost tables you provide.
- The example notebook workflow can download DAKAR2 boost tables externally, but the core module itself does not depend on network access.

### CosmoPowerJAXPerturbations

Fast JAX-based emulator for linear power spectra using [cosmopower-jax](https://github.com/dpiras/cosmopower-jax).

**Location**: `cloelib/cosmology/cosmopower_jax_cosmology.py`

**When to use**: Fast linear predictions with full JAX compatibility, gradient-based inference, MCMC sampling

**Features**:

- Full JAX compatibility enabling automatic differentiation and JIT compilation
- Fast emulation of linear power spectra
- Native support for gradient-based samplers (e.g. HMC/NUTS)
- Limited to linear regime

### JAXPerturbations

Pure JAX implementation for automatic differentiation.

**Location**: `cloelib/cosmology/jax_cosmology.py`

**When to use**: Computing gradients, Fisher forecasts, HMC sampling

## Adding Your Own Perturbations Implementation

To add a new Perturbations implementation, follow these steps.

### Step 1: Create Your Class

Create `cloelib/cosmology/my_solver_cosmology.py`:

```python
from cloelib.cosmology.cosmology import Perturbations, Background
import numpy as np

class MySolverPerturbations:
    """Interface to MySolver for perturbation calculations."""

    def __init__(
        self,
        background: Background,
        nonlinear_model: str = "halofit",
        kmax: float = 10.0,
        zmax: float = 5.0,
        **kwargs
    ):
        """
        Initialize perturbation calculator.

        Args:
            background: Background object (any implementation)
            nonlinear_model: Which non-linear model to use
            kmax: Maximum wavenumber in 1/Mpc
            zmax: Maximum redshift
        """
        self._background = background
        self.nonlinear_model = nonlinear_model

        # Initialize your solver
        self._solver = MySolver(
            H0=background.H0,
            Omega_b=background.Omega_b0,
            # ... pass cosmological parameters from background
        )

        # Set up k and z grids
        self._solver.set_k_grid(kmin=1e-4, kmax=kmax, nk=200)
        self._solver.set_z_grid(zmin=0.0, zmax=zmax, nz=50)

        # Compute power spectrum
        self._solver.compute_power_spectrum(nonlinear=nonlinear_model)

    @property
    def background(self) -> Background:
        """Return the background object."""
        return self._background

    def matter_power_spectrum(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """
        Compute the total matter power spectrum P_mm(k, z).

        Args:
            zs: Redshifts (1D array)
            ks: Wavenumbers in 1/Mpc (1D array)

        Returns:
            P(k, z) in Mpc³, shape (len(zs), len(ks))
        """
        # Ensure inputs are arrays
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)

        # Call solver and interpolate
        P_k_z = self._solver.get_power_spectrum(zs, ks)

        return P_k_z

    def matter_power_spectrum_cb(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """
        Compute the CDM+baryons matter power spectrum P_cb(k, z).

        Args:
            zs: Redshifts (1D array)
            ks: Wavenumbers in 1/Mpc (1D array)

        Returns:
            P_cb(k, z) in Mpc³, shape (len(zs), len(ks))
        """
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)

        # If the solver distinguishes cb from total matter, use the dedicated method;
        # otherwise fall back to the total matter power spectrum
        P_cb_k_z = self._solver.get_cb_power_spectrum(zs, ks)

        return P_cb_k_z

    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Calculate linear growth factor D(z, k)."""
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)

        # Most codes compute scale-independent D(z)
        # If scale-dependent, use ks; otherwise ignore
        D_z = self._solver.get_growth_factor(zs)

        # Return shape (len(zs), len(ks)) even if scale-independent
        return D_z[:, np.newaxis] * np.ones((len(zs), len(ks)))

    def growth_rate(self, zs: np.ndarray | None = None, ks: np.ndarray | None = None) -> np.ndarray:
        """Calculate growth rate f(z) = d ln D / d ln a."""
        if zs is None:
            raise ValueError("zs must be provided")

        zs = np.atleast_1d(zs)
        f_z = self._solver.get_growth_rate(zs)

        # If ks provided, expand to 2D array
        if ks is not None:
            ks = np.atleast_1d(ks)
            f_z = f_z[:, np.newaxis] * np.ones((len(zs), len(ks)))

        return f_z

    def sigma8_0(self) -> float:
        """Compute σ₈ at z=0."""
        return self._solver.get_sigma8()
```

### Step 2: Connect to Background

The key pattern: your Perturbations wraps around a Background object.

```python
# Users do this:
bg = CAMBBackground(...)  # Or any Background implementation
pert = MySolverPerturbations(background=bg)

# Now pert can access:
H_z = pert.background.hubble_parameter(z)
chi = pert.background.comoving_distance(z)
```

This means implementations can be mixed and matched — for example, using a CAMB background with a custom perturbations class.

### Step 3: Handle Array Shapes

Pay attention to array shapes—it's easy to get confused! `cloelib` will always return arrays!

```python
def matter_power_spectrum(self, zs, ks):
    """
    Output shape: (len(zs), len(ks))

    Example:
        zs = [0.0, 0.5, 1.0]  # 3 redshifts
        ks = [0.1, 0.2, ...]  # 100 wavenumbers
        P_k_z has shape (3, 100)
    """
    pass
```

### Step 4: Add Tests

Create `tests/test_my_solver_cosmology.py`:

```python
import pytest
import numpy as np
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.my_solver_cosmology import MySolverPerturbations

@pytest.fixture
def background():
    """Create a test background."""
    return CAMBBackground(
        H0=67.5,
        Omega_b0=0.0492,
        Omega_cdm0=0.2650,
        As=2.1e-9,
        ns=0.965,
        # ... minimal parameters
    )

def test_initialization(background):
    """Test Perturbations initialization."""
    pert = MySolverPerturbations(background=background)
    assert pert.background is background

def test_matter_power_spectrum(background):
    """Test P(k, z) calculation."""
    pert = MySolverPerturbations(background=background)

    z = np.array([0.0, 0.5, 1.0])
    k = np.logspace(-2, 1, 50)
    P_k_z = pert.matter_power_spectrum(z, k)

    # Check shape
    assert P_k_z.shape == (len(z), len(k))

    # Check positivity
    assert np.all(P_k_z > 0)

    # Check power decreases at high z (structure less developed)
    assert np.all(P_k_z[0, :] > P_k_z[-1, :])

def test_sigma8(background):
    """Test σ₈ calculation."""
    pert = MySolverPerturbations(background=background)
    sigma8 = pert.sigma8_0()

    # Reasonable range for σ₈
    assert 0.5 < sigma8 < 1.2

def test_growth_factor(background):
    """Test growth factor calculation."""
    pert = MySolverPerturbations(background=background)

    z = np.array([0.0, 1.0, 2.0])
    k = np.array([0.1, 1.0])
    D = pert.growth_factor(z, k)

    # Check shape
    assert D.shape == (len(z), len(k))

    # Check D decreases with z (earlier = less grown)
    assert np.all(D[0, :] > D[1, :])
    assert np.all(D[1, :] > D[2, :])
```

### Step 5: Update Documentation

- Add to `docs/api.md`
- Update supported codes table in `README.md`
- Add usage examples within the `playrgound` repository

## Interface Patterns

### Working with External Codes

Most Perturbations implementations wrap external Boltzmann codes:

```python
class MyPerturbations:
    def __init__(self, background, ...):
        # Extract parameters from background
        params = {
            'h': background.h,
            'Omega_b': background.Omega_b0,
            'Omega_c': background.Omega_cdm0,
            # ... more parameters
        }

        # Initialize external code
        self._external = ExternalCode(**params)
        self._external.compute()

        # Store for later use
        self._z_grid = ...
        self._k_grid = ...
        self._P_k_z_grid = ...
```

## Tips & Tricks

### Non-linear vs. Linear Power Spectra

Make it clear what you are computing linear or non-linear power spectra at the level of the protocol. The protocols should always return this method and their corresponding CDM and baryon only matter power spectrum `_cb`.

```python
def matter_power_spectrum(self, zs, ks):
    """
    Compute matter power spectrum.
    """
    pass
```

### Parameter Validation

Check inputs early:

```python
def matter_power_spectrum(self, zs, ks):
    if np.any(ks <= 0):
        raise ValueError("Wavenumbers must be positive")
    if np.any(zs < 0):
        raise ValueError("Redshifts must be non-negative")
    if np.any(ks > self.kmax):
        raise ValueError(f"k exceeds kmax={self.kmax}")
```

## Next Steps

Now that you have a grounding in Perturbations, explore:

- [Observables](observables.md) – Connect structure to survey measurements
- [Summary Statistics](summary_statistics.md) – Compute final data products
- [Background](background.md) – Review the foundation
- [API Reference](../api.md) – Full technical details
