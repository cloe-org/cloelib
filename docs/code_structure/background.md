# Background: The Cosmological Foundation

The **Background** module provides the cosmological foundation for structure formation calculations.

## Overview

The Background module is the foundational layer that computes quantities for a smooth, homogeneous universe, including:

- Comoving distances to objects at various redshifts
- Hubble parameter evolution with redshift
- Matter density fractions as functions of redshift

This module does not include structure formation or clustering calculations.

## The Background Protocol

**Protocol Definition**: `cloelib.cosmology.cosmology.Background`

The Background protocol defines the interface that all background implementations must satisfy.

### Required Properties

Every Background implementation must provide:

- **`H0`**: Hubble parameter at z=0 in km/s/Mpc
- **`h`**: Dimensionless Hubble constant (H0/100)
- **`Omega_b0`**: Baryon density parameter at z=0
- **`Omega_cdm0`**: Cold dark matter density parameter at z=0
- **`Omega_cb0`**: Combined cold baryon + cold dark matter density parameter at z=0
- **`Omega_k0`**: Curvature density parameter
- **`mnu`**: Neutrino masses (in eV)
- **`N_ur`**: Effective number of ultra-relativistic species
- **`N_eff`**: Effective number of relativistic species
- **`N_mnu`**: Number of massive neutrino species
- **`As`**: Primordial power spectrum amplitude
- **`ns`**: Primordial power spectrum spectral index
- **`alpha_s`**: Running of the spectral index (d ns / d ln k)
- **`w0`**: Dark energy equation of state parameter
- **`wa`**: Dark energy evolution parameter
- **`gamma_MG`**: Modified gravity parameter
- **`rdrag`**: Sound horizon radius at last scattering (Mpc)
- **`interface_args`**: Dictionary storing interface-specific parameters
- **`z_star`**: Redshift of photon decoupling.

### Required Methods

Every Background implementation must provide these methods:

#### `Omega_b(zs)`

Compute baryon density as a function of redshift.

#### `Omega_m(zs)`

Compute total matter density as a function of redshift.

#### `Omega_cb(zs)`

Compute combined cold baryon + cold dark matter density as a function of redshift.

This is typically Omega_cb(z) = Omega_b(z) + Omega_cdm(z), which is useful when massive neutrinos are present since they don't count as "cold" matter at late times.

#### `hubble_parameter(zs, units="km/s/Mpc")`

Compute the Hubble parameter H(z).

#### `comoving_distance(zs)`

Calculate comoving distance to given redshifts (in Mpc).

#### `transverse_comoving_distance(zs)`

Calculate transverse comoving distance (accounts for curvature).

#### `angular_diameter_distance(zs)`

Calculate angular diameter distance (in Mpc).

## Existing Implementations

### CAMBBackground

Interfaces with the [CAMB](https://camb.readthedocs.io) Boltzmann solver.

**Location**: `cloelib/cosmology/camb_cosmology.py`

**When to use**: Production runs, well-tested, fast performance

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground

bg = CAMBBackground(
    H0=67.5,
    Omega_b0=0.0492,
    Omega_cdm0=0.2650,
    Omega_k0=0.0,
    As=2.1e-9,
    ns=0.965,
    mnu=0.06,  # Sum of neutrino masses
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.55,
    N_mnu=1,
)

# Now use it.
z = np.array([0.5, 1.0, 1.5])
chi = bg.comoving_distance(z)  # Distances in Mpc
H_z = bg.hubble_parameter(z)    # H(z) in km/s/Mpc
```

### CLASSBackground

Interfaces with the [CLASS](https://github.com/lesgourg/class_public) Boltzmann solver.

**Location**: `cloelib/cosmology/class_cosmology.py`

**When to use**: When you need CLASS-specific features or comparing with CLASS-based pipelines

**Example**:

```python
from cloelib.cosmology.class_cosmology import CLASSBackground

bg = CLASSBackground(
    H0=67.5,
    Omega_b0=0.0492,
    Omega_cdm0=0.2650,
    # ... similar parameters to CAMB
)
```

### JAXBackground

Pure JAX implementation for automatic differentiation.

**Location**: `cloelib/cosmology/jax_cosmology.py`

**When to use**: When you need gradients, GPU acceleration, or JIT compilation

**Example**:

```python
from cloelib.cosmology.jax_cosmology import JAXBackground
import jax

bg = JAXBackground(
    H0=67.5,
    Omega_b0=0.0492,
    # ... parameters
)

# Compute gradients.
grad_fn = jax.grad(lambda h0: bg.comoving_distance(jnp.array([1.0]))[0])
dchi_dH0 = grad_fn(67.5)
```

### hi_classBackground

Interfaces with the [hi_class](https://github.com/emiliobellini/hi_class_public) Boltzmann solver.

**Location**: `cloelib/cosmology/hi_class_cosmology.py`

**When to use**: When you need hi_class-specific features or comparing with hi_class-based pipelines

**Example**:

```python
from cloelib.cosmology.hi_class_cosmology import hi_classBackground

bg = hi_classBackground(
    H0=67.5,
    Omega_b0=0.0492,
    Omega_cdm0=0.2650,
    params_smg = {...}, # modified gravity parameters enter here
    # ... similar parameters to CLASS
)
```

## Adding Your Own Background Implementation

To add a new Background implementation, follow these steps.

### Step 1: Create Your Class

Create a new file in `cloelib/cosmology/`, e.g., `my_solver_cosmology.py`:

```python
from cloelib.cosmology.cosmology import Background
import numpy as np

class MySolverBackground:
    """Interface to MySolver for background calculations."""

    def __init__(
        self,
        H0: float,
        Omega_b0: float,
        Omega_cdm0: float,
        Omega_k0: float,
        As: float,
        ns: float,
        mnu: float,
        w0: float,
        wa: float,
        gamma_MG: float,
        N_mnu: int,
        N_ur: float | None = None,
    ):
        """Initialize with cosmological parameters."""
        # Store parameters
        self._H0 = H0
        self._Omega_b0 = Omega_b0
        # ... store all parameters

        # Initialize your solver
        self._solver = MySolver(H0=H0, Omega_b=Omega_b0, ...)
        self._solver.compute()  # If needed

    @property
    def H0(self) -> float:
        """Hubble parameter at z=0 in km/s/Mpc."""
        return self._H0

    @property
    def h(self) -> float:
        """Dimensionless Hubble constant."""
        return self.H0 / 100.0

    @property
    def Omega_cb0(self) -> float:
        """Combined cold baryon + cold dark matter density at z=0."""
        return self._Omega_b0 + self._Omega_cdm0

    # Implement ALL other required properties...

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """Compute Hubble parameter at given redshifts."""
        # Call your solver's methods
        H_z = self._solver.get_hubble(zs)

        # Handle unit conversion if needed
        if units == "1/Mpc":
            H_z = H_z * 1000.0 / self.h  # Convert from km/s/Mpc to 1/Mpc

        return H_z

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """Calculate comoving distance for given redshifts."""
        return self._solver.get_comoving_distance(zs)

    def Omega_cb(self, zs: np.ndarray) -> np.ndarray:
        """Calculate combined cold baryon + cold dark matter density at redshifts."""
        # This is Omega_b(z) + Omega_cdm(z)
        # Useful when you have massive neutrinos
        Omega_b_z = self.Omega_b(zs)
        Omega_cdm_z = self._solver.get_omega_cdm(zs)  # Or implement Omega_cdm(zs) method
        return Omega_b_z + Omega_cdm_z

    # Implement ALL other required methods...
```

### Step 2: Verify Protocol Compliance

Make sure your implementation satisfies the protocol:

```python
from cloelib.cosmology.cosmology import Background

# Python's runtime protocol checking
bg = MySolverBackground(H0=67.5, ...)
assert isinstance(bg, Background)  # Should pass.
```

### Step 3: Add Tests

Create `tests/test_my_solver_cosmology.py`:

```python
import pytest
import numpy as np
from cloelib.cosmology.my_solver_cosmology import MySolverBackground

def test_my_solver_background_init():
    """Test initialization."""
    bg = MySolverBackground(
        H0=67.5,
        Omega_b0=0.0492,
        Omega_cdm0=0.2650,
        Omega_k0=0.0,
        As=2.1e-9,
        ns=0.965,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.55,
        N_mnu=1,
    )
    assert bg.H0 == 67.5
    assert bg.h == 0.675

def test_hubble_parameter():
    """Test Hubble parameter calculation."""
    bg = MySolverBackground(...)
    z = np.array([0.0, 0.5, 1.0])
    H_z = bg.hubble_parameter(z)

    # Check shape
    assert H_z.shape == z.shape

    # Check H(0) = H0
    assert np.isclose(H_z[0], bg.H0)

    # Check H(z) increases with z (for ΛCDM)
    assert H_z[1] > H_z[0]
    assert H_z[2] > H_z[1]

def test_comoving_distance():
    """Test comoving distance calculation."""
    bg = MySolverBackground(...)
    z = np.array([0.1, 0.5, 1.0])
    chi = bg.comoving_distance(z)

    # Check monotonicity
    assert np.all(np.diff(chi) > 0)
```

### Step 4: Update Documentation

- Add your implementation to `docs/api.md`
- Update the table in `README.md` and `docs/home.md`
- Add usage examples

## Interface with External Codes

Background implementations typically wrap external Boltzmann solvers. Here's the pattern:

```python
class MyBackground:
    def __init__(self, ...):
        # Initialize the external solver
        self._external_solver = ExternalSolver(...)

        # Run calculations if needed
        self._external_solver.compute_background()

        # Cache results if beneficial
        self._cache = {}

    def comoving_distance(self, zs):
        # Check cache first (optional but recommended)
        cache_key = tuple(zs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Call external code
        result = self._external_solver.get_distance(zs)

        # Cache and return
        self._cache[cache_key] = result
        return result
```

## Tips & Tricks

### Unit Conversions

Always check units. The protocol specifies:

- Distances: Mpc
- Hubble parameter: km/s/Mpc (default) or 1/Mpc (if `units="1/Mpc"`)
- Masses: eV

### Edge Cases

Handle edge cases gracefully:

```python
def comoving_distance(self, zs):
    # Ensure zs is an array
    zs = np.atleast_1d(zs)

    # Check for negative redshifts
    if np.any(zs < 0):
        raise ValueError("Redshifts must be non-negative")

    # Handle z=0 specially if needed
    result = self._solver.get_distance(zs)
    result[zs == 0] = 0.0  # Distance to z=0 is 0

    return result
```

## Next Steps

Now that you understand Background, you are ready for:

- [Perturbations](perturbations.md) – Add structure formation on top of your background
- [Observables](observables.md) – Connect background to survey measurements
- [API Reference](../api.md) – Full technical documentation
