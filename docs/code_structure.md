# Code Structure

This guide explains the modular architecture of **cloelib** and provides instructions for contributors to extend the library by adding new implementations of **Perturbations** and **Observables**.

## Overview

**cloelib** follows a layered architecture that separates cosmological calculations into distinct components:

```
Background → Perturbations → Observables → Summary Statistics
```

Each layer depends on the previous one, creating a flexible pipeline for computing cosmological observables.

---

## Core Components

### 1. Background

The **Background** protocol defines the interface for computing cosmological background quantities such as distances, Hubble parameters, and matter densities as functions of redshift.

**Protocol Definition**: `cloelib.cosmology.cosmology.Background`

**Key Methods**:

- `Omega_b(z)`: Baryon density as a function of redshift
- `Omega_m(z)`: Matter density as a function of redshift
- `hubble_parameter(z)`: Hubble parameter H(z)
- `comoving_distance(z)`: Comoving distance to redshift z
- `angular_diameter_distance(z)`: Angular diameter distance to redshift z

**Existing Implementations**:

- `CAMBBackground` (from [CAMB](https://camb.readthedocs.io))
- `CLASSBackground` (from [CLASS](https://github.com/lesgourg/class_public))
- `JAXBackground` (JAX-based for automatic differentiation)

**Location**: `cloelib/cosmology/`

---

### 2. Perturbations

The **Perturbations** protocol defines the interface for computing perturbation theory quantities, including matter power spectra, growth factors, and growth rates.

**Protocol Definition**: `cloelib.cosmology.cosmology.Perturbations`

**Key Methods**:

- `matter_power_spectrum(z, k)`: Linear or non-linear matter power spectrum P(k, z)
- `growth_factor(z, k)`: Linear growth factor D(z)
- `growth_rate(z, k)`: Linear growth rate f(z)
- `sigma8_0()`: σ₈ at redshift z=0

**Key Property**:

- `background`: Reference to the associated Background object

**Existing Implementations**:

- `CAMBPerturbations` (from [CAMB](https://camb.readthedocs.io))
- `CLASSPerturbations` (from [CLASS](https://github.com/lesgourg/class_public))
- `HMCode2020EmuPerturbations` (from [HMCode2020Emu](https://github.com/MariaTsedrik/HMcode2020Emu.git))
- `JAXPerturbations` (JAX-based for automatic differentiation)

**Location**: `cloelib/cosmology/`

---

### 3. Observables

The **Observables** layer defines protocols for computing different types of cosmological observables. There are two main protocols:

#### 3.1 Tracer Protocol

The **Tracer** protocol is used for photometric observables, defining window functions for different types of galaxy tracers.

**Protocol Definition**: `cloelib.observables.tracer.Tracer`

**Key Methods**:

- `get_window(z)`: Compute the window function at redshift z
- `_window_integrand(z, zprime)`: Window integrand for integration
- `_get_prefactor(ell)`: Prefactor for Limber approximation

**Key Property**:

- `perturbations`: Reference to the Perturbations object

**Existing Implementations**:

- `ShearTracer`: Cosmic shear (weak lensing) observables
- `PositionsTracer`: Galaxy clustering observables

**Location**: `cloelib/observables/photo.py`

#### 3.2 SpectroPower Protocol

The **SpectroPower** protocol defines the interface for spectroscopic power spectrum calculations, including redshift-space distortions.

**Protocol Definition**: `cloelib.observables.spectro.SpectroPower`

**Key Methods**:

- `Pk2d_rsd(k, mu, **args)`: 2D power spectrum P(k, μ) with redshift-space distortions
- `Pk2d_term_rsd(k, mu, **args)`: Individual terms of the loop expansion

**Key Property**:

- `background`: Reference to the Background object

**Existing Implementations**:

- `CometEFT_spectro`: Spectroscopic power spectrum using [comet-emu](https://comet-emu.readthedocs.io) with EFT model
- `CometVDG_spectro`: Spectroscopic power spectrum using comet-emu with VDG model
- `PBJ_spectro`: Spectroscopic power spectrum using PBJ (not publicly available)

**Location**: `cloelib/observables/`

---

### 4. Summary Statistics

The **Summary Statistics** module provides functions to compute final observable quantities from the tracers and power spectra.

**Available Modules**:

- `angular_two_point`: Compute angular power spectra C_ℓ from tracers
- `angular_correlation_function`: Compute angular correlation functions
- `angular_correlation_function_wigner`: Angular correlation with Wigner symbols
- `legendre_multipoles`: Compute Legendre multipoles for spectroscopic surveys
- `bao_alphas`: Compute BAO distortion parameters (α_parallel, α_perpendicular)
- `APDistortion`: Alcock-Paczynski distortion effects

**Key Classes**:

- `AngularTwoPoint`: Compute two-point correlation functions from two `Tracer` objects
- `LegendreMultipoles`: Compute multipoles of the correlation function from `SpectroPower` objects

**Location**: `cloelib/summary_statistics/`

---

## How to Contribute

### Adding a New Perturbations Implementation

To add support for a new Boltzmann solver or emulator, follow these steps:

#### 1. Create a new file in `cloelib/cosmology/`

Name it according to your implementation, e.g., `my_solver_cosmology.py`.

#### 2. Implement the Background class

```python
from cloelib.cosmology.cosmology import Background
import numpy as np

class MySolverBackground:
    """A wrapper for MySolver background calculations."""
    
    def __init__(self, H0: float, Omega_b0: float, Omega_cdm0: float, ...):
        """Initialize with cosmological parameters."""
        # Initialize your solver with the given parameters
        pass
    
    @property
    def H0(self) -> float:
        """Hubble parameter at z=0 in km/s/Mpc."""
        return self._H0
    
    @property
    def h(self) -> float:
        """Dimensionless Hubble constant."""
        return self.H0 / 100.0
    
    # Implement all other required properties...
    
    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """Compute Hubble parameter at given redshifts."""
        # Call your solver's methods
        pass
    
    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """Calculate comoving distance for given redshifts."""
        # Call your solver's methods
        pass
    
    # Implement all other required methods...
```

#### 3. Implement the Perturbations class

```python
from cloelib.cosmology.cosmology import Perturbations, Background
import numpy as np

class MySolverPerturbations:
    """A wrapper for MySolver perturbation calculations."""
    
    def __init__(self, background: Background, ...):
        """Initialize with a Background object and additional parameters."""
        self._background = background
        # Initialize your solver's perturbation module
        pass
    
    @property
    def background(self) -> Background:
        """Return the background object."""
        return self._background
    
    def matter_power_spectrum(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Compute matter power spectrum P(k, z)."""
        # Call your solver's methods
        pass
    
    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Calculate growth factor D(z)."""
        # Call your solver's methods
        pass
    
    # Implement all other required methods...
```

#### 4. Verify Protocol Compliance

Ensure your implementation satisfies the protocols:

```python
from cloelib.cosmology.cosmology import Background, Perturbations

# Check that your classes implement the protocols
assert isinstance(MySolverBackground(...), Background)
assert isinstance(MySolverPerturbations(...), Perturbations)
```

#### 5. Add Tests

Create a test file in `tests/` to verify your implementation:

```python
# tests/test_my_solver_cosmology.py
import pytest
from cloelib.cosmology.my_solver_cosmology import MySolverBackground, MySolverPerturbations

def test_my_solver_background():
    """Test MySolverBackground implementation."""
    bg = MySolverBackground(H0=67.5, Omega_b0=0.049, ...)
    # Test key methods
    assert bg.H0 == 67.5
    # Add more tests...

def test_my_solver_perturbations():
    """Test MySolverPerturbations implementation."""
    bg = MySolverBackground(...)
    pert = MySolverPerturbations(background=bg, ...)
    # Test key methods
    # Add more tests...
```

#### 6. Update Documentation

- Add your implementation to the API reference in `docs/api.md`
- Update the table of supported codes in `README.md` and `docs/home.md`

---

### Adding a New Observable

To add a new type of observable (e.g., a new tracer or spectroscopic observable), follow these steps:

#### 1. Choose the appropriate protocol

- For photometric observables (window functions): Implement the `Tracer` protocol
- For spectroscopic observables (2D power spectra): Implement the `SpectroPower` protocol

#### 2. Create your implementation

##### Example: Adding a New Tracer

```python
# cloelib/observables/my_new_tracer.py
from cloelib.observables.tracer import Tracer
from cloelib.cosmology.cosmology import Perturbations
import numpy as np
import jax.numpy as jnp

class MyNewTracer:
    """A new tracer for [describe your observable]."""
    
    def __init__(self, perturbations: Perturbations, ...):
        """Initialize the tracer with perturbations and other parameters."""
        self.perturbations = perturbations
        self.background = perturbations.background
        # Initialize other attributes
    
    def get_window(self, z: np.ndarray) -> np.ndarray:
        """Compute the window function at redshift z."""
        # Implement your window function calculation
        pass
    
    def _window_integrand(self, z: np.ndarray, zprime: np.ndarray) -> np.ndarray:
        """Window integrand for integration."""
        # Implement the integrand
        pass
    
    def _get_prefactor(self, ell: np.ndarray) -> np.ndarray:
        """Compute prefactor for Limber approximation."""
        # Implement the prefactor
        pass
```

##### Example: Adding a New SpectroPower Implementation

```python
# cloelib/observables/my_spectro_power.py
from cloelib.observables.spectro import SpectroPower
from cloelib.cosmology.cosmology import Background
import numpy as np

class MySpectroPower:
    """A new spectroscopic power spectrum implementation."""
    
    NLcode: str = "MyCode"
    
    def __init__(self, background: Background, ...):
        """Initialize with background and other parameters."""
        self._background = background
        # Initialize your spectro power module
    
    @property
    def background(self) -> Background:
        """Return the background object."""
        return self._background
    
    def Pk2d_rsd(self, k: np.ndarray, mu: np.ndarray, **args) -> np.ndarray:
        """Compute 2D power spectrum with redshift-space distortions."""
        # Implement P(k, μ) calculation
        pass
    
    def Pk2d_term_rsd(self, k: np.ndarray, mu: np.ndarray, **args) -> np.ndarray:
        """Compute individual terms of the loop expansion."""
        # Implement term-by-term calculation if applicable
        pass
```

#### 3. Add to the observables package

Update `cloelib/observables/__init__.py` to export your new observable:

```python
from cloelib.observables.my_new_tracer import MyNewTracer
from cloelib.observables.my_spectro_power import MySpectroPower

__all__ = [
    # ... existing exports
    "MyNewTracer",
    "MySpectroPower",
]
```

#### 4. Add Tests

Create tests for your new observable:

```python
# tests/test_my_new_observable.py
import pytest
from cloelib.observables.my_new_tracer import MyNewTracer
from cloelib.cosmology.camb_cosmology import CAMBPerturbations

def test_my_new_tracer():
    """Test MyNewTracer implementation."""
    # Set up background and perturbations
    perturbations = CAMBPerturbations(...)
    tracer = MyNewTracer(perturbations=perturbations, ...)
    
    # Test window function
    z = np.array([0.5, 1.0, 1.5])
    window = tracer.get_window(z)
    assert window.shape == (...)
    # Add more tests...
```

#### 5. Update Documentation

- Add your observable to the API reference in `docs/api.md`
- Document the use case and example usage
- Update `docs/home.md` if it's a major addition

---

## Design Principles

### Protocol-Based Design

**cloelib** uses Python protocols (PEP 544) to define interfaces. This allows:

- **Flexibility**: Any class that implements the required methods can be used, without needing explicit inheritance
- **Type Safety**: Static type checkers can verify that implementations satisfy the protocol
- **Clear Contracts**: Protocols document exactly what methods and properties are required

### Separation of Concerns

Each layer has a specific responsibility:

- **Background**: Pure cosmology (no perturbations)
- **Perturbations**: Linear and non-linear structure formation
- **Observables**: Survey-specific calculations (selection functions, biases)
- **Summary Statistics**: Final data products for comparison with observations

### JAX Compatibility

Most implementations support both NumPy and JAX arrays, enabling:

- **Automatic Differentiation**: Compute gradients for parameter inference
- **GPU Acceleration**: Run calculations on GPUs for speed
- **JIT Compilation**: Just-in-time compilation for optimal performance

---

## Common Patterns

### Passing Objects Down the Chain

The typical pattern is:

```python
# 1. Create background
background = CAMBBackground(H0=67.5, Omega_b0=0.049, ...)

# 2. Create perturbations with background
perturbations = CAMBPerturbations(background=background, ...)

# 3. Create observables with perturbations (or background)
tracer = ShearTracer(perturbations=perturbations, dndz=..., z=..., ...)
spectro = CometEFT_spectro(background=background, ...)

# 4. Compute summary statistics
two_point = AngularTwoPoint(tracer1=tracer, tracer2=tracer)
C_ell = two_point.compute_Cl(ells=...)
```

### Caching and Performance

For performance-critical applications:

- Use the `@jax.jit` decorator for JAX functions
- Cache expensive computations using `cloelib.auxiliary.cache`
- Precompute quantities that don't change between calculations

---

## Additional Resources

- **API Reference**: See [api.md](api.md) for detailed API documentation
- **Contributing Guide**: See [contributing.md](contributing.md) for general contribution guidelines
- **Examples**: Check the [cloe-org/playground](https://github.com/cloe-org/playground) repository for usage examples

---

## Questions?

If you have questions about the code structure or need help implementing a new component:

- Open an issue on [GitHub Issues](https://github.com/cloe-org/cloelib/issues)
- Tag `@cloe-maintainers` for assistance
- Join the discussion on [GitHub Discussions](https://github.com/cloe-org/cloelib/discussions)
