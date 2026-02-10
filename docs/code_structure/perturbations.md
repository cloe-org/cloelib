# 🌊 Perturbations: Where Structure Forms

Welcome to **Perturbations**—where the universe gets lumpy! 🌌

If Background is the smooth cosmological stage, Perturbations is where galaxies, clusters, and cosmic web structure emerge. This is where things get exciting!

## What are Perturbations?

Perturbations computes how density fluctuations grow and evolve over cosmic time. It answers questions like:

- "What's the matter power spectrum at z=1 and k=0.1 h/Mpc?"
- "How fast are structures growing at cosmic noon?"
- "What's σ₈ for my cosmology?"

This is the bridge between smooth background cosmology and the clustered universe we observe! 🌠

## The Perturbations Protocol

**Protocol Definition**: `cloelib.cosmology.cosmology.Perturbations`

The Perturbations protocol defines what every perturbation calculator must provide. It's intimately connected to Background—you can't have structure without a universe to put it in!

### Required Property

- **`background`**: Reference to the associated Background object

This is key! Perturbations _always_ needs a Background to compute distances, densities, etc.

### Required Methods

#### `matter_power_spectrum(zs, ks)`

Compute the matter power spectrum P(k, z).

**Inputs**:

- `zs`: Redshifts (can be array)
- `ks`: Wavenumbers in h/Mpc (can be array)

**Returns**: Power spectrum in (Mpc/h)³

**Note**: Can be linear or non-linear depending on implementation!

#### `growth_factor(zs, ks)`

Calculate the linear growth factor D(z, k).

Normalized such that D(z=0) ≈ 1 in matter-dominated era.

#### `growth_rate(zs, ks=None)`

Calculate the linear growth rate f(z) = d ln D / d ln a.

For scale-independent models, `ks` can be `None`.

#### `sigma8_0()`

Compute σ₈ at redshift z=0.

The RMS matter fluctuation in 8 Mpc/h spheres—a key cosmological parameter!

## Existing Implementations

### CAMBPerturbations

Interfaces with [CAMB](https://camb.readthedocs.io) for perturbation calculations.

**Location**: `cloelib/cosmology/camb_cosmology.py`

**When to use**: Fast, accurate, production-ready

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
    nonlinear_model="halofit",  # or "mead2020", "mead", etc.
    kmax=10.0,  # Maximum k in h/Mpc
    zmax=5.0,   # Maximum redshift
)

# Compute power spectrum
z = np.array([0.0, 0.5, 1.0])
k = np.logspace(-3, 1, 100)  # k in h/Mpc
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
    nonlinear="hmcode",  # CLASS's non-linear option
)
```

### HMCode2020EmuPerturbations

Fast emulator for non-linear power spectra using [HMCode2020Emu](https://github.com/MariaTsedrik/HMcode2020Emu.git).

**Location**: `cloelib/cosmology/HMcode2020Emu_cosmology.py`

**When to use**: Fast non-linear predictions, MCMC sampling

**Features**:

- Lightning-fast (emulator!)
- Accurate non-linear P(k)
- Limited parameter range

### JAXPerturbations

Pure JAX implementation for automatic differentiation.

**Location**: `cloelib/cosmology/jax_cosmology.py`

**When to use**: Computing gradients, Fisher forecasts, HMC sampling

**Example**:

```python
import jax
from cloelib.cosmology.jax_cosmology import JAXBackground, JAXPerturbations

bg = JAXBackground(...)
pert = JAXPerturbations(background=bg)

# Compute gradient of σ₈ with respect to Omega_m
grad_fn = jax.grad(lambda Om: JAXPerturbations(
    background=JAXBackground(Omega_m0=Om, ...)
).sigma8_0())

dsigma8_dOm = grad_fn(0.3)
```

## Adding Your Own Perturbations Implementation

Ready to add your own structure formation code? Let's do it! 🚀

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
            kmax: Maximum wavenumber in h/Mpc
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
        Compute matter power spectrum P(k, z).

        Args:
            zs: Redshifts (1D array)
            ks: Wavenumbers in h/Mpc (1D array)

        Returns:
            P(k, z) in (Mpc/h)³, shape (len(zs), len(ks))
        """
        # Ensure inputs are arrays
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)

        # Call solver and interpolate
        P_k_z = self._solver.get_power_spectrum(zs, ks)

        return P_k_z

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

The key pattern: your Perturbations wraps around a Background object!

```python
# Users do this:
bg = CAMBBackground(...)  # Or any Background implementation
pert = MySolverPerturbations(background=bg)

# Now pert can access:
H_z = pert.background.hubble_parameter(z)
chi = pert.background.comoving_distance(z)
```

This means you can mix and match! Want CAMB background with your custom perturbations? Done! ✨

### Step 3: Handle Array Shapes

Pay attention to array shapes—it's easy to get confused!

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
- Add usage examples

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

### Caching Results

Perturbation calculations can be expensive. Cache aggressively!

```python
from functools import lru_cache

class MyPerturbations:
    @lru_cache(maxsize=128)
    def matter_power_spectrum(self, zs_tuple, ks_tuple):
        """Cache based on tuple of z and k values."""
        zs = np.array(zs_tuple)
        ks = np.array(ks_tuple)
        # ... compute
        return result

    # Wrapper to handle arrays
    def matter_power_spectrum(self, zs, ks):
        return self._matter_power_spectrum_cached(
            tuple(zs.flat),
            tuple(ks.flat)
        )
```

### Interpolation

Pre-compute on a grid, then interpolate:

```python
from scipy.interpolate import RectBivariateSpline

class MyPerturbations:
    def __init__(self, background, ...):
        # Pre-compute on grid
        z_grid = np.linspace(0, 5, 100)
        k_grid = np.logspace(-3, 2, 200)
        P_grid = self._compute_on_grid(z_grid, k_grid)

        # Set up interpolator
        self._interpolator = RectBivariateSpline(
            z_grid,
            np.log10(k_grid),
            P_grid,
            kx=3, ky=3  # Cubic interpolation
        )

    def matter_power_spectrum(self, zs, ks):
        # Interpolate
        return self._interpolator(zs, np.log10(ks), grid=True)
```

## Tips & Tricks

### Units Matter! 📏

Standard units in cloelib:

- Wavenumbers: **h/Mpc** (not 1/Mpc!)
- Power spectrum: **(Mpc/h)³** (not Mpc³!)
- Always check external code's convention and convert if needed

### Non-Linear vs Linear 🌊

Make it clear what you're computing:

```python
def matter_power_spectrum(self, zs, ks, nonlinear=True):
    """
    Compute matter power spectrum.

    Args:
        nonlinear: If True, include non-linear corrections.
                   If False, return linear theory P(k).
    """
    pass
```

### Parameter Validation ✅

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

Now that you've mastered Perturbations, explore:

- 🔭 [Observables](observables.md) - Connect structure to survey measurements
- 📊 [Summary Statistics](summary_statistics.md) - Compute final data products
- 🌌 [Background](background.md) - Review the foundation
- 📖 [API Reference](../api.md) - Full technical details

Keep building! 🌊✨
