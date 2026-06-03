# SpectroPower Protocol (Spectroscopic Observables)

**Protocol Definition**: `cloelib.observables.spectro.SpectroPower`

SpectroPower handles 3D power spectra with redshift-space distortions—what you measure in spectroscopic galaxy surveys.

## Required Property

- **`background`**: Reference to Background object

> Note: SpectroPower can use Background directly in some cases, since emulators often bypass traditional perturbation calculations, so these are not always necessary (e.g. COMET), while the Background object is always needed for the parameters.

## Required Attribute

- **`NLcode`**: String identifying the non-linear code/emulator

## Required Methods

### `Pk2d_rsd(k, mu, **args)`

Compute 2D power spectrum P(k, μ) with redshift-space distortions.

**Inputs**:

- `k`: Wavenumbers (1D array)
- `mu`: Cosine of angle to line-of-sight (1D array)
- `**args`: Additional parameters (redshift, cosmological parameters, etc.)

**Returns**: P(k, μ), shape (len(k), len(mu))

### `Pk2d_term_rsd(k, mu, **args)`

Compute individual terms of the perturbation theory expansion.

Useful for checking contributions of different terms.

## Existing SpectroPower Implementations

### CometEFT_SpectroPower

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

### CometVDG_SpectroPower

Comet emulator with VDG_infty model.

**Location**: `cloelib/observables/CometVDG_spectro.py`

**When to use**: Alternative RSD modeling

### PBJSpectroPower

Perturbation theory code interfaced with `Background` and
`LinearPerturbation` objects, its speed depends on the computation of
linear quantities (i.e. on which `LinearPerturbation` backend is
selected).

**Location**: `cloelib/observables/PBJ_spectro.py`

**When to use**: Predictions of nonlinear galaxy power spectrum for
spectroscopic observables, beyond $\Lambda$CDM models, MCMC sampling.

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.PBJ_spectro import PBJSpectroPower

zs = np.asarray([1.])
bg = CAMBBackground(H0=67.5, ...)
linear_perturbations = CAMBLinearPerturbations(bg, zs)

RSD_parameters = {'b1': 1.412, ...} # Biases and counterterms
spectro = PBJSpectroPower(
    linear_perturbations,
	RSD_parameters
	)

k = np.logspace(-2, 0, 50)  # k in 1/Mpc
mu = np.linspace(0, 1, 20)  # μ from 0 (perpendicular) to 1 (parallel)

P_k_mu = spectro.Pk2d_rsd(k, mu)
```

## Adding Your Own SpectroPower

To add a new SpectroPower implementation, follow these steps.

### Step 1: Create Your Class

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

### Step 2: Handle Different Input Styles

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

### Step 3: Write Tests

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

## Tips & Tricks

### Protocol Compliance

Always verify your implementation:

```python
from cloelib.observables.spectro import SpectroPower

assert isinstance(my_spectro, SpectroPower)
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
