# Summary Statistics: Final Data Products

The **Summary Statistics** module produces final data products for likelihood analysis.

This module completes the cosmological pipeline by converting tracers and power spectra into statistical quantities for comparison with observations. Currently, `cloelib` supports observables for large-scale structure multi-probe experiments.

## Overview

This module computes final statistical quantities for likelihood evaluation, including:

- **C_ℓ**: Angular power spectra for photometric surveys
- **ξ(θ)**: Angular two-point photometric correlation functions
- **COSEBIs**: Complete Orthogonal Sets of E/B-Integrals for photometric surveys
- **P_ℓ(k)**: Legendre multipoles for spectroscopic surveys
- **α*∥, α*⊥**: BAO distortion parameters for spectroscopic surveys

These quantities are directly measurable and form the basis for cosmological parameter inference.

**Performance Note**: cloelib does not use internal interpolations. Keep redshift and wavenumber arrays to a maximum of 1500 elements for optimal performance. Otherwise, memory problems might arise. See [Performance Tips](#performance-tips) for details.

## Available Summary Statistics

**Location**: `cloelib/summary_statistics/`

### For Photometric Surveys (Using Tracers)

#### AngularTwoPoint

Compute angular power spectra C_ℓ from two tracers.

**Location**: `cloelib/summary_statistics/angular_two_point.py`

**What it does**:

- Takes two `Tracer` objects
- Integrates over redshift using Limber approximation
- Outputs C_ℓ as a function of multipole ℓ

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

You can correlate different tracers:

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

# Shape: (3, 3, len(ells))
# C_ell[0, 0, :] = bin 1 × bin 1
# C_ell[0, 1, :] = bin 1 × bin 2
# C_ell[1, 1, :] = bin 2 × bin 2
# etc.
```

#### AngularCorrelationFunction

Compute real-space angular correlation functions ξ(θ).

**Location**: `cloelib/summary_statistics/angular_correlation_function.py`

**What it does**: Transforms C_ℓ → ξ(θ) using Hankel transforms

**Example**:

```python
from cloelib.summary_statistics.angular_correlation_function import AngularCorrelationFunction

# After computing C_ell as above...
ang_corr = AngularCorrelationFunction(tracer1, tracer2)

theta = np.logspace(-2, 1, 30)  # Angles in degrees
xi_plus, xi_minus = ang_corr.compute_xi_pm(theta, ells, C_ell)

# ξ₊(θ): E-mode correlation
# ξ₋(θ): B-mode correlation
```

#### AngularCorrelationFunctionWigner

Alternative implementation using Wigner 3j symbols.

**Location**: `cloelib/summary_statistics/angular_correlation_function_wigner.py`

**When to use**: More accurate for certain configurations

### For Spectroscopic Surveys (Using SpectroPower)

#### LegendreMultipoles

Compute multipoles P_ℓ(k) from 2D power spectrum P(k, μ).

**Location**: `cloelib/summary_statistics/legendre_multipoles.py`

**What it does**:

- Takes a `SpectroPower` object
- Integrates P(k, μ) over μ with Legendre polynomials
- Outputs P₀(k), P₂(k), P₄(k), ...

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
multipoles = leg_multi.compute_multipoles(
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

**Physical Meaning**:

- **P₀(k)**: Monopole—average clustering strength
- **P₂(k)**: Quadrupole—RSD signature (infall/outflow)
- **P₄(k)**: Hexadecapole—higher-order RSD effects

#### BAOAlphas

Compute BAO distortion parameters α*∥ and α*⊥.

**Location**: `cloelib/summary_statistics/bao_alphas.py`

**What it does**: Extract Alcock-Paczynski distortions from BAO

**Example**:

```python
from cloelib.summary_statistics.bao_alphas import compute_bao_alphas

# These measure geometric distortions
alphas = compute_bao_alphas(
    P_multipoles=multipoles,
    k=k,
    r_drag_fid=147.0,  # Fiducial r_s value
    r_drag_true=150.0,  # True r_s value
)

alpha_parallel = alphas['alpha_parallel']     # Along LOS
alpha_perpendicular = alphas['alpha_perp']    # Transverse
```

#### APDistortion

Compute full Alcock-Paczynski distortion matrix.

**Location**: `cloelib/summary_statistics/APDistortion.py`

**What it does**: Handles AP effect more generally

### COSEBIs and Other Statistics

**COSEBIs** (Complete Orthogonal Sets of E/B-Integrals) are specialized statistics for cosmic shear.

**Location**: Various modules with `cosebi` in the name

**When to use**:

- Separating E/B modes cleanly
- Dealing with survey boundaries
- Optimal filtering

Requires optional dependencies (`pylevin`, `mpmath`).

## Creating Custom Summary Statistics

To implement a new statistic, follow these steps.

### Step 1: Decide What You Need

**For photometric statistics**: You'll work with `Tracer` objects
**For spectroscopic statistics**: You'll work with `SpectroPower` objects

### Step 2: Implement Your Calculator

```python
# cloelib/summary_statistics/my_custom_statistic.py
from cloelib.observables.tracer import Tracer
import numpy as np

class MyCustomStatistic:
    """Custom statistic for photometric data."""

    def __init__(self, tracer1: Tracer, tracer2: Tracer):
        """
        Initialize with two tracers.

        Args:
            tracer1: First tracer
            tracer2: Second tracer
        """
        self.tracer1 = tracer1
        self.tracer2 = tracer2
        self.perturbations = tracer1.perturbations
        self.background = self.perturbations.background

    def compute_statistic(
        self,
        scales: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute your custom statistic.

        Args:
            scales: Physical scales (e.g., angles, distances)
            **kwargs: Additional parameters

        Returns:
            Your statistic evaluated at scales
        """
        # Get window functions
        z = kwargs.get('z_grid', np.linspace(0.1, 3.0, 100))
        W1 = self.tracer1.get_window(z)
        W2 = self.tracer2.get_window(z)

        # Get power spectrum
        k = kwargs.get('k_grid', np.logspace(-3, 1, 200))
        P_k = self.perturbations.matter_power_spectrum(z, k)

        # Your custom integration/transformation
        result = self._integrate_custom(scales, W1, W2, P_k, z, k)

        return result

    def _integrate_custom(self, scales, W1, W2, P_k, z, k):
        """Your custom integration kernel."""
        # Implement your math here.
        # This might involve:
        # - Limber approximation
        # - Hankel transforms
        # - Special function evaluations
        # - etc.

        result = np.zeros(len(scales))
        for i, scale in enumerate(scales):
            # Compute statistic for this scale
            integrand = self._compute_integrand(scale, W1, W2, P_k, z, k)
            result[i] = np.trapz(integrand, z)

        return result
```

### Step 3: Add Tests

```python
# tests/test_my_custom_statistic.py
import pytest
import numpy as np
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBPerturbations
from cloelib.observables.photo import ShearTracer
from cloelib.summary_statistics.my_custom_statistic import MyCustomStatistic

def test_custom_statistic():
    """Test custom statistic calculation."""
    # Set up cosmology and tracers
    bg = CAMBBackground(...)
    pert = CAMBPerturbations(background=bg)

    z = np.linspace(0.1, 2.0, 50)
    dndz = np.exp(-((z - 1.0) / 0.3)**2)
    dndz = dndz / np.trapz(dndz, z)

    tracer = ShearTracer(
        perturbations=pert,
        dndz=dndz[np.newaxis, :],
        z=z,
        nuisance_params={...}
    )

    # Compute statistic
    stat = MyCustomStatistic(tracer, tracer)
    scales = np.logspace(-2, 1, 20)
    result = stat.compute_statistic(scales)

    # Basic sanity checks
    assert result.shape == scales.shape
    assert np.all(np.isfinite(result))

    # Add physics-based checks
    # e.g., positivity, monotonicity, etc.
```

### Step 4: Document

Add to the summary statistics section of the docs with:

- What it computes
- When to use it
- Example usage
- Physical interpretation

## Integration Techniques

### Limber Approximation

Most angular statistics use Limber:

```python
def get_Cl_limber(self, ell, W1, W2, z_grid, k_grid, P_k_z):
    """
    Compute C_ℓ using Limber approximation.

    C_ℓ = ∫ dz [W1(z) W2(z) / χ²(z)] P(k=ℓ/χ, z)
    """
    chi = self.background.comoving_distance(z_grid)
    H_z = self.background.hubble_parameter(z_grid, units="1/Mpc")

    # Limber: k = (ℓ + 0.5) / χ
    k_limber = (ell + 0.5) / chi

    # Interpolate P(k, z) at Limber k values
    P_limber = self._interpolate_power(k_limber, z_grid, k_grid, P_k_z)

    # Integrate
    integrand = W1 * W2 * P_limber / (chi**2 * H_z)
    C_ell = np.trapz(integrand, z_grid)

    return C_ell
```

### Hankel Transforms

Convert between C_ℓ and ξ(θ):

```python
from scipy.special import jv  # Bessel functions

def compute_xi_from_Cl(theta, ells, C_ell):
    """
    Transform C_ℓ → ξ(θ) using Hankel transform.

    ξ(θ) = (1/2π) ∫ dℓ ℓ C_ℓ J₀(ℓθ)
    """
    xi = np.zeros(len(theta))

    for i, th in enumerate(theta):
        # Bessel function J₀(ℓθ)
        bessel = jv(0, ells * np.radians(th))

        # Integrate
        integrand = ells * C_ell * bessel
        xi[i] = np.trapz(integrand, ells) / (2 * np.pi)

    return xi
```

### Legendre Integration

Extract multipoles from P(k, μ):

```python
from scipy.special import legendre

def compute_multipole(k, mu, P_k_mu, ell):
    """
    Compute P_ℓ(k) from P(k, μ).

    P_ℓ(k) = (2ℓ+1)/2 ∫₋₁¹ dμ P(k, μ) Lℓ(μ)
    """
    # Legendre polynomial
    L_ell = legendre(ell)

    # Integrate over μ
    integrand = P_k_mu * L_ell(mu)
    P_ell = (2 * ell + 1) / 2.0 * np.trapz(integrand, mu)

    return P_ell
```

## Performance Tips

### Array Size Limits

**Important**: cloelib does not use internal interpolations for n(z), matter power spectra, or other quantities. All calculations are performed on the provided grids directly.

**Recommended array sizes**:

- Redshift arrays (z): **Maximum 1500 elements**
- Wavenumber arrays (k): **Maximum 1500 elements**

Using arrays larger than 1500 elements will significantly degrade performance without meaningful improvement in accuracy. The lack of internal interpolation means that oversized arrays lead to:

- Excessive memory usage
- Longer computation times
- Potential numerical instabilities

**Example of appropriate array sizing**:

```python
# Good: Reasonable array sizes
z = np.linspace(0.01, 3.0, 100)  # 100 points is sufficient
k = np.logspace(-3, 1, 200)  # 200 points for k-space

# Acceptable: Higher resolution when needed
z = np.linspace(0.01, 3.0, 500)  # Still within limits
k = np.logspace(-3, 1, 1000)  # Fine-grained k sampling

# ❌ Avoid: Excessively large arrays
z = np.linspace(0.01, 3.0, 5000)  # Too many points, will be slow
k = np.logspace(-3, 1, 3000)  # Unnecessarily high resolution
```

For more details, see [Issue #375](https://github.com/cloe-org/cloelib/issues/375).

### Vectorization

Always vectorize over ℓ or k:

```python
# Slow: loop over ells
C_ell = np.array([get_Cl_single(ell) for ell in ells])

# Fast: vectorized
C_ell = get_Cl_vectorized(ells)  # All ells at once
```

### Caching

Cache expensive computations:

```python
from functools import lru_cache

class MySummaryStatistic:
    @lru_cache(maxsize=32)
    def _compute_windows_cached(self, z_tuple):
        z = np.array(z_tuple)
        return self.tracer.get_window(z)
```

### JAX JIT

Use JAX for ultimate speed:

```python
import jax
import jax.numpy as jnp

@jax.jit
def get_Cl_jax(ell, W1, W2, P_k, chi, H_z):
    """JIT-compiled C_ℓ calculation."""
    k_limber = (ell + 0.5) / chi
    # ... rest of calculation
    return C_ell

# First call compiles, subsequent calls are blazing fast.
```

## Common Patterns

### Data Vector Construction

Build your likelihood data vector:

```python
# Combine different statistics
data_vector = []

# Angular power spectra
C_ell_shear = two_point_shear.get_Cl(ells)
data_vector.extend(C_ell_shear.flatten())

# Correlation functions
xi_plus = corr_func.compute_xi_plus(theta)
data_vector.extend(xi_plus)

# Multipoles
P0, P2 = multipoles.compute_multipoles(k, ells=[0, 2])
data_vector.extend(P0)
data_vector.extend(P2)

# Convert to array
data_vector = np.array(data_vector)
```

### Scale Cuts

Apply physical scale cuts:

```python
# Only use ℓ range where Limber is valid
ell_min = 30  # Limber breaks down at low ℓ
ell_max = 3000  # Beyond survey resolution

ells = ells[(ells >= ell_min) & (ells <= ell_max)]

# Only use linear scales for multipoles
k_max_linear = 0.2  # h/Mpc
k = k[k <= k_max_linear]
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
