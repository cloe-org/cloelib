# HaloAbundance Protocol (Cluster Observables)

**Protocol Definition**: `cloelib.observables.cluster.halo_abundance.halo_abundance.HaloAbundance`

HaloAbundance handles the halo mass and the halo bias.

## Required Attribute

- **`common_halo_properties`**: Reference to common dark matter halo properties object

## Required Methods

## `f_sigma_nu(z, M)`

Compute the halo multiplicity function.

**Inputs**:

- `z`: Redshift points (1D array)
- `M`: Mass points in h^{-1} Msun (1D array)

**Returns**: f(z, M), shape (len(z), len(M))

## `dn_dm(z, M)`

Derivative of the number density.

**Inputs**:

- `z`: Redshift points (1D array)
- `M`: Mass points in h^{-1} Msun (1D array)

**Returns**: dn_dm(z, M), shape (len(z), len(M)), units: h^4 Mpc^{-3} Ms^{-1}.

## `bias(z, M)`

Compute the halo bias.

**Inputs**:

- `z`: Redshift points (1D array)
- `M`: Mass points in h^{-1} Msun (1D array)

**Returns**: b(z, M), shape (len(z), len(M))

## Existing HaloAbundance Implementations

### TinkerHaloAbundance

Tinker et al. 2008 halo mass function and bias.

**Location**: `cloelib/observables/clusters/halo_abundance/tinker_halo_abundance.py`

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.halo_abundance import TinkerHaloAbundance
from cloelib.observables.clusters.common_halo_properties import CommonHaloProperties


bg = CAMBBackground(H0=67.5, ...)
pert = CAMBLinearPerturbations(bg, np.linspace(0.0, 2.0, 100))

common_prop = CommonHaloProperties(perturbations)
halo_abundance = TinkerHaloAbundance(common_prop)

z = np.linspace(0.01, 1.0, 10)
M = np.logspace(14, 15, 5) # in Msun

dn_dm_tinker = halo_abundance.dn_dm(z, M) # Shape: (10, 5)
bias_tinker = halo_abundance.dn_dm(z, M) # Shape: (10, 5)
```

### CastroHaloAbundance

Castro et al. 2021 halo mass function and bias.

**Location**: `cloelib/observables/clusters/halo_abundance/castro_halo_abundance.py`

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.halo_abundance import CastroHaloAbundance
from cloelib.observables.clusters.common_halo_properties import CommonHaloProperties


bg = CAMBBackground(H0=67.5, ...)
pert = CAMBLinearPerturbations(bg, np.linspace(0.0, 2.0, 100))

common_prop = CommonHaloProperties(perturbations)
halo_abundance = CastroHaloAbundance(common_prop)

z = np.linspace(0.01, 1.0, 10)
M = np.logspace(14, 15, 5) # in Msun

dn_dm_castro = halo_abundance.dn_dm(z, M) # Shape: (10, 5)
bias_castro = halo_abundance.dn_dm(z, M) # Shape: (10, 5)
```

## Adding Your Own HaloAbundance

To add a new HaloAbundance implementation, follow these steps.

### Step 1: Create Your Class

```python
# cloelib/observables/clusters/halo_abundance/my_halo_abundance.py
from .halo_abundance_base import HaloAbundanceBase
import numpy as np


class MyHaloAbundance(HaloAbundanceBase):
    """My implementation for halo mass abundance models."""
    def __init__(self, common_halo_properties: CommonHaloProperties):
        HaloAbundanceBase.__init__(self, common_halo_properties)

    def f_sigma_nu(self, z, M):
        r"""
        Computation of the multiplicity function.

        Computes MY multiplicity function at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points.
        M: numpy.ndarray
            Mass points in h^{-1} Msun.

        Returns
        -------
        f_sigma_nu: numpy.ndarray
            f_sigma_nu[i,j], where i is the redshift axis and j the mass axis
        """
        # add your implementation here

    def bias(self, z, M):
        r"""
        Computation of the halo bias.

        Computes MY halo bias at the requested redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            Redshift points
        M: numpy.ndarray
            Mass points in h^{-1} Msun

        Returns
        -------
        bias: numpy.ndarray
            bias[i,j], where i is the redshift axis and j the mass axis
        """
        # add your implementation here
```

### Step 2: Handle Different Initialization Styles

As HaloAbundance is an child class of HaloAbundanceBase,
is also needs `CommonHaloProperties` for initialization,
but it can also have extra arguments:

```python
def __init__(self, common_halo_properties: CommonHaloProperties, args):
    HaloAbundanceBase.__init__(self, common_halo_properties)
    # do something with args
    ...
```

Just make sure it runs `HaloAbundanceBase.__init__` internally.

> Note: If you don't need extra arguments, a `__init__` function
> does not have to be defined in your class.

### Step 3: Add to Package

Update `cloelib/observables/clusters/halo_abundance/__init__.py`:

```python
from cloelib.observables.cluster.halo_abundance.my_halo_abundance import MyHaloAbundance

__all__ = [
    # ... existing exports
    "MyHaloAbundance",
]
```

### Step 4: Write Tests

```python
# tests/test_my_cluster_halo_abundance.py
import numpy as np
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.halo_abundance import MyHaloAbundance
from cloelib.observables.clusters.common_halo_properties import CommonHaloProperties

def test_my_halo_abundance():
    bg = CAMBBackground(H0=67.5, ...)
    pert = CAMBLinearPerturbations(bg, np.linspace(0.0, 2.0, 100))

    common_prop = CommonHaloProperties(perturbations)
    halo_abundance = MyHaloAbundance(common_prop)

    z = np.linspace(0.01, 1.0, 10)
    M = np.logspace(14, 15, 5) # in Msun

    rtol = 5e-7 # add relative precision

    known_dn_dm = ... # add known dn_dm
    assert_allclose(halo_abundance.dn_dm(z, M), known_dn_dm, rtol=rtol)

    known_bias = ... # add known bias
    assert_allclose(halo_abundance.bias(z, M), known_bias, rtol=rtol)
```

## Tips & Tricks

### Protocol Compliance

Always verify your implementation:

```python
from cloelib.observables.cluster.halo_abundance import HaloAbundance

assert isinstance(my_halo_abundance, HaloAbundance)
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
