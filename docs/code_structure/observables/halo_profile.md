# HaloProfile Protocol (Cluster Observables)

**Protocol Definition**: `cloelib.observables.halos.halo_profile.halo_profile.HaloProfile`

HaloProfile handles the halo mass density profiles: the surface mass density
$\Sigma(R)$ and the excess surface mass density $\Delta\Sigma(R)$ used in cluster
weak lensing, optionally including the 2-halo term.

## Required Attribute

- **`halo_model_properties`**: Reference to the `HaloModelProperties` object (cosmology, background, 2-halo terms)

## Required Methods

## `surface_mass_density(R, z, M, c, halo_bias, radius_units)`

Compute the total surface mass density profile $\Sigma(R)$, including the 2-halo term if requested.

**Inputs**:

- `R`: Radial points (1D array), in units of `radius_units`
- `z`: Redshift points (1D array)
- `M`: Mass points in $h^{-1}\,M_\odot$ (1D array)
- `c`: Concentration (float)
- `halo_bias`: Halo bias for the 2-halo term, shape (len(z), len(M)) (optional)
- `radius_units`: One of `"Mpc/h"`, `"radians"`, `"degrees"`, `"arcmin"`, `"arcsec"`

**Returns**: $\Sigma(z, M, R)$, shape (len(z), len(M), len(R)), units: $h\,M_\odot\,\mathrm{pc}^{-2}$

## `excess_surface_mass_density(R, z, M, c, halo_bias, radius_units)`

Compute the total excess surface mass density profile
$\Delta\Sigma(R) = \langle\Sigma\rangle(<R) - \Sigma(R)$,
including the 2-halo term if requested.

**Inputs**: same as `surface_mass_density`

**Returns**: $\Delta\Sigma(z, M, R)$, shape (len(z), len(M), len(R)), units: $h\,M_\odot\,\mathrm{pc}^{-2}$

## `_surface_mass_density_1h(R, RDelta, Delta, c)`

One-halo surface mass density $\Sigma_{1h}(R)$.

**Inputs**:

- `R`: Radial points in $h^{-1}\,\mathrm{Mpc}$
- `RDelta`: Overdensity radius in $h^{-1}\,\mathrm{Mpc}$
- `Delta`: Critical overdensity threshold
- `c`: Concentration (float)

**Returns**: $\Sigma_{1h}$, shape (len(z), len(M), len(R)), units: $h\,M_\odot\,\mathrm{pc}^{-2}$

## `_mean_surface_mass_density_1h(R, RDelta, Delta, c)`

One-halo mean surface mass density within radius R,
$\langle\Sigma_{1h}\rangle(<R)$.

**Inputs**: same as `_surface_mass_density_1h`

**Returns**: $\langle\Sigma_{1h}\rangle$, shape (len(z), len(M), len(R)), units: $h\,M_\odot\,\mathrm{pc}^{-2}$

## `_f_term(x)` and `_g_term(x)`

Profile-specific analytic F and G terms, functions of the dimensionless radius $x = R / R_s$.

## Existing HaloProfile Implementations

### NFWHaloProfile

Navarro-Frenk-White profile (F and G terms from Golse et al. 2002).

**Location**: `cloelib/observables/halos/halo_profile/nfw_halo_profile.py`

**Example**:

```python
import numpy as np

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.halos.halo_model_properties import HaloModelProperties
from cloelib.observables.halos.halo_profile import NFWHaloProfile


bg = CAMBBackground(H0=67.7, ...)
pert = CAMBLinearPerturbations(bg, np.linspace(0.0, 2.0, 100))

halo_props = HaloModelProperties(pert)
halo_profile = NFWHaloProfile(halo_props, two_halo="None")

R = np.linspace(0.01, 1.0, 9)  # in Mpc/h
z = np.linspace(0.01, 0.5, 4)
M = np.linspace(1e14, 5e14, 6)  # in Msun/h
c = 4.0

Sigma_nfw = halo_profile.surface_mass_density(R, z, M, c)  # Shape: (4, 6, 9)
DeltaSigma_nfw = halo_profile.excess_surface_mass_density(R, z, M, c)  # Shape: (4, 6, 9)
```

### BMOHaloProfile

Truncated NFW profile of Baltz, Marshall & Oguri 2009, with truncation radius
$\tau R_\Delta$, where $\tau$ is set by `trunc_fact` (default 3).

**Location**: `cloelib/observables/halos/halo_profile/bmo_halo_profile.py`

**Example**:

```python
from cloelib.observables.halos.halo_profile import BMOHaloProfile

halo_profile = BMOHaloProfile(halo_props, two_halo="sum", trunc_fact=3.0)

halo_bias = np.ones((z.size, M.size))  # required when two_halo != "None"

Sigma_bmo = halo_profile.surface_mass_density(R, z, M, c, halo_bias=halo_bias)
DeltaSigma_bmo = halo_profile.excess_surface_mass_density(
    R, z, M, c, halo_bias=halo_bias
)  # Shape: (4, 6, 9)
```

### EmulatorMiscenteredHaloProfile

Miscentered profiles $\Sigma_\mathrm{off}$ and $\Delta\Sigma_\mathrm{off}$, evaluated
with neural-network emulators trained on the BMO profile and a Rayleigh
miscentering PDF (Johnston et al. 2007).
It takes an extra `sigma_off` argument instead of `halo_bias`, and the weights
must be loaded with `set_weights` first.

**Location**: `cloelib/observables/halos/halo_profile/emulator_miscentered_halo_profile.py`

**Example**:

```python
from cloelib.observables.halos.halo_profile import EmulatorMiscenteredHaloProfile

halo_profile = EmulatorMiscenteredHaloProfile(halo_props)
halo_profile.set_weights()  # default weights from Zenodo

Sigma_off = halo_profile.surface_mass_density(R, z, M, c, sigma_off=0.3)
DeltaSigma_off = halo_profile.excess_surface_mass_density(R, z, M, c, sigma_off=0.3)
```

## Adding Your Own HaloProfile

To add a new HaloProfile implementation, follow these steps.

### Step 1: Create Your Class

```python
# cloelib/observables/halos/halo_profile/my_halo_profile.py
import numpy as np

from cloelib.observables.halos.halo_model_properties import HaloModelProperties

from .halo_profile_base import HaloProfileBase


class MyHaloProfile(HaloProfileBase):
    """My implementation of a halo mass density profile."""

    def __init__(
        self,
        halo_model_properties: HaloModelProperties,
        overdensity_type: str = "vir",
        overdensity: int = 200,
        two_halo: str = "None",
    ):
        HaloProfileBase.__init__(
            self,
            halo_model_properties,
            overdensity_type=overdensity_type,
            overdensity=overdensity,
        )
        self.two_halo = two_halo

    def _f_term(self, x):
        # add your implementation here
        ...

    def _g_term(self, x):
        # add your implementation here
        ...

    def _surface_mass_density_1h(self, R, RDelta, Delta, c):
        r"""
        One-halo surface mass density.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units: Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units: Mpc / h)
        Delta: np.ndarray
            Critical overdensity
        c: float
            Concentration

        Returns
        -------
        Sigma: np.ndarray
            Surface mass density (units: h * Msun / pc**2),
            shape (z.size, M.size, R.size)
        """
        # add your implementation here

    def _mean_surface_mass_density_1h(self, R, RDelta, Delta, c):
        # add your implementation here
        ...

    def surface_mass_density(self, R, z, M, c, halo_bias=None, radius_units="Mpc/h"):
        Sigma = self._surface_mass_density_1h(
            *self.surface_mass_density_args(R, z, M, radius_units=radius_units), c=c
        )
        if self.two_halo != "None":
            Sigma = self.include_surface_mass_density_2h(
                Sigma, self.two_halo, R, z, halo_bias, radius_units
            )
        self.check_profile_shape(R, z, M, Sigma)
        return Sigma

    def excess_surface_mass_density(
        self, R, z, M, c, halo_bias=None, radius_units="Mpc/h"
    ):
        # same pattern: Sigma_mean - Sigma (1h), then include_excess_surface_mass_density_2h
        ...
```

### Step 2: Handle Different Initialization Styles

As MyHaloProfile is a child class of HaloProfileBase,
it also needs `HaloModelProperties` for initialization,
but it can have extra arguments (e.g. `trunc_fact` in `BMOHaloProfile`):

```python
def __init__(self, halo_model_properties: HaloModelProperties, args):
    HaloProfileBase.__init__(self, halo_model_properties)
    # do something with args
    ...
```

Just make sure it runs `HaloProfileBase.__init__` internally.

`HaloProfileBase` already provides helpers you can reuse:

- `surface_mass_density_args`: converts R to Mpc/h and builds $R_\Delta$ and the density threshold with the right shapes
- `include_surface_mass_density_2h` / `include_excess_surface_mass_density_2h`: add the 2-halo term (`"sum"` or `"max"`)
- `check_profile_shape`: checks the (z, M, R) output shape
- `sigma_crit`, `sigma_crit_inv_eff`: critical surface mass density for lensing

### Step 3: Add to Package

Update `cloelib/observables/halos/halo_profile/__init__.py`:

```python
from .my_halo_profile import MyHaloProfile

__all__ = [
    # ... existing exports
    "MyHaloProfile",
]
```

### Step 4: Write Tests

```python
# tests/test_my_halo_profile.py
import numpy as np
from numpy.testing import assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.halos.halo_model_properties import HaloModelProperties
from cloelib.observables.halos.halo_profile import MyHaloProfile


def test_my_halo_profile():
    bg = CAMBBackground(H0=67.7, ...)
    pert = CAMBLinearPerturbations(bg, np.linspace(0.0, 2.0, 100))

    halo_profile = MyHaloProfile(HaloModelProperties(pert))

    R = np.linspace(0.01, 1.0, 9)
    z = np.linspace(0.01, 0.5, 4)
    M = np.linspace(1e14, 5e14, 6)
    c = 4.0

    rtol = 5e-7  # add relative precision

    known_Sigma = ...  # add known Sigma
    assert_allclose(halo_profile.surface_mass_density(R, z, M, c), known_Sigma, rtol=rtol)

    known_DeltaSigma = ...  # add known DeltaSigma
    assert_allclose(
        halo_profile.excess_surface_mass_density(R, z, M, c), known_DeltaSigma, rtol=rtol
    )
```

## Tips & Tricks

### Protocol Compliance

Always verify your implementation:

```python
from cloelib.observables.halos.halo_profile import HaloProfile

assert isinstance(my_halo_profile, HaloProfile)
```

### Output Shapes

All public methods must return arrays with shape (len(z), len(M), len(R)).
Use `check_profile_shape` before returning.

### 2-Halo Term

If `two_halo` is `"sum"` or `"max"`, `halo_bias` with shape (len(z), len(M))
must be passed explicitly; otherwise a `ValueError` is raised.

### JAX Compatibility

If using JAX, avoid Python control flow:

```python
# ❌ Bad (won't JIT)
if x < 1.0:
    result = f_inner(x)
else:
    result = f_outer(x)

# Good (JIT-able)
result = jnp.where(x < 1.0, f_inner(x), f_outer(x))
```

## Next Steps

- [HaloAbundance Protocol](halo_abundance.md) – Halo mass function and halo bias
- [Summary Statistics](../summary_statistics/index.md) – Combine tracers into C_ℓ and multipoles
- [API Reference](../../api.md) – Full technical details
- [Back to Observables](index.md) – Review all observables
- [Back to Overview](../index.md) – Review the architecture
