# Gravitational-Wave Observables

Gravitational-wave (GW) tracers describe the radial windows used for angular
GW number counts and weak lensing. Both implementations live in
`cloelib/observables/gw.py` and provide the `get_window(z)` interface consumed
by `AngularTwoPoint`.

Both tracers require an evenly spaced, strictly increasing redshift grid that
does not contain zero. The normalized source distribution must have shape
`(n_bins, n_z)`, with its last dimension matching the redshift grid. Shift and
width nuisance parameters use one-based bin numbers, such as `dz_gw_1` and
`width_gw_1`.

## Shared example setup

The examples below use one source bin and a linear CAMB matter power spectrum.
Install cloelib with its `camb` extra, then run the blocks in order.

```python
import numpy as np

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
)

z = np.linspace(0.01, 3.0, 101)
background = CAMBBackground(
    H0=67.5,
    Omega_b0=0.049,
    Omega_cdm0=0.265,
    Omega_k0=0.0,
    As=2.1e-9,
    ns=0.965,
    mnu=0.06,
    w0=-1.0,
    wa=0.0,
    gamma_MG=0.545,
    N_mnu=1,
)
perturbations = CAMBLinearPerturbations(background, redshifts=z)

gw_dndz = np.exp(-0.5 * ((z - 1.0) / 0.25) ** 2)[None, :]
gw_dndz /= np.trapezoid(gw_dndz, z, axis=1)[:, None]

gw_nuisance = {
    "b1_gw_bin0": 1.5,
    "dz_gw_1": 0.0,
    "width_gw_1": 1.0,
}
```

## GWNumberCountsTracer

`GWNumberCountsTracer` represents angular fluctuations in the number density
of GW sources.

**What it does**:

- Computes the source-density window $W_i^{\mathrm{GWNC}}(z)$
- Applies a GW source-bias model
- Applies redshift-distribution shift and width nuisance parameters
- Behaves as a scalar field in angular power spectra

Its radial window is

$$
W_i^{\mathrm{GWNC}}(z) =
b_i^{\mathrm{GW}}(z)n_i^{\mathrm{GW}}(z)\frac{H(z)}{c}.
$$

The supported `gw_bias_model` values are:

- `per_bin`: one constant per bin, read from zero-based keys such as
  `b1_gw_bin0`
- `per_bin_int`: the per-bin values interpolated between the redshifts where
  the source distributions peak
- `poly`: a cubic function with coefficients `b1_gw_poly0` through
  `b1_gw_poly3`

Missing bias parameters default to `1.0`. Shift and width parameters are
required for every bin.

**Example**:

```python
from cloelib.observables.gw import GWNumberCountsTracer

gw_number_counts = GWNumberCountsTracer(
    perturbations=perturbations,
    dndz=gw_dndz,
    z=z,
    gw_bias_model="per_bin",
    nuisance_params=gw_nuisance,
)

number_count_window = gw_number_counts.get_window(z)
assert number_count_window.shape == (1, len(z))
```

## GWWeakLensingTracer

`GWWeakLensingTracer` represents the scalar convergence contribution to GW
luminosity-distance fluctuations.

**What it does**:

- Computes the GW lensing-efficiency and convergence windows
- Applies redshift-distribution shift and width nuisance parameters
- Uses the scalar-convergence angular response
- Excludes galaxy intrinsic alignment, magnification bias, and multiplicative
  shear bias

Its radial window is

$$
W_i^{\mathrm{GWWL}}(z) =
\frac{3}{2}\left(\frac{H_0}{c}\right)^2
\Omega_{\mathrm{m},0}(1+z)\chi(z)
\int_z^{z_{\max}}\!\mathrm{d}z'\,
n_i^{\mathrm{GW}}(z')
\frac{\chi(z')-\chi(z)}{\chi(z')}.
$$

`AngularTwoPoint` applies the scalar-convergence response
$\ell(\ell+1)/(\ell+1/2)^2$ when forming a power spectrum.

**Example**:

```python
from cloelib.observables.gw import GWWeakLensingTracer

gw_weak_lensing = GWWeakLensingTracer(
    perturbations=perturbations,
    dndz=gw_dndz,
    z=z,
    nuisance_params=gw_nuisance,
)

lensing_window = gw_weak_lensing.get_window(z)
assert lensing_window.shape == (1, len(z))
```

Use these tracers with `AngularTwoPoint` as shown in the
[GW summary-statistics example](../summary_statistics/gw.md).

## Next Steps

- [Gravitational-Wave Summary Statistics](../summary_statistics/gw.md) – Compute angular power spectra
- [Photometric Observables](photo.md) – Review the photometric tracer implementations
- [API Reference](../../api.md) – Full class and method details
- [Back to Observables](index.md) – Review all observable interfaces
