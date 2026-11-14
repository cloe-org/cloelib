# Gravitational-Wave Summary Statistics

`AngularTwoPoint` computes Limber angular power spectra from
`GWNumberCountsTracer` and `GWWeakLensingTracer` in the same way it consumes
other `Tracer` implementations.

## Angular Power Spectra

**Location**: `cloelib/summary_statistics/angular_two_point.py`

**What it does**:

- Takes two compatible tracer objects
- Integrates their radial windows against the matter power spectrum
- Applies the angular response associated with each field
- Returns one `cosmolib.AngularPowerSpectrum` per tomographic bin pair

The following GW auto- and cross-correlations are supported:

| Tracers                             | Output key prefix  | Spectrum array shape |
| ----------------------------------- | ------------------ | -------------------- |
| GW number counts × GW number counts | `("GWNC", "GWNC")` | `(n_ell,)`           |
| GW weak lensing × GW weak lensing   | `("GWWL", "GWWL")` | `(n_ell,)`           |
| GW number counts × GW weak lensing  | `("GWNC", "GWWL")` | `(n_ell,)`           |
| Galaxy positions × GW number counts | `("POS", "GWNC")`  | `(n_ell,)`           |
| Galaxy positions × GW weak lensing  | `("POS", "GWWL")`  | `(n_ell,)`           |
| Galaxy shear × GW number counts     | `("SHE", "GWNC")`  | `(2, n_ell)`         |
| Galaxy shear × GW weak lensing      | `("SHE", "GWWL")`  | `(2, n_ell)`         |

Complete dictionary keys also contain one-based tomographic bin indices. For
example, `("GWNC", "GWWL", 1, 1)` selects the first number-count bin crossed
with the first weak-lensing bin. Same-observable spectra contain the upper
triangle of bin pairs; different-observable spectra contain the full Cartesian
product. Reversing the two input tracers preserves the canonical key order in
the table.

## Example

Run the shared setup and both tracer examples on the
[GW observables page](../observables/gw.md) first. The variables used here are
the same: `perturbations`, `gw_number_counts`, and `gw_weak_lensing`.

```python
import numpy as np

from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

ells = np.geomspace(10.0, 1000.0, 20)
ks = perturbations.k

gwnc_auto = AngularTwoPoint(
    gw_number_counts, gw_number_counts
).get_Cl(ells, nl=0, ks=ks)

gwwl_auto = AngularTwoPoint(
    gw_weak_lensing, gw_weak_lensing
).get_Cl(ells, nl=0, ks=ks)

gwnc_gwwl = AngularTwoPoint(
    gw_number_counts, gw_weak_lensing
).get_Cl(ells, nl=0, ks=ks)

cross_spectrum = gwnc_gwwl[("GWNC", "GWWL", 1, 1)]
assert cross_spectrum.array.shape == (len(ells),)
```

The `nl` argument is currently reserved and is set to zero in this example.
These calculations return signal spectra without adding a noise model.

## GW Weak-Lensing Response

The `GWWeakLensingTracer` radial window contains the scalar convergence
geometry. For each GWWL field, `AngularTwoPoint` applies

$$
R_\ell^{\mathrm{GWWL}} =
\frac{\ell(\ell+1)}{(\ell+1/2)^2}.
$$

This response appears once in a spectrum containing one GWWL field and is
squared in a GWWL auto-spectrum. The radial window retains the standard
$3\Omega_{\mathrm{m},0}H_0^2/(2c^2)$ normalization.

!!! note
GW tracers are supported by the unmasked Limber `get_Cl` calculation. The
current `get_pseudo_Cl` mixing-matrix path supports photometric `POS` and
`SHE` pairs only.

## Next Steps

- [Gravitational-Wave Observables](../observables/gw.md) – Construct the GW tracers
- [Photometric Summary Statistics](photo.md) – Review galaxy angular statistics
- [API Reference](../../api.md) – Full technical documentation
- [Back to Summary Statistics](index.md) – Review all summary statistics
