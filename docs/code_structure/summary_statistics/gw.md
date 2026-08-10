# Gravitational-Wave Summary Statistics (Using Tracers)

Gravitational-wave (GW) tracers use the same `AngularTwoPoint` Limber
integration as photometric tracers. `GWNumberCountsTracer` represents angular
source number counts (`GWNC`), while `GWWeakLensingTracer` represents weak
lensing of GW luminosity distances (`GWWL`).

## Angular Power Spectra

`AngularTwoPoint.get_Cl(ells, nl, ks)` supports the following GW auto- and
cross-correlations:

| Tracers                             | Output key prefix  | Spectrum array shape |
| ----------------------------------- | ------------------ | -------------------- |
| GW number counts × GW number counts | `("GWNC", "GWNC")` | `(n_ell,)`           |
| GW weak lensing × GW weak lensing   | `("GWWL", "GWWL")` | `(n_ell,)`           |
| GW number counts × GW weak lensing  | `("GWNC", "GWWL")` | `(n_ell,)`           |
| Galaxy positions × GW number counts | `("POS", "GWNC")`  | `(n_ell,)`           |
| Galaxy positions × GW weak lensing  | `("POS", "GWWL")`  | `(n_ell,)`           |
| Galaxy shear × GW number counts     | `("SHE", "GWNC")`  | `(2, n_ell)`         |
| Galaxy shear × GW weak lensing      | `("SHE", "GWWL")`  | `(2, n_ell)`         |

The complete dictionary key also contains the one-based tomographic bin
indices, for example `("GWNC", "GWWL", 1, 2)`. Each value is a cosmolib
`AngularPowerSpectrum`. Correlations involving galaxy shear reserve a second
component for the B-mode and currently fill it with zeros.

```python
import jax.numpy as jnp

from cloelib.observables.gw import (
    GWNumberCountsTracer,
    GWWeakLensingTracer,
)
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

gw_number_counts = GWNumberCountsTracer(...)
gw_weak_lensing = GWWeakLensingTracer(...)

ells = jnp.geomspace(10.0, 1000.0, 30)
ks = gw_number_counts.perturbations.k

gwnc_auto = AngularTwoPoint(gw_number_counts, gw_number_counts).get_Cl(
    ells, nl=0, ks=ks
)
gwwl_auto = AngularTwoPoint(gw_weak_lensing, gw_weak_lensing).get_Cl(
    ells, nl=0, ks=ks
)
gwnc_gwwl = AngularTwoPoint(gw_number_counts, gw_weak_lensing).get_Cl(
    ells, nl=0, ks=ks
)

spectrum = gwnc_gwwl[("GWNC", "GWWL", 1, 1)]
print(spectrum.array)
```

For cross-correlations with photometric tracers, use the same interface:

```python
pos_gwnc = AngularTwoPoint(pos_tracer, gw_number_counts).get_Cl(ells, 0, ks)
pos_gwwl = AngularTwoPoint(pos_tracer, gw_weak_lensing).get_Cl(ells, 0, ks)
shear_gwnc = AngularTwoPoint(shear_tracer, gw_number_counts).get_Cl(ells, 0, ks)
shear_gwwl = AngularTwoPoint(shear_tracer, gw_weak_lensing).get_Cl(ells, 0, ks)
```

## GW Weak-Lensing Response

The `GWWeakLensingTracer` window contains the scalar convergence geometry.
For every GWWL field in a power spectrum, `AngularTwoPoint` additionally
applies

$$
R_\ell^{\mathrm{GWWL}} =
\frac{2\ell(\ell+1)}{(\ell+1/2)^2}.
$$

The factor therefore appears once in a cross-spectrum containing one GWWL
field and twice in a GWWL auto-spectrum.

!!! note
GW tracers are supported by full-sky `get_Cl`. The current `get_pseudo_Cl`
mixing-matrix path supports photometric `POS` and `SHE` pairs only.

## Next Steps

- [Gravitational-Wave Observables](../observables/gw.md) – Configure the GW tracers
- [Photometric Summary Statistics](photo.md) – Compute galaxy angular statistics
- [API Reference](../../api.md) – Full technical documentation
- [Back to Summary Statistics](index.md) – Review all summary statistics
