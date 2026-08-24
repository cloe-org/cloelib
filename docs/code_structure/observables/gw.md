# Gravitational-Wave Observables (Tracer Protocol)

Gravitational-wave (GW) observables use the shared
`cloelib.observables.tracer.Tracer` interface, but are implemented separately
from the photometric tracers in `cloelib/observables/gw.py`.

Both GW tracers require a non-zero redshift grid and a normalized source
distribution with shape `(n_bins, n_z)`. The last dimension of `dndz` must
match `z`; the weak-lensing integration also assumes that this grid is evenly
sampled. Redshift-distribution nuisance parameters are one-based (`dz_gw_1`,
`width_gw_1`, and so on).

## GWNumberCountsTracer

`GWNumberCountsTracer` describes angular fluctuations in the number density of
GW sources. Its window is

$$
W_i^{\mathrm{GWNC}}(z) =
b_i^{\mathrm{GW}}(z)n_i^{\mathrm{GW}}(z)\frac{H(z)}{c}.
$$

The available `gw_bias_model` values are:

- `per_bin`: a constant bias per tomographic bin, read from zero-based keys
  such as `b1_GW_bin0` and `b1_GW_bin1`.
- `per_bin_int`: the same per-bin values interpolated as a function of the
  redshift at which each bin's source distribution peaks.
- `poly`: a cubic bias in redshift, with coefficients `b1_GW_poly0` through
  `b1_GW_poly3`.

Missing bias parameters default to `1.0`.

```python
from cloelib.observables.gw import GWNumberCountsTracer

gw_nuisance = {
    "b1_GW_bin0": 1.5,
    "b1_GW_bin1": 1.8,
    "dz_gw_1": 0.0,
    "dz_gw_2": 0.0,
    "width_gw_1": 1.0,
    "width_gw_2": 1.0,
}

gw_number_counts = GWNumberCountsTracer(
    perturbations=pert,
    dndz=gw_dndz_bins,
    z=z,
    gw_bias_model="per_bin",
    nuisance_params=gw_nuisance,
)

number_count_window = gw_number_counts.get_window(z)
```

## GWWeakLensingTracer

`GWWeakLensingTracer` describes the weak-lensing contribution to GW luminosity
distances. It uses the scalar convergence geometry

$$
W_i^{\mathrm{GWWL}}(z) =
\frac{3}{2}\left(\frac{H_0}{c}\right)^2
\Omega_{\mathrm{m},0}(1+z)\chi(z)
\int_z^{z_{\max}}\!\mathrm{d}z'\,
n_i^{\mathrm{GW}}(z')
\frac{\chi(z')-\chi(z)}{\chi(z')}.
$$

The geometric kernel is analogous to the one used by `ShearTracer`, but the GW
tracer is scalar. It does not include galaxy intrinsic alignments,
magnification bias, or multiplicative shear bias. `AngularTwoPoint` applies the
GW-specific harmonic response when it computes a power spectrum.

```python
from cloelib.observables.gw import GWWeakLensingTracer

gw_weak_lensing = GWWeakLensingTracer(
    perturbations=pert,
    dndz=gw_dndz_bins,
    z=z,
    nuisance_params={
        "dz_gw_1": 0.0,
        "dz_gw_2": 0.0,
        "width_gw_1": 1.0,
        "width_gw_2": 1.0,
    },
)

lensing_window = gw_weak_lensing.get_window(z)
```

## Computing GW Correlations

Pass either GW tracer to `AngularTwoPoint` to compute auto-correlations,
GWNC–GWWL cross-correlations, or supported cross-correlations with photometric
position and shear tracers. See [Gravitational-Wave Summary
Statistics](../summary_statistics/gw.md) for the supported pairings and output
keys.

## Next Steps

- [Gravitational-Wave Summary Statistics](../summary_statistics/gw.md) – Compute angular power spectra
- [Photometric Observables](photo.md) – Review the other Tracer implementations
- [API Reference](../../api.md) – Full class and method details
- [Back to Observables](index.md) – Review all observable interfaces
