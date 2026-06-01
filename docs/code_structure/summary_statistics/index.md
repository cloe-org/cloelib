# Summary Statistics: Final Data Products

The **Summary Statistics** module produces final data products for likelihood analysis.

This module completes the cosmological pipeline by converting tracers and power spectra into statistical quantities for comparison with observations. Currently, `cloelib` supports observables for large-scale structure multi-probe experiments. Observable outputs are formatted — or can be requested to be formatted — following the [cosmolib](https://github.com/astro-ph/cosmolib) data format, which is used by downstream packages such as the reading and writing package [euclidlib](https://github.com/euclidlib/euclidlib). To understand the format, we kindly request the user to check [`euclidlib` docs](https://euclidlib.readthedocs.io/en/latest/format.html).

To see examples of usage, please check the example notebooks for observable computation at [`playground`](https://github.com/cloe-org/playground/tree/main/tutorials/observables).

## Overview

**Location**: `cloelib/summary_statistics/`

This module computes final statistical quantities for likelihood evaluation, including:

- $C_\ell$: Angular power spectra for photometric surveys, either full sky or convolved with the mask
- $\xi_+(\theta)$, $\xi_-(\theta)$, $w(\theta)$, $\gamma_T$, $\gamma_\times$: Angular two-point photometric correlation functions
- COSEBIs: Complete Orthogonal Sets of E/B-Integrals for photometric surveys as in [Asgari et al., 2018.](https://arxiv.org/pdf/1201.2669)
- $P_\ell(k)$: Legendre multipoles for spectroscopic surveys, either full sky or convolved with the mask
- $\xi(r)$: Two-point correlation function from Legendre multipoles, also supported as polar two-point correlation function
- $\alpha_\parallel$, $\alpha_\perp$: BAO distortion parameters for spectroscopic surveys

These quantities are directly measurable and form the basis for cosmological parameter inference.

!!! warning cloelib does not use internal interpolations. Keep redshift and wavenumber arrays to a maximum of 1500 elements for optimal performance. Otherwise, memory problems may arise.

## Performance Tips

`cloelib` does not use internal interpolations. Keep redshift and wavenumber arrays to a maximum of 1500 elements for optimal performance. Otherwise, memory problems may arise.

## Performance Tips

For expensive summary-statistic evaluations, prefer moderate redshift and wavenumber grids, especially when scanning parameter space repeatedly. In practice, keeping these arrays at or below roughly 1500 elements avoids unnecessary memory pressure in the current implementation.

## Available Summary Statistics

- [For Photometric Surveys (Using Tracers)](photo.md)
- [For Spectroscopic Surveys (Using SpectroPower)](spectro.md)

## Next Steps

The pipeline is now complete.
From here:

- [API Reference](../../api.md) – Full technical documentation
- [Contributing Guide](../../contributing.md) – General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground) – Real usage examples
- [Back to Overview](../index.md) – Review the architecture

Or return to any component:

- [Background](../background.md)
- [Perturbations](../perturbations.md)
- [Observables](../observables.md)
