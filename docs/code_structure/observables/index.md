# Observables: Connecting Theory to Observations

The **Observables** module connects theoretical predictions to observational measurements.

This module handles the transformation from theoretical quantities to observable measurements, accounting for survey-specific effects.

## Overview

This module computes survey-specific large scale structure (selection functions, window functions, and bias parameters) and galaxy cluster (halo mass function) quantities that distinguish real observations from idealized theoretical predictions.

This module addresses:

- Window functions for weak lensing surveys and CMB lensing
- Galaxy bias modeling and corrections
- Redshift-space power spectra P(k, μ)
- Halo mass function

## Different Flavors of Observables

**cloelib** has different types of observable protocols, each serving different purposes for:

- Large Scale Structure
  - [**Tracer Protocol**](photo.md): For photometric observables (angular correlations, weak lensing).
  - [**SpectroPower Protocol**](spectro.md): For spectroscopic observables (3D clustering, redshift-space distortions).
- Galaxy Clusters
  - [**HaloAbundance Protocol**](halo_abundance.md): For halo mass function and halo bias.

## Next Steps

Ready to compute final statistics with your observables?

- [Summary Statistics](../summary_statistics/index.md) – Combine tracers into C_ℓ and multipoles
- [Perturbations](../perturbations.md) – Review structure formation
- [Background](../background.md) – Review the foundation
- [API Reference](../../api.md) – Full technical details
- [Back to Overview](../index.md) – Review the architecture
