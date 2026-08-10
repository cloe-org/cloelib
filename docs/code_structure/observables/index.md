# Observables: Connecting Theory to Observations

The **Observables** module connects theoretical predictions to observational measurements.

This module handles the transformation from theoretical quantities to observable measurements, accounting for survey-specific effects.

## Overview

This module computes survey-specific quantities including selection functions, window functions, and bias parameters that distinguish real observations from idealized theoretical predictions.

This module addresses:

- Window functions for weak lensing surveys and CMB lensing
- Galaxy bias modeling and corrections
- Gravitational-wave source number counts and luminosity-distance weak lensing
- Redshift-space power spectra P(k, μ)

## Observable Interfaces

**cloelib** uses two observable protocols. The `Tracer` protocol covers
projected angular observables in both photometric and gravitational-wave
analyses, while `SpectroPower` covers three-dimensional spectroscopic
observables.

### **Tracer Protocol**

For projected observables such as angular number counts and weak lensing:

- [Photometric observables](photo.md)
- [Gravitational-wave observables](gw.md)

### **SpectroPower Protocol**

For spectroscopic observables (3D clustering, redshift-space distortions)

- [Learn more about SpectroPower Protocol](spectro.md)

## Next Steps

Ready to compute final statistics with your observables?

- [Summary Statistics](../summary_statistics/index.md) – Combine tracers into C_ℓ and multipoles
- [Perturbations](../perturbations.md) – Review structure formation
- [Background](../background.md) – Review the foundation
- [API Reference](../../api.md) – Full technical details
- [Back to Overview](../index.md) – Review the architecture
