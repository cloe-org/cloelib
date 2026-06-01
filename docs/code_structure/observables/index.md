# Observables: Connecting Theory to Observations

The **Observables** module connects theoretical predictions to observational measurements.

This module handles the transformation from theoretical quantities to observable measurements, accounting for survey-specific effects.

## Overview

This module computes survey-specific quantities including selection functions, window functions, and bias parameters that distinguish real observations from idealized theoretical predictions.

This module addresses:

- Window functions for weak lensing surveys and CMB lensing
- Galaxy bias modeling and corrections
- Redshift-space power spectra P(k, μ)

## Two Flavors of Observables

**cloelib** has two types of observable protocols, each serving different purposes:

### **Tracer Protocol**

For photometric observables (angular correlations, weak lensing)

- [Learn more about Tracer Protocol](photo.md)

### **SpectroPower Protocol**

For spectroscopic observables (3D clustering, redshift-space distortions)

- [Learn more about SpectroPower Protocol](spectro.md)


## Next Steps

Ready to compute final statistics with your observables?

- [Summary Statistics](../summary_statistics.md) – Combine tracers into C_ℓ and multipoles
- [Perturbations](../perturbations.md) – Review structure formation
- [Background](../background.md) – Review the foundation
- [API Reference](../../api.md) – Full technical details
