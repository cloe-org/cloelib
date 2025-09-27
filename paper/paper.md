---
title: 'cloelib: A Flexible Python Library for Computing Cosmological Observables in the Euclid Era'
tags:
  - Python
  - cosmology
  - Euclid
  - JAX
  - automatic differentiation
  - Boltzmann solvers
  - observables
authors:
  - name: Marco Bonici
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Guadalupe Cañas-Herrera
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Pedro Carrilho
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Santiago Casas
    orcid: 0000-0000-0000-0000
    affiliation: 4
  - name: Chiara Moretti
    orcid: 0000-0000-0000-0000
    affiliation: 5
  - name: Andrea Pezzotta
    orcid: 0000-0000-0000-0000
    affiliation: 6
affiliations:
 - name: University of Waterloo, Canada
   index: 1
 - name: European Space Agency, Netherlands
   index: 2
 - name: University of Edinburgh, United Kingdom
   index: 3
 - name: RWTH Aachen University, Germany
   index: 4
 - name: SISSA, Italy
   index: 5
 - name: INAF, Italy
   index: 6
date: 26 September 2025
bibliography: paper.bib
---

# Summary

`cloelib` is a Python library designed to compute cosmological observables for the Cosmology Likelihood for Observables in Euclid (CLOE) project. As we enter an era of precision cosmology with missions like Euclid, the need for flexible, efficient, and differentiable tools for cosmological analysis has become paramount. `cloelib` addresses this need by providing a modular framework that seamlessly interfaces with established Boltzmann solvers while enabling modern computational techniques through JAX-based automatic differentiation. The library implements protocols for background cosmology, linear perturbations, and non-linear clustering, supporting both photometric and spectroscopic observables crucial for next-generation surveys. By combining traditional numerical cosmology with gradient-based optimization capabilities, `cloelib` facilitates both standard analyses and novel machine learning approaches to cosmological inference.

# Statement of need

The landscape of observational cosmology is rapidly evolving with the advent of Stage IV dark energy experiments, particularly the Euclid space mission [@Euclid:2024]. These surveys will provide unprecedented volumes of high-quality data, requiring sophisticated computational tools that can handle complex theoretical models while maintaining computational efficiency. Current cosmological software packages, while powerful, often lack the flexibility to integrate multiple theoretical frameworks or the capability to leverage modern automatic differentiation techniques essential for gradient-based inference methods.

`cloelib` fills this gap by providing a unified interface to multiple cosmological codes through Python protocols, enabling researchers to seamlessly switch between different theoretical implementations without modifying their analysis pipelines. The library interfaces with established Boltzmann solvers including CAMB [@Lewis:2000] and CLASS [@Blas:2011], as well as state-of-the-art emulators such as HMCode2020Emu [@Mead:2021], COMET [@Eggemeier:2022; @Pezzotta:2025] for clustering observables with massive neutrinos, CosmoPower [@SpurioMancini:2021], BACCO [@Angulo:2020], Capse.jl [@Bonici2024Capse], and Effort.jl [@Bonici:2025]. These emulators provide orders-of-magnitude speedups for cosmological calculations while maintaining percent-level accuracy, making them essential for modern inference pipelines. This flexibility is crucial for systematic studies and cross-validation of cosmological results.

A key innovation of `cloelib` is its native support for JAX [@jax2018github], enabling automatic differentiation of cosmological observables. This capability transforms traditionally fixed computational pipelines into differentiable programs, opening new avenues for parameter inference through gradient-based optimization and neural network integration. The library has already been employed in developing neural network emulators for cosmological observables and testing novel inference techniques that would be computationally prohibitive with traditional methods.

The modular architecture of `cloelib` separates concerns into distinct components: cosmological backgrounds, perturbation theory, observables, and summary statistics. This design philosophy allows researchers to mix and match different theoretical models and approximations, facilitating both standard analyses and experimental approaches. For instance, users can compute angular power spectra using Limber approximation with any combination of supported Boltzmann solvers and non-linear models, or implement custom window functions for specific survey configurations.

Furthermore, `cloelib` addresses the practical needs of the Euclid collaboration and the broader cosmological community by providing implementations of survey-specific systematics, Alcock-Paczynski corrections, and BAO feature extraction. The library seamlessly interfaces with `cloelike`, a companion likelihood module that enables the computation of likelihoods for Euclid observables, including 2×2pt and 3×2pt photometric correlations, spectroscopic galaxy clustering, and their combinations. This integration facilitates end-to-end cosmological analyses from observable computation through likelihood evaluation to Monte Carlo sampling for parameter inference. The library includes comprehensive testing infrastructure and performance profiling tools, ensuring reliability and efficiency for production-level analyses. With its combination of theoretical flexibility, computational efficiency, and modern programming paradigms, `cloelib` represents a significant contribution to the computational infrastructure needed for precision cosmology in the coming decade.

# Usage Examples

The power of `cloelib` lies in its intuitive API that allows researchers to quickly set up complex cosmological calculations. Here we demonstrate key features through practical examples.

## Initializing Cosmological Models

`cloelib` provides a consistent interface for different cosmological backends. Users can instantiate cosmological models using standard parameters:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.cosmology.jax_cosmology import JAXBackground, JAXNonLinearPerturbations
import numpy as np

# Define cosmological parameters
H0, Omega_cdm0, Omega_b0 = 70.0, 0.25, 0.05
As, ns = 2e-9, 0.96
w0, wa = -1.0, 0.0

# Initialize different backends with the same parameters
camb_bg = CAMBBackground(H0=H0, Omega_cdm0=Omega_cdm0, Omega_b0=Omega_b0,
                         As=As, ns=ns, w0=w0, wa=wa, Omega_k0=0.0,
                         mnu=0.06, gamma_MG=0.545)

jax_bg = JAXBackground(H0=H0, Omega_cdm0=Omega_cdm0, Omega_b0=Omega_b0,
                       As=As, ns=ns, w0=w0, wa=wa, Omega_k0=0.0,
                       mnu=0.06, gamma_MG=0.545)

# Compute background quantities at various redshifts
z = np.linspace(0, 3, 100)
H_z_camb = camb_bg.hubble_parameter(z)
chi_z_jax = jax_bg.comoving_distance(z)
```

This demonstrates how different backends can be used interchangeably, allowing for easy cross-validation of results.

## Computing Power Spectra

The library supports both linear and non-linear perturbation theories:

```python
# Initialize perturbations
camb_linear = CAMBLinearPerturbations(background=camb_bg, redshifts=z)
jax_nonlinear = JAXNonLinearPerturbations(background=jax_bg)

# Compute matter power spectra
ks = np.logspace(-4, np.log10(5), 100)
linear_pk = camb_linear.matter_power_spectrum(z, ks)
nonlinear_pk = jax_nonlinear.matter_power_spectrum(z, ks)
```

## Photometric Observables

`cloelib` excels at computing observables for photometric surveys, including galaxy clustering and weak lensing:

```python
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

# Define galaxy redshift distributions (normalized)
my_dndz = np.array([...])  # Shape: (n_bins, n_z_points)

# Create tracers with survey-specific nuisance parameters
tracer_pos = PositionsTracer(
    perturbations=nonlinear_perturbations,
    dndz=my_dndz,
    z=z,
    galaxy_bias_model='poly',
    nuisance_params={'b1_photo_poly0': 1.0, 'magnification_bias_1': 0.0}
)

tracer_she = ShearTracer(
    perturbations=nonlinear_perturbations,
    dndz=my_dndz,
    z=z,
    nuisance_params={'AIA': 1.72, 'CIA': 0.0134, 'multiplicative_bias_1': 0.001}
)

# Compute angular power spectra
twopoint = AngularTwoPoint(tracer_she, tracer_pos)
ells = np.logspace(1, np.log10(3000), 100)
Cl_galaxy_shear = twopoint.get_Cl(ells, 0, ks)
```

## Automatic Differentiation with JAX

One of `cloelib`'s unique features is its support for automatic differentiation:

```python
import jax
import jax.numpy as jnp

def compute_observable(params):
    """Function computing an observable from cosmological parameters."""
    H0, Omega_m = params
    bg = JAXBackground(H0=H0, Omega_cdm0=Omega_m-0.05, Omega_b0=0.05, ...)
    return bg.angular_diameter_distance(jnp.array([1.0]))[0]

# Compute gradients with respect to cosmological parameters
grad_fn = jax.grad(compute_observable)
gradients = grad_fn(jnp.array([70.0, 0.3]))
```

This capability enables efficient parameter estimation using gradient-based methods and facilitates the development of differentiable cosmological pipelines.

# Design and Implementation

The architecture of `cloelib` leverages Python protocols (PEP 544) to define interfaces for cosmological calculations, ensuring type safety and extensibility. The library is organized into specialized modules: cosmology backends implementing the Background and Perturbations protocols, observables providing window functions and power spectrum interfaces, summary statistics for angular correlations and multipoles, and auxiliary utilities for mathematical operations and caching. Performance-critical sections utilize JAX's just-in-time compilation, while the caching system optimizes repeated calculations. The library seamlessly integrates with the broader Python scientific ecosystem through NumPy and SciPy, while maintaining compatibility with JAX arrays for differentiable computations. This design enables researchers to construct complex analysis pipelines while maintaining computational efficiency and code maintainability.

# Acknowledgements

We acknowledge the support of the Euclid Consortium and thank the CLOE software development team for their foundational work. We are grateful to S. Farrens and N. Tessore for technical guidance, and to the broader community for testing and feedback.

# References