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
 - name: University of Edinburgh, United Kingdom
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

The field of observational cosmology is rapidly evolving with the advent of next-generation galaxy surveys, such as the European Space Agency (ESA) Euclid mission [@Euclid:2024], the Dark Energy Spectroscopic Instrument, and the Vera C. Rubin Observatory. These surveys will deliver unprecedented volumes of high-quality data mapping the large-scale structure of the Universe. Extracting scientific insights from this data requires the efficient computation of theoretical predictions that can be statistically compared to observations in order to constrain cosmological models. This, in turn, demands advanced computational tools capable of modeling complex theoretical frameworks while maintaining computational efficiency. However, existing cosmological software—despite being powerful—often lacks the flexibility to integrate diverse software components for exploring a broad range of theoretical models, or to make use of modern automatic differentiation techniques. These techniques are essential for gradient-based inference methods, which have shown great promise in exploring the cosmological and nuisance parameter space of theoretical models [CITE].

`cloelib` fills this gap by providing a uniquely flexible and extensible framework for cosmological inference. Similarly to [CITE PYCCL, CAMB SOURCES, COSMOSIS], it supports the computation of large-scale structure main observables, including cosmic shear and galaxy clustering, using both photometric and spectroscopic redshifts. Yet, `cloelib` is the first and only large-scale structure code in the cosmology community to implement a unified interface to multiple cosmological backends using Python protocols [CITE PROTOCOLS]. This design enables researchers to seamlessly switch between different theoretical implementations—such as Boltzmann solvers or emulators—without modifying their analysis pipelines or the internal workings of `cloelib` for computing theoretical predictions. This level of modularity and interoperability is unprecedented, significantly lowering the barrier to the inclusion of other pipelines for rapid experimentation in cosmological analyses.

The library interfaces with several well-established Boltzmann solvers, including CAMB [@Lewis:2000], CLASS [@Blas:2011], and their extensions [CITE, e.g., EFTCAMB]. It also supports state-of-the-art emulators, such as HMCode2020Emu [@Mead:2021; CITE MARIA], COMET [@Eggemeier:2022; @Pezzotta:2025] for clustering observables with massive neutrinos, CosmoPower [@SpurioMancini:2021], BACCO [@Angulo:2020], Capse.jl [@Bonici2024Capse], and Effort.jl [@Bonici:2025]. These emulators offer orders-of-magnitude speedups in cosmological computations while maintaining percent-level accuracy, making them essential tools for modern inference pipelines. This modularity and performance make `cloelib` particularly well-suited for systematic studies, model comparison, and robust cross-validation of cosmological results.

Furthermore, a key innovation of `cloelib` is its native integration with JAX [@jax2018github], which enables automatic differentiation of cosmological observables. This transforms traditionally rigid, black-box cosmological pipelines into fully differentiable programs, unlocking entirely new capabilities for modern inference. In particular, it makes `cloelib` uniquely suited for gradient-based optimization, Hamiltonian Monte Carlo, and the seamless integration of neural networks into cosmological models. These methods are often infeasible with traditional software due to the lack of differentiability and performance constraints. `cloelib` has already been used to develop neural network emulators for cosmological observables and to prototype novel inference techniques that would be computationally prohibitive using existing tools.

The modular architecture of `cloelib` is built around a clear separation of concerns, organizing functionality into distinct components: cosmological backgrounds (e.g., background expansion and distances), perturbation theory (e.g., matter power spectra), observables (e.g., cosmic shear and galaxy clustering), and summary statistics (e.g., angular power spectra and correlation functions). This design allows researchers to flexibly mix and match different theoretical models and numerical approximations, supporting both standard analyses and experimental workflows. For example, users can compute angular power spectra using the Limber approximation with any combination of supported Boltzmann solvers and non-linear models, or define custom window functions for specific survey geometries.
Crucially, each module in `cloelib` is defined by a Python protocol, enabling a "plug-and-play" approach: users can include only the components they need, choose among interchangeable backends, and match them as desired—all without altering the core logic of their pipeline. This architecture enforces modularity by design, ensuring robustness, reusability, and ease of experimentation. 

In addition, `cloelib` serves the practical needs of both the Euclid collaboration and the wider cosmology community by offering implementations of survey-specific systematics, Alcock–Paczynski corrections, and Baryon Acoustic Oscillation (BAO) feature extraction. The library works seamlessly with `cloelike`, its companion likelihood module, which supports the computation of likelihoods for Euclid observables such as 2×2pt and 3×2pt photometric correlations, spectroscopic galaxy clustering and BAO, as well as their combinations. Together, these tools enable end-to-end cosmological analyses, covering the full chain from observable computation to likelihood evaluation and Monte Carlo sampling for parameter inference.

Beyond its scientific scope, `cloelib` is optimized for efficiency, with native source code implementations of theoretical predictions and advanced caching mechanisms that accelerate computation rather than hinder it. It also integrates comprehensive testing infrastructure and performance profiling tools, aligned with state-of-the-art software development practices, to ensure reliability and scalability in production-level applications. By combining theoretical flexibility, computational performance, and modern programming standards with an Open Science ethos, `cloelib` makes a substantial contribution to the computational framework required for precision cosmology and is poised to become a cornerstone for large-scale structure analyses in the decade ahead.


# Usage Examples

The power of `cloelib` lies in its intuitive API that allows researchers to quickly set up complex cosmological calculations using this "plug-and-play" approach. Here we demonstrate key features through practical examples.

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

`cloelib` excels at computing observables for photometric surveys, including galaxy clustering and cosmic shear including state-of-the art modelling of systematics:

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

## Spectroscopic Observables

## Protocol Compliance of Interfaces

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

## Scaling tests and time performance

| Backend / Emulator | Observable Type        | Configuration | Runtime (s) | Speedup vs. Baseline | Memory Usage (MB) | Notes |
| ------------------ | ---------------------- | ------------- | ----------- | -------------------- | ----------------- | ----- |
| CAMB               | Cℓ                     | Default       |             |                      |                   |       |
| CLASS              | Cℓ                     | Default       |             |                      |                   |       |
| HMCode2020Emu      | Cℓ                     | Default       |             |                      |                   |       |
| COMET              | Pℓ                     | Default       |             |                      |                   |       |
| CosmoPower         | Cℓ                     | Default       |             |                      |                   |       |
| BACCO              | Cℓ                     | Default       |             |                      |                   |       |
| Capse.jl           | Cℓ                     | Default       |             |                      |                   |       |
| Effort.jl          | Pℓ                     | Default       |             |                      |                   |       |


## Author Contributions

In accordance with JOSS guidelines, we provide a description of individual contributions. The authors are listed in alphabetical order. Tier 1 corresponds to the core maintainers of the **cloe-org** team, who are responsible for the long-term sustainability of the `cloelib` package, including reviewing pull requests and leading technical discussions.

* **M. Bonici**: TBA
* **G. Cañas-Herrera**: TBA
* **P. Carrilho**: TBA
* **S. Casas**: TBA
* **C. Moretti**: TBA
* **A. Pezzotta**: TBA



# Acknowledgements

We acknowledge the support of the Euclid Consortium and thank the CLOE software development team for their foundational work. We are grateful to S. Farrens and N. Tessore for technical guidance, and to the broader community for testing and feedback.
(GCH: every contributor that has modified code should be added to the paper authors)

# References
