---
title: "cloelib: A Flexible Python Library for Computing Cosmological Observables in the Euclid Era"
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
    orcid: 0000-0002-8430-126X
    affiliation: 1
  - name: Guadalupe Cañas-Herrera
    orcid: 0000-0003-2796-2149
    affiliation: 2
  - name: Pedro Carrilho
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Santiago Casas
    orcid: 0000-0000-0000-0000
    affiliation: 4
  - name: Chiara Moretti
    orcid: 0000-0003-3314-8936
    affiliation: 5
  - name: Andrea Pezzotta
    orcid: 0000-0003-0726-2268
    affiliation: 6
  - name: Michel Aguena
    orcid: 0000-0000-0000-0000
    affiliation: 7
  - name: Zahra Baghkhani
    orcid: 0000-0000-0000-0000
    affiliation: 8
  - name: Matteo Baratto
    orcid: 0000-0000-0000-0000
    affiliation: 9
  - name: Ben Bose
    orcid: 0000-0000-0000-0000
    affiliation: 10
  - name: Pierre Burger
    orcid: 0000-0000-0000-0000
    affiliation: 11
  - name: Carmelita Carbone
    orcid: 0000-0000-0000-0000
    affiliation: 12
  - name: Chaitanya Chawak
    orcid: 0000-0000-0000-0000
    affiliation: 13
  - name: Jose Coloma Nadal
    orcid: 0000-0000-0000-0000
    affiliation: 14
  - name: Martin Crocce
    orcid: 0000-0000-0000-0000
    affiliation: 15
  - name: Stefano Davini
    orcid: 0000-0000-0000-0000
    affiliation: 16
  - name: Samuel Farrens
    orcid: 0000-0000-0000-0000
    affiliation: 17
  - name: Nastassia Grim
    orcid: 0000-0000-0000-0000
    affiliation: 18
  - name: Alex Hall
    orcid: 0000-0000-0000-0000
    affiliation: 19
  - name: Raphael Kou
    orcid: 0000-0000-0000-0000
    affiliation: 20
  - name: Laila Linke
    orcid: 0000-0000-0000-0000
    affiliation: 21
  - name: Arthur Loureiro
    orcid: 0000-0000-0000-0000
    affiliation: 22
  - name: Dida Markovic
    orcid: 0000-0000-0000-0000
    affiliation: 23
  - name: David Navarro Gironés
    orcid: 0000-0000-0000-0000
    affiliation: 24
  - name: Filippo Oppizzi
    orcid: 0000-0000-0000-0000
    affiliation: 25
  - name: Gabriele Parimbelli
    orcid: 0000-0000-0000-0000
    affiliation: 26
  - name: Robert Reischke
    orcid: 0000-0000-0000-0000
    affiliation: 27
  - name: Fabrice Roy
    orcid: 0000-0000-0000-0000
    affiliation: 28
  - name: Jaime Ruiz Zapatero
    orcid: 0000-0000-0000-0000
    affiliation: 29
  - name: Ziad Sakr
    orcid: 0000-0000-0000-0000
    affiliation: 30
  - name: Davide Sciotti
    orcid: 0000-0000-0000-0000
    affiliation: 31
  - name: Ivan Sladoljev
    orcid: 0000-0000-0000-0000
    affiliation: 32
  - name: Arghavan Souki
    orcid: 0000-0000-0000-0000
    affiliation: 33
  - name: Konstantinos Tanidis
    orcid: 0000-0000-0000-0000
    affiliation: 34
  - name: Peter Taylor
    orcid: 0000-0000-0000-0000
    affiliation: 35
  - name: Nicolas Tessore
    orcid: 0000-0000-0000-0000
    affiliation: 36
  - name: Maria Tsedrik
    orcid: 0000-0000-0000-0000
    affiliation: 37
  - name: Isaac Tutusaus
    orcid: 0000-0000-0000-0000
    affiliation: 38
  - name: Casper Vedder
    orcid: 0000-0000-0000-0000
    affiliation: 39
  - name: Angus H. Wright
    orcid: 0000-0000-0000-0000
    affiliation: 40
affiliations:
  - name: University of Waterloo, Canada
    index: 1
  - name: Leiden Observatory, the Netherlands
    index: 2
  - name: University of Edinburgh, United Kingdom
    index: 3
  - name: RWTH Aachen University, Germany
    index: 4
  - name: INAF - Osservatorio Astronomico di Trieste, Italy
    index: 5
  - name: INAF - Osservatorio Astronomico di Brera, Italy
    index: 6
  - name: TBD
    index: 7
  - name: TBD
    index: 8
  - name: TBD
    index: 9
  - name: TBD
    index: 10
  - name: TBD
    index: 11
  - name: TBD
    index: 12
  - name: TBD
    index: 13
  - name: TBD
    index: 14
  - name: TBD
    index: 15
  - name: TBD
    index: 16
  - name: TBD
    index: 17
  - name: TBD
    index: 18
  - name: TBD
    index: 19
  - name: TBD
    index: 20
  - name: TBD
    index: 21
  - name: TBD
    index: 22
  - name: TBD
    index: 23
  - name: TBD
    index: 24
  - name: TBD
    index: 25
  - name: TBD
    index: 26
  - name: TBD
    index: 27
  - name: TBD
    index: 28
  - name: TBD
    index: 29
  - name: TBD
    index: 30
  - name: TBD
    index: 31
  - name: TBD
    index: 32
  - name: TBD
    index: 33
  - name: TBD
    index: 34
  - name: TBD
    index: 35
  - name: TBD
    index: 36
  - name: TBD
    index: 37
  - name: TBD
    index: 38
  - name: TBD
    index: 39
  - name: TBD
    index: 40
date: 18 March 2026
bibliography: paper.bib
---

# Summary

`cloelib` is a Python library designed to compute cosmological observables for the Cosmology Likelihood for Observables in Euclid (CLOE) project. As we enter an era of precision cosmology with missions like Euclid, the need for flexible, efficient, and differentiable tools for cosmological analysis has become paramount. `cloelib` addresses this need by providing a modular framework that seamlessly interfaces with established Boltzmann solvers while enabling modern computational techniques through JAX-based automatic differentiation. The library implements protocols for background cosmology, linear perturbations, and non-linear clustering, supporting both photometric and spectroscopic observables crucial for next-generation surveys. By combining traditional numerical cosmology with gradient-based optimization capabilities, `cloelib` facilitates both standard analyses and novel machine learning approaches to cosmological inference.

# Statement of need

The field of observational cosmology is rapidly evolving with the advent of next-generation galaxy surveys, such as the European Space Agency (ESA) Euclid mission [@Euclid:2024], the Dark Energy Spectroscopic Instrument, and the Vera C. Rubin Observatory. These surveys will deliver unprecedented volumes of high-quality data mapping the large-scale structure of the Universe. Extracting scientific insights from this data requires the efficient computation of theoretical predictions that can be statistically compared to observations in order to constrain cosmological models. This, in turn, demands advanced computational tools capable of modeling complex theoretical frameworks while maintaining computational efficiency. However, existing cosmological software—despite being powerful—often lacks the flexibility to integrate diverse software components for exploring a broad range of theoretical models, or to make use of modern automatic differentiation techniques. These techniques are essential for gradient-based inference methods, which have shown great promise in exploring the cosmological and nuisance parameter space of theoretical models [CITE].

`cloelib` fills this gap by providing a uniquely flexible and extensible framework for cosmological inference. Similarly to [CITE PYCCL, CAMB SOURCES, COSMOSIS], it supports the computation of large-scale structure main observables, including cosmic shear and galaxy clustering, using both photometric and spectroscopic redshifts. Yet, `cloelib` is the first and only large-scale structure code in the cosmology community to implement a unified interface to multiple cosmological backends using Python protocols [CITE PROTOCOLS]. This design enables researchers to seamlessly switch between different theoretical implementations—such as Boltzmann solvers or emulators—without modifying their analysis pipelines or the internal workings of `cloelib` for computing theoretical predictions. This level of modularity and interoperability is unprecedented, significantly lowering the barrier to the inclusion of other pipelines for rapid experimentation in cosmological analyses.

The library interfaces with several well-established Boltzmann solvers, including CAMB [@Lewis:2000], CLASS [@Blas:2011], and their extensions [CITE, e.g., EFTCAMB]. Moreoever, it interfaces with PBJ [CITE] for spectroscopic clustering. It also supports state-of-the-art emulators, such as HMCode2020Emu [@Mead:2021; CITE MARIA], COMET [@Eggemeier:2022; @Pezzotta:2025] for clustering observables with massive neutrinos, CosmoPower [@SpurioMancini:2021], BACCO [@Angulo:2020], Capse.jl [@Bonici2024Capse], and Effort.jl [@Bonici:2025]. These emulators offer orders-of-magnitude speedups in cosmological computations while maintaining percent-level accuracy, making them essential tools for modern inference pipelines. This modularity and performance make `cloelib` particularly well-suited for systematic studies, model comparison, and robust cross-validation of cosmological results.

Furthermore, a key innovation of `cloelib` is its native integration with JAX [@jax2018github], which enables automatic differentiation of cosmological observables. This transforms traditionally rigid, black-box cosmological pipelines into fully differentiable programs, unlocking entirely new capabilities for modern inference. In particular, it makes `cloelib` uniquely suited for gradient-based optimization, Hamiltonian Monte Carlo, and the seamless integration of neural networks into cosmological models. These methods are often infeasible with traditional software due to the lack of differentiability and performance constraints. `cloelib` has already been used to develop neural network emulators for cosmological observables and to prototype novel inference techniques that would be computationally prohibitive using existing tools.

The modular architecture of `cloelib` is built around a clear separation of concerns, organizing functionality into distinct components: cosmological backgrounds (e.g., background expansion and distances), perturbation theory (e.g., matter power spectra), observables (e.g., cosmic shear and galaxy clustering), and summary statistics (e.g., angular power spectra and correlation functions). This design allows researchers to flexibly mix and match different theoretical models and numerical approximations, supporting both standard analyses and experimental workflows. For example, users can compute angular power spectra using the Limber approximation with any combination of supported Boltzmann solvers and non-linear models, or define custom window functions for specific survey geometries.
Crucially, each module in `cloelib` is defined by a Python protocol, enabling a "plug-and-play" approach: users can include only the components they need, choose among interchangeable backends, and match them as desired—all without altering the core logic of their pipeline. This architecture enforces modularity by design, ensuring robustness, reusability, and ease of experimentation.

In addition, `cloelib` serves the practical needs of both the Euclid collaboration and the wider cosmology community by offering implementations of survey-specific systematics, Alcock–Paczynski corrections, and Baryon Acoustic Oscillation (BAO) feature extraction. The library works seamlessly with `cloelike`, its companion likelihood module, which supports the computation of likelihoods for Euclid observables such as 2×2pt and 3×2pt photometric correlations, spectroscopic galaxy clustering and BAO, as well as their combinations. Together, these tools enable end-to-end cosmological analyses, covering the full chain from observable computation to likelihood evaluation and Monte Carlo sampling for parameter inference.

Beyond its scientific scope, `cloelib` is optimized for efficiency, with native source code implementations of theoretical predictions and advanced caching mechanisms that accelerate computation rather than hinder it. It also integrates comprehensive testing infrastructure and performance profiling tools, aligned with state-of-the-art software development practices, to ensure reliability and scalability in production-level applications. By combining theoretical flexibility, computational performance, and modern programming standards with an Open Science ethos, `cloelib` makes a substantial contribution to the computational framework required for precision cosmology and is poised to become a cornerstone for large-scale structure analyses in the decade ahead.

# Design and Implementation

The architecture of `cloelib` leverages Python protocols (PEP 544) to define interfaces for cosmological calculations, ensuring type safety and extensibility. The library is organized into specialized modules: cosmology backends implementing the Background and Perturbations protocols, observables providing window functions and power spectrum interfaces, summary statistics for angular correlations and multipoles, and auxiliary utilities for mathematical operations and caching. Performance-critical sections utilize JAX's just-in-time compilation, while the caching system optimizes repeated calculations. The library seamlessly integrates with the broader Python scientific ecosystem through NumPy and SciPy, while maintaining compatibility with JAX arrays for differentiable computations. This design enables researchers to construct complex analysis pipelines while maintaining computational efficiency and code maintainability.

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
                         mnu=0.06, N_mnu=1, gamma_MG=0.545)

jax_bg = JAXBackground(H0=H0, Omega_cdm0=Omega_cdm0, Omega_b0=Omega_b0,
                       As=As, ns=ns, w0=w0, wa=wa, Omega_k0=0.0,
                       mnu=0.06, N_mnu=1, gamma_MG=0.545)

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
# PositionsTracer requires per-bin photo-z shifts and magnification bias
tracer_pos = PositionsTracer(
    perturbations=nonlinear_perturbations,
    dndz=my_dndz,
    z=z,
    galaxy_bias_model='poly',
    nuisance_params={
        'b1_photo_poly0': 1.2, 'b1_photo_poly1': 0.0,
        'b1_photo_poly2': 0.0, 'b1_photo_poly3': 0.0,
        'magnification_bias_1': 0.0,
        'dz_pos_1': 0.0,
    }
)

# ShearTracer requires intrinsic alignment (IA) and photo-z shift parameters
tracer_she = ShearTracer(
    perturbations=nonlinear_perturbations,
    dndz=my_dndz,
    z=z,
    nuisance_params={
        'AIA': 1.72, 'CIA': 0.0134, 'EtaIA': 0.0,
        'multiplicative_bias_1': 0.0,
        'dz_shear_1': 0.0,
    }
)

# Compute angular power spectra using the Limber approximation
twopoint = AngularTwoPoint(tracer_she, tracer_pos)
ells = np.logspace(1, np.log10(3000), 100)
Cl_galaxy_shear = twopoint.get_Cl(ells, nl=0, ks=ks)
```

## Spectroscopic Observables

`cloelib` computes redshift-space power spectrum multipoles for spectroscopic galaxy clustering via the `SpectroPower` protocol. The COMET emulator [@Eggemeier:2022; @Pezzotta:2025] is used here as an example of the two available perturbation-theory models (`EFT` and `VDG`); the PBJ emulator is also supported.

```python
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles
import numpy as np

# EFT bias and nuisance parameters for a single redshift bin
RSD_parameters = {
    'b1': 1.8, 'b2': 0.0, 'bG2': 0.0, 'bGam3': 0.0,
    'c0': 0.0, 'c2': 0.0, 'c4': 0.0,
    'b1-b1-cnlo': 0.0, 'b1-cnlo': 0.0, 'cnlo': 0.0,
}

# Spectroscopic power spectrum at a single effective redshift
spectro_power = CometEFT_SpectroPower(
    background=camb_bg,
    RSD_parameters=RSD_parameters,
    redshift=1.0,
)

# Compute Legendre multipoles with Alcock-Paczynski corrections
nbar = 1e-3  # galaxy number density [h/Mpc]^3
multipoles = LegendreMultipoles(
    spectro_power=spectro_power,
    background_fiducial=camb_bg,
    parameters={},
    nbar=nbar,
)

k = np.logspace(-2, np.log10(0.5), 80)
Pk_ell = multipoles.power_multipoles(k, ells=np.array([0, 2, 4]))
# Pk_ell is a dict: {'ell0': array, 'ell2': array, 'ell4': array}
```

The same interface is used to compute two-point correlation function multipoles via an FFTLog transform, and to apply survey window function convolutions.

## Protocol Compliance of Interfaces

`cloelib` uses Python structural subtyping (PEP 544 `Protocol`) to define callable interfaces for all cosmological backends. The `Background` and `Perturbations` protocols are declared with `@runtime_checkable`, enabling explicit compliance checks at the start of an analysis:

```python
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.cosmology.class_cosmology import CLASSBackground, CLASSLinearPerturbations

# Protocol compliance is verified at runtime
assert isinstance(camb_bg, Background)
assert isinstance(CLASSBackground(
    H0=70.0, Omega_b0=0.05, Omega_cdm0=0.25, Omega_k0=0.0,
    As=2e-9, ns=0.96, mnu=0.06, N_mnu=1, w0=-1.0, wa=0.0,
    gamma_MG=0.545
), Background)

# Both CAMB and CLASS objects satisfy the same Perturbations protocol
camb_lin = CAMBLinearPerturbations(background=camb_bg, redshifts=z)
class_lin = CLASSLinearPerturbations(background=CLASSBackground(...), redshifts=z)

assert isinstance(camb_lin, Perturbations)
assert isinstance(class_lin, Perturbations)
```

Because both objects conform to the same protocol, any downstream `cloelib` computation—such as window functions, angular power spectra, or multipoles—can operate on either without requiring modification. This structural approach, rather than relying on inheritance hierarchies, enables the seamless integration of external codes without altering their source. As a result, it provides a straightforward pathway for the community to connect their own tools, provided they adhere to the protocol. In particular, the `Background` and `Perturbations` protocols are compliant with the cosmology.API [CITE].

## Automatic Differentiation with JAX

One of `cloelib`'s unique features is its support for automatic differentiation:

```python
import jax
import jax.numpy as jnp
from cloelib.cosmology.jax_cosmology import JAXBackground

def compute_observable(params):
    """Compute the angular diameter distance at z=1 from (H0, Omega_m)."""
    H0, Omega_m = params
    bg = JAXBackground(
        H0=H0, Omega_cdm0=Omega_m - 0.05, Omega_b0=0.05,
        As=2e-9, ns=0.96, w0=-1.0, wa=0.0, Omega_k0=0.0,
        mnu=0.0, N_mnu=0, gamma_MG=0.545,
    )
    return bg.angular_diameter_distance(jnp.array([1.0]))[0]

# Compute gradients with automatic differentiation
grad_fn = jax.grad(compute_observable)
gradients = grad_fn(jnp.array([70.0, 0.3]))
# gradients[0] = dD_A/dH0,  gradients[1] = dD_A/dOmega_m
```

The `JAXBackground` and `JAXLinearPerturbations`/`JAXNonLinearPerturbations` classes are fully JIT-compilable and differentiable through `jax.grad`, `jax.jacobian`, and `jax.hessian`. This enables Hamiltonian Monte Carlo samplers, variational inference, and the training of neural-network emulators whose inputs are cosmological parameters.

## Scaling tests and time performance

| Backend / Emulator | Observable Type | Runtime (s) | Memory Usage (MB) |
| ------------------ | --------------- | ----------- | ----------------- |
| CAMB               | Cell            |             |                   |
| CAMB               | $\xi$           |             |                   |
| CLASS              | Cell            |             |                   |
| HMCode2020Emu      | Cell            |             |                   |
| COMET              | Pell            |             |                   |
| CosmoPower         | Cell            |             |                   |

## Author Contributions

In accordance with JOSS guidelines, we describe individual contributions below. Authors are listed in alphabetical order. All Tier 1 authors are core maintainers of the **cloe-org** organisation, responsible for the long-term sustainability of `cloelib`, the review of pull requests, and leadership of technical discussions.

- **M. Bonici**: TBA
- **G. Cañas-Herrera**: TBA
- **P. Carrilho**: TBA
- **S. Casas**: TBA
- **C. Moretti**: TBA
- **A. Pezzotta**: TBA

The contributions of all remaining authors have been tracked using the [all-contributors](https://github.com/all-contributors/all-contributors) bot, following the specification of the same name. A full, categorised breakdown of each contributor's role—including code, documentation, testing, ideas, project management, and more—is available in the `README` of the `cloelib` repository.

# Acknowledgements

We acknowledge the support of the Euclid Consortium. We thank the broader CLOE software development team for foundational work that motivated this library. M.B. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC). acknowledges that this project is part of the project UNICORN with file number VI.Veni.242.110 of the research programme Talent Programme Veni Science domain 2024 which is (partly) financed by the Dutch Research Council (NWO) under the grant https://doi.org/10.61686/ZCPQI32997.

# References
