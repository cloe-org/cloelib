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
    orcid: 0000-0003-1339-0194
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
  - name: on behalf of the Euclid Consortium
affiliations:
  - name: University of Waterloo, Canada
    index: 1
  - name: Leiden Observatory, the Netherlands
    index: 2
  - name: Centre for Astrophysics Research, University of Hertfordshire, United Kingdom
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
date: 9 April 2026
bibliography: paper.bib
---

# Summary

\texttt{cloelib} is a Python library developed to compute cosmological observables within the Cosmology Likelihood for Observables in Euclid (\texttt{CLOE}) project\footnote{\href{https://github.com/cloe-org}{https://github.com/cloe-org}}. As cosmology enters a precision era driven by galaxy survey missions such as _Euclid_, there is a growing need for flexible, efficient, and differentiable software capable of supporting next-generation inference pipelines. \texttt{cloelib} addresses these demands through a modular architecture that interfaces seamlessly with established Boltzmann solvers whilst incorporating JAX-based automatic differentiation to enable gradient-based methods. The library defines consistent protocols for background evolution, perturbations, and non-linear structure formation, and supports a wide range of observables, including photometric and spectroscopic large-scale structure probes, as well as cross-correlations with the Cosmic Microwave Background and galaxy clusters. In its finalised form, \texttt{cloelib} is intended to serve as the reference theory computation infrastructure for Euclid's first cosmological release, bridging traditional numerical cosmology with modern optimisation techniques and emerging machine learning approaches to inference.

# Statement of need

The field of observational cosmology is undergoing a rapid transformation, driven by the advent of next-generation galaxy surveys such as the European Space Agency’s _Euclid_ mission [@Euclid:2024], the Dark Energy Spectroscopic Instrument (DESI) [@DESI_review], the \textit{Vera C. Rubin} Observatory’s Large Synoptic Survey Telescope [@LSST], and NASA’s \textit{Nancy Grace Roman} Space Telescope\footnote{\href{https://roman.gsfc.nasa.gov/science/ccs/ROTAC-Report-20250424-v1.pdf}{https://roman.gsfc.nasa.gov/science/ccs/ROTAC-Report-20250424-v1.pdf}}. These projects will generate vast volumes of high-quality data, mapping the large-scale structure of the Universe with unprecedented precision. Extracting scientific results from these data requires efficient computation of theoretical predictions that can be robustly compared with observations to constrain cosmological models. Meeting this challenge places stringent demands on computational tools, which must accurately model complex theoretical scenarios whilst remaining computationally efficient. Despite their strengths, existing cosmological software frameworks often lack the flexibility needed to seamlessly integrate diverse, pre-existing components and to explore a wide range of theoretical models alongside comprehensive treatments of systematic effects. This is the limitation that \texttt{cloelib} addresses by offering a highly flexible and extensible platform for cosmological inference, designed to meet the challenges posed by the next generation of precision cosmology experiments.

In this context, \texttt{cloelib} represents a natural evolution of the structural formalism originally developed in the Cosmology Likelihood for Observables in Euclid (\texttt{CLOE}) software, extending it towards more advanced use cases and significantly enhanced capabilities beyond those presented in [@EP-CLOE2]. The original \texttt{CLOE}\footnote{\href{https://github.com/cloe-org/CLOE}{https://github.com/cloe-org/CLOE}} has played a central role in numerous Euclid analyses—see [@Euclid:2024], [@EP-CLOE3], [@EP-CLOE4], [@EP-CLOE5], [@EP-CLOE6]—demonstrating its robustness and scientific impact. However, the increasing complexity, scale, and methodological demands of next-generation cosmological analyses, as well as the possible combination of all these datasets, have exposed structural limitations in its original design. Notably, \texttt{CLOE} was not conceived with the level of modularity, extensibility, and interoperability now required to efficiently address the broader landscape of theoretical models and systematic effects demanded by the incoming datasets. As a result, a substantial restructuring became necessary to meet these new challenges. \texttt{cloelib} builds directly on the conceptual and practical foundations laid by \texttt{CLOE}, whilst introducing a redesigned architecture that enables greater flexibility, scalability, and integration of heterogeneous components. In doing so, it provides a forward-looking framework tailored to the demands of next-generation precision cosmology.

Similarly to \texttt{CCL} [@pyccl], \texttt{CosmoSIS} [@CosmoSIS], \texttt{CAMB} [@Lewis:2000], \texttt{CLASS} [@Blas:2011], CosmoLike [@CosmoLike], and \texttt{CoCoA}\footnote{\href{https://github.com/CosmoLike/cocoa}{https://github.com/CosmoLike/cocoa}}, it supports the computation of large-scale structure probes, including cosmic shear and galaxy clustering, using both photometric and spectroscopic redshifts. Yet, \texttt{cloelib} is the first and only large-scale structure code in the cosmology community to implement a unified interface to multiple cosmological backends using Python protocols. This design enables researchers to seamlessly switch between different theoretical implementations—such as Boltzmann solvers or emulators—without modifying their analysis pipelines or the internal workings of \texttt{cloelib} for computing theoretical predictions. This level of modularity and interoperability is unprecedented, significantly lowering the barrier to the inclusion of other pipelines for rapid experimentation in cosmological analyses.

Within this protocol-based framework, the library interfaces with several well-established Boltzmann solvers, including CAMB, CLASS, their extensions (e.g., \texttt{hi\_class} [@hi_class_1] [@hi_class_2], \texttt{mgclass} [@mgclass], \texttt{mochi\_class} [@mochi_class]), and other non-linear model extensions emulators (i.e: \texttt{ReACT} [@ReACT]). Moreover, it interfaces with \texttt{PBJ} and \texttt{comet-emu} [@Eggemeier:2022] [@Pezzotta:2025] for nonlinear spectroscopic galaxy clustering. It also supports state-of-the-art emulators, such as \texttt{CosmoPower} [@SpurioMancini:2021], \texttt{BACCOemu} [@Angulo:2020] [@bacco-original] [@bacco-full-power] [@bacco-emu-baryons] [@bacco-euclid], \texttt{EuclidEmulator2} [cite], and \texttt{HMCode2020Emu} [@Mead:2021] [@Tsedrik2024]. These emulators offer orders-of-magnitude speed-ups in cosmological computations whilst maintaining percent-level accuracy, making them essential tools for modern inference pipelines. This modularity and performance make \texttt{cloelib} particularly well-suited for systematic studies, model comparison, and robust cross-validation of cosmological results.

A key innovation of \texttt{cloelib} is its native integration with JAX [@jax2018github], which provides automatic differentiation for cosmological observables and supports efficient, gradient-based computation. This transforms conventional cosmological pipelines into fully differentiable programmes, making advanced inference techniques—such as Hamiltonian Monte Carlo and neural network-based modelling—readily accessible. Whilst such methods are often difficult to implement efficiently in traditional frameworks, \texttt{cloelib} is designed to facilitate these workflows, offering a robust and flexible platform for developing neural network emulators and exploring new inference methodologies.

In addition, \texttt{cloelib} serves the practical needs of both the Euclid collaboration and the wider cosmology community by offering implementations of survey-specific systematics, Alcock–Paczynski corrections, and Baryon Acoustic Oscillation (BAO) feature extraction. The library works seamlessly with `cloelike`, its companion likelihood module, which supports the computation of likelihoods for Euclid observables such as cosmic shear, 2×2pt, and 3×2pt photometric correlations, spectroscopic galaxy clustering and BAO, as well as their combinations [TBD]. Together, these tools enable end-to-end cosmological analyses, covering the full chain from observable computation to likelihood evaluation and posterior sampling for parameter inference.

Beyond its scientific scope, \texttt{cloelib} is optimised for efficiency, with native source code implementations of theoretical predictions and advanced just-in-time (`jit`) caching mechanisms that accelerate computation rather than hinder it. It also integrates comprehensive testing infrastructure and performance profiling tools, aligned with state-of-the-art software development practices, to ensure reliability and scalability in production-level applications. By combining theoretical flexibility, computational performance, and modern programming standards with an Open Science ethos, \texttt{cloelib} makes a substantial contribution to the computational framework required for precision cosmology and is poised to become a cornerstone for large-scale structure analyses in the decade ahead.

# Design style, architecture and implementation

The architecture of \texttt{cloelib} is built around a clear separation of concerns, organizing functionality into four distinct layers: cosmological backgrounds (e.g., expansion history and distances), perturbation theory (e.g., linear and non-linear matter power spectra), observables (e.g., cosmic shear and galaxy clustering window functions), and summary statistics (e.g., angular power spectra and correlation functions). This layered design allows researchers to flexibly mix and match different theoretical models and numerical approximations, supporting both standard analyses and experimental workflows. For example, users can compute angular power spectra using the Limber approximation with any combination of supported Boltzmann solvers and non-linear models, or define custom window functions for specific survey geometries.

Each layer is defined by a Python protocol (PEP 544), enforcing a "plug-and-play" approach to modularity. Concretely, the library is organized into specialized modules: cosmology backends implementing the `Background` and `Perturbations` protocols, observable modules providing window functions and power spectrum interfaces through the `Tracer` and `SpectroPower` protocols, summary statistics for angular correlations and Legendre multipoles, and auxiliary utilities for mathematical operations and caching. Performance-critical sections utilize JAX's just-in-time compilation, while the caching system avoids redundant evaluations across repeated calculations. In this sense, `Background`, `Perturbations`, `Tracer`, and `SpectroPower` are structural interfaces that guarantee type safety and extensibility without relying on inheritance hierarchies. Users can include only the components they need, choose among interchangeable backends, and combine them freely—all without altering the core logic of their pipeline. This architecture ensures robustness, reusability, and ease of experimentation by design.

The library integrates with the broader Python scientific ecosystem through NumPy and SciPy, while maintaining full compatibility with JAX arrays for differentiable computations. This enables researchers to construct complex, end-to-end analysis pipelines that are simultaneously computationally efficient, maintainable, and—where needed—fully differentiable.

# Usage Examples

The power of \texttt{cloelib} lies in its intuitive API that allows researchers to quickly set up complex cosmological calculations using this "plug-and-play" approach. Here we demonstrate key features through practical examples.

## Initializing Cosmological Models

\texttt{cloelib} provides a consistent interface for different cosmological backends. Users can instantiate cosmological models using standard parameters:

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

\texttt{cloelib} excels at computing observables for photometric surveys, including galaxy clustering and cosmic shear including state-of-the art modelling of systematics:

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

\texttt{cloelib} computes redshift-space power spectrum multipoles for spectroscopic galaxy clustering via the `SpectroPower` protocol. In this example, we use `comet-emu` for producting perturbation-theory models (`EFT` and `VDG`):

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

\texttt{cloelib} natively supports Python structural subtyping (PEP 544); the `Background` and `Perturbations` protocols are marked with @runtime_checkable, allowing explicit compliance checks at the beginning of an analysis.

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

Because both objects conform to the same protocol, any downstream \texttt{cloelib} computation—such as window functions, angular power spectra, or multipoles—can operate on either without requiring modification. This structural approach, rather than relying on inheritance hierarchies, enables the seamless integration of external codes without altering their source. As a result, it provides a straightforward pathway for the community to connect their own tools, provided they adhere to the protocol. In particular, the `Background` and `Perturbations` protocols are compliant with the cosmology.API\footnote{\href{https://cosmology.readthedocs.io/projects/api/latest/}{https://cosmology.readthedocs.io/projects/api/latest/}}.

## Automatic Differentiation with JAX

One of \texttt{cloelib}'s unique features is its support for automatic differentiation:

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

## Computational times

We provide below

| Backend / Emulator | Observable Type | Runtime (s) | Memory Usage (MB) |
| ------------------ | --------------- | ----------- | ----------------- |
| CAMB               | Cell            |             |                   |
| CAMB               | $\xi$           |             |                   |
| CLASS              | Cell            |             |                   |
| HMCode2020Emu      | Cell            |             |                   |
| COMET              | Pell            |             |                   |
| CosmoPower         | Cell            |             |                   |

## Performance Profiling

\texttt{cloelib} includes function-level profiling via the `@profile_function` decorator, configurable sampling (default interval: 0.001 s) to balance overhead and granularity, and timestamped interactive HTML reports for run-to-run comparison. Profiling can be controlled through environment variables or function calls (including enable/disable and output configuration), and it includes safeguards to avoid redundant profiling in nested decorated calls.

**Usage example:**

```python
from cloelib.profiling import enable_profiling, disable_profiling, profile_function

enable_profiling()
set_output("./my_profiles")

@profile_function
def compute_observables(cosmology):
    # Your computation here
    pass

compute_observables(cosmo)  # Generates profiling_results/cloelib_compute_observables_YYYYMMDD_HHMMSS.html
disable_profiling()
```

This lightweight profiling infrastructure allows users to optimize their analysis pipelines by understanding where computational time is spent across different backends and observable calculations.

# Author Contributions

In accordance with JOSS guidelines, we describe individual contributions below. Authors are listed in alphabetical order. All Tier 1 authors are core maintainers of the **cloe-org** organisation, responsible for the long-term sustainability of \texttt{cloelib}, the review of pull requests, and leadership of technical discussions.

- **M. Bonici**: Core architecture and protocol design; implementation of the JAX cosmology backends; lensing tracer kernels (including massive neutrino contributions); correlation function module and performance optimisation; caching system with JAX `lax` conditional compatibility; license and project governance.
- **G. Cañas-Herrera**: Project overview and release management; Core architecture of the software and protocol design; CI pipeline configuration; pre-commit and code-quality tooling; issue and pull-request templates; README, documentation, and community contribution tracking (`all-contributors`); pyproject.toml versioning and release workflows.
- **P. Carrilho**: Photometric observable module linear galaxy bias models with JAX-compatible conditional logic; HMCode2020Emu baryonic feedback support and further extrapolation support; CAMB dark-energy model configuration (PPF); mixing-matrix and pseudo-$C_\ell$ corrections; `interpax`-based interpolation in the extrapolator; growth-rate and matter power spectrum redshift/scale interfaces, implementation of EuclidEmulator2.
- **S. Casas**: Implementation of the CLASS cosmology backend and its integration with the `Background` and `Perturbations` protocols; fixes to transverse-distance computations across CAMB, CLASS, and JAX backends; cosmology protocol refinements; CI pipeline and dependency updates.
- **C. Moretti**: Spectroscopic analysis infrastructure: PBJ interface and RSD power spectrum fixes; BAO $\alpha$-parameter module and Alcock–Paczynski distortion utilities; extraction of $r_\mathrm{drag}$ from the background for BAO analyses; Legendre multipole summary statistics; version management and repository clean-up of deprecated directories.
- **A. Pezzotta**: Spectroscopic analysis infrastructure: \texttt{comet-emu} interface and EFT and VDG spectroscopic power spectrum implementations; survey window-function convolution of power spectrum building blocks; Legendre multipole computation optimisation (`np.einsum`); documentation of the spectroscopic observable interface.

The contributions of all remaining authors have been tracked using the [all-contributors](https://github.com/all-contributors/all-contributors) bot, following the specification of the same name. A full, categorised breakdown of each contributor's role—including code, documentation, testing, ideas, project management, and more—is available in the `README` of the \texttt{cloelib} repository, full portraited within the \texttt{cloelib} docs.

# Acknowledgements

We acknowledge the support of the Euclid Consortium, including its provision of scientific coordination, data access, and computational infrastructure essential for this work. We thank the broader CLOE software development team for foundational work that motivated this library. GCH acknowledges that this project is part of the project UNICORN with file number VI.Veni.242.110 of the research programme Talent Programme Veni Science domain 2024 which is (partly) financed by the Dutch Research Council (NWO) under the grant https://doi.org/10.61686/ZCPQI32997. M.B. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC). We acknowledge EuroHPC Joint Undertaking for awarding the project ID EHPC-EXT-2024E02-083 access to Leonardo hosted by CINECA, Italy.

# References
