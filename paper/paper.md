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
    orcid: 0000-0002-4751-5138
    affiliation: 4

  - name: Chiara Moretti
    orcid: 0000-0003-3314-8936
    affiliation: 5

  - name: Andrea Pezzotta
    orcid: 0000-0003-0726-2268
    affiliation: 6

  - name: Michel Aguena
    orcid: 0000-0001-5679-6747
    affiliation: 5

  - name: Giovanni Arico
    orcid: 0000-0002-2802-2928
    affiliation: 7

  - name: Zahra Baghkhani
    orcid: 0000-0002-6632-2614
    affiliation: 8

  - name: Matteo Baratto
    orcid: 0009-0000-8702-9591
    affiliation: [9, 10]

  - name: Emilio Bellini
    orcid: 0000-0003-4762-0795
    affiliation: 11

  - name: Jip de Buck
    orcid: 0000-0000-0000-0000
    affiliation: 2

  - name: Klara Bertmann
    orcid: 0009-0004-6700-2470
    affiliation: 12

  - name: Ben Bose
    orcid: 0000-0003-1965-8614
    affiliation: [13, 14]

  - name: Jeger C. Broxterman
    orcid: 0000-0002-8155-5977
    affiliation: [2, 15]

  - name: Pierre Burger
    orcid: 0000-0002-6374-5208
    affiliation: 1

  - name: Carmelita Carbone
    orcid: 0000-0003-0125-3563
    affiliation: 10

  - name: Chaitanya Chawak
    affiliation: 16

  - name: Jose Coloma-Nadal
    orcid: 0009-0003-0538-4349
    affiliation: 8

  - name: Martin Crocce
    orcid: 0000-0002-9745-6228
    affiliation: 8

  - name: Stefano Davini
    orcid: 0000-0003-3269-1718
    affiliation: 17

  - name: Christopher A. J. Duncan
    affiliation: 13

  - name: Samuel Farrens
    orcid: 0000-0002-9594-9387
    affiliation: 10

  - name: Lisa Goh
    orcid: 0000-0002-0104-8132
    affiliation: [13, 14]

  - name: Nastassia Grim
    orcid: 0000-0001-9602-0599
    affiliation: 18

  - name: Alex Hall
    orcid: 0000-0002-3139-8651
    affiliation: 13

  - name: Ryusei R. Kano
    orcid: 0009-0002-9108-8396
    affiliation: [10, 19]

  - name: Felicitas Keil
    orcid: 0000-0002-8108-1679
    affiliation: 20

  - name: Raphaël Kou
    orcid: 0000-0003-3408-3062
    affiliation: 21

  - name: Laila Linke
    orcid: 0000-0002-2622-8113
    affiliation: 22

  - name: Arthur Loureiro
    orcid: 0000-0002-4371-0876
    affiliation: 22, 23

  - name: Katarina Markovic
    orcid: 0000-0001-6764-073X
    affiliation: 24

  - name: David Navarro-Gironés
    orcid: 0000-0003-0507-372X
    affiliation: 2

  - name: Filippo Oppizzi
    orcid: 0000-0003-3904-8370
    affiliation: 17

  - name: Gabriele Parimbelli
    orcid: 0000-0002-2539-2472
    affiliation: 8

  - name: Anna Porredon
    orcid: 0000-0002-2762-2024
    affiliation: 26

  - name: Robert Reischke
    orcid: 0000-0001-5404-8753
    affiliation: 27

  - name: Jaime Ruiz Zapatero
    orcid: 0000-0002-7951-4391
    affiliation: 28

  - name: Iñigo Sáez-Casares
    orcid: 0000-0003-0013-5266
    affiliation: 6

  - name: Ziad Sakr
    orcid: 0000-0002-4823-3757
    affiliation: 29

  - name: Neel Shah
    orcid: 0009-0001-4424-6489
    affiliation: 30

  - name: Davide Sciotti
    orcid: 0009-0008-4519-2620
    affiliation: 31

  - name: Matthieu Schaller
    orcid: 0000-0002-2395-4902
    affiliation: [15, 2]

  - name: Ivan Sladoljev
    orcid: 0009-0002-9702-2101
    affiliation: 32

  - name: Arghavan Souki
    orcid: 0009-0000-4771-7728
    affiliation: 26

  - name: Konstantinos Tanidis
    orcid: 0000-0001-9843-5130
    affiliation: 33

  - name: Peter L. Taylor
    orcid: 0000-0001-6999-4718
    affiliation: 34

  - name: Nicolas Tessore
    orcid: 0000-0002-9696-7931
    affiliation: 35

  - name: Linus Thummel
    orcid: 0000-0002-9807-5494
    affiliation: [13, 14]

  - name: Maria Tsedrik
    orcid: 0000-0002-0020-5343
    affiliation: [13, 14]

  - name: Isaac Tutusaus
    orcid: 0000-0002-3199-0399
    affiliation: [15, 38, 19]

  - name: Casper Vedder
    orcid: 0009-0007-6341-4648
    affiliation: 2

  - name: Angus H. Wright
    orcid: 0000-0001-7363-7932
    affiliation: 12

  - name: Miguel Zumalacarregui
    orcid: 0000-0002-9943-6490
    affiliation: 37

  - name: Joe Zuntz
    orcid: 0000-0001-9789-9646
    affiliation: 13
    note: on behalf of the Euclid Consortium

affiliations:
  - name: Waterloo Centre for Astrophysics, University of Waterloo, Waterloo, ON N2L 3G1, Canada
    index: 1
  - name: Leiden Observatory, Leiden University, PO Box 9513, 2300 RA, Leiden, the Netherlands
    index: 2
  - name: Centre for Astrophysics Research, University of Hertfordshire, United Kingdom
    index: 3
  - name: German Aerospace Center (DLR), Scientific Information, Linder Höhe, D-51147 Köln, Germany
    index: 4
  - name: INAF - Osservatorio Astronomico di Trieste, Italy
    index: 5
  - name: INAF - Osservatorio Astronomico di Brera, Italy
    index: 6
  - name: INFN - Sezione di Bologna, Viale C. Berti Pichat, 6/2 – 40127 Bologna, Italy
    index: 7
  - name: TBD
    index: 8
  - name: Departement of Physics “Aldo Pontremoli”, Università degli Studi di Milano, Via G. Celoria 16, 20133, Milano, Italy
    index: 9
  - name: Division of Particle and Astrophysical Science, Nagoya University, Furo-cho, Chikusa-ku, Nagoya, 464–8602, Japan
    index: 10
  - name: Lorentz Institute for Theoretical Physics, Leiden University, PO Box 9506, NL-2300 RA Leiden, The Netherlands
    index: 11
  - name: TBD
    index: 12
  - name: INAF - Institute of Space Astrophysics and Cosmic Physics (IASF Milano), Via Corti 12, I-20133 Milano (MI), Italy
    index: 13
  - name: TBD
    index: 14
  - name: Institute of Space Sciences (ICE, CSIC), Campus UAB, Carrer de Can Magrans, s/n, 08193 Barcelona, Spain
    index: 15
  - name: TBD
    index: 16
  - name: INFN-Sezione di Genova, Via Dodecaneso 33, 16146, Genova, Italy
    index: 17
  - name: TBD
    index: 18
  - name: Institut de Recherche en Astrophysique et Planétologie (IRAP), Université de Toulouse, CNRS, UPS, CNES, 14 Av. Edouard Belin, 31400 Toulouse, France
    index: 19
  - name: Department of Physics & Astronomy, University of Sussex, Brighton BN1 9QH, UK
    index: 20
  - name: Universität Innsbruck, Institut für Astro- und Teilchenphysik, Technikerstr. 25/8, 6020 Innsbruck, Austria
    index: 21
  - name: Oskar Klein Centre for Cosmoparticle Physics, Department of Physics, Stockholm University, Stockholm, SE-106 91, Sweden
    index: 22
  - name: Astrophysics Group, Blackett Laboratory, Imperial College London, London SW7 2AZ, UK
    index: 23
  - name: Jet Propulsion Laboratory, California Institute of Technology, 4800 Oak Grove Drive, Pasadena, CA, 91109, USA
    index: 24
  - name: INFN, Sezione di Genova, Via Dodecaneso 33, 16146, Genova, Italy
    index: 25
  - name: TBD
    index: 26
  - name: Argelander-Institut für Astronomie, Universität Bonn, Auf dem Hügel 71, D-53121 Bonn, Germany
    index: 27
  - name: TBD
    index: 28
  - name: Advanced Research Computing Centre, University College London, 90 High Holborn, London WC1V 6LJ, UK
    index: 29
  - name: TBD
    index: 30
  - name: TBD
    index: 31
  - name: Department of Physics, Royal Holloway, University of London, Egham Hill, Egham, UK
    index: 32
  - name: Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas (CIEMAT), Avenida Complutense 40, 28040 Madrid, Spain
    index: 33
  - name: Center for Astrophysics and Cosmology, University of Nova Gorica, 1280 Nova Gorica, Slovenia
    index: 34
  - name: Center for Cosmology and AstroParticle Physics (CCAPP),The Ohio State University, Columbus, OH 43210, USA
    index: 35
  - name: TBD
    index: 36
  - name: TBD
    index: 37
  - name: Institut d'Estudis Espacials de Catalunya (IEEC),  Edifici RDIT, Campus UPC, 08860 Castelldefels, Barcelona, Spain
    index: 38
  - name: TBD
    index: 39
  - name: TBD
    index: 40
  - name: TBD
    index: 41
  - name: Ruhr University Bochum, Faculty of Physics and Astronomy, Astronomical Institute (AIRUB), German Centre for Cosmological Lensing, 44780 Bochum, Germany
    index: 42
  - name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Am Mühlenberg 1, D-14476 Potsdam-Golm, Germany
    index: 43
  - name: Institute for Astronomy, University of Edinburgh, Royal Observatory, Blackford Hill, Edinburgh, EH9 3HJ, UK
    index: 44
  - name: Higgs Centre for Theoretical Physics, School of Physics and Astronomy, Edinburgh, EH9 3FD, UK
    index: 45

date: 9 April 2026
bibliography: paper.bib
---

# Summary

\texttt{cloelib}, available at \href{https://github.com/cloe-org/cloelib}{cloe-org/cloelib}, is a Python library developed to compute cosmological observables within the Cosmology Likelihood for Observables in Euclid (\texttt{CLOE}) project, hosted by \href{https://github.com/cloe-org}{\textbf{cloe-org}}\footnote{\url{https://github.com/cloe-org}}. As cosmology enters a precision era driven by galaxy survey missions such as \emph{Euclid}, there is a growing need for flexible, efficient, and differentiable software capable of supporting next-generation inference pipelines. \texttt{cloelib} addresses these demands through a modular architecture that interfaces seamlessly with established Boltzmann solvers while incorporating JAX-based automatic differentiation to enable gradient-based methods. The library defines consistent protocols for background evolution, perturbations, and non-linear structure formation, and supports a wide range of observables, including photometric and spectroscopic large-scale structure probes, as well as cross-correlations with the Cosmic Microwave Background and galaxy clusters. In its finalised form, \texttt{cloelib} is intended to serve as the reference theory computation infrastructure for Euclid's first cosmological release, bridging traditional numerical cosmology with modern optimisation techniques and emerging machine learning approaches to inference.

# Statement of need

The field of observational cosmology is undergoing a rapid transformation, driven by the advent of next-generation galaxy surveys such as the European Space Agency's *Euclid* mission [Euclid:2024], the Dark Energy Spectroscopic Instrument (DESI; [DESI_review]), the *Vera C. Rubin* Observatory's Large Synoptic Survey Telescope [LSST], and NASA's *Nancy Grace Roman* Space Telescope ([link](https://roman.gsfc.nasa.gov/science/ccs/ROTAC-Report-20250424-v1.pdf)). These projects are generating vast volumes of high-quality data, mapping the large-scale structure of the Universe with unprecedented precision. Extracting scientific results from these data requires efficient computation of theoretical predictions that can be robustly compared with observations to constrain cosmological models. Meeting this challenge places stringent demands on computational tools, which must accurately model complex theoretical scenarios whilst remaining computationally efficient. Despite their strengths, existing cosmological software frameworks often lack the flexibility needed to seamlessly integrate diverse, pre-existing components and to explore a wide range of theoretical models alongside comprehensive treatments of systematic effects. \texttt{cloelib} addresses this limitation by offering a highly flexible and extensible platform for cosmological inference, designed to meet the challenges posed by the next generation of precision cosmology experiments.

In this context, \texttt{cloelib} represents a natural evolution of the structural formalism originally developed in the Cosmology Likelihood for Observables in Euclid (\texttt{CLOE}) software, extending it towards more advanced use cases and significantly enhanced capabilities beyond those presented in Euclid Collaboration: [EP-CLOE2]. The original \texttt{CLOE} ([link](https://github.com/cloe-org/CLOE)) software has played a central role in numerous Euclid analyses—see [Euclid:2024], [EP-CLOE3], [EP-CLOE4], [EP-CLOE5], and [EP-CLOE6]—demonstrating its robustness and scientific impact to forecast and validate *Euclid* scientific performance. However, the increasing complexity, scale, and methodological demands of next-generation cosmological analyses, as well as the possible combination of all these datasets, have exposed structural limitations in its original design. Notably, \texttt{CLOE} was not conceived with the level of modularity, extensibility, and interoperability now required to efficiently address the broader landscape of theoretical models and systematic effects demanded by the incoming datasets. As a result, a substantial restructuring became necessary to meet these new challenges. \texttt{cloelib} builds directly on the conceptual and practical foundations laid by \texttt{CLOE}, whilst introducing a redesigned novel architecture that enables greater flexibility, scalability, and integration of heterogeneous components. In doing so, it offers a framework designed to address the requirements of upcoming precision cosmology analyses.

# State of the Field

Similarly to \texttt{CCL} [pyccl], \texttt{CosmoSIS} [CosmoSIS], \texttt{CAMB} [Lewis:2000], \texttt{CLASS} [Blas:2011], CosmoLike [CosmoLike], and \texttt{CoCoA} ([link](https://github.com/CosmoLike/cocoa)), it supports the computation of large-scale structure probes in the form of angular or spatial two-point correlations, and associated observables such as cosmic shear (and cross-correlations with the cosmic microwave background), galaxy clustering or spectroscopic power spectrum multipoles. Yet, \texttt{cloelib} is the first and only large-scale structure code in the cosmology community to implement a unified interface to multiple cosmological codes using Python protocols. This design enables researchers to seamlessly switch between different theoretical implementations (backends)—such as Boltzmann solvers or emulators—without modifying their analysis pipelines or the internal workings of \texttt{cloelib} for computing theoretical predictions. This modularity and interoperability facilitate the inclusion of additional pipelines, supporting rapid experimentation in cosmological analyses. In fact, following this protocol-based framework, \texttt{cloelib} already interfaces with several well-established cosmological codes in the community.

Among these tools, for the computation of background quantities and the matter power spectrum, \texttt{cloelib} provides interfaces to widely used Boltzmann solvers such as \texttt{CAMB} and \texttt{CLASS}, as well as their extensions (e.g. \texttt{hi\_class} [hi_class_1, hi_class_2], \texttt{mgclass II} [Sakr_2022], and \texttt{mochi\_class} [mochi_class]).

To accelerate matter power spectrum evaluations, \texttt{cloelib} also supports a range of state-of-the-art emulators, including \texttt{CosmoPower} [SpurioMancini:2021, Piras23], \texttt{BACCOemu} [bacco-full-power, bacco-original, bacco-tracers, bacco-euclid, bacco-emu-baryons], [\texttt{EuclidEmulator2}](https://github.com/PedroCarrilho/EuclidEmulator2/tree/pywrapper) [EE2], and \texttt{HMCode2020Emu} [Mead:2021, Tsedrik2024], as well as nonlinear model extensions such as \texttt{ReACT} [ReACT] and baryonic effects through the \texttt{FlamingoBaryonResponseEmulator} [flamingo-emu]. These emulators provide orders-of-magnitude speed-ups while retaining percent-level accuracy, making them well suited for modern cosmological inference pipelines. To compute spectroscopic observables, \texttt{cloelib} interfaces with the state-of-the-art theory code \texttt{PBJ} and the emulator \texttt{comet-emu} [Eggemeier:2022, Pezzotta:2025] for fast computation of nonlinear spectroscopic galaxy clustering. This modular design, combined with its computational performance, makes \texttt{cloelib} particularly well suited for systematic studies, model comparison, and robust cross-validation of cosmological results.

A key feature of \texttt{cloelib} is its native integration with JAX [jax2018github], enabling automatic differentiation for cosmological observables, efficient gradient-based computation, and advanced just-in-time (\texttt{jit}) compilation. This transforms conventional cosmological pipelines into fully differentiable programmes, making advanced inference techniques—such as Hamiltonian Monte Carlo and neural network-based modelling—readily accessible. Whilst such methods are often difficult to implement efficiently in traditional frameworks, \texttt{cloelib} is designed to facilitate these workflows, offering a robust and flexible platform for developing neural network emulators and exploring new inference methodologies.

In addition, \texttt{cloelib} serves the practical needs of both the Euclid Collaboration and the wider cosmology community by offering implementations of survey-specific systematics, such as Alcock–Paczynski corrections, shear and photometric redshift calibration parameters, and spectroscopic purity in surveys. The library works seamlessly with \texttt{cloelike}, its companion likelihood module, which supports the computation of likelihoods for Euclid observables such as cosmic shear, 2x2-pt, and 3x2-pt photometric analyses, spectroscopic galaxy clustering and BAO, as well as their combinations. Together, these tools enable end-to-end cosmological analyses, covering the full chain from observable computation to likelihood evaluation and posterior sampling for parameter inference.

Beyond its scientific scope, \texttt{cloelib} is designed for efficiency and consistency in cosmological applications, featuring native source code implementations of theoretical predictions and \texttt{jit} caching mechanisms that accelerate the computation of otherwise expensive integrals. The library is structured to support portability and reproducibility across different environments, facilitating tasks such as running inference chains and integrating with broader analysis pipelines. It also incorporates testing infrastructure and performance profiling tools to ensure robustness and scalability. By combining theoretical flexibility, computational performance, and modern programming practices within an Open Science framework, \texttt{cloelib} contributes to the computational toolkit for precision cosmology and is well suited for large-scale structure analyses in the coming decade.

# Software Design

The architecture of \texttt{cloelib} is built around a clear separation of concerns to seamlessly compute theoretical predictions of cosmological observables. It allows wrapping theoretical predictions around common cosmological frameworks such as \texttt{Cobaya} [Cobaya]. \texttt{cloelib} organises functionality into four distinct layers: cosmological backgrounds (e.g., expansion history and distances), perturbation theory (e.g., linear and non-linear matter power spectra), observables (e.g., cosmic shear and photometric galaxy clustering, spectroscopic clustering full shape and post-reconstruction BAO), and summary statistics (e.g., angular power spectra and correlation functions). This layered design allows researchers to flexibly mix and match different theoretical models and numerical approximations, supporting both standard analyses and experimental workflows. For example, users can compute angular power spectra using the Limber approximation with any combination of supported Boltzmann solvers and non-linear models, or define custom window functions for specific survey geometries.

Each layer is defined by a Python protocol (PEP 544), enforcing a "plug-and-play" approach to modularity. Concretely, the library is organised into specialised modules: cosmology backends implementing the \texttt{Background} and \texttt{Perturbations} protocols, observable modules providing window functions and power spectrum interfaces through the \texttt{Tracer} and \texttt{SpectroPower} protocols, summary statistics for angular correlations and Legendre multipoles, and auxiliary utilities for mathematical operations and caching. The \texttt{Perturbations} protocol supports Python mixins to enable modular class composition, allowing users to extend or modify matter power spectrum functionality—such as adding baryonic effects—without altering the underlying implementation.

Performance-critical sections utilise \texttt{jit} compilation, while the caching system avoids redundant evaluations across repeated calculations. In this sense, \texttt{Background}, \texttt{Perturbations}, \texttt{Tracer}, and \texttt{SpectroPower} are structural interfaces that guarantee type safety and extensibility without relying on inheritance hierarchies. Users can include only the components they need, choose among interchangeable backends, and combine them freely—all without altering the core logic of their pipeline. This architecture ensures robustness, reusability, and ease of experimentation by design.

The library integrates with the broader Python scientific ecosystem through \texttt{NumPy} and \texttt{SciPy}, while maintaining full compatibility with JAX arrays for differentiable computations. This enables researchers to construct complex, end-to-end analysis pipelines that are simultaneously computationally efficient, maintainable, and—where needed—fully differentiable.

# Usage Examples

The power of \texttt{cloelib} lies in its intuitive API that allows researchers to quickly set up complex cosmological calculations using this "plug-and-play" approach. Here we demonstrate key features through practical examples.

## Initializing Cosmological Models

\texttt{cloelib} provides a consistent interface for different cosmological backends. Users can instantiate cosmological models using standard parameters:

```python
from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
)
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
)
import numpy as np

# --- Cosmological parameters ---
H0 = 70.0
Omega_cdm0 = 0.25
Omega_b0 = 0.05
As = 2e-9
ns = 0.96
w0 = -1.0
wa = 0.0
Omega_k0 = 0.0
mnu = 0.06
N_mnu = 1
gamma_MG = 0.545

# --- Initialize background objects ---
camb_bg = CAMBBackground(
    H0=H0, Omega_cdm0=Omega_cdm0, Omega_b0=Omega_b0,
    As=As, ns=ns, w0=w0, wa=wa, Omega_k0=Omega_k0,
    mnu=mnu, N_mnu=N_mnu, gamma_MG=gamma_MG
)
jax_bg = JAXBackground(
    H0=H0, Omega_cdm0=Omega_cdm0, Omega_b0=Omega_b0,
    As=As, ns=ns, w0=w0, wa=wa, Omega_k0=Omega_k0,
    mnu=mnu, N_mnu=N_mnu, gamma_MG=gamma_MG
)

# --- Compute background quantities ---
z = np.linspace(0, 3, 256)
H_z_camb = camb_bg.hubble_parameter(z)
H_z__jax = jax_bg.hubble_parameter(z)
```

hiThis demonstrates how different backends can be used interchangeably, allowing for straightforward cross-validation of results.

## Computing Power Spectra

The library supports both linear and non-linear perturbation theories:

```python
# Initialize perturbations

from cloelib.cosmology.camb_cosmology import (
    CAMBLinearPerturbations,
)
from cloelib.cosmology.jax_cosmology import (
    JAXNonLinearPerturbations,
)
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations, HMemuNonLinearPerturbations,
)

# Linear perturbations
camb_linear = CAMBLinearPerturbations(background=camb_bg,
                                      redshifts=z)
hmcode2020_linear = HMemuLinearPerturbations(background=camb_bg, redshifts=z)

# Non-linear perturbations
jax_nonlinear = JAXNonLinearPerturbations(background=jax_bg)
hmcode2020emu_nonlinear = HMemuNonLinearPerturbations(background=camb_bg, linearperturbations=hmcode2020_linear, redshifts=z, log10TAGN=7.8)

# Compute matter power spectra
ks = np.logspace(-4, np.log10(5), 256)
linear_pk_camb = camb_linear.matter_power_spectrum(z, ks)
linear_pk_hmcode2020 = hmcode2020_linear.matter_power_spectrum(z, ks)
nonlinear_pk_jax = jax_nonlinear.matter_power_spectrum(z, ks)
nonlinear_pk_hmcode2020emu = hmcode2020emu_nonlinear.matter_power_spectrum(z, ks)
```

## Photometric Observables

 \texttt{cloelib} provides robust computation of observables for photometric surveys, including galaxy clustering and cosmic shear, with advanced modelling of systematic effects. Specifically, the calculation of shear and position window functions is managed within the `photo` module. Each tracer must be initialised with a `Perturbations`-compatible instance, the galaxy density distribution in redshift bins, and the relevant systematic models, such as intrinsic alignments, galaxy bias, magnification, shear multiplicative bias calibration nuisance parameters, and photometric calibration nuisance parameters. Every computed window function is accessible from each tracer instance. Tracers are combined into two-point summary statistics, both in real and harmonic space, using the functions available in the `summary_statistics` module.

```python
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

# Define galaxy redshift distributions (normalized)
#my_dndz = my_dndz  # Shape: (n_bins, n_z_points)

# Create tracers with survey-specific nuisance parameters
# PositionsTracer requires per-bin photo-z shifts
# and magnification bias
tracer_pos = PositionsTracer(
    perturbations=hmcode2020emu_nonlinear,
    dndz=my_dndz_pos_norm,
    z=zs,
    galaxy_bias_model='poly',
    nuisance_params={
    'b1_photo_poly0': 1.2, 'b1_photo_poly1': 0.0,
    'b1_photo_poly2': 0.0, 'b1_photo_poly3': 0.0,
    'magnification_bias_1': 0.0,
    'magnification_bias_2': 0.0,
    'magnification_bias_3': 0.0,
    'magnification_bias_4': 0.0,
    'magnification_bias_5': 0.0,
    'magnification_bias_6': 0.0,
    'dz_pos_1': 0.0,
    'dz_pos_2': 0.0,
    'dz_pos_3': 0.0,
    'dz_pos_4': 0.0,
    'dz_pos_5': 0.0,
    'dz_pos_6': 0.0,
    'width_pos_1': 1.0,
    'width_pos_2': 1.0,
    'width_pos_3': 1.0,
    'width_pos_4': 1.0,
    'width_pos_5': 1.0,
    'width_pos_6': 1.0,
    }
)

# ShearTracer requires intrinsic alignment (IA)
# and photo-z shift parameters
tracer_she = ShearTracer(
    perturbations=hmcode2020emu_nonlinear,
    dndz=my_dndz_pos_norm,
    z=zs,
    nuisance_params={
    'AIA': 1.72, 'CIA': 0.0134, 'EtaIA':-0.41,
    'multiplicative_bias_1': 0.0,
    'multiplicative_bias_2': 0.0,
    'multiplicative_bias_3': 0.0,
    'multiplicative_bias_4': 0.0,
    'multiplicative_bias_5': 0.0,
    'multiplicative_bias_6': 0.0,
    'dz_shear_1': 0.0,
    'dz_shear_2': 0.0,
    'dz_shear_3': 0.0,
    'dz_shear_4': 0.0,
    'dz_shear_5': 0.0,
    'dz_shear_6': 0.0,
    'width_shear_1': 1.0,
    'width_shear_2': 1.0,
    'width_shear_3': 1.0,
    'width_shear_4': 1.0,
    'width_shear_5': 1.0,
    'width_shear_6': 1.0
    }
)

# Compute angular power spectra using the Limber approximation
twopoint_posshe = AngularTwoPoint(tracer_pos, tracer_she)
twopoint_pospos = AngularTwoPoint(tracer_pos, tracer_pos)
twopoint_sheshe = AngularTwoPoint(tracer_she, tracer_she)


ells = np.arange(2, 3000)
cls_sheshe = twopoint_sheshe.get_Cl(ells, nl=0, ks=ks)
cls_posshe = twopoint_posshe.get_Cl(ells, nl=0, ks=ks)
cls_pospos = twopoint_pospos.get_Cl(ells, nl=0, ks=ks)
cls = {**cls_sheshe, **cls_posshe, **cls_pospos}
```

## Spectroscopic Observables

Calculation of the Legendre multipoles (both in Fourier and configuration space) is handled by the `LegendreMultipoles` module, which interfaces with objects that comply with the `SpectroPower` protocol. This module implements shared modelling layers that are handled coherently by cloelib, rather than relying on individual implementations of external pipelines. Modelled effects include shot-noise corrections, Alcock-Paczynski distortions, and the convolution with the survey window function, as well as a number of observational systematic effects, such as spectroscopic redshift errors and the presence of contaminants. In addition, this module can compute the two-point correlation function and projects it – or  $P(k,\mu)$ – to Legendre multipoles. As an example, we show below how to obtain a prediction for the power spectrum multipoles using the \texttt{comet-emu} package.

```python
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles
import numpy as np

# EFT bias and nuisance parameters for a single redshift bin
RSD_parameters = { # Bias and EFT counterterms
    'b1': 1.8, 'b2': 0.0, 'bG2': 0.0, 'bGam3': 0.0,
    'c0': 0.0, 'c2': 0.0, 'c4': 0.0
}

# Spectroscopic power spectrum at a single effective redshift
spectro_power = CometEFT_SpectroPower(
    background=camb_bg,
    RSD_parameters=RSD_parameters,
    redshift=1.0,
)

# Compute Legendre multipoles with Alcock-Paczynski corrections
noise_systematics_parameters = {
    'NP0': 1.0, 'NP20': 0.0, 'NP22': 0.0,# Shot-noise parameters
    'sigmaz': 0.0, # Redshift error for AP effect
    'fout': 0.0 # Purity
}
nbar = 1e-3  # galaxy number density [h/Mpc]^3
multipoles = LegendreMultipoles(
    spectro_power=spectro_power,
    background_fiducial=camb_bg,
    parameters=noise_systematics_parameters,
    nbar=nbar,
)

k = np.logspace(-2, np.log10(0.5), 80)
Pk_ell = multipoles.power_multipoles(k, ells=np.array([0, 2, 4]))
# Pk_ell is a dict: {'ell0': array, 'ell2': array, 'ell4': array}
```

The same interface is used to compute two-point correlation function multipoles via an FFTLog transform, and to apply convolutions with survey window functions.

## Protocol Compliance of Interfaces

\texttt{cloelib} natively supports Python structural subtyping (PEP 544); the `Background` and `Perturbations` protocols are marked with `@runtime_checkable`, allowing explicit compliance checks at the beginning of an analysis.

```python
from cloelib.cosmology.cosmology import (
    Background,
)
# Protocol compliance is verified at runtime
assert isinstance(camb_bg, Background)
if isinstance(camb_bg, Background):
    print("CAMB background is compliant with the Background protocol.")
```

Because both objects conform to the same protocol, any downstream \texttt{cloelib} computation—such as window functions, angular power spectra, or multipoles—can operate on either without requiring modification. This structural approach, rather than relying on inheritance hierarchies, enables the seamless integration of external codes without altering their source. As a result, it provides a clear pathway for the community to connect their own tools, provided they comply with the protocol.

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

\texttt{cloelib} exhibits performance comparable to other tools available in the community, despite being implemented exclusively in Python. The computational cost of the \texttt{Background}-compatible classes is negligible (effectively instantaneous), whereas the runtime of the \texttt{Perturbation}-compatible classes depends on the choice of backend, namely whether a Boltzmann solver or an emulator is employed.

Below, we provide representative estimates of the computational time required to evaluate key cosmological observables (these tests were run on a MacBook Pro M4 from 2025). These comprise photometric probes—cosmic shear, photometric galaxy clustering, and galaxy–galaxy lensing, the so-called 3x2-pt analysis—calculated in harmonic space (angular power spectra). We also present corresponding estimates for full-shape analyses of spectroscopic galaxy clustering, in Fourier space (Legendre multipoles), utilising \texttt{comet-emu} as the backend. For all cases, \texttt{CAMB} is employed to compute \texttt{Background} quantities, while for photometric probes, \texttt{HMCode2020emu} is used as the \texttt{Perturbations} backend.

For photometric analyses, initialising the shear and position tracer classes—each conforming to the \texttt{Tracer} protocol—typically requires approximately 0.4 seconds to compute the lensing efficiency (utilised for both shear and the magnification systematic effect in the position tracer). This quantity is cached, ensuring that all subsequent calls to either tracer are effectively instantaneous.

The computation of angular power spectra for a 3x2-pt analysis—using 3000 multipole values, 512 $k$-values, 500 redshift $z$-values, and linear galaxy and magnification bias—with six redshift bins (78 spectra in total), using the Limber approximation, takes approximately 0.06 seconds per call. This follows a one-second initialization phase, during which \texttt{jit}-compiled quantities are cached to accelerate subsequent integral evaluations. For a *Euclid*-like final data release configuration as described in Euclid Collaboration: Mellier et al. (2025), with thirteen redshift bins, the computation of 351 angular power spectra for a 3x2-pt analysis requires approximately 0.1 seconds per call after initialization and caching. This represents a significant speed-up compared to the former \texttt{CLOE} software.

For full-shape spectroscopic Legendre multipoles, the computation takes approximately 3 milliseconds for a single redshift bin.

## Performance Profiling

\texttt{cloelib} includes function-level profiling via the \texttt{@profile\_function} decorator, configurable sampling (default interval: 0.001 s) to balance overhead and granularity, and timestamped interactive HTML reports for run-to-run comparison. Profiling can be controlled through environment variables or function calls (including enable/disable and output configuration), and it includes safeguards to avoid redundant profiling in nested decorated calls.

**Usage example:**

```python
from cloelib.profiling import (
  profile_function,
  enable_profiling,
  disable_profiling
)

from cloelib.summary_statistics.angular_correlation_function_wigner import (
    AngularCorrelationFunctionWigner
)

# Define at which angular separation to evaluate the correlation function
theta_rad = np.deg2rad(np.geomspace(5, 100, 20)/60)
ells_integration = np.arange(2, 60000)

# Initialize correlation function calculation object
correlation_function_GG = AngularCorrelationFunctionWigner(twopoint_pospos, ells_integration, hmcode2020emu_nonlinear.k)

#apply the wrapper
get_xi_profiled = profile_function(correlation_function_GG.get_xi)

# Enable the profiling
enable_profiling()

xi_pospos = get_xi_profiled(theta_rad)

# Disable the profiling
disable_profiling()
```

This lightweight profiling infrastructure allows users to optimize their analysis pipelines by understanding where computational time is spent across different backends and observable calculations.

# Documentation

Comprehensive documentation for \texttt{cloelib} is available at [cloe-org.github.io/cloelib/dev/home/](https://cloe-org.github.io/cloelib/dev/home/). The documentation includes detailed API references, installation instructions, explanations about the software structure, and guides for integrating cloelib into your analysis workflows.

For practical examples, example scripts, and interactive tutorials, visit the [cloe-org/playground](https://github.com/cloe-org/playground) repository, which hosts a collection of Jupyter notebooks showcasing typical use cases and advanced features.

\texttt{cloelib} output formats for both photometric and spectroscopic observables are compliant with \texttt{euclidlib}\footnote{\href{https://euclidlib.readthedocs.io/en/latest/}{https://euclidlib.readthedocs.io/en/latest/}} formats, using \texttt{cosmolib}\footnote{\href{https://github.com/astro-ph/cosmolib}{https://github.com/astro-ph/cosmolib}} dataclasses.

# Availability

**Source:** [github.com/cloe-org/cloelib](https://github.com/cloe-org/cloelib)
**License:** MIT
**Install (PyPI):** \texttt{pip install cloelib}
**Documentation:** [https://cloe-org.github.io/cloelib/dev/home/](https://cloe-org.github.io/cloelib/dev/home/)
**\texttt{conda/mamba} environments**  [github.com/cloe-org/cloe-org-environments](https://github.com/cloe-org/cloe-org-environments)
**Examples:** [github.com/cloe-org/playground](https://github.com/cloe-org/playground)

# Author Contributions

In accordance with JOSS guidelines, we describe individual contributions below. Authors are listed in alphabetical order. All Tier 1 authors are core maintainers of the **cloe-org** organisation, responsible for the long-term sustainability of \texttt{cloelib}, the review of pull requests, and leadership of technical discussions.

- **M. Bonici**: Core architecture and protocol design; implementation of the JAX cosmology backends; lensing tracer kernels (including massive neutrino contributions); correlation function module and performance optimisation; caching system with JAX `lax` conditional compatibility; licence and project governance.
 - **G. Cañas-Herrera**: Project overview and release management; core architecture of the software, protocol and class inheritance design; CI pipeline configuration; pre-commit and code-quality tooling; issue and pull-request templates; README, documentation, and community contribution tracking (`all-contributors`); pyproject.toml versioning and release workflows. Developed models for systematic shear and calibration of photometric redshift nuisance parameters. Homogenisation of output format for observables.
- **P. Carrilho**: Photometric observable module linear galaxy bias models with JAX-compatible conditional logic; HMCode2020Emu baryonic feedback support and further extrapolation support; \texttt{CAMB} dark-energy model configuration (PPF); mixing-matrix and pseudo-$C_\ell$ corrections; `interpax`-based interpolation in the extrapolator; growth-rate and matter power spectrum redshift/scale interfaces, implementation of EuclidEmulator2.
- **S. Casas**: Implementation of the \texttt{CLASS} cosmology backend and its integration with the `Background` and `Perturbations` protocols; fixes to transverse-distance computations across \texttt{CAMB}, \texttt{CLASS}, and JAX backends; cosmology protocol refinements; CI pipeline and dependency updates.
- **C. Moretti**: Spectroscopic analysis infrastructure: PBJ interface and RSD power spectrum fixes; BAO $\alpha$-parameter module and Alcock–Paczynski distortion utilities; extraction of $r_\mathrm{drag}$ from the background for BAO analyses; Legendre multipole summary statistics; version management and repository clean-up of deprecated directories.
- **A. Pezzotta**: Spectroscopic analysis infrastructure: \texttt{comet-emu} interface and EFT and VDG spectroscopic power spectrum implementations; survey window-function convolution of power spectrum building blocks; Legendre multipole computation optimisation (`np.einsum`); documentation of the spectroscopic observable interface.

The contributions of all remaining authors have been tracked using the [all-contributors](https://github.com/all-contributors/all-contributors) bot, following the specification of the same name. A full, categorised breakdown of each contributor's role—including code, documentation, testing, ideas, project management, and more—is available in the `README` of the \texttt{cloelib} repository, fully detailed within the \texttt{cloelib} docs.

# Acknowledgements

We thank the broader CLOE software development team for the foundational work that motivated this library. We thank Fabrice Roy for helping deploy the documentation. G.C.H. acknowledges that this project is part of the UNICORN project (file number VI.Veni.242.110) within the Talent Programme Veni Science domain 2024, which is partly financed by the Dutch Research Council (NWO) under grant [https://doi.org/10.61686/ZCPQI32997](https://doi.org/10.61686/ZCPQI32997). M.B. acknowledges support from the Natural Sciences and Engineering Research Council of Canada (NSERC). C.M. is supported by the Agenzia Spaziale Italiana project “Attività scientifica per la missione Euclid – fase E ACCORDO ATTUATIVO n. 2024-10-HH.0.” B.B. is supported by a UK Research and Innovation Stephen Hawking Fellowship (EP/W005654/2). E.B. and K.T. acknowledge support from the European Union’s Horizon Europe research and innovation programme under the Marie Skłodowska-Curie COFUND Postdoctoral Programme (grant agreement No. 101081355 – SMASH), as well as from the Republic of Slovenia and the European Regional Development Fund. A.H. acknowledges the support of a Royal Society University Research Fellowship. A.H.W. is supported by the Deutsches Zentrum für Luft- und Raumfahrt (DLR) under project 50QE2305, funded by the Bundesministerium für Wirtschaft und Klimaschutz, and also acknowledges funding from the German Science Foundation (DFG) via the Collaborative Research Center SFB1491 “Cosmic Interacting Matters – From Source to Signal.” R.K. is supported by UK STFC grant ST/X001040/1.

We acknowledge the EuroHPC Joint Undertaking for awarding project ID EHPC-EXT-2024E02-083 access to Leonardo, hosted by CINECA (Italy). We acknowledge the use of the Spanish Supercomputing Network (RES) resources provided by the Barcelona Supercomputing Center (BSC) on MareNostrum 5 under allocations AECT-2024-3-0020, 2025-1-0045, 2025-2-0046, and 2025-3-0036. We also acknowledge support from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 101053992) for computational resources.

The Euclid Consortium acknowledges the European Space Agency and a number of agencies and institutes that have supported its development, in particular: the Agenzia Spaziale Italiana; the Austrian Forschungsförderungsgesellschaft funded through BMIMI; the Belgian Science Policy; the Canadian Euclid Consortium; the Deutsches Zentrum für Luft- und Raumfahrt; DTU Space and the Niels Bohr Institute (Denmark); the French Centre National d’Études Spatiales; the Fundação para a Ciência e a Tecnologia; the Hungarian Academy of Sciences; the Ministerio de Ciencia, Innovación y Universidades; the National Aeronautics and Space Administration; the National Astronomical Observatory of Japan; the Netherlands Research School for Astronomy; the Norwegian Space Agency; the Research Council of Finland; the Romanian Space Agency; the Swiss Space Office (SSO) at the State Secretariat for Education, Research, and Innovation (SERI); and the United Kingdom Space Agency. A complete and detailed list is available at [www.euclid-ec.org/consortium/community/](https://www.euclid-ec.org/consortium/community/).

# References
