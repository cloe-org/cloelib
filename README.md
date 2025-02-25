# cloelib

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Library of models for the CLOE (Cosmology Likelihood for Observables in Euclid) org.  This package is made efficient and user-friendly by transforming it into the Cosmology Library for Observables in Euclid (compatible with classic Boltzmann Solvers and JAX). This is a work in progress, and will benefit from the feedback of the Euclid community (and the whole cosmology community in general).

## Features
The main features of `cloelib` are:
- **User-friendly approach**: plot Euclid-like observables and other products (theoretical predictions for photometric and spectroscopic functions, window tracers, power spectra) in 3 minutes.
- **Automatic differentiation**: `cloelib` includes a toy-example of how to use autodiff _à la `Jax`_ to allow for gradient computations.
- **Extra modularity**: the class hierarchy of `cloelib` allows to import your external emulator or Boltzmann solver easily. It realies on python `Protocols` to interface external codes.  

The main structure of `cloelib` is based on:
- Preparing your Cosmology using **Background & Perturbations** protocols
- Decide which observables you would like to compute using **Tracer** or **SpectroPower** protocols.
- Compute final summary statistics like angular power spectra or Legendre Multiples.

## Installation
Clone the repository `main` branch and pip install it:
   ```sh
   pip install .
   ```

### Prerequisites
- `python`, `jax`, `jaxlib`, `interpax`

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/gcanasherrera/cloelib.git
   ```
2. Navigate to the project directory:
   ```sh
   cd cloelib
   ```

## Usage
Check out the `notebooks` folder to see how to compute observables and other quantities.

## Contributing
If you would like to contribute, follow the steps below:

1. Open an issue to let the `cloelib` maintainers know about your contribution plans
2. Fork the repository
3. Create a new branch:
   ```sh
   git checkout -b feature/your-feature-name
   ```
4. Commit your changes:
   ```sh
   git commit -m 'Add some feature'
   ```
5. Push to the branch:
   ```sh
   git push origin feature/your-feature-name
   ```
6. Open a pull request

## License
This project is licensed under the GLG License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This project relies on previous work by the Euclid Consortium and the `jaxcosmo` project.
