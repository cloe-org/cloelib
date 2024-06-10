# cloelite

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
Proof of concept of how to make CLOE (Cosmology Likelihood for Observables in Euclid) more efficient and user-friendly by transforming it into the Cosmology Library for Observables in Euclid (compatible with classic Boltzmann Solvers and JAX).
This is a work in progress, and will benefit from the feedback of the Euclid community (and the whole cosmology community in general). 

`cloelite` does not contain all the main features actual `cloe` has, and it should be seen as a toy repository to test further functionalities currently not implemented in `cloe` (and that would be painful to get with its current architecture).

## Features
The main features of `cloelite` vs. `cloe` are:
- **Independence from `Cobaya`**: `cloelite` computes the Euclid observables without making an evaluation of the posterior distribution, nor importing `Cobaya` at all.
- **User-friendly approach**: plot Euclid-like quantities (theoretical predictions for observables, window tracers, power spectra) in 3 minutes.
- **Automatic differentiation**: `cloelite` includes a toy-example of how to use autodiff _à la `Jax`_ to allow for gradient computations.
- **Extra modularity**: the class hierarchy of `cloelite` allows to import your external emulator or Boltzmann solver easily.

The main differences with respect to `cloe` are:
- **Background & Perturbations vs. former cosmology.py**
- **Introduction of tracers**
- **Full integration of the nonlinear models in the code**

## Installation
Clone the repository `main` branch and pip install it:
   ```sh
   pip install .
   ```

### Prerequisites
- TBA

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/gcanasherrera/cloelite.git
   ```
2. Navigate to the project directory:
   ```sh
   cd cloelite
   ```

## Usage
TBA


## Contributing
If you would like to contribute, follow the steps below:

1. Open an issue to let the `cloelite` maintainers know about your contribution plans
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

See the [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This project highly benefits from all the contributions of the amazing members of Euclid Consortium IST:Likelihood and IST:Nonlinear (if you are Euclidean and you are playing with this, do not panic! we don't aim to take credit whatsoever about this, as we just wanted to demostrate that better times can actually be ahead of us!)
- This project is partially inspired by the great work of [`jax-cosmo`](https://jax-cosmo.readthedocs.io/en/latest/) people (in particular, the `Background` and `Perturbations` classes)
