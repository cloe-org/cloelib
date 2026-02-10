# Code Structure: Guide to cloelib's Architecture

This guide explains the modular architecture of **cloelib** and provides instructions for contributors to extend the library by adding new implementations of Perturbations and Observables.

## Overview

**cloelib** follows a layered architecture that separates cosmological calculations into distinct components:

```
Background → Perturbations → Observables → Summary Statistics
```

Each layer depends on the previous one, creating a flexible pipeline from fundamental cosmology to final data products.

### Design Principles

**Modularity**: The architecture allows easy swapping of implementations (e.g., CAMB for CLASS).

**Flexibility**: New components can be added by implementing the appropriate protocol.

**Reproducibility**: Clear interfaces ensure consistent behavior across implementations.

**Extensibility**: Protocol-based design enables contributors to extend the library without modifying existing code.

## Core Components

### 1. [Background](background.md)

The Background module provides the cosmological foundation, computing distances, Hubble parameters, and matter densities as functions of redshift.

**Purpose**: Compute background quantities as functions of redshift

**Dependencies**: None (foundational layer)

**Use case**: Implementing new Boltzmann solvers or emulators

[Learn more about Background](background.md)

### 2. [Perturbations](perturbations.md)

The Perturbations module computes structure formation quantities including matter power spectra, growth factors, and growth rates.

**Purpose**: Calculate perturbation theory quantities

**Dependencies**: Background module

**Use case**: Adding non-linear models or new structure formation codes

[Learn more about Perturbations](perturbations.md)

### 3. [Observables](observables.md)

The Observables module connects theoretical predictions to survey measurements, handling selection functions, biases, and window functions.

**Purpose**: Compute survey-specific observables

**Dependencies**: Perturbations (for tracers) or Background (for spectroscopic)

**Use case**: Adding new measurement types or survey configurations

[Learn more about Observables](observables.md)

### 4. [Summary Statistics](summary_statistics.md)

The Summary Statistics module produces final data products for comparison with observations, including angular power spectra, correlation functions, and multipoles.

**Purpose**: Compute final statistical quantities

**Dependencies**: Observables module

**Use case**: Implementing new statistical estimators

[Learn more about Summary Statistics](summary_statistics.md)

## Typical Workflow

The standard workflow for computing observables follows this pattern:

```python
# 1. Initialize background cosmology
background = CAMBBackground(H0=67.5, Omega_b0=0.049, ...)

# 2. Initialize perturbations with background
perturbations = CAMBPerturbations(background=background, ...)

# 3. Define observables with perturbations or background
tracer = ShearTracer(perturbations=perturbations, dndz=..., z=..., ...)

# 4. Compute summary statistics
two_point = AngularTwoPoint(tracer1=tracer, tracer2=tracer)
C_ell = two_point.compute_Cl(ells=...)
```

## For Contributors

Each module page includes:

- Protocol definitions specifying required methods and properties
- Step-by-step guides for adding new implementations
- Code examples demonstrating proper usage
- Interface specifications for connecting with external codes
- Testing recommendations

The protocol-based design means contributors only need to implement the required methods without inheriting from base classes or understanding the entire codebase.

## Design Philosophy

### Protocol-Based Design

**cloelib** uses Python protocols (PEP 544) instead of traditional inheritance:

- **Type safety**: Static type checkers can verify implementations satisfy the protocol
- **Flexibility**: Any class implementing required methods is valid
- **Clear contracts**: Protocols explicitly document requirements

### Separation of Concerns

Each module has a specific responsibility:

- **Background**: Pure cosmology (no structure formation)
- **Perturbations**: Structure growth (no survey details)
- **Observables**: Survey specifics (no final statistics)
- **Summary Statistics**: Final products (no cosmology details)

This separation improves code maintainability, testability, and extensibility.

### JAX Compatibility

Most implementations support both NumPy and JAX arrays:

- **Automatic differentiation**: Enables gradient computation for parameter inference
- **GPU acceleration**: Allows scaling to larger problems
- **JIT compilation**: Provides improved performance
- **NumPy compatibility**: Maintains familiar interface

## Additional Resources

- [API Reference](../api.md): Detailed technical documentation
- [Contributing Guide](../contributing.md): General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground): Usage demonstrations

## Support

For questions about the code structure or implementing new components:

- [GitHub Issues](https://github.com/cloe-org/cloelib/issues): Report bugs or request features
- [GitHub Discussions](https://github.com/cloe-org/cloelib/discussions): Ask questions and share ideas
- Tag `@cloe-maintainers` for assistance from the core team
