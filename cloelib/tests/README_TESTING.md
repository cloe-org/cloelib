# Testing Guide for CLoELib Cosmology Backends

This guide explains how to run tests for specific cosmology backends (CAMB, CLASS, JAX) in CLoELib.

## Overview

CLoELib supports multiple cosmology backends through a protocol-based design. The test suite is designed to run parametrized tests across all available backends or specific backends of your choice.

## Available Backends

- **CAMB**: Boltzmann code for computing cosmological observables
- **CLASS**: Cosmic Linear Anisotropy Solving System
- **JAX**: JAX-based cosmology implementation for differentiable computing

## Running Tests for Specific Backends

### Command Line Options

Use the `--backend` option to select which backend to test:

```bash
# Run tests for CAMB only
pytest --backend=camb

# Run tests for CLASS only  
pytest --backend=class

# Run tests for JAX only
pytest --backend=jax

# Run tests for all available backends (default)
pytest --backend=all
# or simply
pytest
```

### Marker-Based Selection

Each backend has an associated pytest marker that allows fine-grained test selection:

```bash
# Run only CAMB tests using markers
pytest -m camb

# Run only CLASS tests using markers
pytest -m class

# Run only JAX tests using markers
pytest -m jax

# Run all tests except JAX
pytest -m "not jax"

# Run CAMB or CLASS tests (but not JAX)
pytest -m "camb or class"
```

### Combining with Other Pytest Options

You can combine backend selection with other pytest options:

```bash
# Run CAMB tests with verbose output
pytest --backend=camb -v

# Run CLASS tests for specific test file
pytest --backend=class test_cosmology_protocols.py

# Run JAX tests with coverage
pytest --backend=jax --cov=cloelib

# Run specific test method for CAMB
pytest --backend=camb test_cosmology_protocols.py::TestBackgroundMethods::test_omega_m
```

## Handling Missing Dependencies

If a requested backend is not installed or available, the tests will be automatically skipped with an informative message:

```bash
$ pytest --backend=camb
# If CAMB is not installed:
# SKIPPED [1] conftest.py:136: Backend 'camb' is not available or not installed
```

## Checking Available Backends

To see which backends are currently available on your system, you can run a simple Python check:

```python
# From the cloelib/tests/ directory
import conftest
implementations = conftest.get_backend_implementations()
print("Available backends:", list(implementations.keys()))
```

## Test Structure

The test suite includes several types of tests:

1. **Protocol Compliance Tests** (`test_cosmology_protocols.py`):
   - Verify backends implement required interfaces
   - Test method signatures and return types
   - Validate protocol adherence

2. **Backend-Specific Tests**:
   - `test_camb_cosmology.py` - CAMB-specific functionality
   - `test_class_cosmology.py` - CLASS-specific functionality  
   - `test_jax_cosmology.py` - JAX-specific functionality

3. **Cross-Backend Validation Tests** (`test_migration_validation.py`):
   - Compare outputs between different backends
   - Ensure consistency across implementations

## Examples

### Development Workflow

When developing or debugging a specific backend:

```bash
# Focus on CAMB development
pytest --backend=camb -v

# Run only background tests for CAMB
pytest --backend=camb -k "background" -v

# Run with immediate failure on first error
pytest --backend=camb -x
```

### Continuous Integration

For CI/CD pipelines that may have different backend availability:

```bash
# Test all available backends (gracefully handles missing ones)
pytest --backend=all

# Test specific backends if available, skip if not
pytest --backend=camb || true
pytest --backend=class || true  
pytest --backend=jax || true
```

### Performance Testing

Test specific backends for performance characteristics:

```bash
# Run JAX tests (typically faster due to JIT compilation)
pytest --backend=jax --benchmark-only

# Compare performance between backends
pytest -m "camb or class" --benchmark-compare
```

## Configuration

The test configuration is managed through:

- `conftest.py`: Shared fixtures and backend discovery logic
- `pytest.ini`: Pytest configuration and marker registration
- Environment variables or config files for backend-specific settings

### Fixture Scoping Considerations

The test suite uses specific fixture scopes to balance test isolation and performance:

- **Module-scoped fixtures** (`backend_classes`, `background_instance`, `perturbation_instances`): 
  - Created once per test module (file) rather than per test function
  - Provides good test isolation (different modules get fresh instances)
  - Significantly reduces backend initialization overhead
  - Safe for read-only operations which dominate the test suite
  
- **Session-scoped fixtures** (`standard_cosmology`, `test_redshifts`, `test_wavenumbers`):
  - Created once per test session for efficiency
  - Used for immutable test data that doesn't vary between backends
  - Safe to reuse across all test functions

**Important**: When developing new fixtures that depend on `backend_classes`, use module scope for performance while maintaining adequate isolation. Parametrized fixtures should never use session scope as it can cause fixture reuse problems when running tests with different backend selections.

#### Performance Considerations

The fixture scoping has significant performance implications:

- **Module scope** (current): ~0.5s initialization overhead
  - Tests within the same module share backend instances
  - Acceptable since most operations are read-only
  - Reduces total test runtime by ~90%
  
- **Function scope** (previous): ~4.4s initialization overhead  
  - Each of the 93 tests creates fresh backend instances
  - Necessary only if tests modify backend state
  - Causes significant performance degradation

For complete test isolation, users can still run individual test files or specific tests, which will create fresh instances. The module scope provides the optimal balance between performance and isolation for the current test suite.

## Troubleshooting

### Common Issues

1. **Backend not available**: Ensure the backend package is installed
   ```bash
   pip install camb  # for CAMB
   pip install classy # for CLASS
   pip install jax    # for JAX
   ```

2. **Import errors**: Check that backend modules are properly installed and importable

3. **Test failures**: Use `-v` flag for verbose output to identify specific issues

### Debug Mode

Run tests with maximum verbosity and immediate failure:

```bash
pytest --backend=camb -v -s -x --tb=long
```

This provides detailed output for debugging failing tests.